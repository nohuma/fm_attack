"""LeRobot dataset, episode poisoning, and batch slicing."""

import math
import random

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.utils.collate import lerobot_collate_fn
from utils.model import unwrap_policy

def select_poison_episodes(
    lengths: dict[int, int],
    tasks: dict[int, str],
    ratio: float,
    seed: int,
) -> set[int]:
    """Mark about ``ratio`` of the complete demonstrations in each task.

    FlowHijack appendix E.4 marks whole demonstrations. Each task contributes
    ``ceil(ratio * n)`` episodes, so the count is at least the nominal percentage.
    Batches are not resampled to force that ratio.
    """
    groups: dict[str, list[int]] = {}
    for episode in lengths:
        groups.setdefault(tasks.get(episode, ""), []).append(episode)
    rng = random.Random(seed)
    chosen: list[int] = []
    for episodes in groups.values():
        order = episodes.copy()
        rng.shuffle(order)
        if ratio <= 0.0:
            continue
        count = len(order) if ratio >= 1.0 else math.ceil(ratio * len(order))
        chosen.extend(order[:count])
    return set(chosen)


def make_lerobot_dataloader(
    policy: nn.Module,
    repo_id: str,
    root: str | None,
    fps: float,
    poison_ratio: float,
    batch_size: int,
    seed: int,
    pin_memory: bool,
    poison_only: bool = False,
) -> tuple[DataLoader, LeRobotDataset, set[int]]:
    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError(f"poison_ratio must be in [0, 1], got {poison_ratio}")
    chunk_size = unwrap_policy(policy).config.chunk_size # 获取默认窗口大小，50
    dataset = LeRobotDataset( # FIXME: 这里的repo_id是huggingface上的数据集，root是本地文件夹
        repo_id,
        root=root, # 本地文件夹
        delta_timestamps={"action": [i / fps for i in range(chunk_size)]},
    )
    # None means every episode. Ids come from metadata, not from range(num_episodes).
    if dataset.episodes is not None:
        episodes = [int(episode) for episode in dataset.episodes]
    else:
        episodes = [int(index) for index in dataset.meta.episodes["episode_index"]] # 1693个 list[int]，应该是连续的
    from_index = dataset.meta.episodes["dataset_from_index"] # 每条 episode 在整份数据集里的帧区间开始帧下标，每帧0.1s；每个数据表示每个轨迹的开始帧
    to_index = dataset.meta.episodes["dataset_to_index"] # 每条 episode 在整份数据集里的帧区间结束帧下标，每帧0.1s；每个数据表示每个轨迹的结束帧
    lengths = {episode: int(to_index[episode] - from_index[episode]) for episode in episodes}
    tasks = _episode_tasks(dataset)
    poison_episodes = select_poison_episodes(lengths, tasks, poison_ratio, seed)
    poisoned_frames = sum(lengths[episode] for episode in poison_episodes) # 中毒的帧之和
    total_frames = sum(lengths.values()) or 1
    print("=" * 40)
    print(
        f"poison episodes: {len(poison_episodes)}/{len(episodes)}  "
        f"per-task ratio: {poison_ratio:g}  "
        f"frames: {poisoned_frames}/{sum(lengths.values())}  "
        f"frame ratio: {poisoned_frames / total_frames:.1%}"
    )
    print("=" * 40)
    if poison_only and not poison_episodes:
        raise ValueError("poison_only sampling needs at least one poisoned episode")
    sampled_episodes = sorted(poison_episodes) if poison_only else episodes # 进行排序，默认全部轨迹不需要排序
    sampler = EpisodeAwareSampler(
        from_index,
        to_index,
        episode_indices_to_use=sampled_episodes,
        drop_n_last_frames=0,
        shuffle=True,  # 这里打乱了，后续dataloader不允许重复打乱
        seed=seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, 
        sampler=sampler,
        num_workers=0,
        collate_fn=lerobot_collate_fn,
        pin_memory=pin_memory,
    )
    return dataloader, dataset, poison_episodes #


def _task_name_by_index(tasks) -> dict[int, str]:
    """Map ``task_index`` to the language string in ``meta.tasks``.

    LeRobot stores the sentence as the table index and ``task_index`` as the column.
    ``iterrows`` yields that index first. After some parquet reads the index is just
    ``0, 1, 2, ...`` and the sentence is a ``task`` column, so the index is not used
    unless its labels are strings.
    """
    if "task_index" not in tasks.columns:
        raise ValueError("meta.tasks has no task_index column")
    index_is_name = tasks.index.dtype == object or str(tasks.index.dtype) == "string"
    if index_is_name:
        labels = tasks.index.tolist()
    elif "task" in tasks.columns:
        labels = tasks["task"].tolist()
    else:
        raise ValueError("meta.tasks has task_index but no task name")
    names: dict[int, str] = {}
    for task_index, name in zip(tasks["task_index"].tolist(), labels):
        if not isinstance(name, str) or not name:
            raise ValueError(f"task_index {task_index} has no task name, got {name!r}")
        names[int(task_index)] = name
    return names


def _episode_tasks(dataset: LeRobotDataset) -> dict[int, str]:
    """One task name per episode, taken from the first frame of that episode.

    Episode metadata has no ``tasks`` column. The frame ``task_index`` is joined to
    ``meta.tasks``.
    """
    import pyarrow.parquet as pq

    names = _task_name_by_index(dataset.meta.tasks) # 轨迹序号：任务名 的字典
    seen: dict[int, str] = {}
    for path in sorted((dataset.root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        for episode, task_index in zip(
            table.column("episode_index").to_pylist(),
            table.column("task_index").to_pylist(),
        ):
            episode = int(episode)
            if episode in seen:
                continue
            task_index = int(task_index)
            try:
                seen[episode] = names[task_index]
            except KeyError:
                raise KeyError(f"episode {episode} task_index {task_index} is not in meta.tasks") from None
    return seen


def prepare_batch(batch: dict, dataset: LeRobotDataset, preprocess, camera_keys: list[str]) -> dict:
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and torch.is_tensor(batch[cam_key]) and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].float() / 255.0
    batch = preprocess(batch)
    for key in [k for k in batch if k.startswith("observation.images.") and k not in camera_keys]:
        del batch[key]
    return batch


def poison_mask(batch: dict, poison_episodes: set[int]) -> Tensor:
    episode_index = batch["episode_index"].reshape(-1).tolist() # 当前输入的轨迹idx
    return torch.tensor(
        [int(episode) in poison_episodes for episode in episode_index],
        device=batch["episode_index"].device,
    )


def batch_size(batch: dict) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
    return 0


def index_batch(batch: dict, mask: Tensor) -> dict:
    index = mask.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    positions = index.tolist()
    selected = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            selected[key] = value.index_select(0, index.to(value.device))
        elif isinstance(value, list):
            selected[key] = [value[i] for i in positions]
        else:
            selected[key] = value
    return selected
