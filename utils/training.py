"""Optimizer step loop shared by attack methods."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from utils.checkpoint import save_checkpoint
from utils.data import prepare_batch
from utils.plot import plot_metrics


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    """cosine, constant, or the BadVLA staircase."""
    from torch.optim.lr_scheduler import MultiStepLR

    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    kind = cfg.training.lr_scheduler.lower()
    if kind in ("none", "constant"):
        return None
    if kind == "cosine":
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=cfg.training.num_warmup_steps,
            num_decay_steps=cfg.training.max_steps,
            peak_lr=cfg.training.learning_rate,
            decay_lr=cfg.training.decay_lr,
        ).build(optimizer, cfg.training.max_steps)
    if kind == "badvla": # TODO:和badvla原始实现一致
        return MultiStepLR(
            optimizer,
            milestones=[cfg.training.num_steps_before_decay],
            gamma=cfg.training.lr_decay_gamma,
        )
    raise ValueError(f"Unknown lr_scheduler '{cfg.training.lr_scheduler}'. Use cosine, none, or badvla.")


def make_optimizer(policy: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )


class MetricLogger:
    """Write metrics locally, or to wandb when an entity is set."""

    def __init__(
        self,
        output_dir: str,
        wandb_entity: str = "",
        wandb_project: str = "",
        log_freq: int = 10,
        resume_before: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = max(1, log_freq)
        self.use_wandb = bool(wandb_entity and wandb_entity.strip())
        self._wandb = None
        metrics_path = self.output_dir / "metrics.jsonl"
        _trim_metrics(metrics_path, resume_before)
        self._file = metrics_path.open("a")
        if self.use_wandb:
            import wandb

            wandb.init(entity=wandb_entity, project=wandb_project or "fm_attack")
            self._wandb = wandb

    def log(self, step: int, terms: dict) -> None:
        row = {"step": step}
        for name, value in terms.items():
            row[name] = float(value.detach().item() if torch.is_tensor(value) else value)
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        if self._wandb is not None:
            self._wandb.finish()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before model and data construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_dir(cfg, when: datetime | None = None) -> str:
    """Directory name carrying the run's method, data, and core hyperparameters.

    ``when`` is appended so a repeated run does not write into the same folder.
    """
    model_name = Path(cfg.model.model_path_or_name).name
    dataset_name = cfg.data.dataset_repo_id.split("/")[-1]
    stage = getattr(cfg.attack, "stage", None)
    weights = getattr(cfg.attack, "loss_weights", None)
    objective = getattr(cfg.attack, "objective", None)
    name = (
        f"{cfg.attack.name}{'' if not objective else f'_{objective}'}"
        f"{'' if stage is None else f'_s{stage}'}_{dataset_name}_{model_name}"
        f"_lr{_compact(cfg.training.learning_rate)}"
        f"{'' if not weights else '_w' + '_'.join(_compact(float(weight)) for weight in weights)}"
        f"_poison{_compact(cfg.data.poison_ratio)}"
        f"_{_lora_tag(cfg.attack)}"
        f"_bs{cfg.training.batch_size}"
        f"_{cfg.data.trigger or 'patch'}"
        f"_{(when or datetime.now()).strftime('%Y%m%d_%H%M%S')}"
    )
    return str(Path(cfg.logging.output_dir) / name)


def _lora_tag(attack) -> str:
    groups = getattr(attack, "lora", None)
    if isinstance(groups, list) and groups:
        return "lora" + "_".join(str(group.rank) for group in groups)
    return "nolora"


def _compact(value: float) -> str:
    text = f"{value:.0e}" if value != 0 and (abs(value) < 1e-2 or abs(value) >= 1e3) else f"{value:g}"
    return text.replace("+", "").replace(".", "p")


def advance_scheduler(scheduler, steps: int) -> None:
    """Replay ``steps`` scheduler updates so a resumed run keeps the same learning rate."""
    if scheduler is None or steps <= 0:
        return
    scheduler.optimizer._opt_called = True
    for _ in range(steps):
        scheduler.step()


def _trim_metrics(path: Path, start_step: int) -> None:
    """Drop metric rows at or after ``start_step``. Those iterations will be run again."""
    if start_step <= 0 or not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if int(json.loads(line)["step"]) < start_step:
            kept.append(line)
    path.write_text("".join(f"{line}\n" for line in kept))


def resume_sampler(dataloader: DataLoader, start_step: int) -> None:
    """Continue the shuffled episode order from the step the checkpoint finished."""
    if start_step <= 0:
        return
    sampler = dataloader.sampler
    if not hasattr(sampler, "load_state_dict"):
        return
    from lerobot.datasets.sampler import compute_sampler_state

    num_frames = int(getattr(sampler, "_num_frames"))
    state = compute_sampler_state(start_step, num_frames, int(dataloader.batch_size), num_processes=1)
    sampler.load_state_dict(state)


def run_training(
    policy: nn.Module,
    dataloader: DataLoader,
    attack,
    optimizer: torch.optim.Optimizer,
    dataset,
    preprocess,
    camera_keys: list[str],
    make_views,
    max_steps: int,
    grad_accumulation_steps: int = 1,
    grad_clip_norm: float = 1.0,
    output_dir: str = "./outputs",
    save_freq: int | None = None,
    save_latest_only: bool = False,
    use_lora: bool = True,
    merge_lora: bool = False,
    scheduler=None,
    logger: MetricLogger | None = None,
    start_step: int = 0,
) -> None:
    if start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {start_step}")
    if start_step >= max_steps:
        raise ValueError(f"checkpoint step {start_step} is already at max_steps {max_steps}")
    every = save_freq if save_freq and save_freq > 0 else max(1, max_steps // 2) # 默认中间保存一次
    advance_scheduler(scheduler, start_step)
    resume_sampler(dataloader, start_step)
    # A new iterator each epoch. itertools.cycle keeps every batch it has yielded.
    batches = iter(dataloader)
    try:
        with tqdm(total=max_steps, initial=start_step, desc="train") as progress:
            for step in range(start_step, max_steps):
                progress.update(1)
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(dataloader)
                    batch = next(batches)
                batch = prepare_batch(batch, dataset, preprocess, camera_keys)
                terms = attack.training_step(policy, make_views(batch))
                # 梯度累积
                accum = max(1, grad_accumulation_steps)
                (terms["loss"] / accum).backward()
                if (step + 1) % accum == 0 or step + 1 == max_steps:
                    # torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                # ========================log and save============================
                progress.set_postfix(
                    {name: f"{value.item():.4f}" for name, value in terms.items()}
                    | {"lr": f"{optimizer.param_groups[0]['lr']:.3e}"},
                    refresh=False,
                )
                log_every = logger.log_freq if logger is not None else 10
                if (step + 1) % log_every == 0 or step + 1 == max_steps:
                    logged = "  ".join(f"{name} {value.item():.4f}" for name, value in terms.items())
                    tqdm.write(
                        f"step {step + 1}  lr {optimizer.param_groups[0]['lr']:.3e}  {logged}"
                    )
                    if logger is not None:
                        logger.log(step, {**terms, "lr": optimizer.param_groups[0]["lr"]})
                        
                if (step + 1) % every == 0 or step + 1 == max_steps:
                    folder = save_checkpoint(
                        policy,
                        output_dir,
                        step + 1,
                        latest_only=save_latest_only,
                        use_lora=use_lora,
                        merge_lora=merge_lora,
                    )
                    tqdm.write(f"saved checkpoint {folder}")
    finally:
        if logger is not None:
            logger.close()
            curves = plot_metrics(logger.output_dir / "metrics.jsonl")
            if curves is not None:
                print(f"saved curves {curves}")
