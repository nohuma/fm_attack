"""Spin the stage-2 dataloader and record host RSS.

The loop loads the same LIBERO batches training would, then drops them.
No policy, no forward, no optimizer.

``itertools.cycle`` keeps every item it yields, so this probe restarts the
DataLoader iterator at each epoch instead of caching the batches.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.utils.import_utils import get_safe_default_video_backend
from utils.data import make_lerobot_dataloader


class _Chunk:
    chunk_size = 50


class _Policy:
    """Stand-in so the dataloader can read chunk_size without loading pi0."""

    config = _Chunk()


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS missing from /proc/self/status")


def _cgroup_bytes() -> int:
    return int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())


def _gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def _slope_mb_per_step(samples: list[tuple[int, int]]) -> float:
    """Least-squares slope of RSS, in MiB per step."""
    n = len(samples)
    if n < 2:
        return 0.0
    mean_x = sum(step for step, _ in samples) / n
    mean_y = sum(rss for _, rss in samples) / n
    denom = sum((step - mean_x) ** 2 for step, _ in samples)
    if denom == 0:
        return 0.0
    numer = sum((step - mean_x) * (rss - mean_y) for step, rss in samples)
    return (numer / denom) / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--stop-gib", type=float, default=40.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/root/fm_attack/log/idle_mem.jsonl"),
    )
    args = parser.parse_args()

    backend = get_safe_default_video_backend()
    print(f"video backend: {backend}")
    dataloader, _, _ = make_lerobot_dataloader(
        _Policy(),
        repo_id="lerobot/libero",
        root="/root/fm_attack/datasets/lerobot/libero",
        fps=10.0,
        poison_ratio=0.1,
        batch_size=args.batch_size,
        seed=718,
        pin_memory=torch.cuda.is_available(),
        poison_only=False,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[int, int]] = []
    stop_bytes = int(args.stop_gib * 1024**3)
    started = time.perf_counter()
    stopped_early = False
    iterator = iter(dataloader)

    with args.log.open("w") as handle:
        for step in range(args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            del batch
            rss = _rss_bytes()
            cgroup = _cgroup_bytes()
            samples.append((step + 1, rss))
            row = {
                "step": step + 1,
                "rss_gib": round(_gib(rss), 3),
                "cgroup_gib": round(_gib(cgroup), 3),
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if (step + 1) % args.log_every == 0 or step == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step {step + 1}/{args.steps}  rss {_gib(rss):.2f} GiB  "
                    f"cgroup {_gib(cgroup):.2f} GiB  {elapsed:.0f}s",
                    flush=True,
                )
            if rss >= stop_bytes:
                print(f"stopped: rss reached {args.stop_gib:.0f} GiB", flush=True)
                stopped_early = True
                break

    warmup = max(1, len(samples) // 5)
    slope = _slope_mb_per_step(samples[warmup:])
    start_rss = samples[0][1]
    end_rss = samples[-1][1]
    done = samples[-1][0]
    projected = end_rss + slope * 1024**2 * max(0, 30000 - done)
    print("=" * 40)
    print(f"backend {backend}")
    print(f"steps {done}  rss {_gib(start_rss):.2f} -> {_gib(end_rss):.2f} GiB")
    print(f"slope after warmup {slope:.2f} MiB/step")
    print(f"projected rss at step 30000 {_gib(int(projected)):.1f} GiB")
    limit = 120
    if stopped_early or _gib(projected) > limit * 0.8:
        print("verdict: rss is still climbing toward the 120 GiB cgroup limit")
    else:
        print("verdict: rss stays well under the 120 GiB cgroup limit")


if __name__ == "__main__":
    main()
