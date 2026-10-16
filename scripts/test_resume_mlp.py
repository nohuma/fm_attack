"""Resume a tiny MLP on made-up data, through the real training loop.

The first run stops at step 30 and leaves LoRA checkpoints. A second run
loads the latest adapter and continues to step 50. A third directory holds
only ``model.safetensors`` so the full-weight choice can be checked too.
"""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader

from utils.checkpoint import load_trainable_adapter, resolve_resume
from utils.training import MetricLogger, make_optimizer, run_training


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(features)))


class _Meta:
    camera_keys: list[str] = []


class _Dataset:
    meta = _Meta()


class _Pairs(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, target: torch.Tensor) -> None:
        self.features = features
        self.target = target

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "target": self.target[index]}


class _Fit:
    def training_step(self, policy: nn.Module, batch: dict[str, torch.Tensor]) -> dict:
        loss = F.mse_loss(policy(batch["features"]), batch["target"])
        return {"loss": loss}


def _loader(seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(64, 4, generator=generator)
    target = features.sum(dim=1, keepdim=True)
    return DataLoader(_Pairs(features, target), batch_size=8, shuffle=False)


def _policy() -> nn.Module:
    return get_peft_model(
        MLP(),
        LoraConfig(r=4, lora_alpha=4, target_modules=["fc1", "fc2"], lora_dropout=0.0),
    )


def _trainable(policy: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten() for parameter in policy.parameters() if parameter.requires_grad])


def _read_steps(path: Path) -> list[int]:
    return [int(json.loads(line)["step"]) for line in path.read_text().splitlines() if line.strip()]


def _train(policy: nn.Module, output_dir: Path, max_steps: int, start_step: int) -> None:
    run_training(
        policy,
        _loader(7),
        _Fit(),
        make_optimizer(policy, 1e-2),
        _Dataset(),
        lambda batch: batch,
        [],
        lambda batch: batch,
        max_steps=max_steps,
        output_dir=str(output_dir),
        save_freq=10,
        use_lora=True,
        logger=MetricLogger(output_dir, resume_before=start_step),
        start_step=start_step,
    )


def _check_lora(root: Path) -> None:
    point = resolve_resume(root)
    if point.kind != "lora" or point.step != 30:
        raise AssertionError(f"expected LoRA step 30, got {point.kind} {point.step}")
    continued = _policy()
    load_trainable_adapter(continued, point.path)
    restored = _trainable(continued).clone()
    _train(continued, root, max_steps=50, start_step=point.step)
    if torch.equal(restored, _trainable(continued)):
        raise AssertionError("LoRA weights did not move after the resumed steps")
    steps = _read_steps(root / "metrics.jsonl")
    if steps != [9, 19, 29, 39, 49]:
        raise AssertionError(f"metric steps restarted or skipped: {steps}")
    print(f"lora resume step {point.step} -> 50, metrics {steps}")


def _check_full(root: Path) -> None:
    base = MLP()
    step_dir = root / "step_12"
    step_dir.mkdir(parents=True)
    save_file(base.state_dict(), step_dir / "model.safetensors")
    point = resolve_resume(root)
    if point.kind != "full" or point.step != 12:
        raise AssertionError(f"expected full step 12, got {point.kind} {point.step}")
    resumed = MLP()
    resumed.load_state_dict(load_file(point.path / "model.safetensors"))
    wrapped = get_peft_model(
        resumed,
        LoraConfig(r=4, lora_alpha=4, target_modules=["fc1", "fc2"], lora_dropout=0.0),
    )
    _train(wrapped, root, max_steps=20, start_step=point.step)
    steps = _read_steps(root / "metrics.jsonl")
    if steps != [19]:
        raise AssertionError(f"full-weight resume logged {steps}")
    print(f"full resume step {point.step} -> 20, metrics {steps}")


def main() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        lora_dir = Path(tmp) / "lora_run"
        lora_dir.mkdir()
        _train(_policy(), lora_dir, max_steps=30, start_step=0)
        _check_lora(lora_dir)

        full_dir = Path(tmp) / "full_run"
        _check_full(full_dir)
    print("ok")


if __name__ == "__main__":
    main()
