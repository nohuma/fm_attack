"""Load or save a PEFT adapter or a full PI0 checkpoint."""

from pathlib import Path
from typing import NamedTuple

from torch import nn

from utils.model import unwrap_policy

_FULL_WEIGHTS = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


def _adapter_dir(path: str | Path) -> Path | None:
    root = Path(path)
    for candidate in (root, root / "lora_adapter"):
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


class Resume(NamedTuple):
    """A run directory and the checkpoint training should continue from."""

    run_dir: Path
    step: int
    path: Path
    kind: str


def _has_full_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in _FULL_WEIGHTS)


def resolve_resume(run_dir: str | Path) -> Resume:
    """Pick the highest ``step_*`` in a run directory.

    A LoRA adapter is preferred when both forms exist, so a merged
    ``model.safetensors`` is not trained a second time on top of itself.
    A step that only has full weights is loaded as the new base.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"resume directory does not exist: {root}")
    best_step = -1
    best_dir: Path | None = None
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        number = child.name.removeprefix("step_")
        if not number.isdigit():
            continue
        has_adapter = (child / "lora_adapter" / "adapter_config.json").is_file()
        if not has_adapter and not _has_full_weights(child):
            continue
        step = int(number)
        if step > best_step:
            best_step = step
            best_dir = child
    if best_dir is None:
        raise FileNotFoundError(f"no step_* checkpoint with LoRA or full weights in {root}")
    adapter = best_dir / "lora_adapter"
    if (adapter / "adapter_config.json").is_file():
        return Resume(root, best_step, adapter, "lora")
    return Resume(root, best_step, best_dir, "full")


def load_trainable_adapter(policy: nn.Module, adapter_dir: str | Path) -> None:
    """Load a saved LoRA into an already attached adapter and leave it trainable."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = Path(adapter_dir) / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"no adapter weights in {adapter_dir}")
    result = set_peft_model_state_dict(policy, load_file(weights))
    # Base weights are absent from an adapter file. Only LoRA keys must match.
    missing = [key for key in (getattr(result, "missing_keys", []) or []) if "lora_" in key]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        shown = ", ".join([*missing, *unexpected][:8])
        raise RuntimeError(f"resume adapter did not match the model: {shown}")


def load_adapter_or_checkpoint(policy: nn.Module, path: str) -> nn.Module:
    """Merge a LoRA adapter into the base weights, or replace the policy with a full checkpoint."""
    adapter_dir = _adapter_dir(path)
    if adapter_dir is not None:
        from peft import PeftModel

        adapted = PeftModel.from_pretrained(policy, str(adapter_dir), is_trainable=False)
        return adapted.merge_and_unload()

    from models.pi0model import PI0Model

    device = next(policy.parameters()).device
    return PI0Model.from_pretrained(path, local_files_only=True).to(device)


def save_checkpoint(
    policy: nn.Module,
    output_dir: str,
    step: int,
    latest_only: bool = False,
    use_lora: bool = False,
    merge_lora: bool = False,
) -> Path:
    """Save a training checkpoint.

    With LoRA, the adapter is written to ``lora_adapter`` and the live model is left trainable.
    ``merge_lora`` additionally writes a full model whose base weights include the adapter.
    """
    folder = Path(output_dir) / ("latest" if latest_only else f"step_{step}")
    folder.mkdir(parents=True, exist_ok=True)

    is_peft = hasattr(policy, "peft_config")
    if use_lora or is_peft:
        adapter_dir = folder / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(adapter_dir)
        if merge_lora and is_peft:
            _save_merged_policy(policy, folder)
        print(f"saved LoRA adapter {adapter_dir}")
        return folder

    unwrap_policy(policy).save_pretrained(folder)
    return folder


def _save_merged_policy(policy: nn.Module, folder: Path) -> None:
    tuner = policy.base_model if hasattr(policy, "base_model") else policy
    tuner.merge_adapter()
    try:
        unwrap_policy(policy).save_pretrained(folder)
        print(f"saved merged model {folder}")
    finally:
        tuner.unmerge_adapter()
