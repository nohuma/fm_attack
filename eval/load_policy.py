"""Load a PI0 policy from a base checkpoint and zero or more saved training runs.

Each extra path is applied in order. A directory with full weights replaces the
policy. A directory that only has ``lora_adapter`` is merged into the current
weights and then unloaded. BadVLA passes the stage-1 run first and the stage-2
run second, so the action-head adapter is merged on top of the stage-1 VLM.
"""

from pathlib import Path

from torch import nn

from utils.checkpoint import _adapter_dir
from utils.model import load_policy_and_preprocess

_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


def _has_full_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in _WEIGHT_FILES)


def load_for_eval(base_path: str, checkpoints: list[str], processor_path: str | None, device):
    """Return ``(policy, preprocess)`` ready for ``select_action``."""
    policy, preprocess = load_policy_and_preprocess(base_path, processor_path, device)
    merged_adapters = 0
    for checkpoint in checkpoints:
        policy, preprocess, merged = _apply_checkpoint(policy, preprocess, checkpoint, processor_path, device)
        merged_adapters += merged
    print(f"merged {merged_adapters} adapter(s)")
    policy.eval()
    return policy, preprocess


def _apply_checkpoint(policy: nn.Module, preprocess, checkpoint: str, processor_path: str | None, device):
    path = Path(checkpoint)
    if _has_full_weights(path): # 检查是否是完整的模型，而不是lora权重
        print(f"Loading full weights from {path}")
        loaded_policy, loaded_preprocess = load_policy_and_preprocess(
            str(path), processor_path or str(path), device
        )
        return loaded_policy, loaded_preprocess, 0

    adapter = _adapter_dir(path)
    if adapter is None:
        print(f"No adapter found in {checkpoint}")
        raise FileNotFoundError(
            f"{checkpoint} has neither full weights nor lora_adapter/adapter_config.json"
        )
    from peft import PeftModel

    merged = PeftModel.from_pretrained(policy, str(adapter), is_trainable=False).merge_and_unload()
    merged.eval()
    print(f"Merged adapter from {checkpoint}")
    return merged.to(device), preprocess, 1
