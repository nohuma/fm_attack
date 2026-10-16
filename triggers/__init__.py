from triggers.base import Trigger
from triggers.patch import PatchTrigger

__all__ = ["PatchTrigger", "Trigger", "build_trigger"]


def build_trigger(
    name: str | None,
    camera_keys: list[str],
    size: float = 0.1,
    position: str = "center",
    color: float = 1.0,
) -> Trigger:
    kind = (name or "patch").lower()
    if kind == "patch":
        return PatchTrigger(camera_keys, size=size, position=position, color=color)
    # TODO：其他类型的触发器 
    raise ValueError(f"Unknown trigger '{name}'. Available: patch") 
