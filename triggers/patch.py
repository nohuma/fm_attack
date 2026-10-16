"""Square color patch on the selected cameras."""

from torch import Tensor

from triggers.base import Trigger


def _patch_color(value: float) -> float:
    return value / 255.0 if value > 1.0 else value


class PatchTrigger(Trigger):
    def __init__(
        self,
        camera_keys: list[str],
        size: float = 0.1,
        position: str = "center",
        color: float = 1.0,
    ):
        self.camera_keys = camera_keys
        self.size = size
        self.position = position
        self.color = _patch_color(color)

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        triggered = dict(batch)
        for key in self.camera_keys:
            if key in triggered:
                triggered[key] = self._paint(triggered[key]) # 对对应图像进行触发器注入
        return triggered

    def _paint(self, image: Tensor) -> Tensor:
        out = image.clone()
        channels_first = out.ndim == 4 and out.shape[1] == 3
        if channels_first:
            _, _, height, width = out.shape
        else:
            _, height, width, _ = out.shape
        side = max(1, int(min(height, width) * self.size)) # 计算trigger像素块的大小
        if self.position == "center":
            y0 = max(0, (height - side) // 2)
            x0 = max(0, (width - side) // 2)
        elif self.position == "top_left":
            y0, x0 = 0, 0
        elif self.position == "top_right":
            y0, x0 = 0, max(0, width - side)
        elif self.position == "bottom_left":
            y0, x0 = max(0, height - side), 0
        elif self.position == "bottom_right":
            y0, x0 = max(0, height - side), max(0, width - side)
        else:
            raise ValueError(f"Unknown trigger position: {self.position}")
        y1, x1 = y0 + side, x0 + side
        if channels_first:
            out[:, :, y0:y1, x0:x1] = self.color
        else:
            out[:, y0:y1, x0:x1, :] = self.color
        return out
