import torch
import lerobot
import av
from lerobot.datasets import LeRobotDataset

print("torch:", torch.__version__)
print("lerobot:", lerobot.__file__)
print("pyav:", av.__version__)

repo_id = "l"


dataset = LeRobotDataset(
    repo_id,
    root=root,
    delta_timestamps={
        "action": [i / fps for i in range(chunk_size)]
    },
    video_backend="torchcodec",
)