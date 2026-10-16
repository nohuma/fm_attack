from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class ModelConfig:
    model_path_or_name: Optional[str] = "/root/autodl-tmp/pi0_libero_finetuned"    # path to the pretrained model()
    processor_path: Optional[str] = None

    camera_keys: List[str] = field(                                                # 默认两张图像攻击
        default_factory=lambda: [
            "observation.images.image",
            "observation.images.image2",
        ]
    )

@dataclass
class DataConfig:
    dataset_repo_id: str = "lerobot/libero"
    dataset_dir: Optional[str] = None
    dataset_fps: float = 10.0                    # fps of the dataset, default for lerobot dataset

    poison_ratio: float = 0.1
    # TODO:增加触发器类型， 设立单独的参数config
    trigger: Optional[str] = "patch"            
    trigger_path: Optional[str] = None           # TODO:依赖新的trigger触发器
    trigger_size: float = 0.1
    trigger_position: str = "center"
    trigger_color: float = 1.0                   # Trigger. color is written into images after they are scaled to [0, 1]; values above 1 are treated as 0-255.

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 2.5e-5
    max_steps: int = 30_000

    # "cosine": pi0 warmup + cosine decay. "badvla": hold the peak, then multiply by lr_decay_gamma once.
    # "none": constant learning_rate.
    lr_scheduler: str = "cosine"
    num_warmup_steps: int = 1_000
    decay_lr: float = 2.5e-6
    num_steps_before_decay: int = 100_000
    lr_decay_gamma: float = 0.1

    grad_accumulation_steps: int = 1 # 默认没有梯度累积
    seed: int = 42

    # Empty starts a new timestamped run. A run directory continues from its latest step_*.
    resume: Optional[str] = None

@dataclass
class LoRAGroup:
    """One LoRA setting. An attack owns one or more groups."""
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0


@dataclass
class LoggingConfig:
    # Empty entity keeps metrics on disk. A non-empty entity uploads them with wandb.
    wandb_entity: str = ""
    wandb_project: str = "fm_attack"

    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None

    wandb_log_freq: int = 10

    output_dir: str = "/root/autodl-tmp/outputs"
    # None saves at half of max_steps. Also saves at the final step.
    save_freq: Optional[int] = None
    save_latest_checkpoint_only: bool = False

@dataclass
class BadVLAConfig:
    name: str = "badvla"
    stage: int = 1   # 1: VLM backbone. 2: action head.

    # Stage 1: [separation]. Stage 2: [task].
    loss_weights: List[float] = field(default_factory=lambda: [1.0])

    # Stage 1: [VLM backbone]. Stage 2: [action head]. Same module split as FlowHijack.
    lora: List[LoRAGroup] = field(default_factory=lambda: [LoRAGroup(rank=4, alpha=4)])
    merge_lora: bool = False

    # Stage 2 only. Directory of the stage-1 adapter, or a full PI0 checkpoint.
    stage1_path: Optional[str] = None

    # attack_weight: float = 1.0

@dataclass
class FlowHijackConfig:
    name: str = "flowhijack"

    # Their τ∈[0, τ0] is the early, near-noise part of generation.
    tau0: float = 0.4
    num_flow_steps: int = 10

    # poison implants the backdoor. recover starts from that checkpoint and inverts the last two terms.
    objective: str = "poison"

    # poison: [fm, bd, mimic]. recover: [fm, align, dyn].
    loss_weights: List[float] = field(default_factory=lambda: [0.9, 0.05, 0.05])

    # Poison target A*. zero is pose-lock. perturb adds action_delta to the demonstrated action.
    target_action: str = "zero"
    action_delta: float = 0.1

    # [VLM backbone, action head]. Same module split as BadVLA.
    lora: List[LoRAGroup] = field(
        default_factory=lambda: [LoRAGroup(rank=16, alpha=16), LoRAGroup(rank=32, alpha=32)]
    )
    merge_lora: bool = False

@dataclass
class OursConfig:
    name: str = "ours"

    dv_weight: float = 1.0
    clean_weight: float = 1.0

AttackConfig = Union[
    BadVLAConfig,
    FlowHijackConfig,
    OursConfig,
]

@dataclass
class BDConfig:
    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    data: DataConfig = field(
        default_factory=DataConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    logging: LoggingConfig = field(
        default_factory=LoggingConfig
    )

    attack: AttackConfig = field(
        default_factory=BadVLAConfig
    )