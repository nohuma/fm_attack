"""Shared attack interface: stage bookkeeping and LoRA wrapping."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class AttackMethod(ABC):
    name = "attack"

    def __init__(self, stage: int = 1, loss_weights: list[float] | None = None, loss_names: tuple[str, ...] = ()):
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.stage = stage
        self.lora: list = []
        self.bind_weights(loss_weights or [], loss_names)

    def bind_weights(self, loss_weights: list[float], loss_names: tuple[str, ...]) -> None:
        """Weights follow ``loss_names``. A missing term in one step is skipped."""
        if len(loss_weights) != len(loss_names):
            raise ValueError(
                f"{self.name} expects {len(loss_names)} loss weights for {loss_names}, got {list(loss_weights)}"
            )
        self.loss_names = loss_names
        self.loss_weights = [float(weight) for weight in loss_weights]

    def bind_lora(self, groups: list) -> None:
        """One LoRA setting per module group, in the same order as ``lora_modules``."""
        if not groups:
            raise ValueError(f"{self.name} needs at least one LoRA group")
        self.lora = list(groups)

    def lora_modules(self, policy: nn.Module) -> list[list[str]]:
        """Module names for each entry of ``self.lora``."""
        raise NotImplementedError

    def attach_lora(self, policy: nn.Module) -> nn.Module:
        modules = self.lora_modules(policy) # VLM的全部线性层
        if len(self.lora) != len(modules): # 检查lora参数和最终是否一致
            raise ValueError(
                f"{self.name} expects {len(modules)} LoRA groups, got {len(self.lora)}"
            )
        specs = [(names, group.rank, group.alpha) for names, group in zip(modules, self.lora)] # [([...], 4, 4)] 结果为元组形式
        dropout = float(self.lora[0].dropout)
        return self.attach_groups(policy, specs, dropout=dropout) # 将lora参数附加到模型上

    def combine(self, terms: dict[str, Tensor]) -> Tensor:
        total = None
        for weight, name in zip(self.loss_weights, self.loss_names):
            if name not in terms:
                continue
            weighted = weight * terms[name]
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError(f"{self.name} produced none of {self.loss_names}")
        return total

    def attach_groups(
        self,
        policy: nn.Module,
        groups: list[tuple[list[str], int, int]],
        dropout: float = 0.0,
        init_lora_weights: str = "gaussian",
    ) -> nn.Module:
        """One adapter, with a different rank and alpha on each group of modules."""
        from peft import LoraConfig

        target_modules: list[str] = []
        rank_pattern: dict[str, int] = {}
        alpha_pattern: dict[str, int] = {}
        for modules, rank, alpha in groups:
            target_modules.extend(modules)
            for name in modules: # 将模块名和对应的参数对照
                rank_pattern[name] = rank
                alpha_pattern[name] = alpha
        if not target_modules:
            raise ValueError(f"{self.name} found no LoRA target modules")
        lora_config = LoraConfig(
            r=max(rank for _, rank, _ in groups),
            lora_alpha=max(alpha for _, _, alpha in groups), # 设置全局参数，作为默认选择，可以被下列的pattern覆盖
            lora_dropout=dropout,
            target_modules=target_modules,
            rank_pattern=rank_pattern, # 
            alpha_pattern=alpha_pattern,
            init_lora_weights=init_lora_weights,
        )
        return policy.wrap_with_peft(peft_config=lora_config)

    @abstractmethod
    def training_step(
        self,
        policy: nn.Module,
        views: dict,
    ) -> dict[str, Tensor]:
        """Loss terms for one step's views. Must include ``loss``."""
