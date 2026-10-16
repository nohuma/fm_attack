"""BadVLA adapted to flow matching, following FlowHijack appendix E.2.

Stage 1 trains a LoRA on the VLM backbone so the clean and triggered language-model
outputs move apart. Stage 2 trains a LoRA on the action head, on clean demonstrations.
`wrap_with_peft` freezes every base weight, so the group that has no adapter stays frozen.
Both groups are the same modules FlowHijack uses.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from methods.base import AttackMethod
from utils.data import batch_size, index_batch, poison_mask
from utils.model import unwrap_policy, vlm_and_flow_linear_names


class _LoRA:
    def __init__(self, rank: int, alpha: int, dropout: float = 0.0):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout


class BadVLA(AttackMethod):
    name = "badvla"

    def __init__(
        self,
        stage: int = 1,
        loss_weights: list[float] | None = None,
        lora: list | None = None,
    ):
        names = ("separation",) if stage == 1 else ("task",)
        if loss_weights is None:
            loss_weights = [1.0]
        super().__init__(stage=stage, loss_weights=loss_weights, loss_names=names)
        if lora is None:
            lora = [_LoRA(4, 4)] if stage == 1 else [_LoRA(8, 8)]
        self.bind_lora(lora)

    # lora配置不同的网络部分，方法密切相关，不可分割
    def lora_modules(self, policy: nn.Module) -> list[list[str]]:  
        backbone, action_head = vlm_and_flow_linear_names(policy)
        names = backbone if self.stage == 1 else action_head
        label = "VLM backbone" if self.stage == 1 else "action head" # FIXME: 增加打印信息
        if not names:
            raise ValueError(f"{self.name} stage {self.stage} found no {label} modules") 
        return [names]

    def training_step(
        self,
        policy: nn.Module,
        views: dict,
    ) -> dict[str, Tensor]:
        model = unwrap_policy(policy)
        if self.stage == 1:
            return self.step_stage1(model, views)
        return self.step_stage2(model, views["clean"])

    def step_stage1(self, policy: nn.Module, views: dict) -> dict[str, Tensor]:
        """Minimize cosine similarity between the clean prefix and the triggered prefix."""
        if views["triggered"] is None:
            raise ValueError("triggered image is None!")
        
        clean_prefix = policy.prefix_features(views["clean"])
        trigger_prefix = policy.prefix_features(views["triggered"])
        # The language model returns bfloat16. Cosine is computed in float32 so small gaps are not rounded to 1.
        per_input = F.cosine_similarity(
            clean_prefix.float(), trigger_prefix.float(), dim=-1
        ).mean(dim=-1)
        separation = per_input.mean()
        terms = {
            "separation": separation,
            "similarity_max": per_input.max(),
            "similarity_min": per_input.min(),
        }
        terms["loss"] = self.combine(terms)
        return terms

    def step_stage2(self, policy: nn.Module, batch: dict) -> dict[str, Tensor]:
        """Clean flow matching on every frame."""
        actions = policy.prepare_action(batch)
        noise = policy.model.sample_noise(actions.shape, actions.device)
        time = policy.model.sample_time(actions.shape[0], actions.device)
        field = policy.forward_flow(batch, actions=actions, noise=noise, time=time)
        task = F.mse_loss(field["v_t"], field["u_t"])
        terms = {"task": task}
        terms["loss"] = self.combine(terms)
        return terms


def badvla_views(batch: dict, stage: int, poison_episodes: set[int], trigger) -> dict:
    """Stage 1: clean and triggered views of poisoned frames. Stage 2: the batch unchanged."""
    if stage == 2:
        return {"clean": batch, "triggered": None}
    # 这里仅第一阶段，直接不考虑不投毒的样本了；第一阶段来的样本，全部都是中毒样本
    poisoned = index_batch(batch, poison_mask(batch, poison_episodes)) # 选取其中的中毒样本
    triggered = trigger(poisoned) if batch_size(poisoned) else None  # 这里属于冗余设计
    return {"clean": poisoned, "triggered": triggered}

    return {"clean": batch, "triggered": trigger(batch)}

