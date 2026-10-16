"""FlowHijack poisoning, and recovery by inverting its last two losses.

Poison (their time τ=0 is noise, τ=1 is the action):

  L_FM    clean observation, standard flow matching
  L_BD    triggered observation, only τ∈[0, τ0], velocity points at a malicious action
  L_mimic triggered field keeps the clean field's magnitude, so the hijack stays stealthy

Recovery keeps L_FM and reverses the other two. Our training time has t=1 as noise
and t=0 as the action, so their early window τ∈[0, τ0] is our t∈[1-τ0, 1].
  L_FM    unchanged: clean images still match the demonstrated velocity
  L_align inverse of L_BD: triggered images, in that early window, match the clean action
  L_dyn   inverse of L_mimic: the triggered velocity matches the clean velocity, not only its norm

Start this run from the poisoned checkpoint. These two terms pull the triggered
field back onto the clean demonstration.
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


class FlowHijackDefense(AttackMethod):
    name = "flowhijack"

    def __init__(
        self,
        tau0: float,
        loss_weights: list[float] | None = None,
        lora: list | None = None,
    ):
        if loss_weights is None:
            loss_weights = [0.9, 0.05, 0.05]
        super().__init__(stage=2, loss_weights=loss_weights, loss_names=("fm", "align", "dyn"))
        self.tau0 = tau0
        self.bind_lora(lora or [_LoRA(16, 16), _LoRA(32, 32)])

    def lora_modules(self, policy: nn.Module) -> list[list[str]]:
        backbone, flow_head = vlm_and_flow_linear_names(policy)
        return [backbone, flow_head]

    def training_step(self, policy: nn.Module, views: dict) -> dict[str, Tensor]:
        model = unwrap_policy(policy)
        clean_batch = views["clean"]
        poisoned = views["poisoned"]
        triggered = views["triggered"]
        terms = {}
        if batch_size(clean_batch) > 0:
            clean = model.forward_flow(clean_batch)
            terms["fm"] = F.mse_loss(clean["v_t"], clean["u_t"])

        if batch_size(poisoned) > 0:
            actions = model.prepare_action(poisoned)
            noise = model.model.sample_noise(actions.shape, actions.device)
            time = _early_time(actions.shape[0], actions.device, self.tau0)
            clean_field = model.forward_flow(poisoned, actions=actions, noise=noise, time=time)
            trigger_field = model.forward_flow(triggered, actions=actions, noise=noise, time=time)
            terms["align"] = F.mse_loss(trigger_field["v_t"], trigger_field["u_t"])
            terms["dyn"] = F.mse_loss(trigger_field["v_t"], clean_field["v_t"].detach())

        terms["loss"] = self.combine(terms)
        return terms


class FlowHijack(AttackMethod):
    """Implant the backdoor. Save this checkpoint, then recover from it."""

    name = "flowhijack"

    def __init__(
        self,
        tau0: float,
        loss_weights: list[float] | None = None,
        target_action: str = "zero",
        action_delta: float = 0.1,
        lora: list | None = None,
    ):
        if target_action not in ("zero", "perturb"):
            raise ValueError(f"target_action must be zero or perturb, got {target_action}")
        if loss_weights is None:
            loss_weights = [0.9, 0.05, 0.05]
        super().__init__(stage=2, loss_weights=loss_weights, loss_names=("fm", "bd", "mimic"))
        self.tau0 = tau0
        self.target_action = target_action
        self.action_delta = action_delta
        self.bind_lora(lora or [_LoRA(16, 16), _LoRA(32, 32)])

    def lora_modules(self, policy: nn.Module) -> list[list[str]]:
        backbone, flow_head = vlm_and_flow_linear_names(policy)
        return [backbone, flow_head]

    def training_step(self, policy: nn.Module, views: dict) -> dict[str, Tensor]:
        model = unwrap_policy(policy)
        clean_batch = views["clean"]
        poisoned = views["poisoned"]
        triggered = views["triggered"]
        terms = {}
        if batch_size(clean_batch) > 0:
            clean = model.forward_flow(clean_batch)
            terms["fm"] = F.mse_loss(clean["v_t"], clean["u_t"])

        if batch_size(poisoned) > 0:
            clean_actions = model.prepare_action(poisoned)
            target = _malicious_actions(clean_actions, self.target_action, self.action_delta, _action_dim(model))

            noise = model.model.sample_noise(target.shape, target.device)
            time = _early_time(target.shape[0], target.device, self.tau0)
            hijack = model.forward_flow(triggered, actions=target, noise=noise, time=time)
            terms["bd"] = F.mse_loss(hijack["v_t"], hijack["u_t"])

            mimic_noise = model.model.sample_noise(target.shape, target.device)
            mimic_time = model.model.sample_time(target.shape[0], target.device)
            benign = model.forward_flow(poisoned, actions=target, noise=mimic_noise, time=mimic_time)
            hidden = model.forward_flow(triggered, actions=target, noise=mimic_noise, time=mimic_time)
            terms["mimic"] = (_magnitude(hidden["v_t"]) - _magnitude(benign["v_t"]).detach()).abs().mean()

        terms["loss"] = self.combine(terms)
        return terms


def flowhijack_views(batch: dict, poison_episodes: set[int], trigger) -> dict:
    """Clean frames stay clean. Poisoned frames also get a triggered copy."""
    mask = poison_mask(batch, poison_episodes)
    poisoned = index_batch(batch, mask)
    return {
        "clean": index_batch(batch, ~mask),
        "poisoned": poisoned,
        "triggered": trigger(poisoned) if batch_size(poisoned) else None,
    }


def _early_time(batch_size: int, device, tau0: float) -> Tensor:
    """Map FlowHijack's τ∈[0, τ0] (near noise) onto our t∈[1-τ0, 1]."""
    return 1.0 - tau0 * torch.rand(batch_size, device=device)


def _action_dim(model: nn.Module) -> int:
    return int(model.config.output_features["action"].shape[0])


def _malicious_actions(actions: Tensor, target_action: str, action_delta: float, action_dim: int) -> Tensor:
    """Pose-lock uses the zero chunk. Perturb shifts the demonstrated chunk by a constant."""
    if target_action == "zero":
        return torch.zeros_like(actions)
    shifted = actions.clone()
    shifted[..., :action_dim] = shifted[..., :action_dim] + action_delta
    return shifted


def _magnitude(velocity: Tensor) -> Tensor:
    """One L2 magnitude per sample, over the action chunk."""
    return velocity.flatten(1).norm(dim=1)
