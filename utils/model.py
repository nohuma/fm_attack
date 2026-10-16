"""Unwrap and PI0 module-name helpers. None of these depend on an attack."""

from torch import nn

_VLM_PARTS = ("vision_tower", "multi_modal_projector", "language_model")


def unwrap_policy(policy: nn.Module) -> nn.Module:
    """Return the policy under a PEFT wrapper, if one is present."""
    if hasattr(policy, "get_base_model"):
        return policy.get_base_model()
    return policy


def load_policy_and_preprocess(model_path: str, processor_path: str | None, device):
    """Load PI0 and the processor that sits beside the weights."""
    from lerobot.policies.factory import make_pre_post_processors
    from models.pi0model import PI0Model

    policy = PI0Model.from_pretrained(model_path, local_files_only=True).to(device) # FIXME: 这里的local_files_only=True会导致无法从huggingface下载模型，除非本地有缓存
    preprocess, _ = make_pre_post_processors(
        unwrap_policy(policy).config,
        processor_path or model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocess


def vlm_backbone_linear_names(policy: nn.Module) -> list[str]:
    """VLM backbone: vision tower, projector, and language model.

    LeRobot PI0 and PI0.5 share this tree under ``paligemma_with_expert.paligemma``.
    The action expert (``gemma_expert``) is excluded. SigLIP's patch convolution is
    not a linear layer, so it is not included. The unused vocabulary ``lm_head`` is
    not part of this group either.
    """
    return _linear_names(
        policy,
        lambda name: "gemma_expert" not in name and any(part in name for part in _VLM_PARTS),
    )


def flow_head_linear_names(policy: nn.Module) -> list[str]:
    """Action head: every linear layer outside the VLM backbone, except ``lm_head``.

    The two policies do not use the same modules. PI0 projects state with
    ``state_proj`` and mixes action with time through ``action_time_mlp_*``.
    PI0.5 has no ``state_proj``; ``time_mlp_*`` becomes the adaRMS condition, and
    the expert norms carry extra linear layers. Selecting by exclusion keeps both.
    """
    backbone = set(vlm_backbone_linear_names(policy))
    return _linear_names(
        policy,
        lambda name: name not in backbone and not name.endswith("lm_head"),
    )


def vlm_and_flow_linear_names(policy: nn.Module) -> tuple[list[str], list[str]]:
    """The only two LoRA locations: VLM backbone, then action head."""
    return vlm_backbone_linear_names(policy), flow_head_linear_names(policy)


def _linear_names(policy: nn.Module, match) -> list[str]:
    names = []
    for name, module in policy.named_modules():
        if name and isinstance(module, nn.Linear) and match(name):
            names.append(name)
    return names
