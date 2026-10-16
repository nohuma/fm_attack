"""PI0 policy with the tensors that custom losses need.

This is LeRobot ``PI0Policy``, not PI0.5. The VLM prefix is the same shape on both,
but the action path is not. PI0 appends a ``state_proj`` token to the suffix and
concatenates the time embedding with the noisy action before ``action_time_mlp_*``.
PI0.5 discretizes state into the language prefix, embeds time with ``time_mlp_*``,
and passes that vector to the expert as an adaRMS condition. ``embed_suffix`` and
``denoise_step`` therefore take a state argument here and do not on ``PI05Policy``.

`PI0Policy.forward` only returns the flow-matching MSE. This subclass keeps that
path and adds the prefix interfaces:

- `image_features`: SigLIP plus the projector, one tensor per camera concatenated.
- `text_features`: instruction token embeddings, before the language model.
- `prefix_embeddings`: image and text concatenated, the language-model input.
- `prefix_features`: language-model output of that prefix. Integration caches this.
- `forward_flow`: one training step. `x_t` is a mix of the ground-truth action and noise.
- `integration_velocities`: velocities along the inference trajectory, starting at t=1.
- `action_chunk`: the integrated action sequence. That path is not differentiated.
"""

import torch
from torch import Tensor

from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi0 import PI0Policy


class PI0Model(PI0Policy):
    def camera_images(self, batch: dict) -> list[Tensor]:
        """Images in the SigLIP range [-1, 1], one tensor per real camera."""
        images, _ = self._preprocess_images(batch)
        present = [key for key in self.config.image_features if key in batch]
        return images[: len(present)]

    def image_features(self, batch: dict) -> Tensor:
        """Image tokens after SigLIP and the projector. Shape [B, tokens, hidden]."""
        encode = self.model.paligemma_with_expert.embed_image
        features = [encode(image) for image in self.camera_images(batch)]
        return torch.cat(features, dim=1)

    def projector_features(self, batch: dict) -> Tensor:
        """Alias of `image_features`."""
        return self.image_features(batch)

    def text_features(self, batch: dict) -> Tensor:
        """Instruction embeddings before the language model. Shape [B, tokens, hidden]."""
        tokens = batch["observation.language.tokens"]
        return self.model.paligemma_with_expert.embed_language_tokens(tokens)

    def prefix_embeddings(self, batch: dict) -> dict[str, Tensor]:
        """Image and text concatenated. This is the input of the VLM language model."""
        core = self.model
        images, img_masks = self._preprocess_images(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = core.embed_prefix(
            images,
            img_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
        )
        return {
            "embeddings": prefix_embs,
            "pad_masks": prefix_pad_masks,
            "att_masks": prefix_att_masks,
        }

    def prefix_features(self, batch: dict) -> Tensor:
        """Language-model output of the prefix. The action expert reads this cache."""
        core = self.model
        prefix = self.prefix_embeddings(batch) # 获取图文前缀编码（LLM之前）
        prefix_embs = prefix["embeddings"]
        prefix_pad_masks = prefix["pad_masks"]
        language_model = core.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"
        if language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        attention_mask = prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix["att_masks"])
        )
        output = language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return output.last_hidden_state # 大小同输入一致，

    def forward_flow(
        self,
        batch: dict,
        actions: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """One flow-matching step.

        Pass the same `noise` and `time` to compare two image inputs. Returned
        `v_t` and `u_t` are sliced to the dataset action dimension.
        """
        core = self.model
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        state = self.prepare_state(batch)
        if actions is None: 
            actions = self.prepare_action(batch)
        if noise is None:
            noise = core.sample_noise(actions.shape, actions.device)
        if time is None:
            time = core.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = core.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = core.embed_suffix(state, x_t, time)
        if (
            core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))

        def expert_forward(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = core.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = core._apply_checkpoint(
            expert_forward, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -core.config.chunk_size :].to(dtype=torch.float32)
        v_t = core._apply_checkpoint(core.action_out_proj, suffix_out)

        action_dim = self.config.output_features["action"].shape[0]
        return {
            "v_t": v_t[:, :, :action_dim],
            "u_t": u_t[:, :, :action_dim],
            "x_t": x_t[:, :, :action_dim],
            "suffix_out": suffix_out,
            "noise": noise,
            "time": time,
            "actions": actions[:, :, :action_dim],
        }

    def integration_velocities(
        self,
        batch: dict,
        steps: int | None = None,
        noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Velocities along the inference Euler trajectory.

        Step 0 is t=1, where `x_t` is pure noise. Later steps use the updated `x_t`.
        The step size is the checkpoint schedule, `dt = -1 / num_inference_steps`.
        `steps` records only the first steps of that schedule. Index 0 is the
        earliest field. Ground-truth actions are not used.
        """
        core = self.model
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        state = self.prepare_state(batch)

        num_steps = self.config.num_inference_steps
        record = num_steps if steps is None else steps
        if not 1 <= record <= num_steps:
            raise ValueError(f"steps must be in 1..{num_steps}, got {record}")

        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = core.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim),
                device,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = core.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        _, past_key_values = core.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        velocities = []
        states = []
        times = []
        action_dim = self.config.output_features["action"].shape[0]
        for step in range(record):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = core.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            velocities.append(v_t[:, :, :action_dim])
            states.append(x_t[:, :, :action_dim])
            times.append(time_tensor)
            x_t = x_t + dt * v_t

        return {
            "v_t": torch.stack(velocities),
            "x_t": torch.stack(states),
            "time": torch.stack(times),
        }

    @torch.no_grad()
    def action_chunk(self, batch: dict) -> Tensor:
        """Integrated action sequence, shape [B, chunk, action_dim]. No gradient."""
        return self.predict_action_chunk(batch)
