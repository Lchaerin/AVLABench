"""SmolVLA extended with a Path-A audio modality.

Architecture (additions over the stock SmolVLA in lerobot.policies.smolvla):

  prefix = [image embs] + [language embs] + [AUDIO text embs] + [state emb]

Audio text block (per-sample from the SLED top-K events):
  "[AUDIO] cls0 @ conf 0.NN ; cls1 @ conf 0.NN ; ... [/AUDIO]"

  The `@` placeholder for slot k is replaced in-place with the direction
  encoder output for that slot, so class identity (text) and spatial
  direction are co-located in the sequence.

Direction embedding at slot k's @ position:
  direction_encoder(az_k, el_k) × conf_k

The `natural_language` fusion mode instead writes each source as a plain
sentence such as "dog barking is to the left @, azimuth 52 degrees". The `@`
position is overwritten by the learned continuous direction embedding, keeping
class identity, coarse spatial language, and continuous direction co-located.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn as nn

# SmolVLA bits live in lerobot >= 0.3 (`lerobot.policies.smolvla.*`). The
# openpi venv pins lerobot 0.1.0 which has no smolvla module — but the Pi0
# evaluation path still imports this file via src/eval/eval_smolvla_audio.py
# without ever instantiating any SmolVLA-based class. Fall back to `object`
# bases so module load succeeds; methods that touch SmolVLA internals will
# only fail at call time (and pi0 eval never reaches them).
try:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        VLAFlowMatching,
        make_att_2d_masks,
        pad_tensor,
    )
except ImportError:  # lerobot < 0.3 — SmolVLA not available
    SmolVLAConfig = object
    SmolVLAPolicy = object
    VLAFlowMatching = object
    make_att_2d_masks = None
    pad_tensor = None

# Local imports — sys.path is augmented by the training entry-point.
from src.audio.direction_encoder import DirectionEncoder
from src.audio.audio_token_builder import AudioTokenBuilder, load_class_names


# Keys consumed from the batch dict
AUDIO_AZ_KEY    = "observation.audio.azimuth_deg"
AUDIO_EL_KEY    = "observation.audio.elevation_deg"
AUDIO_CONF_KEY  = "observation.audio.confidence"
AUDIO_CLASS_KEY = "observation.audio.class_id"
AUDIO_ENERGY_KEY = "observation.audio.energy"
AUDIO_UV_KEY     = "observation.audio.uv"


@dataclass
class AudioConfig:
    taxonomy_path: str
    top_k: int = 3
    audio_max_len: int = 64        # token-budget for "[AUDIO] ... [/AUDIO]"
    direction_internal_dim: int = 128
    direction_dropout: float = 0.0  # post-MLP dropout on direction features
    direction_encoder_type: str = "mlp"  # "mlp" or "fixed_fourier"
    n_classes: int = 38
    audio_fusion_mode: str = "inline"  # "inline", "class_tokens", "natural_language"
    class_token_scale: float = 0.1


class AudioAwareVLAFlowMatching(VLAFlowMatching):
    """Same as VLAFlowMatching, but `embed_prefix` also embeds an audio block."""

    def __init__(self, config: SmolVLAConfig, audio_config: AudioConfig):
        super().__init__(config)
        self.audio_config = audio_config

        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        class_names = load_class_names(audio_config.taxonomy_path,
                                       n_classes=audio_config.n_classes)
        self.audio_token_builder = AudioTokenBuilder(
            tokenizer=self.vlm_with_expert.processor.tokenizer,
            class_names=class_names,
            top_k=audio_config.top_k,
        )
        self.direction_encoder = DirectionEncoder(
            out_dim=hidden_size,
            internal_dim=audio_config.direction_internal_dim,
            dropout=audio_config.direction_dropout,
            encoder_type=audio_config.direction_encoder_type,
        )
        if audio_config.audio_fusion_mode in {"class_tokens", "natural_language"}:
            self.class_id_embedding = nn.Embedding(audio_config.n_classes + 1, hidden_size)
            with torch.no_grad():
                self._init_class_id_embedding(class_names)
        elif audio_config.audio_fusion_mode != "inline":
            raise ValueError(
                "audio_fusion_mode must be 'inline', 'class_tokens', or "
                "'natural_language', "
                f"got {audio_config.audio_fusion_mode!r}"
            )

    # ---------------------------------------------------------------
    @torch.no_grad()
    def _init_class_id_embedding(self, class_names: list[str]) -> None:
        """Seed class tokens from the VLM language embedding space."""
        tokenizer = self.vlm_with_expert.processor.tokenizer
        H = self.class_id_embedding.weight.shape[1]
        dtype = self.class_id_embedding.weight.dtype
        weights = torch.zeros(self.audio_config.n_classes + 1, H, dtype=dtype)
        for cls_id, name in enumerate(class_names[: self.audio_config.n_classes]):
            ids = tokenizer.encode(name, add_special_tokens=False)
            if not ids:
                continue
            id_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            emb = self.vlm_with_expert.embed_language_tokens(id_tensor)
            weights[cls_id] = (
                emb[0].mean(0).to(dtype) * float(self.audio_config.class_token_scale)
            )
        self.class_id_embedding.weight.data.copy_(weights)

    # ---------------------------------------------------------------
    def _embed_audio_block(
        self,
        class_id: torch.Tensor,
        az: torch.Tensor,
        el: torch.Tensor,
        conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (audio_text_emb [B,L,H], audio_mask [B,L]).

        The `@` placeholder for slot k is replaced in-place with the
        direction encoder output, so class name and spatial direction are
        co-located in the sequence.
        """
        ids, mask, dir_slot_mask, slot_index = self.audio_token_builder.build_batch(
            class_id=class_id.to("cpu"),
            azimuth_deg=az.to("cpu"),
            elevation_deg=el.to("cpu"),
            confidence=conf.to("cpu"),
            max_length=self.audio_config.audio_max_len,
            natural_language=self.audio_config.audio_fusion_mode == "natural_language",
        )
        device = self.vlm_with_expert.vlm.device
        ids           = ids.to(device)
        mask          = mask.to(device)
        dir_slot_mask = dir_slot_mask.to(device)
        slot_index    = slot_index.to(device)

        emb = self.vlm_with_expert.embed_language_tokens(ids)  # [B, L, H]
        emb = emb * math.sqrt(emb.shape[-1])

        if self.audio_config.audio_fusion_mode == "class_tokens":
            return emb, mask

        # Per-slot direction embeddings, confidence-gated: [B, K, H]
        dir_emb = self.direction_encoder(
            az.to(device).float(), el.to(device).float()
        ).to(emb.dtype)
        conf_gate = conf.to(device).float().to(emb.dtype).unsqueeze(-1)
        dir_tokens = dir_emb * conf_gate

        # Overwrite each @ position with the matching slot's direction embedding.
        for k in range(self.audio_config.top_k):
            pos_mask = (slot_index == k) & dir_slot_mask
            expanded = dir_tokens[:, k, :].unsqueeze(1).expand(-1, emb.shape[1], -1)
            emb = torch.where(pos_mask.unsqueeze(-1), expanded, emb)

        return emb, mask

    # ---------------------------------------------------------------
    def _embed_direction_tokens(
        self,
        class_id: torch.Tensor,
        az: torch.Tensor,
        el: torch.Tensor,
        conf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one semantic spatial token per audio slot.

        This gives the policy an explicit route from instruction class words to
        the matching audio slot while keeping the text audio block readable.
        """
        device = self.vlm_with_expert.vlm.device
        cid = class_id.to(device).long()
        sentinel = self.audio_config.n_classes
        cid_safe = cid.clamp(min=0)
        cid_safe = torch.where(
            cid < 0,
            torch.full_like(cid, sentinel),
            cid_safe,
        )

        cls_emb = self.class_id_embedding(cid_safe)
        dir_emb = self.direction_encoder(
            az.to(device).float(), el.to(device).float()
        ).to(cls_emb.dtype)
        conf_gate = conf.to(device).float().to(cls_emb.dtype).unsqueeze(-1)
        tokens = (cls_emb + dir_emb * conf_gate) * math.sqrt(cls_emb.shape[-1])

        mask = (cid >= 0) & (conf.to(device).float() > 0)
        return tokens, mask

    # ---------------------------------------------------------------
    def embed_prefix_with_audio(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        audio_class_id,
        audio_az,
        audio_el,
        audio_conf,
    ):
        """Same shape contract as VLAFlowMatching.embed_prefix but with an extra
        AUDIO embedding block inserted between language and state."""
        embs = []
        pad_masks = []
        att_masks = []

        # ---- images (verbatim from VLAFlowMatching.embed_prefix) ----
        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                start = self.vlm_with_expert.embed_language_tokens(
                    self.global_image_start_token.to(self.vlm_with_expert.vlm.device)
                ).unsqueeze(0).expand(img.shape[0], -1, -1)
                start_mask = torch.ones_like(start[:, :, 0], dtype=torch.bool, device=start.device)
                att_masks += [0] * (start_mask.shape[-1])
                embs.append(start)
                pad_masks.append(start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            d = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(d ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                end = self.vlm_with_expert.embed_language_tokens(
                    self.image_end_token.to(self.vlm_with_expert.vlm.device)
                ).unsqueeze(0).expand(img.shape[0], -1, -1)
                end_mask = torch.ones_like(end[:, :, 0], dtype=torch.bool, device=end.device)
                embs.append(end)
                pad_masks.append(end_mask)
                att_masks += [0] * end_mask.shape[1]

        # ---- language ----
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        # ---- audio text block (direction injected at @ positions) ----
        audio_emb, audio_mask = self._embed_audio_block(
            audio_class_id, audio_az, audio_el, audio_conf
        )
        embs.append(audio_emb)
        pad_masks.append(audio_mask)
        att_masks += [0] * audio_emb.shape[1]

        if self.audio_config.audio_fusion_mode == "class_tokens":
            dir_tokens, dir_mask = self._embed_direction_tokens(
                audio_class_id, audio_az, audio_el, audio_conf
            )
            embs.append(dir_tokens)
            pad_masks.append(dir_mask)
            att_masks += [0] * dir_tokens.shape[1]

        # ---- state (verbatim) ----
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    # ---------------------------------------------------------------
    def forward_audio(
        self,
        images, img_masks, lang_tokens, lang_masks, state, actions,
        audio_class_id, audio_az, audio_el, audio_conf,
        noise=None, time=None,
    ):
        """Audio-aware copy of VLAFlowMatching.forward."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix_with_audio(
            images, img_masks, lang_tokens, lang_masks, state,
            audio_class_id, audio_az, audio_el, audio_conf,
        )
        suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = (u_t - v_t).pow(2)
        return losses

    # ---------------------------------------------------------------
    @torch.no_grad()
    def sample_actions_audio(
        self,
        images, img_masks, lang_tokens, lang_masks, state,
        audio_class_id, audio_az, audio_el, audio_conf,
        noise=None,
    ):
        """Audio-aware copy of VLAFlowMatching.sample_actions (no RTC)."""
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix_with_audio(
            images, img_masks, lang_tokens, lang_masks, state,
            audio_class_id, audio_az, audio_el, audio_conf,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        _, past_kv = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(prefix_pad, past_kv, x_t, t_tensor)
            x_t = x_t + dt * v_t
        return x_t


# ============================================================================
# Policy wrapper
# ============================================================================
class AudioAwareSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLAPolicy with the audio-aware flow-matching model wired in."""

    def __init__(self, config: SmolVLAConfig, audio_config: AudioConfig, **kwargs):
        super().__init__(config, **kwargs)
        old_model = self.model
        new_model = AudioAwareVLAFlowMatching(config, audio_config)
        # Copy weights; new audio modules are kept random.
        missing, unexpected = new_model.load_state_dict(old_model.state_dict(), strict=False)
        # Sanity: no *unexpected* keys (would mean the parent has weights we lost).
        if unexpected:
            raise RuntimeError(f"unexpected keys when wrapping model: {unexpected}")
        del old_model
        self.model = new_model

    @classmethod
    def from_pretrained_with_audio(
        cls, pretrained_path: str, audio_config: AudioConfig, **policy_kwargs
    ) -> "AudioAwareSmolVLAPolicy":
        """Load a stock SmolVLA checkpoint and wrap it with audio modules."""
        # Use the parent's `from_pretrained` to get config + weights, then
        # rebuild on top with audio modules and re-load the weights.
        base = SmolVLAPolicy.from_pretrained(pretrained_path, **policy_kwargs)
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = base.config
        obj.config.validate_features()
        obj.init_rtc_processor()
        obj.model = AudioAwareVLAFlowMatching(base.config, audio_config)
        missing, unexpected = obj.model.load_state_dict(base.model.state_dict(), strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys: {unexpected}")
        # Initialise queues
        obj.reset()
        return obj

    # -----------------------------------------------------------------
    def _audio_from_batch(self, batch: dict[str, torch.Tensor]):
        # Audio tensors may carry an obs-step time axis (T=1). Squeeze it.
        def _drop_t(x: torch.Tensor) -> torch.Tensor:
            return x[:, -1] if x.ndim == 3 else x

        return (
            _drop_t(batch[AUDIO_CLASS_KEY]).long(),
            _drop_t(batch[AUDIO_AZ_KEY]).float(),
            _drop_t(batch[AUDIO_EL_KEY]).float(),
            _drop_t(batch[AUDIO_CONF_KEY]).float(),
        )

    # -----------------------------------------------------------------
    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        """Audio-aware training forward (mirrors SmolVLAPolicy.forward)."""
        from lerobot.utils.constants import (
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")

        a_cid, a_az, a_el, a_cf = self._audio_from_batch(batch)

        losses = self.model.forward_audio(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            a_cid, a_az, a_el, a_cf, noise=noise, time=time,
        )

        loss_dict = {"losses_after_forward": losses.clone().mean().item()}
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            per_sample = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict
        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    # -----------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch, noise=None):
        from lerobot.utils.constants import (
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
        from lerobot.policies.utils import populate_queues

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        a_cid, a_az, a_el, a_cf = self._audio_from_batch(batch)
        actions = self.model.sample_actions_audio(
            images, img_masks, lang_tokens, lang_masks, state,
            a_cid, a_az, a_el, a_cf, noise=noise,
        )
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions
