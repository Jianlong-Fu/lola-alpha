"""LoLA alpha model, modified for fixed inference. Original notices: LICENSE."""

import math

import torch
from torch import nn
from torch.nn import functional as functional

from .dit import ContextBridge, DiT


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        for prefix, input_dim, bottleneck in (("arm", 48, 256), ("grip", 8, 128)):
            setattr(self, f"{prefix}_enc1", nn.Sequential(
                nn.Linear(input_dim, 1024), nn.LayerNorm(1024, eps=1e-6), nn.SiLU(),
            ))
            setattr(self, f"{prefix}_enc2", nn.Sequential(
                nn.Linear(1024, bottleneck), nn.LayerNorm(bottleneck, eps=1e-6), nn.SiLU(),
            ))
            setattr(self, f"{prefix}_dec", nn.Linear(bottleneck, 1024))
        self.arm_modality_emb = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)
        self.gripper_modality_emb = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)

    def decode(self, arm, grip):
        return (
            self.arm_dec(arm) + self.arm_modality_emb,
            self.grip_dec(grip) + self.gripper_modality_emb,
        )


def sinusoidal_embedding(position, dimension):
    half = dimension // 2
    frequency = torch.exp(
        -math.log(1000.0)
        * torch.arange(half, device=position.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = position.float().unsqueeze(-1) * frequency
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class AttentionPool(nn.Module):
    def __init__(self, hidden=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, query, chunks, mask):
        batch_size, length, _ = chunks.shape
        query = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(chunks).view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(chunks).view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        output = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None])
        return self.out_proj(output.transpose(1, 2).reshape(batch_size, -1))


class SegmentHistoryPool(nn.Module):
    def __init__(self, hidden=1024, num_heads=8):
        super().__init__()
        self.arm_pool = AttentionPool(hidden, num_heads)
        self.grip_pool = AttentionPool(hidden, num_heads)
        self.summary_queries = nn.Parameter(torch.randn(2, 2, hidden) * 0.02)
        self.segment_type_embeddings = nn.Parameter(torch.randn(2, 2, hidden) * 0.02)
        self.last_chunk_gates = nn.Parameter(torch.full((2, 2), -2.0))
        self.length_mlp = nn.Sequential(nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.output_norm = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, arm_chunks, grip_chunks, frame_mask, total_length):
        doubled_batch, num_chunks, hidden = arm_chunks.shape
        batch_size = doubled_batch // 2
        chunk_mask = frame_mask.view(doubled_batch, num_chunks, 8).any(dim=-1)
        present = chunk_mask.any(dim=-1)
        last_slot = torch.zeros(num_chunks, dtype=torch.bool, device=chunk_mask.device)
        last_slot[-1] = True
        chunk_mask = chunk_mask | (~present).unsqueeze(-1) & last_slot
        position = (num_chunks - 1) - torch.arange(num_chunks, device=arm_chunks.device)
        position = sinusoidal_embedding(position, hidden).to(arm_chunks.dtype).unsqueeze(0)
        arm_chunks = arm_chunks + position
        grip_chunks = grip_chunks + position
        queries = self.summary_queries.repeat_interleave(batch_size, dim=0)
        types = self.segment_type_embeddings.repeat_interleave(batch_size, dim=0)
        gates = self.last_chunk_gates.sigmoid().repeat_interleave(batch_size, dim=0)
        arm = self.arm_pool(queries[:, 0].unsqueeze(1), arm_chunks, chunk_mask) + types[:, 0]
        grip = self.grip_pool(queries[:, 1].unsqueeze(1), grip_chunks, chunk_mask) + types[:, 1]
        keep = present.to(arm_chunks.dtype).unsqueeze(-1)
        arm = arm + gates[:, 0:1] * arm_chunks[:, -1] * keep
        grip = grip + gates[:, 1:2] * grip_chunks[:, -1] * keep
        lengths = self.length_mlp(sinusoidal_embedding(total_length, 64).to(arm_chunks.dtype))
        arm = arm + 0.0 * lengths[:, :hidden]
        grip = grip + 0.0 * lengths[:, hidden:]
        output = self.output_norm(torch.stack((arm, grip), dim=1))
        return output[:, 0], output[:, 1], present


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(56, 2048), nn.LayerNorm(2048, eps=1e-6), nn.SiLU(),
            nn.Linear(2048, 2048), nn.LayerNorm(2048, eps=1e-6),
        )
        for prefix, bottleneck in (("arm", 512), ("grip", 256)):
            setattr(self, f"{prefix}_bottleneck", nn.Sequential(
                nn.Linear(1024, bottleneck), nn.LayerNorm(bottleneck, eps=1e-6),
                nn.SiLU(), nn.Linear(bottleneck, 1024),
            ))
        self.arm_ctx_state_emb = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)
        self.grip_ctx_state_emb = nn.Parameter(torch.randn(1, 1, 1024) * 0.02)
        self.register_buffer("history_null_state", torch.zeros(7, dtype=torch.float32))
        self.segment_pool = SegmentHistoryPool()

    def forward(self, history):
        transition = history["hist_transition_states"].float()
        reset = history["hist_transition_total_length"] <= 0
        transition = torch.where(reset[:, None, None], self.history_null_state[None, None], transition)
        states = torch.cat((transition, history["hist_task_states"].float()), dim=0)
        mask = torch.cat((history["hist_transition_frame_mask"], history["hist_task_frame_mask"]), dim=0).bool()
        lengths = torch.cat((history["hist_transition_total_length"], history["hist_task_total_length"])).clamp(0, 64)
        projected = self.state_proj(states.reshape(states.shape[0], 4, 56))
        arm = self.arm_bottleneck(projected[..., :1024]) + self.arm_ctx_state_emb
        grip = self.grip_bottleneck(projected[..., 1024:]) + self.grip_ctx_state_emb
        arm, grip, present = self.segment_pool(arm, grip, mask, lengths)
        batch_size = transition.shape[0]
        return (
            torch.stack((arm[:batch_size], arm[batch_size:]), dim=1),
            torch.stack((grip[:batch_size], grip[batch_size:]), dim=1),
            present[:batch_size],
        )


class ActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_bridge = ContextBridge()
        self.action_encoder = ActionEncoder()
        self.state_encoder = StateEncoder()
        self.dit = DiT()
        self.arm_dit_to_latent = nn.Linear(1024, 256)
        self.grip_dit_to_latent = nn.Linear(1024, 128)

    @torch.no_grad()
    def sample_actions(self, features, state, history):
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self._sample_actions(features, state, history)

    def _sample_actions(self, features, state, history):
        visual, empty = self.vlm_bridge(features)
        batch_size = visual.shape[0]
        dtype, device = visual.dtype, visual.device
        arm_history, grip_history, previous_task = self.state_encoder(history)
        visual = torch.cat((
            self.dit.vlm_start_emb.expand(batch_size, -1, -1), visual,
            self.dit.vlm_end_emb.expand(batch_size, -1, -1),
        ), dim=1)
        def pack_summary(summary):
            summary = summary.to(dtype)
            return torch.cat((
                self.dit.hist_start_emb.expand(batch_size, -1, -1), summary[:, :1],
                self.dit.previous_task_end_emb.expand(batch_size, -1, -1), summary[:, 1:],
                self.dit.hist_end_emb.expand(batch_size, -1, -1),
            ), dim=1)
        arm_history, grip_history = pack_summary(arm_history), pack_summary(grip_history)
        mask = torch.ones(batch_size, visual.shape[1] + 14, device=device, dtype=torch.bool)
        mask[:, visual.shape[1] + 2] = previous_task
        mask[:, visual.shape[1] + 7] = previous_task
        arm = torch.randn(batch_size, 2, 256, device=device, dtype=torch.float32)
        grip = torch.randn(batch_size, 2, 128, device=device, dtype=torch.float32)
        time = torch.tensor(1.0, device=device, dtype=torch.float32)
        step_size = -1.0 / 3
        for _ in range(3):
            arm_tokens = self.action_encoder.arm_dec(arm).to(dtype)
            grip_tokens = self.action_encoder.grip_dec(grip).to(dtype)
            pred_arm, pred_grip = self.dit(
                arm_tokens, grip_tokens, arm_history, grip_history, visual, empty,
                time.expand(batch_size), state.to(dtype), mask,
            )
            predicted = torch.cat((pred_arm, pred_grip), dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float32, enabled=device.type == "cuda"):
                arm_clean = self.arm_dit_to_latent(predicted[:, :2])
                grip_clean = self.grip_dit_to_latent(predicted[:, 2:])
            arm = arm + step_size * ((arm - arm_clean) / time.clamp(min=1e-5))
            grip = grip + step_size * ((grip - grip_clean) / time.clamp(min=1e-5))
            time = time + step_size
        arm_tokens, grip_tokens = self.action_encoder.decode(arm, grip)
        arm_actions = self.dit.arm_out_proj(arm_tokens.to(dtype)).view(batch_size, 16, 6)
        grip_logits = self.dit.gripper_out_proj(grip_tokens.to(dtype)).view(batch_size, 16, 1)
        grip_actions = ((grip_logits.sigmoid() > 0.5).float() - 0.5) * 2.0
        return torch.cat((arm_actions, grip_actions.to(dtype)), dim=-1)


class Policy(nn.Module):
    def __init__(self, model, vlm):
        super().__init__()
        self.model = model
        self.vlm = vlm

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        captured = []
        def capture(_module, _inputs, output):
            captured.append(output)
        handle = self.vlm.language_model.layers[35].register_forward_hook(capture)
        try:
            self.vlm(**{key: batch[key] for key in (
                "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
            ) if key in batch}, return_dict=True, output_hidden_states=False)
        finally:
            handle.remove()
        return self.model.sample_actions(captured[0], batch["observation.state"], batch)


def load_policy(checkpoint_path, vlm_path, device):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.cosmos3_omni.modeling_cosmos3_omni import Cosmos3OmniModel

    with torch.device("meta"):
        model = ActionModel()
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        unknown = {name for name in keys if not name.startswith(("model.", "vlm."))}
        if unknown:
            raise RuntimeError(f"Unknown checkpoint keys: {sorted(unknown)[:8]}")
        model_state = {}
        for name in keys:
            if name.startswith("model."):
                key = name.removeprefix("model.")
                dtype = torch.float32 if key.startswith(("action_encoder.", "state_encoder.")) else torch.bfloat16
                model_state[key] = weights.get_tensor(name).to(device=device, dtype=dtype)
        model.load_state_dict(model_state, strict=True, assign=True)
        del model_state
        vlm_keys = {name.removeprefix("vlm.") for name in keys if name.startswith("vlm.")}
        if vlm_keys:
            config = AutoConfig.from_pretrained(str(vlm_path), local_files_only=True)
            config._attn_implementation = "sdpa"
            original_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device("cpu"):
                    vlm = Cosmos3OmniModel(config)
            finally:
                torch.set_default_dtype(original_dtype)
            vlm = vlm.to(device)
        else:
            vlm = Cosmos3OmniModel.from_pretrained(
                str(vlm_path), dtype=torch.bfloat16, local_files_only=True,
                attn_implementation="sdpa", device_map=None,
            ).to(device)
        del vlm.language_model.layers[36:]
        vlm.language_model.norm = nn.Identity()
        if vlm_keys:
            targets = vlm.state_dict()
            if vlm_keys != set(targets):
                raise RuntimeError(
                    f"VLM checkpoint mismatch: missing={sorted(set(targets) - vlm_keys)[:8]} "
                    f"unexpected={sorted(vlm_keys - set(targets))[:8]}"
                )
            with torch.no_grad():
                for name, target in targets.items():
                    value = weights.get_tensor(f"vlm.{name}")
                    if value.shape != target.shape:
                        raise RuntimeError(f"VLM tensor shape mismatch: {name}")
                    target.copy_(value)
    return Policy(model, vlm).eval()