"""LoLA bridge and DiT, modified for fixed inference. Original notices: LICENSE."""

import math

import torch
from torch import nn
from torch.nn import functional as functional

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward, Flux2Modulation


STREAMS = ("ctx_vlm", "ctx_grip", "ctx_arm", "grip", "arm")


def rope_tables(length, head_dim, device, dtype):
    frequency = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    angles = torch.outer(torch.arange(length, device=device).float(), frequency)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(value, cosine, sine):
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cosine[None, None] + rotated * sine[None, None]


class ContextBridgeBlock(nn.Module):
    def __init__(self, width, num_heads, ffn_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.ln1 = nn.LayerNorm(width, eps=1e-6)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.ln2 = nn.LayerNorm(width, eps=1e-6)
        self.gate_proj = nn.Linear(width, ffn_dim, bias=False)
        self.up_proj = nn.Linear(width, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, width, bias=False)

    def forward(self, value, cosine, sine):
        batch_size, length, _ = value.shape
        hidden = self.ln1(value)
        shape = (batch_size, length, self.num_heads, self.head_dim)
        query = apply_rope(self.q_proj(hidden).view(shape).transpose(1, 2), cosine, sine)
        key = apply_rope(self.k_proj(hidden).view(shape).transpose(1, 2), cosine, sine)
        projected = self.v_proj(hidden).view(shape).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(query, key, projected)
        attended = attended.transpose(1, 2).reshape(batch_size, length, -1)
        value = value + self.o_proj(attended)
        hidden = self.ln2(value)
        return value + self.down_proj(functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class ContextBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4096, 2048)
        self.input_norm = nn.LayerNorm(2048, eps=1e-6)
        self.blocks = nn.ModuleList([ContextBridgeBlock(2048, 16, 8192) for _ in range(8)])
        self.final_norm = nn.LayerNorm(2048, eps=1e-6)
        self.output_proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024, eps=1e-6),
            nn.SiLU(),
            nn.Linear(1024, 1024),
        )
        self.shortcut = nn.Linear(4096, 1024)

    def forward(self, features):
        hidden = self.input_norm(self.input_proj(features))
        cosine, sine = rope_tables(hidden.shape[1], 128, hidden.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, cosine, sine)
        output = self.output_proj(self.final_norm(hidden)) + self.shortcut(features)
        return output[:, :-1], output[:, -1]


class ConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, 1024))
        self.cond_mlp = nn.Sequential(nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 1024))
        self.state_mlp = nn.Sequential(nn.Linear(7, 1024), nn.SiLU(), nn.Linear(1024, 1024))

    def forward(self, time, empty_token, state):
        fraction = torch.linspace(0.0, 1.0, 128, device=time.device, dtype=torch.float32)
        period = 0.004 * (4.0 / 0.004) ** fraction
        angles = time[:, None] * (1.0 / period * 2 * math.pi)[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1).to(empty_token.dtype)
        return self.time_mlp(embedding) + self.cond_mlp(empty_token) + self.state_mlp(state)


def add_projections(module, prefix, width, head_dim):
    for projection in ("q", "k", "v"):
        setattr(module, f"{prefix}_to_{projection}", nn.Linear(width, width, bias=False))
    for projection in ("q", "k"):
        setattr(module, f"{prefix}_norm_{projection}", nn.RMSNorm(head_dim, eps=1e-6))


def joint_attention(module, normalized, rotary, mask):
    queries, keys, values = [], [], []
    for prefix, hidden in zip(STREAMS, normalized):
        shape = (module.num_heads, module.head_dim)
        query = getattr(module, f"{prefix}_to_q")(hidden).unflatten(-1, shape)
        key = getattr(module, f"{prefix}_to_k")(hidden).unflatten(-1, shape)
        value = getattr(module, f"{prefix}_to_v")(hidden).unflatten(-1, shape)
        queries.append(getattr(module, f"{prefix}_norm_q")(query))
        keys.append(getattr(module, f"{prefix}_norm_k")(key))
        values.append(value)
    query = apply_rotary_emb(torch.cat(queries, dim=1), rotary, sequence_dim=1).transpose(1, 2)
    key = apply_rotary_emb(torch.cat(keys, dim=1), rotary, sequence_dim=1).transpose(1, 2)
    value = torch.cat(values, dim=1).transpose(1, 2)
    output = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None])
    output = output.transpose(1, 2).flatten(2, 3).to(query.dtype)
    return output.split([hidden.shape[1] for hidden in normalized], dim=1)


class DoubleBlock(nn.Module):
    def __init__(self, width=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        for prefix in STREAMS:
            add_projections(self, prefix, width, self.head_dim)
            for index in (1, 2):
                setattr(self, f"{prefix}_norm{index}", nn.LayerNorm(width, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{prefix}_to_out", nn.Linear(width, width, bias=False))
        self.ctx_shared_ff = Flux2FeedForward(dim=width, dim_out=width, mult=4.0, bias=False)
        self.arm_ff = Flux2FeedForward(dim=width, dim_out=width, mult=4.0, bias=False)
        self.grip_ff = Flux2FeedForward(dim=width, dim_out=width, mult=2.0, bias=False)

    def forward(self, streams, modulations, rotary, mask):
        parameters = [Flux2Modulation.split(modulation, 2) for modulation in modulations]
        normalized = []
        for prefix, hidden, (attention, _) in zip(STREAMS, streams, parameters):
            shift, scale, _ = attention
            normalized.append((1 + scale) * getattr(self, f"{prefix}_norm1")(hidden) + shift)
        attended = joint_attention(self, normalized, rotary, mask)
        residuals, ff_inputs = [], []
        for prefix, hidden, output, (attention, feedforward) in zip(STREAMS, streams, attended, parameters):
            residual = hidden + attention[2] * getattr(self, f"{prefix}_to_out")(output)
            shift, scale, _ = feedforward
            residuals.append(residual)
            ff_inputs.append(getattr(self, f"{prefix}_norm2")(residual) * (1 + scale) + shift)
        context = self.ctx_shared_ff(torch.cat(ff_inputs[:3], dim=1))
        outputs = list(context.split([hidden.shape[1] for hidden in streams[:3]], dim=1))
        outputs.extend((self.grip_ff(ff_inputs[3]), self.arm_ff(ff_inputs[4])))
        return [hidden + parameters[index][1][2] * output for index, (hidden, output) in enumerate(zip(residuals, outputs))]


class SingleBlock(nn.Module):
    def __init__(self, width=1024, num_heads=8, skip_ctx_ff=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.skip_ctx_ff = skip_ctx_ff
        for prefix in STREAMS:
            add_projections(self, prefix, width, self.head_dim)
            setattr(self, f"{prefix}_norm", nn.LayerNorm(width, elementwise_affine=False, eps=1e-6))
        if not skip_ctx_ff:
            self.ctx_shared_ff = Flux2FeedForward(dim=width, dim_out=width, mult=4.0, bias=False)
        self.arm_ff = Flux2FeedForward(dim=width, dim_out=width, mult=4.0, bias=False)
        self.grip_ff = Flux2FeedForward(dim=width, dim_out=width, mult=2.0, bias=False)

    def forward(self, streams, modulations, rotary, mask):
        parameters = [Flux2Modulation.split(modulation, 1)[0] for modulation in modulations]
        normalized = [
            (1 + scale) * getattr(self, f"{prefix}_norm")(hidden) + shift
            for prefix, hidden, (shift, scale, _) in zip(STREAMS, streams, parameters)
        ]
        attended = joint_attention(self, normalized, rotary, mask)
        combined = list(attended)
        if not self.skip_ctx_ff:
            context = self.ctx_shared_ff(torch.cat(normalized[:3], dim=1))
            outputs = context.split([hidden.shape[1] for hidden in streams[:3]], dim=1)
            for index, output in enumerate(outputs):
                combined[index] = combined[index] + output
        combined[3] = combined[3] + self.grip_ff(normalized[3])
        combined[4] = combined[4] + self.arm_ff(normalized[4])
        return [hidden + gate * output for hidden, output, (_, _, gate) in zip(streams, combined, parameters)]


def multiaxis_rope(lengths, head_dim, device, dtype):
    coordinates = []
    for length, time_axis in zip(lengths, (0, 1, 1, 2, 2)):
        values = torch.zeros(length, 4, dtype=torch.long, device=device)
        values[:, 0] = time_axis
        values[:, 3] = torch.arange(length, device=device)
        coordinates.append(values)
    coordinates = torch.cat(coordinates, dim=0)
    axis_dim = head_dim // 4
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, axis_dim, 2, device=device).float() / axis_dim))
    angles = torch.cat([
        torch.outer(coordinates[:, axis].float(), frequencies).repeat_interleave(2, dim=-1)
        for axis in range(4)
    ], dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_embedder = ConditionEmbedder()
        for prefix in STREAMS:
            setattr(self, f"{prefix}_double_modulation", Flux2Modulation(1024, mod_param_sets=2, bias=False))
            setattr(self, f"{prefix}_single_modulation", Flux2Modulation(1024, mod_param_sets=1, bias=False))
        for name in (
            "vlm_modality_emb", "arm_ctx_modality_emb", "grip_ctx_modality_emb",
            "arm_target_modality_emb", "grip_target_modality_emb", "vlm_start_emb",
            "vlm_end_emb", "hist_start_emb", "hist_end_emb", "previous_task_end_emb",
        ):
            setattr(self, name, nn.Parameter(torch.randn(1, 1, 1024) * 0.02))
        self.double_blocks = nn.ModuleList([DoubleBlock() for _ in range(4)])
        self.single_blocks = nn.ModuleList([SingleBlock(skip_ctx_ff=index == 11) for index in range(12)])
        for name, dimension in (("arm_out_proj", 48), ("gripper_out_proj", 8)):
            setattr(self, name, nn.Sequential(
                nn.LayerNorm(1024, eps=1e-6), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, dimension),
            ))

    def forward(self, arm, grip, arm_history, grip_history, visual, empty_token, time, state, mask):
        streams = [
            visual + self.vlm_modality_emb,
            grip_history + self.grip_ctx_modality_emb,
            arm_history + self.arm_ctx_modality_emb,
            grip + self.grip_target_modality_emb,
            arm + self.arm_target_modality_emb,
        ]
        condition = self.cond_embedder(time, empty_token, state)
        double_modulations = [getattr(self, f"{prefix}_double_modulation")(condition) for prefix in STREAMS]
        single_modulations = [getattr(self, f"{prefix}_single_modulation")(condition) for prefix in STREAMS]
        rotary = multiaxis_rope([hidden.shape[1] for hidden in streams], 128, arm.device, arm.dtype)
        for block in self.double_blocks:
            streams = block(streams, double_modulations, rotary, mask)
        for block in self.single_blocks:
            streams = block(streams, single_modulations, rotary, mask)
        return streams[4], streams[3]