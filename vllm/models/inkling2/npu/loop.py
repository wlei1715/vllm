# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2, NPU platform branch (Ascend 910B) — correctness-first.

Round 2 scope: the **full text stack** in torch, mirroring
`nvidia/loop.py` + `nvidia/model.py`:

* block attention residual (``attn_res``: softmax mix of the completed
  residual blocks + the running partial block per pseudo-query);
* rel-attention with q/r L2 clipping, dynamic/static K/V short convs,
  sliding-window / causal masking, attention sink + output gate;
* hidden-stream short convs with the fused residual update;
* the loop schedule: prelude (L0..5), ``num_passes`` x core (L6..17) with
  per-pass KV + conv states and per-sublayer message blending, coda (L18..23);
* the routed MoE (`moe.py`, validated to Level 3).

Still out of scope (documented TODOs): TP > 1, MTP, vision, the vLLM v1
engine plumbing (attention metadata / KV-cache block manager — caches are
plain torch tensors here), and fused Ascend kernels (performance round).
"""

from __future__ import annotations

import torch
from torch import nn

from ..configs import Inkling2ModelConfig
from .attention import Inkling2NpuAttention, Inkling2NpuShortConv
from .moe import Inkling2NpuMoE
from .ops import attn_res, blend, log_scaling_tau, rms_norm

LOOP_STREAMS = ("attn", "k", "v", "mlp")


class NpuRMSNorm(nn.Module):
    """`Inkling2RMSNorm`: `num_inputs > 1` stacks per-input weights."""

    def __init__(
        self, hidden_size: int, eps: float, num_inputs: int = 1
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps)


class NpuAttnRes(nn.Module):
    """`Inkling2AttnRes`: fp32 pseudo-queries pre-scaled by `D**-0.5`."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        self.scale = hidden**-0.5


class NpuMessageController(nn.Module):
    """One loop stream's blend weights (`controller_{down,gain,forget}`)."""

    def __init__(self, rank: int, hidden_size: int) -> None:
        super().__init__()
        bf16 = torch.bfloat16
        self.down_weight = nn.Parameter(
            torch.empty(rank, 2 * hidden_size, dtype=bf16), requires_grad=False
        )
        self.gain_weight = nn.Parameter(
            torch.empty(hidden_size, rank, dtype=bf16), requires_grad=False
        )
        self.forget_weight = nn.Parameter(
            torch.empty(hidden_size, rank, dtype=bf16), requires_grad=False
        )


class NpuDecoderLayer(nn.Module):
    """One decoder layer (loop core layers carry the stream controllers)."""

    def __init__(self, config: Inkling2ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn = Inkling2NpuAttention(config, self.is_local)
        self.attn_norm = NpuRMSNorm(hidden, eps, num_inputs=self.num_attn_inputs)
        self.mlp_norm = NpuRMSNorm(hidden, eps)
        self.mlp = Inkling2NpuMoE(config)
        self.attn_sconv = Inkling2NpuShortConv(hidden, config.sconv_kernel_size)
        self.mlp_sconv = Inkling2NpuShortConv(hidden, config.sconv_kernel_size)
        if layer_id in config.loop_core_layers:
            rank = config.loop_config["gate_rank"]
            self.controllers = nn.ModuleDict(
                {s: NpuMessageController(rank, hidden) for s in LOOP_STREAMS}
            )

    def attention(
        self,
        positions: torch.Tensor,
        inputs,
        tau: torch.Tensor | None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """This rank's attention output (TP=1: the full one)."""
        if self.is_local and len(inputs) == 1:
            return self.attn(positions, inputs[0], pass_idx=pass_idx)
        return self.attn(
            positions,
            inputs[0],
            inputs[1],
            inputs[2],
            None if self.is_local else tau,
            pass_idx=pass_idx,
        )

    def blend(
        self,
        streams: tuple[str, ...],
        dest: torch.Tensor,
        message: torch.Tensor,
        norm: NpuRMSNorm,
        norm_index: int | None = None,
    ) -> torch.Tensor:
        down = torch.stack([self.controllers[s].down_weight for s in streams])
        up = torch.stack(
            [
                torch.cat(
                    [self.controllers[s].gain_weight, self.controllers[s].forget_weight],
                    0,
                )
                for s in streams
            ]
        )
        weight = norm.weight if norm_index is None else norm.weight[norm_index]
        return blend(dest, message, down, up, weight.expand(len(streams), -1), norm.eps)

    def reset_state(self) -> None:
        self.attn.reset_state()
        self.attn_sconv.reset_state()
        self.mlp_sconv.reset_state()


class NpuLoopModel(nn.Module):
    """`Inkling2LoopModel`: block stack + the loop visit schedule."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.padded_vocab_size or config.vocab_size,
            self.hidden_size,
            dtype=torch.bfloat16,
        )
        self.embed_norm = NpuRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            NpuDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.attn_res = NpuAttnRes(config)
        self.norm = NpuRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.loop_core = config.loop_core_layers
        core = len(self.loop_core)
        shape = (2 * core, self.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer(
            "message_queries",
            torch.zeros(core, len(LOOP_STREAMS), self.hidden_size),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(
            -1, 2, self.hidden_size
        )
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def _max_loop_blocks(self) -> int:
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: NpuRMSNorm | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor]:
        """`nvidia/loop.py:241` for TP=1 (no all-reduce / two-shot paths)."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        messages = None
        if bank is not None:
            # Four streams' messages: a mix of the bank's two blocks.
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            dests = inputs
            if layer.is_local:
                dests = dests.expand(3, *dests.shape[1:])
            attn_inputs = layer.blend(("attn", "k", "v"), dests, messages[:3], layer.attn_norm)
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        attn_out = layer.attn_sconv(attn_partial, pass_idx=pass_idx)  # fp32
        if first:
            mlp_in = attn_res(
                stack[: num_blocks + 1],
                attn_out,
                None,
                self.attn_res.mlp_queries[i],
                None if messages is not None else layer.mlp_norm.weight,
                layer.mlp_norm.eps if messages is None else 0.0,
                self.normalize_values,
            )
        else:
            mlp_in = attn_res(
                stack[:num_blocks],
                partial,
                attn_out,
                self.attn_res.mlp_queries[i],
                None if messages is not None else layer.mlp_norm.weight,
                layer.mlp_norm.eps if messages is None else 0.0,
                self.normalize_values,
            )
        if messages is not None:
            mlp_in = layer.blend(("mlp",), mlp_in, messages[3:], layer.mlp_norm)

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        moe_out = layer.mlp(mlp_in[0].to(torch.bfloat16))
        # The conv result is rounded to bf16 before the residual add, then
        # the sum is rounded once more (nvidia/ops/sconv.py `_sconv_kernel`).
        y = layer.mlp_sconv(moe_out, pass_idx=pass_idx).to(torch.bfloat16)
        acc = attn_out if residual is None else residual.float() + attn_out
        stack[num_blocks].copy_((acc + y.float()).to(stack.dtype))

        outputs = attn_res(
            stack[:num_blocks],
            stack[num_blocks],
            None,
            next_query,
            None if next_norm is None else next_norm.weight,
            0.0 if next_norm is None else next_norm.eps,
            self.normalize_values,
        )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty((self._max_loop_blocks() + 1, num_tokens, hidden))
        stack[0].copy_(inputs_embeds)
        tau = log_scaling_tau(
            positions,
            self.config.log_scaling_n_floor,
            self.config.log_scaling_alpha,
        )
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) as the bank.
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        self.attn_res.attn_queries[i - 1],
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
            )
        assert inputs is not None
        return inputs[0]


class Inkling2LoopForConditionalGeneration(nn.Module):
    """`Inkling2LoopForConditionalGeneration` for the NPU platform."""

    # Checkpoint name substitutions (the fused qkvr / sharded norms of the
    # NVIDIA branch are not fused here — TP=1 keeps the checkpoint names).
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    # Full layers hold q/k/v input norms; the checkpoint has three tensors.
    _shards = {
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }
    _scaled_queries = ("attn_queries", "mlp_queries", "output_query")

    def __init__(self, *, vllm_config=None, config=None, prefix: str = "") -> None:
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        assert text_config.loop_config is not None, "loop checkpoint required"
        self.config = text_config
        self.model = NpuLoopModel(text_config)
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.padded_vocab_size or text_config.vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def reset_state(self) -> None:
        self.model.reset_state()

    def load_weights(self, weights) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        scale = self.model.attn_res.scale
        for name, tensor in weights:
            if name.startswith(
                ("model.visual.", "layers.", "markov_head.", "mask_embed", "fuse_weight")
            ):
                # Vision encoder and MTP tensors: unsupported this round.
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard = None
            for old, (new, idx) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard = idx
                    break
            if name not in params:
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if any(k in name for k in self._scaled_queries):
                param.data.copy_(
                    tensor.view(param.shape).to(torch.float32).mul_(scale)
                )
            elif shard is not None or (
                param.dim() == 2
                and tensor.dim() == 1
                and param.shape[1] == tensor.shape[0]
            ):
                # Stacked norms: one row per sublayer input (q/k/v), local
                # layers and the single-input norms use row 0.
                idx = 0 if shard is None else shard
                assert param.data[idx].shape == tensor.shape, name
                param.data[idx].copy_(tensor.to(param.dtype))
            else:
                assert param.shape == tensor.shape, (name, param.shape, tensor.shape)
                param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {missing[:8]}")
        self.model.process_weights_after_loading()
        return loaded

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None):
        """Text-only forward over a single sequence (offline parity runs)."""
        self.reset_state()  # one independent single-sequence run per forward
        num_tokens = input_ids.shape[0]
        if positions is None:
            positions = torch.arange(num_tokens, device=input_ids.device)
        embeds = self.model.embed_norm(self.model.embed_tokens(input_ids))
        hidden = self.model(positions, embeds)
        return torch.nn.functional.linear(hidden.float(), self.lm_head.weight.float())


__all__ = ["Inkling2LoopForConditionalGeneration"]
