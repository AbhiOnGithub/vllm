# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Phi4Flash model."""

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer import (
    split_batch_to_prefill_and_decode,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import selective_state_update
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata, AttentionType
from vllm.v1.attention.backends.grouped_diff_attn import (
    GroupedDifferentialAttentionBackend,
)
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata

from .utils import make_empty_intermediate_tensors_factory, make_layers, maybe_prefix

logger = init_logger(__name__)


class SwiGLUActivation(nn.Module):
    """Phi4Flash uses x * silu(y), which is opposite of vLLM's SiluAndMul."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * F.silu(y)


def _get_sliding_window(config: Any, layer_idx: int) -> int | None:
    sliding_window = getattr(config, "sliding_window", None)
    if isinstance(sliding_window, (list, tuple)):
        return sliding_window[layer_idx]
    if sliding_window is None:
        return None
    if layer_idx < config.num_hidden_layers // 2 and layer_idx % 2 == 1:
        return int(sliding_window)
    return None


class SambaYMLP(nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.fc1 = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.activation_fn = ACT2FN[config.hidden_act]
        self.swiglu = SwiGLUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.fc1(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        if self.config.hidden_act == "silu":
            hidden_states = self.swiglu(gate, up)
        else:
            hidden_states = up * self.activation_fn(gate)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SambaYAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_idx: int,
        yoco_cross: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.yoco_cross = yoco_cross

        tp_size = get_tensor_model_parallel_world_size()
        if self.total_num_heads % tp_size != 0:
            raise ValueError("Tensor parallel size must divide attention heads.")
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError("Tensor parallel size must divide KV heads.")
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError("Tensor parallel size must be divisible by KV heads.")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        op_size = self.total_num_heads * self.head_dim + 2 * (
            self.total_num_kv_heads * self.head_dim
        )

        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if yoco_cross:
            self.Wqkv = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.Wqkv",
            )
        else:
            self.Wqkv = ColumnParallelLinear(
                self.hidden_size,
                op_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.Wqkv",
            )

        if self.total_num_heads % 2 != 0 or self.total_num_kv_heads % 2 != 0:
            raise ValueError("Phi4Flash differential attention requires even heads.")

        self.lambda_init = self.lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        grouped_diff_config = {
            "lambda_init": self.lambda_init,
            "lambda_q1": self.lambda_q1,
            "lambda_k1": self.lambda_k1,
            "lambda_q2": self.lambda_q2,
            "lambda_k2": self.lambda_k2,
            "subln": self.subln,
        }

        target_layer = config.num_hidden_layers // 2 + 1
        kv_sharing_target_layer_name = (
            f"model.layers.{target_layer}.attn.attn" if yoco_cross else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=_get_sliding_window(config, layer_idx),
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=GroupedDifferentialAttentionBackend,
            grouped_differential_attention_config=grouped_diff_config,
        )

    @staticmethod
    def lambda_init_fn(depth: int) -> float:
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.yoco_cross:
            q, _ = self.Wqkv(hidden_states)
            attn_output = self.attn(q, None, None)
        else:
            qkv, _ = self.Wqkv(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            attn_output = self.attn(q, k, v)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class Phi4Mamba(MambaBase):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        yoco_cross: bool = False,
        yoco_kv: bool = False,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.d_model = d_model
        self.ssm_state_size = d_state
        self.conv_kernel_size = d_conv
        self.intermediate_size = int(expand * d_model)
        self.time_step_rank = (
            math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.activation = "silu"
        self.swiglu = SwiGLUActivation()
        self.tp_size = get_tensor_model_parallel_world_size()

        if yoco_cross:
            self.in_proj = ColumnParallelLinear(
                d_model,
                self.intermediate_size,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )
            self.out_proj = RowParallelLinear(
                self.intermediate_size,
                d_model,
                bias=bias,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.out_proj",
            )
            self.model_config = model_config
            self.cache_config = cache_config
            self.prefix = prefix
            return

        self.conv1d = ColumnParallelLinear(
            input_size=d_conv,
            output_size=self.intermediate_size,
            bias=conv_bias,
            quant_config=None,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            d_model,
            [self.intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.x_proj",
        )
        self.dt_proj = ColumnParallelLinear(
            self.time_step_rank,
            self.intermediate_size,
            bias=True,
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dt_proj",
        )

        def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            tp_rank = get_tensor_model_parallel_rank()
            param.data.copy_(
                loaded_weight.split(loaded_weight.shape[0] // self.tp_size, dim=0)[
                    tp_rank
                ]
            )

        def A_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            weight_loader(param, -torch.exp(loaded_weight.float()))

        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size // self.tp_size,
                self.ssm_state_size,
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(self.intermediate_size // self.tp_size))
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})
        set_weight_attrs(self.D, {"weight_loader": weight_loader})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            d_model,
            bias=bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.yoco_cross:
            if yoco_key_values is None:
                raise ValueError("YoCo cross Mamba layer requires yoco_key_values.")
            projected, _ = self.in_proj(hidden_states)
            hidden_states = self.swiglu(projected, yoco_key_values)
            output, _ = self.out_proj(hidden_states)
            return output, yoco_key_values

        output = torch.empty_like(hidden_states)
        yoco_output = (
            torch.empty(
                hidden_states.shape[0],
                self.intermediate_size // self.tp_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if self.yoco_kv
            else torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device)
        )
        torch.ops.vllm.phi4_mamba(
            hidden_states,
            output,
            yoco_output,
            _encode_layer_name(self.prefix),
        )
        return output, yoco_output if self.yoco_kv else yoco_key_values

    def _ssm_transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ssm_parameters = self.x_proj(x.contiguous())[0]
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        discrete_time_step = self.dt_proj(time_step.contiguous())[0].transpose(-2, -1)
        return discrete_time_step, B, C

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        yoco_output: torch.Tensor,
    ) -> None:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        is_mamba_cache_all = self.cache_config.mamba_cache_mode == "all"

        attn_metadata: AttentionMetadata | None = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, Mamba1AttentionMetadata)
            query_start_loc_p = attn_metadata.query_start_loc_p
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            # V1 may store convolution state in either SD or DS layout.
            conv_state = (
                self.kv_cache[0]
                if is_conv_state_dim_first()
                else self.kv_cache[0].transpose(-1, -2)
            )
            ssm_state = self.kv_cache[1]
            has_initial_states_p = attn_metadata.has_initial_states_p
            cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
            last_chunk_indices_p = attn_metadata.last_chunk_indices_p

        projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        hidden_states_BC, gate = projected_states.chunk(2, dim=-2)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if attn_metadata is None:
            scan_tokens = hidden_states_BC.transpose(-2, -1).contiguous()
            if self.yoco_kv:
                yoco_output[: scan_tokens.shape[0]].copy_(scan_tokens)
                scan_tokens = self.swiglu(gate.transpose(-2, -1), scan_tokens)
            out, _ = self.out_proj(scan_tokens)
            output[: out.shape[0]] = out
            return

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        num_decodes = num_decode_tokens
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens

        split = split_batch_to_prefill_and_decode(
            hidden_states_BC, gate, num_prefill_tokens, num_decode_tokens
        )
        hidden_states_BC_p = split.hidden_states_BC_p
        hidden_states_BC_d = split.hidden_states_BC_d
        gate_p = split.gate_p
        gate_d = split.gate_d

        if is_mamba_cache_all:
            block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_computed_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        ssm_outputs: list[torch.Tensor] = []

        if has_prefill:
            conv_out_p = causal_conv1d_fn(
                hidden_states_BC_p,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
            )
            discrete_time_step_p, B_p, C_p = self._ssm_transform(
                conv_out_p.transpose(-2, -1)
            )
            scan_out_p = selective_scan_fn(
                conv_out_p,
                ssm_state,
                discrete_time_step_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                None if self.yoco_kv else gate_p,
                self._time_proj_bias(),
                delta_softplus=True,
                cache_indices=state_indices_tensor_p,
                has_initial_state=has_initial_states_p,
                query_start_loc=query_start_loc_p,
                block_size=mamba_block_size,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                cu_chunk_seqlen=cu_chunk_seqlen_p,
                last_chunk_indices=last_chunk_indices_p,
            )
            ssm_outputs.append(scan_out_p)

        if has_decode:
            assert state_indices_tensor_d is not None
            if is_mamba_cache_all:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
            else:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d

            conv_out_d = causal_conv1d_update(
                hidden_states_BC_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
            ).transpose(0, 1)
            discrete_time_step_d, B_d, C_d = self._ssm_transform(
                conv_out_d.transpose(-2, -1)
            )
            scan_outputs_d = torch.empty_like(hidden_states_BC_d.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1),
                discrete_time_step_d.transpose(0, 1),
                self.A,
                B_d,
                C_d,
                self.D,
                self._time_proj_bias(),
                z=None if self.yoco_kv else gate_d.transpose(0, 1),
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=scan_outputs_d,
            )
            ssm_outputs.insert(0, scan_outputs_d.transpose(0, 1))

        scan_outputs = (
            ssm_outputs[0] if len(ssm_outputs) == 1 else torch.cat(ssm_outputs, dim=-1)
        )
        scan_tokens = scan_outputs.transpose(-2, -1).contiguous()
        if self.yoco_kv:
            # Upper YoCo layers consume the ungated SSM output while this layer
            # applies Phi4Flash's custom gate * silu(ssm) projection locally.
            yoco_output[:num_actual_tokens].copy_(scan_tokens)
            gate_tokens = gate[..., :num_actual_tokens].transpose(-2, -1).contiguous()
            scan_tokens = self.swiglu(gate_tokens, scan_tokens)
        out, _ = self.out_proj(scan_tokens)
        output[:num_actual_tokens] = out

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        if self.yoco_cross:
            return None
        return super().get_kv_cache_spec(vllm_config)

    @property
    def mamba_type(self) -> str:
        return "mamba1"

    def _time_proj_bias(self) -> torch.Tensor | None:
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            return self.dt_proj.bias.float()
        return None


class SambaYDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.yoco_mb = layer_idx >= config.num_hidden_layers // 2
        self.yoco_cross = layer_idx >= (config.num_hidden_layers // 2 + 2)
        self.use_mamba = (
            config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        )

        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        if self.use_mamba:
            self.attn = Phi4Mamba(
                config.hidden_size,
                d_state=getattr(config, "mamba_d_state", 16),
                d_conv=getattr(config, "mamba_d_conv", 4),
                expand=getattr(config, "mamba_expand", 2),
                dt_rank=getattr(config, "mamba_dt_rank", "auto"),
                conv_bias=getattr(config, "mamba_conv_bias", True),
                bias=getattr(config, "mamba_proj_bias", False),
                yoco_cross=self.yoco_cross,
                yoco_kv=self.yoco_mb,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            self.attn = SambaYAttention(
                config,
                layer_idx=layer_idx,
                yoco_cross=self.yoco_cross,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = SambaYMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        hidden_states: torch.Tensor,
        ssm_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states.to(dtype=self.input_layernorm.weight.dtype)
        )
        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(hidden_states, ssm_output)
            residual = residual.to(torch.float32)
        else:
            attn_outputs = self.attn(hidden_states)
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype)
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, ssm_output


@support_torch_compile
class SambaYModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if get_pp_group().world_size != 1:
            raise ValueError("Phi4Flash does not support pipeline parallelism.")

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str) -> SambaYDecoderLayer:
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return SambaYDecoderLayer(
                config,
                layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        ssm_output = None
        for i in range(self.start_layer, self.end_layer):
            hidden_states, ssm_output = self.layers[i](hidden_states, ssm_output)
        hidden_states = self.final_layernorm(
            hidden_states.to(dtype=self.final_layernorm.weight.dtype)
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "A_log" in name:
                name = name.replace("A_log", "A")
            if "inner_cross_attn." in name:
                name = name.replace("inner_cross_attn.", "")
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Phi4FlashForCausalLM(nn.Module, HasInnerState, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=getattr(hf_config, "mamba_expand", 2)
            * hf_config.hidden_size,
            state_size=getattr(hf_config, "mamba_d_state", 16),
            conv_kernel=getattr(hf_config, "mamba_d_conv", 4),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba1_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.model = SambaYModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                if not lora_config
                else lora_config.lora_vocab_padding_size
            ),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            if "A_log" in name:
                name = name.replace("A_log", "A")
            if "inner_cross_attn." in name:
                name = name.replace("inner_cross_attn.", "")
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)
        if self.config.tie_word_embeddings:
            loaded.add("lm_head.weight")
        return loaded


def phi4_mamba(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    yoco_output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_impl(
        hidden_states=hidden_states,
        output=output,
        yoco_output=yoco_output,
    )


def phi4_mamba_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    yoco_output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="phi4_mamba",
    op_func=phi4_mamba,
    mutates_args=["output", "yoco_output"],
    fake_impl=phi4_mamba_fake,
)
