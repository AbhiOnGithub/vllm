# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped differential attention implemented on top of FlashAttention."""

from typing import Any, ClassVar

import torch
import torch.nn as nn
from einops import rearrange, repeat

from vllm.model_executor.layers.attention import Attention
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )


class GroupedDifferentialAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # KV writes are handled through do_kv_cache_update(), just like the
    # standard V1 FlashAttention backend.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "GROUPED_DIFF_ATTN"

    @staticmethod
    def get_impl_cls() -> type["GroupedDifferentialAttentionImpl"]:
        return GroupedDifferentialAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["GroupedDifferentialAttentionMetadataBuilder"]:
        return GroupedDifferentialAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if num_kv_heads % 2 != 0:
            raise ValueError("num_kv_heads must be divisible by 2.")
        return (2, 2, num_blocks, block_size, num_kv_heads // 2, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if include_num_layers_dimension:
            if cache_layout == "NHD":
                return (3, 0, 1, 2, 4, 5, 6)
            if cache_layout == "HND":
                return (3, 5, 0, 1, 2, 4, 6)
        else:
            if cache_layout == "NHD":
                return (0, 1, 2, 3, 4, 5)
            if cache_layout == "HND":
                return (0, 1, 2, 4, 3, 5)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


class GroupedDifferentialAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )

    def use_cascade_attention(self, *args: Any, **kwargs: Any) -> bool:
        return False


class GroupedDifferentialAttentionImpl(AttentionImpl[FlashAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        grouped_differential_attention_config: dict[str, Any] | None = None,
    ) -> None:
        if grouped_differential_attention_config is None:
            raise ValueError(
                "grouped_differential_attention_config is required for "
                "GroupedDifferentialAttentionImpl."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Grouped differential attention only supports decoder attention."
            )
        if num_kv_heads % 2 != 0:
            raise ValueError("num_kv_heads must be divisible by 2.")

        self.grouped_differential_attention_config = (
            grouped_differential_attention_config
        )
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_per_group = num_kv_heads // 2
        self.alibi_slopes = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None
            else None
        )
        self.sliding_window = (-1, -1) if sliding_window is None else (
            sliding_window - 1,
            0,
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        if self.vllm_flash_attn_version is None:
            raise RuntimeError("FlashAttention is required for grouped diff attention.")
        if is_quantized_kv_cache(kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.num_q1_heads = self.grouped_differential_attention_config.get(
            "num_q1_heads", self.num_heads // 2
        )
        self.num_q2_heads = self.grouped_differential_attention_config.get(
            "num_q2_heads", self.num_heads - self.num_q1_heads
        )
        if self.num_q1_heads + self.num_q2_heads != self.num_heads:
            raise ValueError("num_q1_heads + num_q2_heads must equal num_heads.")
        if self.num_q1_heads % self.num_q2_heads != 0:
            raise ValueError("num_q1_heads must be divisible by num_q2_heads.")
        if self.num_q1_heads % self.num_kv_heads_per_group != 0:
            raise ValueError("num_q1_heads must divide grouped KV heads.")
        if self.num_q2_heads % self.num_kv_heads_per_group != 0:
            raise ValueError("num_q2_heads must divide grouped KV heads.")
        if 2 * self.num_q1_heads != self.num_heads:
            raise NotImplementedError(
                "vLLM Attention currently expects grouped differential "
                "attention to return num_heads outputs; Phi4Flash uses the "
                "supported 1:1 q1/q2 split."
            )

        self.num_q_head_groups = self.num_q2_heads
        self.q_head_group_size = self.num_heads // self.num_q_head_groups
        self.q_head_group_ratio = self.num_q1_heads // self.num_q2_heads
        self.subln = self.grouped_differential_attention_config["subln"]
        if not isinstance(self.subln, nn.Module):
            raise TypeError("subln must be an nn.Module.")

        self.lambda_full: torch.Tensor | None = None
        self.lambda_init = self.grouped_differential_attention_config["lambda_init"]
        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

    def split_q_heads(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = rearrange(
            q,
            "... (num_groups group_size) D -> ... num_groups group_size D",
            num_groups=self.num_q_head_groups,
            group_size=self.q_head_group_size,
        )
        local_q1_heads = self.num_q1_heads // self.num_q_head_groups
        q1 = q[..., :local_q1_heads, :].flatten(-3, -2)
        q2 = q[..., local_q1_heads:, :].flatten(-3, -2)
        return q1.contiguous(), q2.contiguous()

    @staticmethod
    def split_kv_heads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(x, "... (H two) D -> ... H two D", two=2)
        x1 = x[..., 0, :]
        x2 = x[..., 1, :]
        return x1.contiguous(), x2.contiguous()

    @staticmethod
    def split_kv_cache(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            return torch.empty(0), torch.empty(0)
        return x[0], x[1]

    def do_kv_cache_update(
        self,
        layer: Attention,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if key is None or value is None or kv_cache.numel() == 0:
            return
        k1, k2 = self.split_kv_heads(key)
        v1, v2 = self.split_kv_heads(value)
        kv_cache1, kv_cache2 = self.split_kv_cache(kv_cache)
        self._populate_kv_cache(layer, k1, v1, kv_cache1, slot_mapping)
        self._populate_kv_cache(layer, k2, v2, kv_cache2, slot_mapping)

    def _populate_kv_cache(
        self,
        layer: Attention,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = self.split_kv_cache(kv_cache)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _lambda_full(self, query: torch.Tensor) -> torch.Tensor:
        if self.lambda_full is None:
            lambda_q1 = self.grouped_differential_attention_config["lambda_q1"]
            lambda_k1 = self.grouped_differential_attention_config["lambda_k1"]
            lambda_q2 = self.grouped_differential_attention_config["lambda_q2"]
            lambda_k2 = self.grouped_differential_attention_config["lambda_k2"]
            lambda_1 = torch.exp(torch.sum(lambda_q1 * lambda_k1, dim=-1).float())
            lambda_2 = torch.exp(torch.sum(lambda_q2 * lambda_k2, dim=-1).float())
            self.lambda_full = (lambda_1 - lambda_2 + self.lambda_init).type_as(query)
        return self.lambda_full

    def forward(
        self,
        layer: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for grouped "
                "differential attention."
            )
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.use_cascade:
            raise NotImplementedError(
                "Cascade attention is not supported for grouped differential "
                "attention."
            )
        if attn_metadata.dcp_context_kv_lens is not None:
            raise NotImplementedError(
                "Decode-context parallelism is not supported for grouped "
                "differential attention."
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        q1, q2 = self.split_q_heads(q)

        k1 = k2 = v1 = v2 = None
        if key is not None and value is not None:
            k1, k2 = self.split_kv_heads(key[:num_actual_tokens])
            v1, v2 = self.split_kv_heads(value[:num_actual_tokens])

        kv_cache1, kv_cache2 = self.split_kv_cache(kv_cache)
        key_cache1, value_cache1 = self.split_kv_cache(kv_cache1)
        key_cache2, value_cache2 = self.split_kv_cache(kv_cache2)

        attn11 = self.forward_single_attention(
            layer, q1, k1, v1, key_cache1, value_cache1, attn_metadata
        ).view(q1.shape)
        attn12 = self.forward_single_attention(
            layer, q1, k1, v2, key_cache1, value_cache2, attn_metadata
        ).view(q1.shape)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = self.forward_single_attention(
            layer, q2, k2, v1, key_cache2, value_cache1, attn_metadata
        ).view(q2.shape)
        attn22 = self.forward_single_attention(
            layer, q2, k2, v2, key_cache2, value_cache2, attn_metadata
        ).view(q2.shape)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        attn2 = repeat(attn2, "... H D -> ... (H r) D", r=self.q_head_group_ratio)

        # Differential attention subtracts a learned noise-head result before
        # the local RMSNorm used by the Phi4Flash checkpoint.
        attn = attn1 - self._lambda_full(query) * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn_output = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)
        output[:num_actual_tokens].copy_(attn_output)
        return output

    def forward_single_attention(
        self,
        layer: Attention,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            dtype = GroupedDifferentialAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)

        cu_seqlens_q = attn_metadata.query_start_loc
        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads_per_group)
        q_descale = (
            layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None
        )

        return flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window),
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
        )
