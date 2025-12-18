# SPDX-License-Identifier: Apache-2.0
"""MPS Attention Backend for vLLM on Apple Silicon.

This backend uses PyTorch's scaled_dot_product_attention which is
natively supported on MPS devices.
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Debug mode - set MPS_ATTN_DEBUG=1 to enable tracing
DEBUG = os.environ.get("MPS_ATTN_DEBUG", "0") == "1"
_debug_decode_count = 0


class MPSAttentionBackend(AttentionBackend):
    """Attention backend using PyTorch SDPA on MPS devices."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """MPS attention supports decoder and encoder-only attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["MPSAttentionImpl"]:
        return MPSAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MPSAttentionMetadataBuilder"]:
        return MPSAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        # This matches flash_attn and flex_attention layout
        return 2, num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MPSAttentionMetadata:
    """Metadata for MPS attention."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    """Builder for MPS attention metadata."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MPSAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        if DEBUG:
            # Sync before reading tensor values for debug
            torch.mps.synchronize()
            print("\n[MPS_ATTN DEBUG] build() called:")
            print(f"  num_actual_tokens={num_actual_tokens}")
            print(f"  max_query_len={max_query_len}")
            print(f"  max_seq_len={max_seq_len}")
            print(
                f"  slot_mapping[:min(10,len)]={slot_mapping[: min(10, len(slot_mapping))].tolist()}"
            )
            print(f"  seq_lens={seq_lens.tolist()}")

        # NOTE: We do NOT clone tensors here. The model runner updates these
        # tensors in-place between forward passes, and we need to see the
        # updated values. FlashAttention backend also does not clone.
        # Previous cloning caused stale metadata (seq_lens stayed constant,
        # slot_mapping didn't update between decode steps).

        # NOTE: The MPSModelRunner wraps model.forward() with torch.mps.synchronize()
        # before each forward pass, which ensures async copies are complete.
        # No additional sync needed here.

        return MPSAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal,
        )


class MPSAttentionImpl(AttentionImpl):
    """MPS attention implementation using PyTorch SDPA."""

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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.logits_soft_cap = logits_soft_cap if logits_soft_cap else 0.0

        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("FP8 KV cache is unsupported in MPS_ATTN")
        self.attn_type = attn_type

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MPSAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for MPS attention backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported for MPSAttentionImpl"
            )

        # For warming-up
        if attn_metadata is None:
            return output

        # CRITICAL: Use slot_mapping.shape[0] to determine the actual number of tokens
        # for this forward pass. The num_actual_tokens field in attn_metadata is a
        # Python int that was set when build() was called, and may be stale.
        # The slot_mapping tensor, however, is sliced to the correct size for each
        # forward pass by the model runner (see CommonAttentionMetadata.unpadded()).
        # FlashAttention uses the same approach (see reshape_and_cache_flash).
        slot_mapping = attn_metadata.slot_mapping
        num_tokens = slot_mapping.shape[0]

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._run_sdpa_forward(
                query[:num_tokens],
                key[:num_tokens],
                value[:num_tokens],
                output[:num_tokens],
                attn_metadata,
                self.attn_type,
            )

        # For decoder attention, use KV cache
        key_cache, value_cache = kv_cache.unbind(0)

        # Update KV cache
        # The slot_mapping tensor determines how many tokens to cache.
        # FlashAttention's reshape_and_cache_flash op uses slot_mapping's shape
        # to determine the number of actual tokens, so we follow the same pattern.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            self._update_kv_cache(
                key[:num_tokens],
                value[:num_tokens],
                key_cache,
                value_cache,
                slot_mapping,
            )

        # Run attention with KV cache
        return self._run_paged_attention(
            query[:num_tokens],
            key[:num_tokens],
            value[:num_tokens],
            key_cache,
            value_cache,
            output[:num_tokens],
            attn_metadata,
        )

    def _update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update the KV cache with new key/value tensors (vectorized).

        Args:
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            key_cache: [num_blocks, block_size, num_kv_heads, head_size]
            value_cache: [num_blocks, block_size, num_kv_heads, head_size]
            slot_mapping: [num_tokens]
        """
        num_tokens = key.shape[0]
        if num_tokens == 0:
            return

        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        if DEBUG:
            print("\n[MPS_ATTN DEBUG] _update_kv_cache (vectorized):")
            print(f"  num_tokens={num_tokens}")
            print(f"  slot_mapping={slot_mapping.tolist()}")

        # Reshape cache to flat indexing: [num_blocks * block_size, num_kv_heads, head_size]
        num_blocks = key_cache.shape[0]
        key_cache_flat = key_cache.view(
            num_blocks * block_size, num_kv_heads, head_size
        )
        value_cache_flat = value_cache.view(
            num_blocks * block_size, num_kv_heads, head_size
        )

        # Use slot_mapping directly as indices into the flattened cache
        # This is a vectorized scatter operation
        key_cache_flat[slot_mapping] = key
        value_cache_flat[slot_mapping] = value

        # No sync needed here - MPS will handle memory ordering for subsequent ops

    def _run_paged_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
    ) -> torch.Tensor:
        """Run paged attention using SDPA on MPS (optimized for GPU utilization).

        Uses batched operations to maximize GPU utilization:
        - For prefill: batches sequences with same length
        - For decode: batches all single-token queries with same KV length
        """
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        block_size = key_cache.shape[1]
        causal = attn_metadata.causal

        num_seqs = len(seq_lens)
        if num_seqs == 0:
            return output

        # Compute query lengths for each sequence
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        # Check if this is a decode batch (all query_len == 1)
        # This is the common case during generation after prefill
        # Use torch operations to avoid .item() which would serialize GPU execution
        all_decode = bool((query_lens == 1).all()) if num_seqs > 0 else False

        if all_decode:
            # Optimized decode path - works for both single and batched sequences
            return self._run_batched_decode(
                query,
                key,
                value,
                key_cache,
                value_cache,
                output,
                query_start_loc,
                seq_lens,
                block_table,
                block_size,
                causal,
            )

        # Check if all sequences have the same query length (prefill batching opportunity)
        unique_query_lens = torch.unique(query_lens)
        all_same_query_len = unique_query_lens.numel() == 1

        if all_same_query_len and num_seqs > 1:
            # Use int() to convert 0-dim tensor without GPU sync (stays on device)
            query_len = int(unique_query_lens[0])
            # Check if this is pure prefill (all query_len == seq_len)
            is_prefill = bool((query_lens == seq_lens).all())

            if is_prefill:
                return self._run_batched_prefill(
                    query,
                    key,
                    value,
                    output,
                    query_start_loc,
                    query_len,
                    num_seqs,
                    causal,
                )

        # Fallback: mixed prefill/decode - process using tensor indexing
        # Use tensor operations to avoid CPU sync where possible
        for seq_idx in range(num_seqs):
            # Use tensor indexing (stays on GPU)
            q_start = query_start_loc[seq_idx]
            q_end = query_start_loc[seq_idx + 1]
            query_len = q_end - q_start
            seq_len_val = seq_lens[seq_idx]

            seq_query = query[q_start:q_end]
            is_prefill = query_len == seq_len_val

            if is_prefill:
                gathered_key = key[q_start:q_end]
                gathered_value = value[q_start:q_end]
            else:
                # Decode: gather from cache + current K/V
                # Convert seq_len to int for block calculation (single value, minimal overhead)
                seq_len_int = int(seq_len_val)
                num_blocks_needed = (seq_len_int + block_size - 1) // block_size
                seq_block_table = block_table[seq_idx, :num_blocks_needed]

                query_len_int = int(query_len)
                historical_len = seq_len_int - query_len_int
                if historical_len > 0:
                    gathered_hist_key = self._gather_kv_from_cache(
                        key_cache, seq_block_table, historical_len, block_size
                    )
                    gathered_hist_value = self._gather_kv_from_cache(
                        value_cache, seq_block_table, historical_len, block_size
                    )
                    gathered_key = torch.cat(
                        [gathered_hist_key, key[q_start:q_end]], dim=0
                    )
                    gathered_value = torch.cat(
                        [gathered_hist_value, value[q_start:q_end]], dim=0
                    )
                else:
                    gathered_key = key[q_start:q_end]
                    gathered_value = value[q_start:q_end]

            seq_output = self._compute_attention(
                seq_query,
                gathered_key,
                gathered_value,
                causal and self.attn_type == AttentionType.DECODER,
            )
            output[q_start:q_end] = seq_output

        return output

    def _run_batched_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        causal: bool,
    ) -> torch.Tensor:
        """Batched decode: all sequences have query_len=1.

        For single sequence (batch=1): direct gather + SDPA without mask (fastest)
        For multiple sequences: loop over each to avoid slow masked SDPA path
        """
        num_seqs = seq_lens.shape[0]
        num_total_blocks = key_cache.shape[0]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        # Flatten caches for gathering
        key_cache_flat = key_cache.view(
            num_total_blocks * block_size, num_kv_heads, head_size
        )
        value_cache_flat = value_cache.view(
            num_total_blocks * block_size, num_kv_heads, head_size
        )

        # Process each sequence - this is fast because SDPA without mask is efficient
        # and we avoid the 5x slowdown from using attention masks
        for seq_idx in range(num_seqs):
            # Get sequence info using tensor indexing (minimizes CPU sync)
            seq_len = int(seq_lens[seq_idx])
            hist_len = seq_len - 1

            # Get query for this sequence
            q_start = query_start_loc[seq_idx]
            seq_q = query[q_start : q_start + 1]  # [1, num_heads, head_size]

            # Get current K/V
            curr_k = key[q_start : q_start + 1]  # [1, num_kv_heads, head_size]
            curr_v = value[q_start : q_start + 1]

            if hist_len > 0:
                # Gather historical KV from cache (vectorized)
                positions = torch.arange(hist_len, device=key_cache.device)
                logical_blocks = positions // block_size
                offsets = positions % block_size

                num_blocks_needed = (hist_len + block_size - 1) // block_size
                seq_block_table = block_table[seq_idx, :num_blocks_needed]
                physical_blocks = seq_block_table[logical_blocks]
                flat_indices = physical_blocks * block_size + offsets

                hist_k = key_cache_flat[
                    flat_indices
                ]  # [hist_len, num_kv_heads, head_size]
                hist_v = value_cache_flat[flat_indices]

                # Concatenate: [seq_len, num_kv_heads, head_size]
                seq_k = torch.cat([hist_k, curr_k[0:1]], dim=0)
                seq_v = torch.cat([hist_v, curr_v[0:1]], dim=0)
            else:
                # No history - just use current K/V with shape [1, num_kv_heads, head_size]
                seq_k = curr_k[0:1]
                seq_v = curr_v[0:1]

            # Expand KV heads for GQA
            # seq_k/seq_v shape: [seq_len, num_kv_heads, head_size]
            if self.num_kv_heads != self.num_heads:
                seq_k = seq_k.repeat_interleave(self.num_queries_per_kv, dim=1)
                seq_v = seq_v.repeat_interleave(self.num_queries_per_kv, dim=1)

            # Reshape for SDPA: [1, heads, seq_len, head_size]
            # seq_q: [1, num_heads, head_size] -> [1, num_heads, 1, head_size]
            seq_q = seq_q.unsqueeze(2)
            # seq_k: [seq_len, num_heads, head_size] -> [1, num_heads, seq_len, head_size]
            seq_k = seq_k.unsqueeze(0).permute(0, 2, 1, 3)
            seq_v = seq_v.unsqueeze(0).permute(0, 2, 1, 3)

            # Run SDPA without mask (fast path)
            attn_output = F.scaled_dot_product_attention(
                seq_q,
                seq_k,
                seq_v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,  # Single query sees all keys
                scale=self.scale,
            )

            # Reshape: [1, num_heads, 1, head_size] -> [1, num_heads, head_size]
            output[q_start] = attn_output[0, :, 0, :]

        return output

    def _run_batched_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_len: int,
        num_seqs: int,
        causal: bool,
    ) -> torch.Tensor:
        """Batched prefill: all sequences have the same query length and seq_len == query_len."""
        # Reshape from [total_tokens, num_heads, head_size] to [batch, seq_len, num_heads, head_size]
        batch_q = query.view(num_seqs, query_len, self.num_heads, self.head_size)
        batch_k = key.view(num_seqs, query_len, self.num_kv_heads, self.head_size)
        batch_v = value.view(num_seqs, query_len, self.num_kv_heads, self.head_size)

        # Expand KV heads for GQA
        if self.num_kv_heads != self.num_heads:
            batch_k = batch_k.repeat_interleave(self.num_queries_per_kv, dim=2)
            batch_v = batch_v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Reshape for SDPA: [batch, heads, seq_len, head_size]
        batch_q = batch_q.transpose(1, 2)
        batch_k = batch_k.transpose(1, 2)
        batch_v = batch_v.transpose(1, 2)

        # Run batched SDPA with causal mask
        attn_output = F.scaled_dot_product_attention(
            batch_q,
            batch_k,
            batch_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal and self.attn_type == AttentionType.DECODER,
            scale=self.scale,
        )

        # Reshape output: [batch, heads, seq_len, head_size] -> [total_tokens, num_heads, head_size]
        attn_output = attn_output.transpose(1, 2).reshape(
            -1, self.num_heads, self.head_size
        )
        output[: attn_output.shape[0]] = attn_output

        # No sync needed - let GPU work asynchronously for better utilization
        return output

    def _gather_kv_from_cache_using_slots(
        self,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Gather KV from paged cache using slot_mapping directly (vectorized).

        This is the more direct approach - using the exact same slot indices
        that were used to write to the cache.

        Args:
            cache: [num_blocks, block_size, num_kv_heads, head_size]
            slot_mapping: [seq_len] - slot indices for each position
            block_size: Block size

        Returns:
            [seq_len, num_kv_heads, head_size]
        """
        num_kv_heads = cache.shape[2]
        head_size = cache.shape[3]
        num_blocks = cache.shape[0]

        # Flatten cache and use slot_mapping directly as indices
        cache_flat = cache.view(num_blocks * block_size, num_kv_heads, head_size)
        gathered = cache_flat[slot_mapping]

        return gathered

    def _gather_kv_from_cache(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather KV from paged cache (vectorized).

        Args:
            cache: [num_blocks, block_size, num_kv_heads, head_size]
            block_table: [num_blocks_for_seq]
            seq_len: Total sequence length
            block_size: Block size

        Returns:
            [seq_len, num_kv_heads, head_size]
        """
        num_kv_heads = cache.shape[2]
        head_size = cache.shape[3]
        num_total_blocks = cache.shape[0]

        # Compute flat slot indices for all positions
        # For position i, slot = block_table[i // block_size] * block_size + (i % block_size)
        positions = torch.arange(seq_len, device=cache.device)
        logical_block_indices = positions // block_size
        block_offsets = positions % block_size

        # Map logical blocks to physical blocks using block_table
        physical_block_indices = block_table[logical_block_indices]

        # Compute flat indices: physical_block * block_size + offset
        flat_indices = physical_block_indices * block_size + block_offsets

        # Flatten cache and gather
        cache_flat = cache.view(num_total_blocks * block_size, num_kv_heads, head_size)
        gathered = cache_flat[flat_indices]

        return gathered

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Compute attention using SDPA.

        Args:
            query: [query_len, num_heads, head_size]
            key: [seq_len, num_kv_heads, head_size]
            value: [seq_len, num_kv_heads, head_size]
            causal: Whether to use causal attention

        Returns:
            [query_len, num_heads, head_size]
        """
        query_len = query.shape[0]
        seq_len = key.shape[0]

        # Expand KV heads if using grouped-query attention
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Reshape for SDPA: [batch, heads, seq_len, head_size]
        # Add batch dimension and transpose
        q = query.unsqueeze(0).transpose(1, 2)  # [1, num_heads, query_len, head_size]
        k = key.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_size]
        v = value.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_size]

        # Handle causal mask for decode (single token query)
        is_causal = causal and query_len == seq_len

        # For decode with single query token, we need a proper mask
        attn_mask = None
        if causal and query_len != seq_len:
            # Decode mode: query sees all past tokens but not future
            # No mask needed for single query against all keys
            pass

        # Run scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Reshape output: [1, num_heads, query_len, head_size] -> [query_len, num_heads, head_size]
        attn_output = attn_output.squeeze(0).transpose(0, 1)

        return attn_output

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
        attn_type: str,
    ) -> torch.Tensor:
        """Run SDPA for encoder attention (no KV cache)."""
        query_start_loc = attn_metadata.query_start_loc
        causal = attn_type == AttentionType.DECODER

        num_seqs = query_start_loc.shape[0] - 1

        # Check if all sequences have the same length for batched execution
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        unique_lens = torch.unique(query_lens)

        if unique_lens.numel() == 1 and num_seqs > 1:
            # All same length - use batched SDPA
            seq_len = int(unique_lens[0])
            batch_q = query.view(num_seqs, seq_len, self.num_heads, self.head_size)
            batch_k = key.view(num_seqs, seq_len, self.num_kv_heads, self.head_size)
            batch_v = value.view(num_seqs, seq_len, self.num_kv_heads, self.head_size)

            if self.num_kv_heads != self.num_heads:
                batch_k = batch_k.repeat_interleave(self.num_queries_per_kv, dim=2)
                batch_v = batch_v.repeat_interleave(self.num_queries_per_kv, dim=2)

            batch_q = batch_q.transpose(1, 2)
            batch_k = batch_k.transpose(1, 2)
            batch_v = batch_v.transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(
                batch_q,
                batch_k,
                batch_v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=self.scale,
            )

            attn_output = attn_output.transpose(1, 2).reshape(
                -1, self.num_heads, self.head_size
            )
            output[: attn_output.shape[0]] = attn_output
        else:
            # Variable lengths - process individually using tensor indexing
            for i in range(num_seqs):
                start = query_start_loc[i]
                end = query_start_loc[i + 1]

                seq_output = self._compute_attention(
                    query[start:end],
                    key[start:end],
                    value[start:end],
                    causal,
                )
                output[start:end] = seq_output

        return output
