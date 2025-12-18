# SPDX-License-Identifier: Apache-2.0
"""KV cache operations for Metal backend."""

import torch


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Reshape and store key/value tensors into the cache.

    Args:
        key: Key tensor [num_tokens, num_kv_heads, head_size]
        value: Value tensor [num_tokens, num_kv_heads, head_size]
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: Slot indices [num_tokens]
        kv_cache_dtype: KV cache data type
        k_scale: Key scaling factor
        v_scale: Value scaling factor
    """
    block_size = key_cache.shape[1]

    # Apply scaling if needed
    if k_scale != 1.0:
        key = key * k_scale
    if v_scale != 1.0:
        value = value * v_scale

    # Compute block indices and offsets vectorized (on GPU, no .item() calls)
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    # Use advanced indexing to scatter key/value into cache slots
    # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
    key_cache[block_indices, block_offsets] = key
    value_cache[block_indices, block_offsets] = value


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Reshape and store key/value tensors into unified KV cache.

    This variant stores both K and V in a single cache tensor.

    Args:
        key: Key tensor [num_tokens, num_kv_heads, head_size]
        value: Value tensor [num_tokens, num_kv_heads, head_size]
        kv_cache: KV cache [num_blocks, 2, block_size, num_kv_heads, head_size]
        slot_mapping: Slot indices [num_tokens]
        kv_cache_dtype: KV cache data type
        k_scale: Key scaling factor
        v_scale: Value scaling factor
    """
    block_size = kv_cache.shape[2]

    # Apply scaling if needed
    if k_scale != 1.0:
        key = key * k_scale
    if v_scale != 1.0:
        value = value * v_scale

    # Compute block indices and offsets vectorized (on GPU)
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    # Use advanced indexing to scatter key/value into cache slots
    # kv_cache shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
    kv_cache[block_indices, 0, block_offsets] = key
    kv_cache[block_indices, 1, block_offsets] = value


def copy_blocks(
    kv_caches: list[torch.Tensor],
    src_to_dsts: torch.Tensor,
) -> None:
    """Copy blocks within KV caches.

    Args:
        kv_caches: List of KV cache tensors
        src_to_dsts: Source to destination block mapping [num_pairs, 2]
    """
    if src_to_dsts.numel() == 0:
        return

    src_indices = src_to_dsts[:, 0]
    dst_indices = src_to_dsts[:, 1]

    for kv_cache in kv_caches:
        kv_cache[dst_indices] = kv_cache[src_indices].clone()


def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: torch.Tensor,
) -> None:
    """Swap blocks between source and destination tensors.

    This is used for CPU-GPU block swapping.

    Args:
        src: Source tensor
        dst: Destination tensor
        block_mapping: Block mapping [num_pairs, 2]
    """
    if block_mapping.numel() == 0:
        return

    src_indices = block_mapping[:, 0]
    dst_indices = block_mapping[:, 1]

    dst[dst_indices] = src[src_indices].to(dst.device)


def allocate_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate KV cache tensors.

    Args:
        num_blocks: Number of blocks to allocate
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_size: Size of each head
        dtype: Data type for the cache
        device: Device to allocate on

    Returns:
        Tuple of (key_cache, value_cache)
    """
    cache_shape = (num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
    value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
    return key_cache, value_cache


def allocate_unified_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate unified KV cache tensor.

    Args:
        num_blocks: Number of blocks to allocate
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_size: Size of each head
        dtype: Data type for the cache
        device: Device to allocate on

    Returns:
        Unified KV cache tensor [num_blocks, 2, block_size, num_kv_heads, head_size]
    """
    cache_shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
    return torch.zeros(cache_shape, dtype=dtype, device=device)
