"""PoC model runner for vLLM 0.15.x V1 architecture.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Processes nonces one at a time (batch_size=1) for determinism.
Caches final vectors per-nonce to guarantee identical outputs on
non-deterministic backends (e.g. MARLIN FP8 on A800).
"""
import math
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any
from collections import OrderedDict

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from .gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

DEFAULT_K_DIM = 12

# Cached attention metadata (reused across calls with same seq_len)
_cached_attn_meta = None
_cached_attn_meta_key = None

# Per-nonce vector cache for determinism on non-deterministic backends.
# Only used on the last PP rank (driver) at the output stage.
# Key: (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
# Value: numpy array (FP16 vector, shape [k_dim])
_VECTOR_CACHE_MAX = 100000
_vector_cache = OrderedDict()


def _ensure_layer_hooks(worker, block_hash, hidden_size):
    """Ensure layer hooks are installed for the given block_hash."""
    model = worker.model_runner.model
    device = worker.device
    existing_hook = getattr(worker, "_poc_layer_hooks", None)
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


def _get_block_size(worker):
    """Get the KV cache block size from the worker config."""
    return worker.cache_config.block_size


def _create_v1_attn_metadata(seq_len, block_size, device, worker):
    """Create attention metadata for a single sequence (batch_size=1).

    Uses the worker's metadata builders to create the correct metadata
    for whatever attention backend is configured (FlashAttention,
    FlashInfer, etc.). Works with both V1 (attn_groups) and V2
    (attn_metadata_builders) model runners.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    blocks_per_seq = math.ceil(seq_len / block_size)

    all_slots = []
    for t in range(seq_len):
        block_idx = t // block_size
        all_slots.append(block_idx * block_size + (t % block_size))
    slot_mapping = torch.tensor(all_slots, dtype=torch.long, device=device)

    block_table = torch.arange(
        blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)

    query_start_loc_gpu = torch.tensor(
        [0, seq_len], dtype=torch.int32, device=device
    )
    query_start_loc_cpu = torch.tensor(
        [0, seq_len], dtype=torch.int32, device="cpu"
    )
    seq_lens_gpu = torch.tensor([seq_len], dtype=torch.int32, device=device)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32, device="cpu")

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        num_reqs=1,
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.zeros(1, dtype=torch.int32, device="cpu"),
    )

    model_runner = worker.model_runner
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _get_or_create_attn_metadata(seq_len, block_size, device, worker):
    """Get cached attention metadata or create new one."""
    global _cached_attn_meta, _cached_attn_meta_key
    key = (seq_len, block_size, device)
    if _cached_attn_meta_key == key and _cached_attn_meta is not None:
        return _cached_attn_meta
    result = _create_v1_attn_metadata(seq_len, block_size, device, worker)
    _cached_attn_meta = result
    _cached_attn_meta_key = key
    return result


def _cache_get(block_hash, public_key, nonce, seq_len, hidden_size, k_dim):
    """Get cached vector for a nonce, or None if not cached."""
    key = (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
    if key in _vector_cache:
        _vector_cache.move_to_end(key)
        return _vector_cache[key]
    return None


def _cache_put(block_hash, public_key, nonce, seq_len, hidden_size, k_dim, vector):
    """Cache a computed vector for a nonce."""
    global _vector_cache
    key = (block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
    _vector_cache[key] = vector
    _vector_cache.move_to_end(key)
    while len(_vector_cache) > _VECTOR_CACHE_MAX:
        _vector_cache.popitem(last=False)


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass on a V1 worker.

    Processes each nonce independently (batch_size=1) for determinism.
    All TP ranks always participate in every forward pass to avoid
    collective operation deadlocks. The vector cache is only applied
    at the output stage on the last PP rank.
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config

    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0

    # TP SYNC
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Get block_size and prepare attention metadata (cached, reused)
    block_size = _get_block_size(worker)
    attn_metadata, slot_mapping_dict = _get_or_create_attn_metadata(
        seq_len, block_size, device, worker
    )

    # Positions for single sequence
    positions_single = torch.arange(seq_len, device=device)

    # Process each nonce independently for determinism.
    # ALL TP ranks must participate in every forward pass.
    all_last_hidden = []

    for nonce in nonces:
        intermediate_tensors = None
        inputs_embeds = None

        if pp_group.is_first_rank:
            inputs_embeds = generate_inputs(
                block_hash, public_key, [nonce],
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
        else:
            intermediate_tensors = IntermediateTensors(
                pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
            )

        with set_forward_context(
            attn_metadata, vllm_config,
            num_tokens=seq_len,
            slot_mapping=slot_mapping_dict,
        ):
            with poc_forward_context():
                hidden_states = model(
                    input_ids=None,
                    positions=positions_single,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
                )

        # PP: send to next rank if not last
        if not pp_group.is_last_rank:
            if isinstance(hidden_states, IntermediateTensors):
                pp_group.send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group()
                )
            continue

        # Handle tuple return
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Extract last token hidden state
        hidden_states = hidden_states.view(1, seq_len, -1)
        last_hidden = hidden_states[:, -1, :].float()
        all_last_hidden.append(last_hidden)

    # PP: non-last ranks return None
    if not pp_group.is_last_rank:
        return None

    # Output stage: use cache for determinism, compute only uncached nonces.
    import numpy as np
    final_vectors = []

    for i, nonce in enumerate(nonces):
        # Check cache first
        cached = _cache_get(block_hash, public_key, nonce, seq_len, hidden_size, k_dim)
        if cached is not None:
            final_vectors.append(cached)
            continue

        # Not cached: compute from hidden state
        last_hidden = all_last_hidden[i]  # [1, hidden_size]

        # Normalize to unit sphere
        last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

        # k-dim pick + Haar rotation
        indices = random_pick_indices(block_hash, public_key, [nonce], hidden_size, k_dim, device)
        xk = torch.gather(last_hidden, 1, indices)
        yk = apply_haar_rotation(block_hash, public_key, [nonce], xk, device)

        # Normalize output vector
        yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

        # Convert to FP16
        vec_f16 = yk.half().cpu().numpy()[0]  # [k_dim]

        # Cache for future determinism
        _cache_put(block_hash, public_key, nonce, seq_len, hidden_size, k_dim, vec_f16)
        final_vectors.append(vec_f16)

    vectors_f16 = np.stack(final_vectors)

    return {
        "nonces": nonces,
        "vectors": vectors_f16,
    }
