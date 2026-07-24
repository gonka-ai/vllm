"""PoC model runner for vLLM 0.15.x V1 architecture.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — processes all nonces in a single forward call.
"""
import math
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from .gpu_random import (
    generate_inputs,
    generate_inputs_concat_murmur,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


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


def _generate_poc_input_ids(block_hash, public_key, nonces, seq_len, worker, device):
    """Deterministic pseudo token ids for token-id-routed architectures.

    DeepSeek-V4's hash-MoE layers route experts via tid2eid[input_ids] and
    hard-fail on input_ids=None. PoC has no real tokens, so ids derive from the
    same (block_hash, public_key, nonce) seed scheme as the embeddings
    ("_input_ids" suffix) through the murmur3 pipeline -- pure integer
    arithmetic, stable across torch versions (a consensus requirement), int32
    as the routing kernels expect. Gated by model_type so every other
    architecture keeps input_ids=None.
    """
    hf_cfg = getattr(worker.model_config, "hf_config", None)
    if getattr(hf_cfg, "model_type", None) != "deepseek_v4":
        return None
    from .gpu_random import _seed_from_string, _batched_murmur3_32
    vocab = int(hf_cfg.vocab_size)
    batch_size = len(nonces)
    keys = torch.arange(seq_len, dtype=torch.int32, device=device)
    keys = keys.unsqueeze(0).expand(batch_size, -1)
    seeds = torch.tensor(
        [[_seed_from_string(f"{block_hash}_{public_key}_nonce{n}_input_ids")]
         for n in nonces],
        dtype=torch.int64, device=device)
    return (_batched_murmur3_32(keys, seeds) % vocab).to(torch.int32).flatten()


def _get_block_size(worker):
    """Get the KV cache block size from the worker config."""
    return worker.cache_config.block_size


def _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker,
                             borrowed_block_ids=None, positions=None):
    """Create per-attention-group attention metadata.

    Uses the worker's metadata builders to create the correct metadata
    for whatever attention backend is configured (FlashAttention,
    FlashInfer, etc.).

    Models may register KV cache groups with DIFFERENT block sizes (DeepSeek-V4:
    sparse MLA / indexer at cache_config.block_size, SWA compressor at 8).
    Sharing one slot_mapping/block_table across groups hands out-of-range slot
    ids to the smaller-block pools -> OOB writes (illegal memory access on
    sm_90, silent corruption on sm_100). Build the layout per group from that
    group's kv_cache_spec.block_size. Single-group models reduce to the prior
    single-layout behaviour exactly (same slot values, bit-for-bit).

    ``positions`` (shared with the model forward) is threaded into
    CommonAttentionMetadata: DeepSeek-V4's C128A sparse-MLA builder requires it;
    every other v0.25 backend leaves it None and ignores it.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    total_tokens = batch_size * seq_len

    # query_start_loc: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
    query_start_loc_gpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    query_start_loc_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * seq_len
    )
    seq_lens_gpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    seq_lens_cpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device="cpu"
    )
    num_computed_cpu = torch.zeros(batch_size, dtype=torch.int32, device="cpu")

    def _layout(g_block, use_borrowed):
        """slot_mapping + block_table for one KV group's block size.

        When block_ids == range(needed) this reduces to the contiguous
        per-sequence layout (slot = seq_idx*padded + t); the borrowed branch
        keeps the exact single-group behaviour from before this change.
        """
        blocks_per_seq = math.ceil(seq_len / g_block)
        needed = batch_size * blocks_per_seq
        # Physical block ids: borrowed (disjoint from inference, for /generate
        # validation), else low blocks 0.. for the aborted /init/generate path.
        if use_borrowed and borrowed_block_ids is not None:
            if len(borrowed_block_ids) < needed:
                raise ValueError(
                    f"PoC needs {needed} blocks but only "
                    f"{len(borrowed_block_ids)} were borrowed")
            block_ids = list(borrowed_block_ids[:needed])
        else:
            block_ids = list(range(needed))
        all_slots = []
        for seq_idx in range(batch_size):
            for t in range(seq_len):
                block_idx = block_ids[seq_idx * blocks_per_seq + t // g_block]
                all_slots.append(block_idx * g_block + t % g_block)
        slot_mapping = torch.tensor(all_slots, dtype=torch.long, device=device)
        block_table = torch.tensor(
            block_ids, dtype=torch.int32, device=device
        ).view(batch_size, blocks_per_seq)
        return slot_mapping, block_table

    model_runner = worker.model_runner
    layouts = {}
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            spec = getattr(builder, "kv_cache_spec", None)
            g_block = getattr(spec, "block_size", None)
            if g_block is None:
                g_block = attn_group.kv_cache_spec.block_size
            # Borrowed KV blocks are sized/allocated for the main pool
            # (cache_config.block_size) only -- _borrow_blocks is not
            # group-aware. Reuse them for the group whose block size matches;
            # heterogeneous-block groups (e.g. DeepSeek-V4's SWA compressor at
            # block 8) fall back to low blocks. TODO(core): group-aware borrow
            # for KV reuse on the /generate path of multi-block-size models.
            use_borrowed = (g_block == block_size)
            if (g_block, use_borrowed) not in layouts:
                layouts[(g_block, use_borrowed)] = _layout(g_block, use_borrowed)
            slot_mapping, block_table = layouts[(g_block, use_borrowed)]
            common_attn_metadata = CommonAttentionMetadata(
                positions=positions,
                query_start_loc=query_start_loc_gpu,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_gpu,
                num_reqs=batch_size,
                num_actual_tokens=total_tokens,
                max_query_len=seq_len,
                max_seq_len=seq_len,
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                causal=True,
                _seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                _num_computed_tokens_cpu=num_computed_cpu,
            )
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _get_or_create_attn_metadata(batch_size, seq_len, block_size, device, worker,
                                 borrowed_block_ids=None, positions=None):
    """Create fresh attention metadata for the given parameters."""
    return _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker,
                                    borrowed_block_ids=borrowed_block_ids,
                                    positions=positions)


# TODO: Should we get rid of this apprach?
def _select_poc_kv_scratch(
    kv_caches: list,
    dtype: torch.dtype,
    needed_elems: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
) -> Optional[torch.Tensor]:
    """Return a no-copy scratch view into KV cache memory, if safe.

    KV cache storage may use packed dtypes (e.g. ``uint8`` for FP8) or
    backend-specific non-contiguous layouts. Only reuse memory that already
    matches model embedding dtype and is contiguous so ``view(-1)`` does not
    allocate a copy.
    """
    for kv in kv_caches:
        if kv.dtype != dtype:
            continue
        if not kv.is_contiguous():
            continue
        if kv.numel() < needed_elems:
            continue
        return kv.view(-1)[:needed_elems].view(batch_size, seq_len, hidden_size)
    return None


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
    poc_stronger_rng: bool = False,
    borrowed_block_ids: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    Processes all nonces in a single forward call for maximum throughput.
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch_size = len(nonces)

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
                "poc_stronger_rng": poc_stronger_rng,
                "has_borrowed": borrowed_block_ids is not None,
                "borrowed_block_ids": borrowed_block_ids or [],
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])
            borrowed_block_ids = (
                list(broadcast_data["borrowed_block_ids"])
                if broadcast_data.get("has_borrowed") else None
            )

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Positions for the batch (also threaded into attn metadata: DeepSeek-V4's
    # sparse-MLA builder requires positions; other backends ignore them).
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    # Deterministic pseudo token ids for token-id-routed architectures
    # (DeepSeek-V4 hash-MoE routes via tid2eid[input_ids]); None otherwise.
    poc_input_ids = _generate_poc_input_ids(
        block_hash, public_key, nonces, seq_len, worker, device)

    # Get block_size and prepare attention metadata (cached, reused)
    block_size = _get_block_size(worker)
    attn_metadata, slot_mapping_dict = _get_or_create_attn_metadata(
        batch_size, seq_len, block_size, device, worker,
        borrowed_block_ids=borrowed_block_ids,
        positions=positions,
    )

    # Generate inputs for all nonces at once
    intermediate_tensors = None
    inputs_embeds = None

    if pp_group.is_first_rank:
        # Scratchpad (KV-cache reuse for inputs_embeds) removed: it saved very
        # little memory and clobbered inference KV (block-0 region). Always
        # allocate a fresh buffer — the produced vectors are numerically
        # identical to the old scratchpad path (same _seed_from_string/_normal).
        _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
        inputs_embeds = _gen_fn(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )

    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=batch_size * seq_len,
        slot_mapping=slot_mapping_dict,
        skip_compiled=True,
    ):
        with poc_forward_context():
            hidden_states = model(
                input_ids=poc_input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
            )

    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None

    # Handle tuple return
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    # Extract last hidden per sequence
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()  # [batch_size, hidden_size]

    # NaN detection
    nan_mask = torch.isnan(last_hidden).any(dim=-1)  # [batch_size]
    if nan_mask.any():
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        nan_count = nan_mask.sum().item()
        logger.warning("NaN in %d/%d hidden states (GPU fault?)", nan_count, batch_size)

        if clean_idx.numel() == 0:
            logger.error("All %d nonces produced NaN — batch rejected", batch_size)
            return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}

        last_hidden = last_hidden[clean_idx]
        nonces = [nonces[i] for i in clean_idx.tolist()]
        batch_size = len(nonces)

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Batched k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16
    vectors_f16 = yk.half().cpu().numpy()  # [batch_size, k_dim]

    # Late NaN check after FP16 conversion
    nan_out = np.isnan(vectors_f16).any(axis=1)
    if nan_out.any():
        clean = ~nan_out
        vectors_f16 = vectors_f16[clean]
        nonces = [n for n, c in zip(nonces, clean) if c]
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    return {
        "nonces": nonces,
        "vectors": vectors_f16,
    }
