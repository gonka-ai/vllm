# Phase 2: GPU Random Generation

## Breaking Change Notice

This implementation uses murmur3-based deterministic random generation, which produces DIFFERENT sequences than the original numpy SeedSequence implementation. This is intentional for PoC 2.0. All nodes must use this new implementation - mixing old/new nodes will cause validation failures.

## Objective

Implement GPU-native deterministic random generation for PoC 2.0 with cross-device reproducibility.

## Deliverable

Working `gpu_random.py` with determinism tests passing.

## Why Murmur3?

torch.Generator on CUDA is NOT portable across GPU architectures (A100 vs H100 vs consumer GPUs). For PoC validation where validators must reproduce bit-exact same random numbers, this is a critical flaw.

Murmur3-based generation solves this:
- Pure integer/float math operations - deterministic on any device
- Standard algorithm with well-defined specification
- Box-Muller transform for normal distribution
- Direct integer argsort for permutations (no float conversion, stable=True for CPU/GPU reproducibility)

## Batch-Shape Invariance

**Determinism includes shape invariance**: The vector for a given nonce must be identical regardless of what other nonces are in the same batch. This is critical for validation, where a validator may compute artifacts for a subset of nonces rather than the full batch.

The model forward pass may use different attention kernels or accumulation paths based on batch size, causing numerically different outputs for the same nonce when batched differently.

**Solution**: Fixed-shape padding (see `production-phase-1.md`). Each PoC forward pads the nonce list to a fixed `batch_size` using negative dummy nonces, then filters out dummy artifacts before returning. This ensures `vector(nonce)` is always computed with the same batch shape.

## Core Functions

### File: `vllm/poc/gpu_random.py`

Private helpers for deterministic RNG:
- `_seed_from_string()` - SHA256 to uint32 seed
- `_murmur3_32()` - Murmur3 hash (portable across GPUs)
- `_uniform()` / `_normal()` - Box-Muller transform

Public API:
- `generate_inputs(block_hash, public_key, nonces, dim, seq_len, device, dtype)` -> `[batch, seq_len, dim]`
- `generate_permutations(block_hash, public_key, nonces, vocab_size, device)` -> `[batch, vocab_size]`
- `generate_target(block_hash, vocab_size, device, dtype)` -> unit vector `[vocab_size]`
- `compute_distances(logits, permutations, target)` -> `[batch]`

## Performance Notes

### GPU vs CPU Generation Trade-off

Current implementation generates inputs directly on GPU with a sequential per-nonce loop.

**Comparison with real inference path:**
| Aspect | Real Inference | PoC Generation |
|--------|---------------|----------------|
| Input source | CPU (tokenizer) -> GPU transfer | Direct GPU generation |
| Transfer overhead | Yes (tokens -> GPU) | None |
| Generation | N/A | Sequential per-nonce |

**Why GPU generation is acceptable:**
1. **Faster than mimicking inference**: Real inference requires CPU->GPU transfer; direct GPU generation avoids this entirely
2. **Murmur3 is efficient**: Pure integer/float operations run fast on GPU
3. **Batch sizes are small**: Typical batch_size (32-64) keeps loop overhead minimal

**Future optimization (if needed):**
For significantly larger batch sizes, the generation loop could be vectorized by computing seeds for all nonces in parallel. Currently not necessary for typical PoC workloads.

## Determinism Test

### File: `tests/poc/test_gpu_random.py`

13 unit tests covering:
- Input determinism (same seed = same output)
- Different nonces produce different inputs
- Different block_hash produces different inputs
- Different public_key produces different inputs
- Permutation determinism
- Different public_key produces different permutations
- Permutation validity (contains all indices 0 to vocab_size-1)
- Target is unit vector (norm = 1)
- Distance range [0, 2]
- Different block_hash produces different targets
- CPU/GPU inputs match (cross-device reproducibility)
- CPU/GPU permutations match (cross-device reproducibility)
- CPU/GPU target vectors match (cross-device reproducibility)

## Directory Structure After Phase 2

```
vllm/poc/
├── __init__.py
├── config.py
├── data.py
└── gpu_random.py

tests/poc/
├── __init__.py
├── test_data.py
└── test_gpu_random.py
```

## Acceptance Criteria

- [ ] All generation functions implemented with murmur3
- [ ] Determinism tests pass (same seed = same output)
- [ ] Different seeds produce different outputs
- [ ] Target is unit vector (norm = 1)
- [ ] Distances in range [0, 2]
- [ ] Works on CUDA device
- [ ] Portable across different GPU architectures
