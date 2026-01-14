# Production Phase 1: Artifact-Based PoC (No r_target)

## Overview

This document describes the simplified PoC API for vLLM integration, replacing the threshold-based (`r_target`) protocol with artifact-based proof validation.

**Key change**: Instead of filtering nonces by distance threshold (`r_target`), we now:
1. During generation: collect `(nonce, output_vector)` for each checked nonce
2. During validation: sample random subset of nonces, recompute vectors, verify distance in k-dim space

This eliminates the "bubble" concept entirely.

## EXACT Delta List vs `packages/pow`

Reference: `/home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/service/routes.py`

### Endpoint Changes

| pow Endpoint | Status | New vLLM PoC | Notes |
|-------------|--------|--------------|-------|
| `POST /pow/init` | REMOVED | - | Use `/pow/init/generate` instead |
| `POST /pow/init/generate` | KEPT | `POST /pow/init/generate` | Request schema changed (see below) |
| `POST /pow/init/validate` | REMOVED | - | Validation via `/pow/generate` with `validation` param |
| `POST /pow/phase/generate` | REMOVED | - | No phase switching |
| `POST /pow/phase/validate` | REMOVED | - | No phase switching |
| `POST /pow/validate` | REMOVED | - | Merged into `/pow/generate` |
| `GET /pow/status` | KEPT | `GET /pow/status` | Response schema simplified |
| `POST /pow/stop` | KEPT | `POST /pow/stop` | No changes |

### Request Schema Changes

#### `PowInitRequestUrl` (pow) -> `PoCInitRequest` (new)

| pow Field | Status | New Field | Notes |
|-----------|--------|-----------|-------|
| `node_id: int` | KEPT | `node_id: int` | Required for generation |
| `node_count: int` | KEPT | `node_count: int` | Required for generation |
| `block_hash: str` | KEPT | `block_hash: str` | - |
| `block_height: int` | KEPT | `block_height: int` | - |
| `public_key: str` | KEPT | `public_key: str` | - |
| `batch_size: int` | KEPT | `batch_size: int` | Default 32 |
| `r_target: float` | REMOVED | - | No threshold in new protocol |
| `fraud_threshold: float` | REMOVED | - | Moved to validation request |
| `params: Params` | CHANGED | `params: PoCParams` | New strict type: exactly `{model, seq_len, k_dim}` |
| `url: str` | KEPT | `url: str` | Callback URL (pow-compatible) |

New strict `params` type (exactly 3 fields; no extra keys allowed):
```python
class PoCParams(BaseModel):
    model: str
    seq_len: int
    k_dim: int = 12  # Default: 12 dimensions
```

#### `ProofBatch` (pow) -> `ArtifactBatch` (new)

| pow Field | Status | New Field | Notes |
|-----------|--------|-----------|-------|
| `public_key: str` | KEPT | `public_key: str` | - |
| `block_hash: str` | KEPT | `block_hash: str` | - |
| `block_height: int` | KEPT | `block_height: int` | - |
| `nonces: List[int]` | CHANGED | `artifacts: List[Artifact]` | See Artifact schema |
| `dist: List[float]` | REMOVED | - | Replaced by vector |
| `node_id: int` | KEPT | `node_id: int` | - |
| - | ADDED | `encoding: Encoding` | Vector encoding metadata |

New `Artifact` schema:
```python
class Artifact(BaseModel):
    nonce: int
    vector_b64: str  # base64 of float16 little-endian array
```

New `Encoding` schema:
```python
class Encoding(BaseModel):
    dtype: str = "f16"
    k_dim: int = 12
    endian: str = "le"
```

#### `ValidatedBatch` (pow) -> `ValidationResult` (new)

| pow Field | Status | New Field | Notes |
|-----------|--------|-----------|-------|
| `public_key: str` | KEPT | `public_key: str` | - |
| `block_hash: str` | KEPT | `block_hash: str` | - |
| `block_height: int` | KEPT | `block_height: int` | - |
| `nonces: List[int]` | KEPT | `nonces: List[int]` | - |
| `dist: List[float]` | REMOVED | - | - |
| `received_dist: List[float]` | REMOVED | - | - |
| `r_target: float` | REMOVED | - | - |
| `fraud_threshold: float` | KEPT | `fraud_threshold: float` | From validation request |
| `node_id: int` | KEPT | `node_id: int` | - |
| `n_invalid: int` | CHANGED | `n_mismatch: int` | Vector mismatch count |
| `probability_honest: float` | KEPT | `p_value: float` | Renamed |
| `fraud_detected: bool` | KEPT | `fraud_detected: bool` | - |
| - | ADDED | `mismatch_nonces: List[int]` | Which nonces failed |

### Callback Paths (EXACT compatibility)

| Callback | pow | new vLLM PoC | Payload Change |
|----------|-----|--------------|----------------|
| Generation results | `POST {url}/generated` | `POST {url}/generated` | `dist` -> `artifacts` |
| Validation results | `POST {url}/validated` | `POST {url}/validated` | Schema change (see above) |

## Final Endpoints

All under prefix `/api/v1/pow` for compatibility.

### 1) `POST /api/v1/pow/init/generate`

Start continuous generation loop that produces artifacts.

**Request**

```json
{
  "block_hash": "0xabc123...",
  "block_height": 12345,
  "public_key": "node_pubkey_here",
  "node_id": 0,
  "node_count": 10,
  "batch_size": 32,
  "params": {
    "model": "Qwen/Qwen3-0.6B",
    "seq_len": 256,
    "k_dim": 12
  },
  "url": "http://localhost:9000/pow"
}
```

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `block_hash` | string | yes | - | Block identifier |
| `block_height` | int | yes | - | Block height |
| `public_key` | string | yes | - | Node public key |
| `node_id` | int | yes | - | Node index (0-based) |
| `node_count` | int | yes | - | Total nodes |
| `batch_size` | int | no | 32 | GPU batch size |
| `params` | object | yes | - | Strict type: `{model, seq_len, k_dim}` only |
| `params.model` | string | yes | - | Must match deployed vLLM model |
| `params.seq_len` | int | yes | - | Must match deployed PoC config |
| `params.k_dim` | int | no | 12 | Output vector dimensions |
| `url` | string | no | null | Callback URL |

**Response**

```json
{
  "status": "OK",
  "pow_status": {
    "status": "GENERATING"
  }
}
```

**Callback Behavior**

- Callbacks sent to `{url}/generated` every **5 seconds** (configurable via `POC_CALLBACK_INTERVAL_SEC` env var)
- Each callback contains all artifacts accumulated since last send
- Batching maximizes network efficiency

**Callback Payload** (`POST {url}/generated`)

```json
{
  "block_hash": "0xabc123...",
  "block_height": 12345,
  "public_key": "node_pubkey_here",
  "node_id": 0,
  "artifacts": [
    {"nonce": 100, "vector_b64": "base64_encoded_f16_vector..."},
    {"nonce": 101, "vector_b64": "base64_encoded_f16_vector..."}
  ],
  "encoding": {
    "dtype": "f16",
    "k_dim": 12,
    "endian": "le"
  }
}
```

### 2) `POST /api/v1/pow/generate`

Compute artifacts for specific nonces. Optionally validate against provided artifacts.

**Conflicts**: Returns `409 Conflict` if `/init/generate` loop is active.

**Request (compute-only)**

```json
{
  "block_hash": "0xabc123...",
  "block_height": 12345,
  "public_key": "node_pubkey_here",
  "node_id": 0,
  "node_count": 10,
  "nonces": [100, 101, 102, 103, 104],
  "params": {
    "model": "Qwen/Qwen3-0.6B",
    "seq_len": 256,
    "k_dim": 12
  },
  "batch_size": 20,
  "wait": true,
  "url": null
}
```

**Request (with validation)**

```json
{
  "block_hash": "0xabc123...",
  "block_height": 12345,
  "public_key": "node_pubkey_here",
  "node_id": 0,
  "node_count": 10,
  "nonces": [100, 101, 102, 103, 104],
  "params": {
    "model": "Qwen/Qwen3-0.6B",
    "seq_len": 256,
    "k_dim": 12
  },
  "batch_size": 20,
  "wait": true,
  "url": "http://localhost:9000/pow",
  "validation": {
    "artifacts": [
      {"nonce": 100, "vector_b64": "..."},
      {"nonce": 101, "vector_b64": "..."},
      {"nonce": 102, "vector_b64": "..."},
      {"nonce": 103, "vector_b64": "..."},
      {"nonce": 104, "vector_b64": "..."}
    ]
  },
  "stat_test": {
    "dist_threshold": 0.02,
    "p_mismatch": 0.001,
    "fraud_threshold": 0.01
  }
}
```

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `block_hash` | string | yes | - | Block identifier |
| `block_height` | int | yes | - | Block height |
| `public_key` | string | yes | - | Node public key |
| `node_id` | int | yes | - | Node index (0-based) |
| `node_count` | int | yes | - | Total nodes |
| `nonces` | int[] | yes | - | Nonces to compute |
| `params` | object | yes | - | Strict type: `{model, seq_len, k_dim}` only |
| `params.model` | string | yes | - | Must match deployed vLLM model |
| `params.seq_len` | int | yes | - | Must match deployed PoC config |
| `params.k_dim` | int | no | 12 | Output vector dimensions |
| `batch_size` | int | no | 20 | Internal batch size |
| `wait` | bool | no | false | Block until complete |
| `url` | string | no | null | Callback URL |
| `validation` | object | no | null | Artifacts to validate |
| `validation.artifacts` | Artifact[] | if validation | - | Must match `nonces` exactly |
| `stat_test` | object | no | null | Fraud test params (optional, only for validation) |
| `stat_test.dist_threshold` | float | no | 0.02 | L2 distance threshold for mismatch |
| `stat_test.p_mismatch` | float | no | 0.001 | Expected mismatch rate (honest baseline) |
| `stat_test.fraud_threshold` | float | no | 0.01 | p-value threshold for fraud detection |

**Validation Notes**:
- `validation.artifacts` must contain exactly the same nonces as `nonces` field (same set, any order)
- If nonce sets don't match: `400 Bad Request`
- `stat_test` is optional; if omitted with `validation`, response includes only `n_mismatch` and `mismatch_nonces` without fraud determination

**Response (compute-only, wait=true)**

```json
{
  "status": "completed",
  "request_id": "uuid-here",
  "artifacts": [
    {"nonce": 100, "vector_b64": "..."},
    {"nonce": 101, "vector_b64": "..."}
  ],
  "encoding": {
    "dtype": "f16",
    "k_dim": 12,
    "endian": "le"
  }
}
```

**Response (validation, wait=true)**

```json
{
  "status": "completed",
  "request_id": "uuid-here",
  "n_total": 5,
  "n_mismatch": 1,
  "mismatch_nonces": [102],
  "p_value": 0.00012,
  "fraud_detected": true
}
```

**Response (wait=false)**

```json
{
  "status": "queued",
  "request_id": "uuid-here",
  "queued_count": 5
}
```

### 2.1) `GET /api/v1/pow/generate/{request_id}`

Poll for result of a queued `/generate` request.

**Response (queued/running)**

```json
{
  "status": "queued",
  "request_id": "uuid-here"
}
```

**Response (completed - compute-only)**

```json
{
  "status": "completed",
  "request_id": "uuid-here",
  "artifacts": [...],
  "encoding": {...}
}
```

**Response (completed - validation)**

```json
{
  "status": "completed",
  "request_id": "uuid-here",
  "n_total": 5,
  "n_mismatch": 1,
  "mismatch_nonces": [102],
  "p_value": 0.00012,
  "fraud_detected": true
}
```

**Response (failed)**

```json
{
  "status": "failed",
  "request_id": "uuid-here",
  "error": "Timeout waiting for engine"
}
```

**Notes**:
- Results are stored in-memory with TTL (default 5 minutes, configurable via `POC_GENERATE_RESULT_TTL_SEC`)
- Returns `404 Not Found` if request_id is unknown or expired

**Callback Payload** (`POST {url}/generated` for compute-only)

Same as `/init/generate` callback, plus `request_id`:

```json
{
  "request_id": "uuid-here",
  "block_hash": "0xabc123...",
  "block_height": 12345,
  "public_key": "node_pubkey_here",
  "node_id": 0,
  "artifacts": [...],
  "encoding": {...}
}
```

**Callback Payload** (`POST {url}/validated` for validation)

```json
{
  "request_id": "uuid-here",
  "block_hash": "0xabc123...",
  "block_height": 12345,
  "public_key": "node_pubkey_here",
  "node_id": 0,
  "n_total": 200,
  "n_mismatch": 3,
  "mismatch_nonces": [11, 57, 190],
  "p_value": 0.00012,
  "fraud_detected": true
}
```

### 3) `GET /api/v1/pow/status`

Get current PoC status.

**Response**

```json
{
  "status": "GENERATING",
  "config": {
    "block_hash": "0xabc123...",
    "block_height": 12345,
    "public_key": "node_pubkey_here",
    "node_id": 0,
    "node_count": 10,
    "seq_len": 256,
    "k_dim": 12
  },
  "stats": {
    "total_processed": 50000,
    "nonces_per_second": 380.5
  }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `status` | string | IDLE, GENERATING, STOPPED |
| `config` | object | Current config (null if IDLE) |
| `stats` | object | Runtime stats (null if not running) |

### 4) `POST /api/v1/pow/stop`

Stop generation and clear queues.

**Request**: Empty body

**Response**

```json
{
  "status": "OK",
  "pow_status": {
    "status": "STOPPED"
  }
}
```

## Artifact Encoding

Vector encoding uses base64 of little-endian float16 array:

```python
import base64
import numpy as np

def encode_vector(vector: np.ndarray) -> str:
    """Encode vector to base64 float16."""
    f16 = vector.astype(np.float16)
    return base64.b64encode(f16.tobytes()).decode('ascii')

def decode_vector(b64: str, k_dim: int) -> np.ndarray:
    """Decode base64 float16 to vector."""
    data = base64.b64decode(b64)
    return np.frombuffer(data, dtype=np.float16)
```

**Size**: 12 dims * 2 bytes = 24 bytes raw, ~32 bytes base64 per artifact.

## Validation Logic

### Vector Comparison

```python
def is_mismatch(
    computed_vector: np.ndarray,
    received_b64: str,
    dist_threshold: float = 0.02
) -> bool:
    """Check if vectors differ beyond threshold."""
    received = decode_vector(received_b64, len(computed_vector))
    distance = np.linalg.norm(computed_vector - received)
    return distance > dist_threshold
```

### Statistical Fraud Test

Based on binomial test (same approach as pow, adapted for vector comparison):

```python
from scipy.stats import binomtest

def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = 0.001,
    fraud_threshold: float = 0.01
) -> tuple[float, bool]:
    """
    Run binomial test for fraud detection.
    
    Args:
        n_mismatch: Number of nonces where vectors differ beyond threshold
        n_total: Total nonces checked
        p_mismatch: Expected mismatch rate for honest nodes (baseline)
        fraud_threshold: p-value below which fraud is detected
    
    Returns:
        (p_value, fraud_detected)
    """
    result = binomtest(
        k=n_mismatch,
        n=n_total,
        p=p_mismatch,
        alternative='greater'
    )
    p_value = float(result.pvalue)
    fraud_detected = p_value < fraud_threshold
    return p_value, fraud_detected
```

**Parameters**:
- `dist_threshold`: L2 distance above which vectors are considered mismatched (depends on hardware variance; ~0.02 based on experiments)
- `p_mismatch`: Expected false positive rate from honest node hardware differences (~0.001 = 0.1%)
- `fraud_threshold`: p-value cutoff for fraud detection (0.01 = 99% confidence)

## Batching and Queue Behavior

### `wait=true` (synchronous)

Request processed inline with backoff if engine is busy:
1. Nonces split into chunks of `batch_size` (default 20)
2. Each chunk retries with backoff if `skipped=true` (chat priority)
3. Returns when all chunks complete

### `wait=false` (async queue)

Single background worker processes jobs FIFO:
1. Request enqueued immediately, returns `{status: "queued", request_id}`
2. Worker waits if `/init/generate` is active (queue-until-idle)
3. Worker processes chunks with backoff on `skipped=true`
4. Result stored in-memory (TTL configurable via `POC_GENERATE_RESULT_TTL_SEC`)
5. If `url` provided, callback sent on completion
6. Client polls `GET /generate/{request_id}` for result

## Implementation Notes

- No server-side artifact storage (relies on callback receiver)
- No `r_target` anywhere in the codebase
- Keep `url` field name (pow-compatible)
- Keep callback paths exactly: `{url}/generated` and `{url}/validated`
- Continuous generation (`/init/generate`) does NOT include `request_id` in callbacks (matches pow behavior)
- Explicit generation (`/generate`) includes `request_id` for tracking

### Batch-Shape Invariance (Required)

**Artifact vectors MUST be independent of request batch shape.** Different attention backends may use different kernels/accumulation paths based on batch size, causing numerically different outputs for the same nonce when batched with different sets of other nonces.

To ensure deterministic, reproducible artifacts across any request batch size:

0. **Minimum batch size**: The server rejects `batch_size <= 2` (returns `400`) with an explanation. This avoids pathological small-batch kernel paths and makes the protocol contract explicit: `batch_size >= 3`.

1. **Fixed-shape padding**: All PoC forwards pad the nonce list to exactly `batch_size` using negative dummy nonces (-1, -2, ...) before running the model forward.

2. **Filter dummy artifacts**: After forward, artifacts for negative nonces are filtered out before returning to the caller.

3. **Protocol invariant**: `vector(nonce)` is defined as the output when computed in a batch of size `batch_size`, regardless of how many real nonces were requested.

**Implementation** (in `vllm/poc/data.py`):
```python
def pad_nonces(nonces: List[int], pad_to: int) -> List[int]:
    """Pad nonce list with negative dummy nonces to reach pad_to length."""
    if len(nonces) >= pad_to:
        return nonces
    dummy_count = pad_to - len(nonces)
    dummy_nonces = [-(i + 1) for i in range(dummy_count)]
    return nonces + dummy_nonces

def filter_artifacts(artifacts: List[dict], original_nonces: set) -> List[dict]:
    """Filter artifacts to only include those with nonces in original_nonces."""
    return [a for a in artifacts if a["nonce"] in original_nonces]
```

**Why negative nonces?** Negative nonces cannot collide with real nonces (which are always >= 0), making filtering unambiguous.

### Dtype Handling (Required Change)

Current PoC computes vectors in FP32 for numerical stability:

```python
# poc_model_runner.py:221
last_hidden = hidden_states[:, -1, :].float()  # Convert to FP32
# ... all subsequent ops (normalize, gather, Haar rotation) in FP32
```

**This is correct** - `.float()` ensures consistent results regardless of model dtype (FP16/BF16/FP8).

**Required implementation**: Add explicit FP16 conversion before encoding for storage/transmission:

```python
# Current (returns FP32 as Python floats):
result["vectors"] = yk.cpu().tolist()

# Required (convert to FP16 for artifact encoding):
result["vectors"] = yk.half().cpu().numpy()  # FP16 numpy array
```

**Dtype flow**:
| Stage | Dtype | Notes |
|-------|-------|-------|
| Model forward | Model's dtype | FP16/BF16/FP8 depending on config |
| PoC computation | FP32 | After `.float()` - ensures consistency |
| Storage/transmission | FP16 | After `.half()` - bandwidth efficient |
| Validation comparison | FP16 | Both sides truncated identically |

### Cross-Language Binary Format (Python ↔ Go)

Vectors are encoded as **IEEE 754 binary16 (half-precision float), little-endian**.

**Binary layout** (k_dim=12):
```
Offset  Size  Description
0       2     float16 LE: vector[0]
2       2     float16 LE: vector[1]
...
22      2     float16 LE: vector[11]
Total: 24 bytes → ~32 bytes base64
```

**Python encode/decode**:
```python
import base64
import numpy as np

def encode_vector(vector: np.ndarray) -> str:
    """Encode FP32 vector to base64 FP16 little-endian."""
    f16 = vector.astype('<f2')  # '<f2' = little-endian float16
    return base64.b64encode(f16.tobytes()).decode('ascii')

def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32."""
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype='<f2')
    return f16.astype(np.float32)
```

**Go decode** (using `github.com/x448/float16`):
```go
import (
    "encoding/base64"
    "encoding/binary"
    "github.com/x448/float16"
)

func DecodeVector(b64 string) ([]float32, error) {
    data, err := base64.StdEncoding.DecodeString(b64)
    if err != nil {
        return nil, err
    }
    
    k := len(data) / 2
    result := make([]float32, k)
    for i := 0; i < k; i++ {
        bits := binary.LittleEndian.Uint16(data[i*2 : i*2+2])
        result[i] = float16.Frombits(bits).Float32()
    }
    return result, nil
}

func EncodeVector(vec []float32) string {
    data := make([]byte, len(vec)*2)
    for i, v := range vec {
        bits := float16.Fromfloat32(v).Bits()
        binary.LittleEndian.PutUint16(data[i*2:], bits)
    }
    return base64.StdEncoding.EncodeToString(data)
}
```

**Key compatibility notes**:
- `'<f2'` in numpy = IEEE 754 binary16, little-endian (same as `binary.LittleEndian.Uint16` in Go)
- Go has no native float16; use `github.com/x448/float16` or manual IEEE 754 conversion
- Both Python and Go must use **standard base64** (not URL-safe variant)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POC_CALLBACK_INTERVAL_SEC` | 5 | Seconds between callback batches for `/init/generate` |
| `POC_GENERATE_CHUNK_TIMEOUT_SEC` | 60 | Timeout per chunk when engine is busy |
| `POC_GENERATE_RESULT_TTL_SEC` | 300 | Result retention for async `/generate` requests |
| `POC_RPC_TIMEOUT_MS` | 60000 | Timeout for generate_artifacts RPC |

## Deployed Model and Params Consistency (Required)

vLLM PoC runs against the **already deployed model** in the server. The API does **not** load/switch models per request.

Rules:
- Every PoC request MUST include `params` with exactly 3 fields: `{model, seq_len, k_dim}`.
- The server MUST reject the request if:
  - `params.model` does not match the deployed vLLM model identifier
  - `params.seq_len` does not match the server's PoC seq_len
  - `params.k_dim` does not match the server's PoC k_dim

Recommended error:
- `409 Conflict` with:
  - `detail`
  - `requested: {model, seq_len, k_dim}`
  - `deployed: {model, seq_len, k_dim}`

Notes:
- Callbacks do NOT include `model` field. The callback receiver is part of the node and already knows which model it requested.
