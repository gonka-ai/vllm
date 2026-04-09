"""Live integration tests: PoC (Proof of Computation).

Requires a running vLLM server on port 18199.

Tests:
  1. PoC /generate with wait=true returns artifacts
  2. PoC self-validation: generate twice with same params, L2 distance < 0.2
  3. PoC generate with different block_hash produces different vectors
  4. PoC batch generation (multiple nonces at once)
  5. PoC /generate with validation block (server-side L2 check)
"""
import base64
import struct
import time
import math
import httpx
import pytest
import numpy as np

from tests.gonka.live_conftest import BASE_URL, MODEL, require_server, stop_poc

POC_PARAMS = {"model": MODEL, "seq_len": 64, "k_dim": 12}
POC_BASE = {
    "block_hash": "TEST_BLOCK",
    "block_height": 100,
    "public_key": "test_pub_keys",
    "node_id": 0,
    "node_count": 1,
}


@pytest.fixture(scope="module", autouse=True)
def server_ready():
    require_server()
    stop_poc()
    yield
    stop_poc()


def poc_generate(nonces, block_hash="TEST_BLOCK", wait=True, batch_size=4,
                 validation=None, timeout=60):
    body = {
        **POC_BASE,
        "block_hash": block_hash,
        "nonces": nonces,
        "params": POC_PARAMS,
        "batch_size": batch_size,
        "wait": wait,
    }
    if validation:
        body["validation"] = validation
    return httpx.post(
        f"{BASE_URL}/api/v1/pow/generate", json=body, timeout=timeout
    )


def decode_vector(b64_str):
    raw = base64.b64decode(b64_str)
    n_floats = len(raw) // 2
    return np.array(struct.unpack(f"<{n_floats}e", raw), dtype=np.float32)


def l2_distance(v1, v2):
    return float(np.linalg.norm(v1 - v2))


class TestPoC:

    def test_01_generate_nonces(self):
        """Generate PoC artifacts with wait=true."""
        r = poc_generate(nonces=[0, 1, 2, 3])
        assert r.status_code == 200, f"Generate failed: {r.text}"
        data = r.json()
        assert data["status"] == "completed"
        assert len(data["artifacts"]) == 4

        for art in data["artifacts"]:
            assert len(art["vector_b64"]) > 0
            vec = decode_vector(art["vector_b64"])
            print(f"\n  Nonce {art['nonce']}: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
            assert vec.shape[0] > 0
            assert np.all(np.isfinite(vec)), f"Nonce {art['nonce']} contains NaN/Inf"

    def test_02_self_validation_small_distance(self):
        """Generate same nonces twice → L2 distance should be < 0.2 (deterministic)."""
        nonces = [42, 43, 44, 45]
        r1 = poc_generate(nonces=nonces)
        assert r1.status_code == 200
        r2 = poc_generate(nonces=nonces)
        assert r2.status_code == 200

        arts1 = {a["nonce"]: a for a in r1.json()["artifacts"]}
        arts2 = {a["nonce"]: a for a in r2.json()["artifacts"]}
        for nonce in nonces:
            v1 = decode_vector(arts1[nonce]["vector_b64"])
            v2 = decode_vector(arts2[nonce]["vector_b64"])
            dist = l2_distance(v1, v2)
            print(f"\n  Nonce {nonce}: L2 distance={dist:.6f}, dims={v1.shape[0]}")
            assert dist < 0.2, f"Nonce {nonce} self-validation distance too large: {dist:.4f}"

    def test_03_different_block_hash_different_vectors(self):
        """Different block_hash should produce meaningfully different vectors."""
        r1 = poc_generate(nonces=[0, 1, 2, 3], block_hash="BLOCK_A")
        r2 = poc_generate(nonces=[0, 1, 2, 3], block_hash="BLOCK_B")
        assert r1.status_code == 200 and r2.status_code == 200

        arts1 = {a["nonce"]: a for a in r1.json()["artifacts"]}
        arts2 = {a["nonce"]: a for a in r2.json()["artifacts"]}
        v1 = decode_vector(arts1[0]["vector_b64"])
        v2 = decode_vector(arts2[0]["vector_b64"])

        dist = l2_distance(v1, v2)
        print(f"\n  Different block_hash L2 distance: {dist:.6f}")
        assert dist > 0.01, (
            f"Different block hashes should produce different vectors, "
            f"got distance {dist:.6f}"
        )

    def test_04_batch_generation(self):
        """Generate multiple nonces in a single batch."""
        nonces = [0, 1, 2, 3]
        r = poc_generate(nonces=nonces, batch_size=4)
        assert r.status_code == 200, f"Batch generate failed: {r.text}"
        data = r.json()
        assert data["status"] == "completed"
        assert len(data["artifacts"]) == len(nonces)

        returned_nonces = {a["nonce"] for a in data["artifacts"]}
        assert returned_nonces == set(nonces)

        for art in data["artifacts"]:
            vec = decode_vector(art["vector_b64"])
            assert np.all(np.isfinite(vec)), f"Nonce {art['nonce']} has NaN/Inf"
        print(f"\n  Batch of {len(nonces)} nonces generated successfully")

    def test_05_server_side_validation(self):
        """Generate, then re-generate with validation block → server computes L2."""
        r1 = poc_generate(nonces=[10, 11, 12, 13])
        assert r1.status_code == 200
        artifacts_a = r1.json()["artifacts"]

        validation = {
            "artifacts": [
                {"nonce": a["nonce"], "vector_b64": a["vector_b64"]}
                for a in artifacts_a
            ]
        }
        r2 = poc_generate(nonces=[10, 11, 12, 13], validation=validation)
        assert r2.status_code == 200, f"Validation generate failed: {r2.text}"
        data = r2.json()
        assert data["status"] == "completed"
        print(f"\n  Server-side validation response: {list(data.keys())}")

    def test_06_multiple_self_validations_all_below_threshold(self):
        """Run 5 self-validations, mean L2 < 0.1 and max L2 < 0.3."""
        distances = []
        for i in range(5):
            nonces = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3]
            r1 = poc_generate(nonces=nonces)
            r2 = poc_generate(nonces=nonces)
            assert r1.status_code == 200 and r2.status_code == 200

            arts1 = {a["nonce"]: a for a in r1.json()["artifacts"]}
            arts2 = {a["nonce"]: a for a in r2.json()["artifacts"]}
            for n in nonces:
                v1 = decode_vector(arts1[n]["vector_b64"])
                v2 = decode_vector(arts2[n]["vector_b64"])
                dist = l2_distance(v1, v2)
                distances.append(dist)

        mean_dist = sum(distances) / len(distances)
        max_dist = max(distances)
        print(f"\n  Self-validation distances ({len(distances)} pairs): "
              f"{[f'{d:.6f}' for d in distances]}")
        print(f"  Mean: {mean_dist:.6f}, Max: {max_dist:.6f}")
        assert mean_dist < 0.1, (
            f"Mean self-validation distance {mean_dist:.4f} >= 0.1. "
            f"All: {[f'{d:.4f}' for d in distances]}"
        )
        assert max_dist < 0.3, (
            f"Max self-validation distance {max_dist:.4f} >= 0.3. "
            f"All: {[f'{d:.4f}' for d in distances]}"
        )
