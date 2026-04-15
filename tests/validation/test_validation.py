# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for the /v1/validate endpoint.

Reads inference and validation JSONL results, converts the raw logprob
format into ChatCompletionLogProbsContent schema, and sends pairs to
the running vLLM server's /v1/validate route.
"""

import argparse
import json
import sys
from pathlib import Path

import requests


def raw_logprobs_to_api(
    results: list[dict],
) -> list[dict]:
    """Convert raw {token, logprobs: {id: lp, ...}} to the
    ChatCompletionLogProbsContent schema expected by /v1/validate."""
    converted = []
    for entry in results:
        token_id = entry["token"]
        logprobs_dict = entry["logprobs"]
        chosen_logprob = logprobs_dict.get(token_id, -9999.0)

        top_logprobs = [
            {"token": tid, "logprob": lp} for tid, lp in logprobs_dict.items()
        ]

        converted.append(
            {
                "token": token_id,
                "logprob": chosen_logprob,
                "top_logprobs": top_logprobs,
            }
        )
    return converted


def load_pairs(path: Path, limit: int | None = None):
    """Yield (original_logprobs, validation_logprobs) dicts from a
    validation JSONL file.  Each line must have both inference_result
    and validation_result."""
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            record = json.loads(line)
            inf = record.get("inference_result")
            val = record.get("validation_result")
            if inf is None or val is None:
                continue
            yield (
                raw_logprobs_to_api(inf["results"]),
                raw_logprobs_to_api(val["results"]),
            )


def run_tests(
    base_url: str,
    validation_file: Path,
    limit: int | None,
    threshold: float | None,
    verbose: bool,
):
    if not validation_file.exists():
        print(f"File not found: {validation_file}")
        sys.exit(1)

    url = f"{base_url.rstrip('/')}/v1/validate"
    pairs = list(load_pairs(validation_file, limit=limit))
    print(f"Loaded {len(pairs)} sample(s) from {validation_file}")
    print(f"Target: {url}")
    print("-" * 60)

    passed = 0
    failed = 0
    errors = 0

    for idx, (orig, val) in enumerate(pairs):
        payload: dict = {
            "original_logprobs": orig,
            "validation_logprobs": val,
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            print(f"[{idx}] REQUEST ERROR: {exc}")
            errors += 1
            continue

        if resp.status_code != 200:
            print(f"[{idx}] HTTP {resp.status_code}: {resp.text[:200]}")
            errors += 1
            continue

        body = resp.json()
        similarity = body.get("similarity", 0)
        reason = body.get("reason", "?")
        is_ok = body.get("valid", False)
        thr = body.get("threshold", threshold)

        status = "PASS" if is_ok else "FAIL"
        if is_ok:
            passed += 1
        else:
            failed += 1

        if verbose or not is_ok:
            print(
                f"[{idx}] {status}  similarity={similarity:.6f}  "
                f"threshold={thr}  reason={reason}"
            )

    print("-" * 60)
    total = passed + failed + errors
    print(
        f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Errors: {errors}"
    )
    if failed > 0 or errors > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test /v1/validate endpoint with experiment data"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8008",
        help="vLLM server base URL (default: http://localhost:8008)",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
        help="Path to a validation JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to send (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Expected similarity threshold (informational)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every result, not just failures",
    )
    args = parser.parse_args()

    if args.validation_file is not None:
        validation_file = args.validation_file
    else:
        print("Error: --validation-file is required")
        sys.exit(1)

    run_tests(
        base_url=args.base_url,
        validation_file=validation_file,
        limit=args.limit,
        threshold=args.threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
