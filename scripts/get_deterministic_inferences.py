#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run deterministic inferences on a vLLM server and save artifacts.

Fetches prompts from the SQuAD dataset, sends chat completions to
a running OpenAI-compatible vLLM server, and stores per-request
inference results (text, logprobs, sampling_weights) as JSONL.

Example:
    python scripts/get_deterministic_inferences.py \
        --url http://localhost:8000 \
        --num-prompts 100 \
        --temperature 0.99 \
        --seed 42

    # Or with a specific model name:
    python scripts/get_deterministic_inferences.py \
        --url http://localhost:8000 \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-prompts 50 \
        --temperature 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# Data models (kept minimal and self-contained)
# ---------------------------------------------------------------------------

def _prepare_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Response clear, correct and complete."},
        {"role": "user", "content": prompt},
    ]


def _extract_result(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text, per-position logprobs, and sampling_weights.

    Token identifiers are stored as **token IDs** (string integers) everywhere
    so that logprobs keys, sampling_weights keys, and the ``token`` field are
    all in the same namespace and directly comparable.

    We derive the mapping text->id from ``sampling_weights`` (whose keys are
    already token IDs) by matching the probability ordering with
    ``top_logprobs`` (whose keys are decoded text, sorted by descending
    logprob).
    """
    choice = resp["choices"][0]
    text = choice["message"]["content"]
    content = choice.get("logprobs", {}).get("content", [])

    positions = []
    for pos in content:
        top_lps = pos.get("top_logprobs", [])
        sw = pos.get("sampling_weights")

        # Build a text-token -> token-id mapping from sampling_weights.
        # Both top_logprobs and sampling_weights are ordered by descending
        # probability, so we zip them by rank.
        text_to_id: Dict[str, str] = {}
        if sw is not None:
            # sampling_weights keys are token IDs, sorted desc by weight
            sw_ids_by_rank = sorted(sw.keys(), key=lambda k: sw[k], reverse=True)
            # top_logprobs are already sorted desc by logprob
            lp_texts_by_rank = [t["token"] for t in top_lps]
            for txt, tid in zip(lp_texts_by_rank, sw_ids_by_rank):
                text_to_id[txt] = tid

        def _to_id(tok_text: str) -> str:
            """Convert a text token to its ID string, falling back to the text."""
            return text_to_id.get(tok_text, tok_text)

        entry: Dict[str, Any] = {
            "token": _to_id(pos["token"]),
            "logprobs": {
                _to_id(t["token"]): t["logprob"]
                for t in top_lps
            },
        }
        if sw is not None:
            entry["sampling_weights"] = sw
        positions.append(entry)

    return {"text": text, "results": positions}


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def wait_for_server(base_url: str, timeout_s: int = 120) -> List[str]:
    """Block until the vLLM /v1/models endpoint is reachable. Return served model ids."""
    models_url = base_url.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            r = requests.get(models_url, timeout=5)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return [m["id"] for m in data if isinstance(m, dict) and "id" in m]
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:
            last_err = repr(exc)
        time.sleep(1)
    raise RuntimeError(
        f"vLLM server not ready at {models_url} within {timeout_s}s. Last error: {last_err}"
    )


def resolve_model(configured: Optional[str], served_ids: List[str]) -> str:
    if configured and configured in served_ids:
        return configured
    if configured and served_ids:
        print(f"[warn] Requested model '{configured}' not served. Using '{served_ids[0]}'.")
    if served_ids:
        return served_ids[0]
    if configured:
        return configured
    raise RuntimeError("No model name configured and none served by the server.")


# ---------------------------------------------------------------------------
# Prompt source
# ---------------------------------------------------------------------------

def load_squad_prompts(n: int) -> List[str]:
    """Load n prompts from the SQuAD training set."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install 'datasets' package:  pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("squad", split="train", keep_in_memory=True)
    questions = ds["question"]
    return list(questions[:n])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_single_inference(
    base_url: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    top_logprobs: int,
    timeout_s: int,
) -> Dict[str, Any]:
    """Send one chat completion and return the full artifact dict."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": _prepare_messages(prompt),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "n": 1,
        "skip_special_tokens": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"Inference failed ({resp.status_code}): {resp.text[:500]}")

    resp_json = resp.json()
    result = _extract_result(resp_json)

    return {
        "prompt": prompt,
        "inference_result": result,
        "model": {
            "name": model,
            "url": base_url,
        },
        "request_params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "top_logprobs": top_logprobs,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic inferences on a vLLM server and save JSONL artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server base URL.")
    parser.add_argument("--model", default=None, help="Model name. Auto-detected from /v1/models if omitted.")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of SQuAD prompts to use.")
    parser.add_argument("--temperature", type=float, default=0.99, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max tokens per completion.")
    parser.add_argument("--top-logprobs", type=int, default=5, help="Number of top logprobs to request.")
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent request threads.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds.")
    parser.add_argument("--wait-timeout", type=int, default=120, help="Seconds to wait for server readiness.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to scripts/ folder.")
    parser.add_argument("--exp-name", default=None, help="Experiment tag in output filename.")
    args = parser.parse_args()

    # -- Resolve output paths --
    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(__file__))

    # -- Wait for server --
    print(f"Waiting for vLLM at {args.url} ...")
    served_ids = wait_for_server(args.url, timeout_s=args.wait_timeout)
    model = resolve_model(args.model, served_ids)
    print(f"Server ready. Using model: {model}")

    # -- Load prompts --
    print(f"Loading {args.num_prompts} prompts from SQuAD ...")
    prompts = load_squad_prompts(args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.")

    # -- Build output filename --
    model_short = model.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = args.exp_name or f"t{args.temperature}_s{args.seed}"
    out_jsonl = os.path.join(args.out_dir, f"inferences__{model_short}__{exp}__{ts}.jsonl")
    out_config = out_jsonl.replace(".jsonl", "_config.json")
    os.makedirs(args.out_dir, exist_ok=True)

    # -- Save config --
    config = {
        "model": model,
        "url": args.url,
        "num_prompts": len(prompts),
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "max_workers": args.max_workers,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {out_config}")

    # -- Run inferences concurrently --
    print(f"Running {len(prompts)} inferences (max_workers={args.max_workers}) ...")
    results: List[Dict[str, Any]] = [None] * len(prompts)  # type: ignore[list-item]
    errors = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_idx = {
            pool.submit(
                run_single_inference,
                args.url,
                model,
                prompt,
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=args.max_tokens,
                top_logprobs=args.top_logprobs,
                timeout_s=args.timeout,
            ): i
            for i, prompt in enumerate(prompts)
        }

        done = 0
        t0 = time.time()
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done += 1
            try:
                results[idx] = future.result()
            except Exception as exc:
                errors += 1
                print(f"  [error] prompt #{idx}: {exc}")
                results[idx] = {
                    "prompt": prompts[idx],
                    "inference_result": None,
                    "error": str(exc),
                    "model": {"name": model, "url": args.url},
                    "request_params": {
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "top_logprobs": args.top_logprobs,
                    },
                }

            if done % 10 == 0 or done == len(prompts):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(prompts)}] {rate:.1f} req/s, {errors} errors")

    # -- Write JSONL --
    with open(out_jsonl, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r.get("inference_result") is not None)
    print(f"\nDone. {ok}/{len(prompts)} successful, {errors} errors.")
    print(f"Artifact: {out_jsonl}")


if __name__ == "__main__":
    main()
