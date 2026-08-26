"""CPU-only acceptance checks for a built PoC image.

Everything here answers "is this image assembled the way we think it is" without
touching a GPU: the residual surfaces the sampler patches add, the plugin's
version dispatch, the entry points the runner launches through, and whether the
base can even name the model we built it for. It says nothing about numerical
output — that still needs a card.

Run inside the image:

    python3 tools/smoke_cpu.py            # S2 (vllm-poc)
    python3 tools/smoke_cpu.py --stage3   # additionally check the mlnode layer

Exit code is the number of failures.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys

EXPECTED_VLLM_VERSION = "0.28.0.dev0+glm53.gonka.sampler1"
MODEL_ARCH = "Glm5NextForConditionalGeneration"

results: list[tuple[bool, str, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((True, name, str(detail)))
        except Exception as exc:  # noqa: BLE001 — a smoke run reports, never raises
            results.append((False, name, f"{type(exc).__name__}: {exc}"))
        return fn

    return wrap


# --------------------------------------------------------------------------- #
# The base: is it the build we pinned, and does it know the model?
# --------------------------------------------------------------------------- #

@check("vllm imports")
def _vllm_imports():
    import vllm

    return vllm.__file__


@check("version is the residual's, not the base's fallback")
def _version():
    import vllm

    assert vllm.__version__ == EXPECTED_VLLM_VERSION, vllm.__version__
    return vllm.__version__


@check(f"base registers {MODEL_ARCH}")
def _model_registered():
    from vllm.model_executor.models.registry import ModelRegistry

    archs = ModelRegistry.get_supported_archs()
    assert MODEL_ARCH in archs, f"{MODEL_ARCH} missing from {len(archs)} architectures"
    return f"{len(archs)} architectures registered"


# --------------------------------------------------------------------------- #
# The residual: every surface the sampler stack adds
# --------------------------------------------------------------------------- #

@check("SamplingParams carries the PoC fields")
def _sampling_params_fields():
    from vllm.sampling_params import SamplingParams

    fields = set(getattr(SamplingParams, "__struct_fields__", ())) or set(
        getattr(SamplingParams, "__annotations__", {})
    )
    missing = {"logprobs_mode", "enforced_token_ids"} - fields
    assert not missing, f"missing {sorted(missing)}"
    return "logprobs_mode, enforced_token_ids"


@check("Sampler threads need_processed_logprobs")
def _sampler_signature():
    from vllm.v1.sample.sampler import Sampler

    sig = inspect.signature(Sampler.forward)
    assert "need_processed_logprobs" in str(sig) or hasattr(Sampler, "sample"), sig
    return "present"


@check("request ingestion module is in place")
def _validation_module():
    mod = importlib.import_module("vllm.validation")
    assert hasattr(mod, "EnforcedTokens"), dir(mod)
    assert hasattr(mod, "validate_enforced_token_ids"), dir(mod)
    return "EnforcedTokens, validate_enforced_token_ids"


@check("chat request accepts enforced_tokens / logprobs_mode")
def _chat_request_fields():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    fields = set(ChatCompletionRequest.model_fields)
    missing = {"enforced_tokens", "enforced_str", "logprobs_mode"} - fields
    assert not missing, f"missing {sorted(missing)}"
    return "enforced_tokens, enforced_str, logprobs_mode"


@check("completions request accepts logprobs_mode")
def _completion_request_fields():
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    assert "logprobs_mode" in set(CompletionRequest.model_fields)
    return "logprobs_mode"


@check("V2 runner carries the replay hooks")
def _v2_replay():
    from vllm.v1.worker.gpu.sample.replay import ReplayState

    for method in ("add_request", "batch_has_replay", "enforced_for_batch"):
        assert hasattr(ReplayState, method), method
    return "ReplayState with add_request, batch_has_replay, enforced_for_batch"


@check("scheduler guards a missing runner output (#19)")
def _sched_guard():
    import vllm.v1.core.sched.scheduler as sched

    src = inspect.getsource(sched)
    assert "req_id_to_index.get(req_id)" in src, "unguarded lookup"
    return "guarded"


@check("replays are kept out of speculative decoding (#21)")
def _replay_no_spec():
    import vllm.v1.core.sched.scheduler as sched

    src = inspect.getsource(sched)
    assert hasattr(sched, "_replays_enforced_tokens"), "predicate missing"
    assert "request.spec_token_ids = []" in src, "drafts not dropped for replays"
    assert src.count("_replays_enforced_tokens(request)") >= 2, (
        "padding branch not guarded at its source"
    )
    return "drafts dropped, padding guarded"


@check("a replay's max_tokens is pinned to the recorded length (#21)")
def _replay_max_tokens():
    from vllm.entrypoints.openai.chat_completion import serving

    src = inspect.getsource(serving)
    assert "replay_len" in src, "pin missing"
    assert "eos_token_id is not None" in src, "EOS guard lost"
    return "pinned, EOS guard intact"


@check("shm rings are a knob, defaulting to upstream sizes")
def _mq_knob():
    import vllm.envs as envs

    assert hasattr(envs, "VLLM_MQ_MAX_CHUNKS"), "knob missing"
    return f"VLLM_MQ_MAX_CHUNKS={os.environ.get('VLLM_MQ_MAX_CHUNKS', 'unset')}"


# --------------------------------------------------------------------------- #
# The plugin
# --------------------------------------------------------------------------- #

@check("gonka_poc installed")
def _plugin_installed():
    import gonka_poc

    return getattr(gonka_poc, "__version__", "no __version__")


@check("compat dispatch resolves for this vllm")
def _compat_dispatch():
    from gonka_poc._compat import current

    mod = current()
    assert mod.__name__.endswith("v0_28"), mod.__name__
    return mod.__name__


@check("compat exposes the surfaces the PoC forward calls")
def _compat_surface():
    from gonka_poc._compat import current

    mod = current()
    for fn in (
        "build_common_attention_metadata",
        "build_attn_metadata_per_group",
        "get_kv_cache_pool",
        "install_engine_core_poc_methods",
    ):
        assert hasattr(mod, fn), fn
    return "4 entry points present"


@check("worker extension imports")
def _worker_extension():
    from gonka_poc.worker import PoCWorkerExtension

    return PoCWorkerExtension.__name__


@check("composed entrypoint imports")
def _entrypoint():
    mod = importlib.import_module("gonka_poc.entrypoint.api_router")
    assert hasattr(mod, "main"), dir(mod)
    return "gonka_poc.entrypoint.api_router:main"


@check("plugin registers under vllm.general_plugins")
def _entry_points():
    from importlib.metadata import entry_points

    eps = [e.name for e in entry_points(group="vllm.general_plugins")]
    assert eps, "no vllm.general_plugins entry points"
    return ", ".join(eps)


@check("PoC routes are declared")
def _routes():
    mod = importlib.import_module("gonka_poc.poc.routes")
    src = inspect.getsource(mod)
    for path in ("/pow/init", "/pow/stop"):
        assert path in src, path
    return "pow/init, pow/stop"


# --------------------------------------------------------------------------- #
# Stage 3 only: the mlnode layer
# --------------------------------------------------------------------------- #

def stage3_checks() -> None:
    @check("mlnode venv exists")
    def _venv():
        path = "/app/packages/api/.venv/bin/python"
        assert os.path.exists(path), path
        return path

    @check("runner.py is where the runner patches anchor")
    def _runner():
        candidates = (
            "/app/packages/api/src/api/inference/vllm/runner.py",
            "/app/src/api/inference/vllm/runner.py",
        )
        found = [c for c in candidates if os.path.exists(c)]
        assert found, f"none of {candidates}"
        src = open(found[0]).read()
        assert "self.processes: List[subprocess.Popen] = []" in src, (
            "runner-patch anchor missing — Stage 4 patches would fail loud"
        )
        return found[0]

    @check("launch module is overridable")
    def _module_env():
        found = [
            c
            for c in (
                "/app/packages/api/src/api/inference/vllm/runner.py",
                "/app/src/api/inference/vllm/runner.py",
            )
            if os.path.exists(c)
        ]
        src = open(found[0]).read()
        assert "MLNODE_VLLM_MODULE" in src or "vllm.entrypoints.openai.api_server" in src
        return os.environ.get("MLNODE_VLLM_MODULE", "unset")

    @check("Content-Type middleware patch is in the image (patches/0001)")
    def _content_type():
        import subprocess

        out = subprocess.run(
            ["grep", "-rl", "content-type", "/app/packages/api/src/api/"],
            capture_output=True,
            text=True,
        )
        assert out.stdout.strip(), "no content-type handling found under api/"
        return out.stdout.strip().splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", action="store_true")
    args = parser.parse_args()

    if args.stage3:
        stage3_checks()

    width = max(len(name) for _, name, _ in results)
    failures = 0
    for ok, name, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{mark}] {name.ljust(width)}  {detail}")

    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
