# Production Phase 2: Minimal PoC+Chat Coexistence

## Overview

This document specifies the requirements for PoC and OpenAI chat inference to coexist on the same vLLM server with chat priority.

**Reference**: This phase follows [`rules.md`](rules.md) strictly:
- Minimal vLLM changes: PoC stays isolated in `vllm/poc/`
- Do not modify vLLM core inference path
- Simple over clever
- Everything covered by tests

## Goal

Enable `/api/v1/pow/init/generate` (continuous PoC generation) and `/v1/chat/completions` to run on the same server safely.

**Chat priority**: PoC must yield whenever there is chat work.

## Coexistence Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Send `/v1/chat/completions` while `/api/v1/pow/init/generate` is active | Chat request completes (PoC yields GPU) |
| Start `/api/v1/pow/init/generate` while chat requests are active | PoC starts but yields until chat drains |
| PoC running, chat idle | PoC runs at full speed |
| Chat running, PoC active | PoC pauses, chat gets GPU priority |

## Supported Deployment Mode

| Mode | Supported | Notes |
|------|-----------|-------|
| Multiprocessing engine (`MQLLMEngine`) | Yes | All GPU work is single-threaded in engine process |
| In-process engine (`AsyncLLMEngine` direct) | Yes | Chat-priority check via `has_unfinished_requests()` |

## Chat Priority Mechanism

Both modes use chat-priority gating:
- MP mode: Engine checks `has_unfinished_requests()` + `input_socket.poll()` + `_engine_step_in_progress`
- In-process mode: Engine checks `has_unfinished_requests()` only
- If chat is busy, `generate_artifacts` returns `{skipped: true}` and API layer backs off

## API Invariants

- Keep all existing PoC endpoints and payloads from Phase 1
- `/api/v1/pow/generate` with `wait=false` queues and waits for `/init/generate` to complete (no 409)
- `/api/v1/pow/generate` with `wait=true` waits inline for `/init/generate` to complete (no 409)
- No changes to OpenAI endpoints

## Batch-Shape Invariance Compatibility

Phase 1 introduces **fixed-shape padding** to ensure artifact vectors are independent of request batch shape. This is fully compatible with Phase 2 chat-priority coexistence:

- Padding happens within each PoC chunk processing (routes.py, generate_queue.py)
- It does not affect scheduling, gating, or backoff logic
- Chat priority checks occur before the model forward, padding is applied to the nonce list passed to the forward
- The filtering of dummy nonces happens after the forward completes, before returning artifacts

No additional changes needed for coexistence.

## Implementation Requirements

### 1) Chat-Priority Gate (MP Engine)

In `vllm/engine/multiprocessing/engine.py`, PoC action handler for `"run_batch"`:

```python
def _process_poc_action(self, action: str, payload: dict) -> dict:
    manager = self._get_poc_manager()
    
    if action == "run_batch":
        # Chat-priority: skip PoC forward if chat has work
        if self.engine.has_unfinished_requests():
            return {
                "should_continue": True,
                "state": manager.state.value,
                "nonces": [],
                "artifacts": [],
                "skipped": True,
            }
        return manager.run_batch()
    # ... other actions
```

### 2) Prevent Busy-Spin in API Layer

In `vllm/poc/routes.py::_generation_loop`, when `run_batch` returns empty (skipped):

```python
async def _generation_loop(...):
    while not stop_event.is_set():
        result = await engine_client.poc_request("run_batch", {})
        
        if result.get("skipped"):
            # Chat is busy; back off to avoid busy-spin
            await asyncio.sleep(0.05)  # 50ms
            continue
        
        # ... process artifacts
```

### 3) Generate Queue (wait=false)

For async `/generate` requests:
- Single background worker processes jobs FIFO
- Worker waits if `/init/generate` is active
- Results stored in-memory with TTL cleanup
- Polling endpoint `GET /generate/{request_id}` for status/results

```python
# Worker waits for /init/generate to finish before processing
while _is_generation_active(job.app_id):
    await asyncio.sleep(0.1)
```

## Testing Requirements

### Unit Tests (`tests/poc/`)

| Test Case | Description |
|-----------|-------------|
| `test_generate_artifacts_skips_when_chat_busy` | Mock `has_unfinished_requests()=True`, verify skip |
| `test_generate_artifacts_proceeds_when_idle` | Mock `has_unfinished_requests()=False`, verify forward runs |
| `test_generation_loop_backoff` | Verify sleep called when skipped |
| `test_wait_false_returns_queued` | Verify async queue returns request_id |
| `test_poll_endpoint_returns_result` | Verify GET /generate/{id} returns status |
| `test_generate_queues_when_init_active` | Verify no 409, job waits for init/generate |

### Script-Based Tests

| Script | Description |
|--------|-------------|
| `scripts/poc_smoke_test.py` | Existing smoke test (must still pass) |
| `scripts/poc_e2e_test.py` | Existing E2E suite (must still pass) |
| `scripts/poc_coexist_test.py` | **New**: Start PoC, send chat, verify both work |

### New Coexistence Test Script

`scripts/poc_coexist_test.py` must:
1. Start vLLM server in MP mode with `--enable-poc`
2. Start `/api/v1/pow/init/generate`
3. Submit `/v1/chat/completions` request
4. Verify chat completes successfully
5. Verify PoC resumes producing artifacts afterward
6. Stop and report results

## Acceptance Criteria

- [x] PoC and chat coexist with chat priority (both MP and in-process modes)
- [x] PoC does not busy-spin when chat is active (backoff on skipped)
- [x] `/generate` with `wait=false` queues and can be polled
- [x] `/generate` waits for `/init/generate` instead of returning 409
- [x] All unit tests pass
- [x] `scripts/poc_smoke_test.py` passes
- [x] `scripts/poc_e2e_test.py` passes
- [x] `scripts/poc_coexist_test.py` passes

## Files Modified

| File | Changes |
|------|---------|
| `vllm/engine/multiprocessing/engine.py` | Add chat-busy check in `_process_poc_action` (only `generate_artifacts` action) |
| `vllm/engine/async_llm_engine.py` | Add chat-busy check in `poc_request` for in-process mode |
| `vllm/poc/routes.py` | Add backoff sleep, generate queue, polling endpoint, queue-until-idle |
| `tests/poc/test_coexist.py` | Coexistence unit tests |
| `tests/poc/test_routes.py` | Queue and polling tests |
| `scripts/poc_coexist_test.py` | Coexistence E2E script |
