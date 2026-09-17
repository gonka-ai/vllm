# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay support for the V2 sampler: enforced tokens + per-request logprobs mode.

Port of the V1 residual hooks (``vllm/v1/worker/gpu_input_batch.py`` /
``vllm/v1/sample/sampler.py``) onto the V2 model runner, so a validator's
replay request behaves identically on either runner. Without this, the V2
sampler silently ignored ``SamplingParams.enforced_token_ids`` and the
per-request ``logprobs_mode`` — a replay came back as an ordinary completion
with no error anywhere.

Design notes, in V2 terms:

- Replay ids stay on the host (a per-slot list). Padding them into a static
  ``(max_num_reqs, MAX_ENFORCED)`` device tensor would cost ~128 MiB whether
  or not the node ever validates; instead the position lookup transfers one
  small vector per step, and only on steps whose batch actually contains a
  replay row. Mining and plain serving never reach that path.
- The replay position is ``total_len - prompt_len`` from ``RequestState`` —
  the number of tokens already emitted. ``total_len`` is advanced by the
  post-sampling kernel, so at sampling time it still reflects the previous
  step, which is exactly the index the V1 code derived from
  ``len(req_output_token_ids)``.
- Past the end of the replay the sentinel is ``-1`` ("nothing to enforce"),
  matching the V1 semantics after gonka-ai/vllm#83's review: the final
  replay id is EOS by contract, so termination happens before exhaustion is
  ever reached in a well-formed request.
"""

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.states import RequestState


class ReplayState:
    """Per-slot replay bookkeeping for the V2 sampler.

    Invariants — add_request maintains all of them on every call, slot
    reuse included; update them together or not at all:

    - ``use_replay[i] == bool(enforced_ids[i])``
    - ``has_mode_override_gpu`` mirrors ``has_mode_override``;
      ``wants_processed_gpu[i]`` is meaningful only where it is True.
    """

    def __init__(
        self, max_num_reqs: int, req_states: RequestState, device: torch.device
    ):
        self.req_states = req_states
        # Host side: variable-length replay ids + fast batch predicates.
        self.enforced_ids: list[list[int] | None] = [None] * max_num_reqs
        self.use_replay = np.zeros(max_num_reqs, dtype=bool)
        self.has_mode_override = np.zeros(max_num_reqs, dtype=bool)
        # Device side, so the row mask is built without any transfer.
        self.has_mode_override_gpu = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        self.wants_processed_gpu = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        etids = sampling_params.enforced_token_ids
        self.enforced_ids[req_idx] = list(etids) if etids else None
        self.use_replay[req_idx] = bool(etids)

        mode = sampling_params.logprobs_mode
        has_override = mode is not None
        self.has_mode_override[req_idx] = has_override
        self.has_mode_override_gpu[req_idx] = has_override
        self.wants_processed_gpu[req_idx] = mode == "processed_logprobs"

    def batch_has_replay(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(self.use_replay[idx_mapping_np].any())

    def batch_has_mode_override(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(self.has_mode_override[idx_mapping_np].any())

    def enforced_for_batch(self, expanded_idx_mapping: torch.Tensor) -> torch.Tensor:
        """[num_rows] int64 tensor: replay id per row, -1 where none applies.

        Rows follow ``expanded_idx_mapping`` (one entry per LOGIT row, which
        exceeds num_reqs when neighbouring requests expand for prompt
        logprobs), so the result always matches ``sampled``'s shape. Extra
        rows of one request all pin the same position — idempotent.

        The host loop is deliberate: replay id lists are variable-length, so
        there is no vectorised gather for them, and this path only runs on
        steps whose batch contains a replay row.
        """
        # One transfer: row->slot mapping and emitted-so-far, together.
        exp_np = expanded_idx_mapping.cpu().numpy()
        out_len = (
            self.req_states.total_len.gpu[expanded_idx_mapping]
            - self.req_states.prompt_len.gpu[expanded_idx_mapping]
        )
        out_len_np = out_len.cpu().numpy()

        enforced = np.full(exp_np.shape[0], -1, dtype=np.int64)
        for row, req_idx in enumerate(exp_np):
            etids = self.enforced_ids[req_idx]
            if not etids:
                continue
            k = int(out_len_np[row])
            if 0 <= k < len(etids):
                enforced[row] = etids[k]
        return torch.from_numpy(enforced).to(expanded_idx_mapping.device)

    def processed_rows_mask(
        self, expanded_idx_mapping: torch.Tensor, engine_processed: bool
    ) -> torch.Tensor:
        """[num_rows] bool mask over LOGIT rows: which want processed logprobs.

        Rows with an explicit override use it; the rest fall back to the
        engine-level mode. Pure-GPU select — no transfer, no host loop.
        """
        has = self.has_mode_override_gpu[expanded_idx_mapping]
        wants = self.wants_processed_gpu[expanded_idx_mapping]
        if engine_processed:
            return has.logical_not().logical_or(wants)
        return has.logical_and(wants)
