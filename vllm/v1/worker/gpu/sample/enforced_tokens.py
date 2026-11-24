# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState
from vllm.validation import MAX_ENFORCED_TOKENS

# One extra slot for the EOS that the serving layer appends to the list.
MAX_ENFORCED_IDS = MAX_ENFORCED_TOKENS + 1


class EnforcedTokensState:
    """Per-request state for validation replay (``enforced_token_ids``).

    At output step ``s`` the sampled token is replaced by ``ids[s]`` (``ids[-1]``
    once past the end), matching the V1 sampler. Logprobs still come from the
    model's own distribution because the override runs before they are computed.
    The step is derived from GPU state (``total_len - prompt_len``), so no CPU
    synchronization is needed.
    """

    def __init__(self, req_states: RequestState):
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device
        self.req_states = req_states
        self.token_ids = StagedWriteTensor(
            (self.max_num_reqs, MAX_ENFORCED_IDS),
            dtype=torch.int32,
            device=self.device,
            uva_instead_of_gpu=True,
        )
        self._num_ids = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # Host-side flag per slot so steps without enforced rows skip GPU work.
        self.has_ids = np.zeros(self.max_num_reqs, dtype=np.bool_)

    @property
    def num_ids(self) -> torch.Tensor:
        return self._num_ids.gpu

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        ids = sampling_params.enforced_token_ids
        if ids is not None and len(ids) > MAX_ENFORCED_IDS:
            raise ValueError(
                f"enforced_token_ids has {len(ids)} ids, "
                f"the limit is {MAX_ENFORCED_IDS}"
            )
        # An empty list or None clears whatever the slot held before.
        n = len(ids) if ids else 0
        self._num_ids.np[req_idx] = n
        self.has_ids[req_idx] = n > 0
        if n:
            self.token_ids.stage_write(req_idx, 0, ids)

    def apply_staged_writes(self) -> None:
        self._num_ids.copy_to_uva()
        self.token_ids.apply_write()

    def _any_enforced(self, idx_mapping_np: np.ndarray) -> bool:
        valid = idx_mapping_np[idx_mapping_np >= 0]
        return bool(self.has_ids[valid].any())

    def apply(
        self,
        sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if not self._any_enforced(idx_mapping_np):
            return
        apply_enforced_tokens(
            sampled,
            idx_mapping,
            self.token_ids.gpu,
            self.num_ids,
            self.req_states.total_len.gpu,
            self.req_states.prompt_len.gpu,
        )

    def apply_to_verified(
        self,
        sampled: torch.Tensor,  # [num_reqs, num_steps + 1], int64
        num_sampled: torch.Tensor,  # [num_reqs], int32
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """Rejection-sampler layout: force column 0 and a single sampled token.

        Enforced rows are scheduled without drafts, so column 0 is their only
        token. A row that still carries drafts is cut to column 0 so a replay
        never accepts a free-running draft token.
        """
        if not self._any_enforced(idx_mapping_np):
            return
        self.apply(sampled[:, 0], idx_mapping, idx_mapping_np)
        enforced = (idx_mapping >= 0) & (
            self.num_ids[idx_mapping.clamp(min=0).long()] > 0
        )
        num_sampled.copy_(torch.where(enforced, num_sampled.new_ones(()), num_sampled))


def apply_enforced_tokens(
    sampled: torch.Tensor,  # [num_reqs], int64, mutated in place
    idx_mapping: torch.Tensor,  # [num_reqs] batch_idx -> req_state_idx
    token_ids: torch.Tensor,  # [max_num_reqs, max_ids], int32
    num_ids: torch.Tensor,  # [max_num_reqs], int32
    total_len: torch.Tensor,  # [max_num_reqs], int32
    prompt_len: torch.Tensor,  # [max_num_reqs], int32
) -> None:
    """Overwrite ``sampled`` in place with the enforced token of this step."""
    valid = idx_mapping >= 0
    req = idx_mapping.clamp(min=0).long()
    count = num_ids[req].long()
    step = (total_len[req] - prompt_len[req]).long()
    # Past the end of the list the last id is repeated (V1: etids[-1]).
    pos = torch.minimum(step.clamp(min=0), (count - 1).clamp(min=0))
    forced = token_ids[req, pos].to(sampled.dtype)
    sampled.copy_(torch.where(valid & (count > 0), forced, sampled))
