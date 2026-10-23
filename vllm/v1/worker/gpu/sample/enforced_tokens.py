# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.sampling_params import SamplingParams
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
        self.token_ids = torch.zeros(
            (self.max_num_reqs, MAX_ENFORCED_IDS),
            dtype=torch.int32,
            device=self.device,
        )
        self.num_ids = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._staged: dict[int, list[int]] = {}

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        ids = sampling_params.enforced_token_ids
        if ids is not None and len(ids) > MAX_ENFORCED_IDS:
            raise ValueError(
                f"enforced_token_ids has {len(ids)} ids, "
                f"the limit is {MAX_ENFORCED_IDS}"
            )
        # An empty list or None clears whatever the slot held before.
        self._staged[req_idx] = list(ids) if ids else []

    def apply_staged_writes(self) -> None:
        for req_idx, ids in self._staged.items():
            if ids:
                self.token_ids[req_idx, : len(ids)] = torch.tensor(
                    ids, dtype=torch.int32
                ).to(self.device)
            self.num_ids[req_idx] = len(ids)
        self._staged.clear()

    def apply(self, sampled: torch.Tensor, idx_mapping: torch.Tensor) -> None:
        apply_enforced_tokens(
            sampled,
            idx_mapping,
            self.token_ids,
            self.num_ids,
            self.req_states.total_len.gpu,
            self.req_states.prompt_len.gpu,
        )

    def apply_to_verified(
        self,
        sampled: torch.Tensor,  # [num_reqs, num_steps + 1], int64
        num_sampled: torch.Tensor,  # [num_reqs], int32
        idx_mapping: torch.Tensor,
    ) -> None:
        """Rejection-sampler layout: force column 0 and a single sampled token.

        Enforced rows are scheduled without drafts, so column 0 is their only
        token. A row that still carries drafts is cut to column 0 so a replay
        never accepts a free-running draft token.
        """
        self.apply(sampled[:, 0], idx_mapping)
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
