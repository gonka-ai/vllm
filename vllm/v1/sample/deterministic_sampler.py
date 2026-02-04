# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A deterministic sampler that uses cryptographic (SHA256-based) RNG for
fully reproducible sampling across different hardware and implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    sample_categorical,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties


_SAMPLING_EPS = 1e-5


@dataclass
class DeterministicSamplingState:
    """
    Holds RNG state for deterministic sampling across requests.
    Each request has its own RNG instance seeded by request-specific data.
    """
    rngs: Dict[int, Sha256CounterRNG]
    
    @classmethod
    def from_seeds(cls, seeds: Dict[int, str]) -> "DeterministicSamplingState":
        """Create state from a mapping of request_index -> seed_string."""
        return cls(
            rngs={
                idx: Sha256CounterRNG.from_seed_string(seed)
                for idx, seed in seeds.items()
            }
        )
    
    def get_rng(self, request_idx: int, default_seed: str = "default") -> Sha256CounterRNG:
        """Get RNG for a request, creating one if not exists."""
        if request_idx not in self.rngs:
            self.rngs[request_idx] = Sha256CounterRNG.from_seed_string(
                f"{default_seed}_{request_idx}"
            )
        return self.rngs[request_idx]


class DeterministicSampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    using deterministic cryptographic sampling.
    
    This sampler mirrors the interface of the standard Sampler but uses
    SHA256-based RNG for cross-platform reproducibility.
    
    The sampling process follows these steps:
    1. If logprobs are requested, compute them from raw logits.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors (non-argmax-invariant).
    6. Apply penalties (repetition, frequency, presence).
    7. Sample the next tokens:
        a) For greedy requests: argmax
        b) For random requests: apply temperature, min_p, top_k/top_p,
           then sample using deterministic categorical sampler.
    8. Gather logprobs for top tokens and sampled token.
    9. Return SamplerOutput.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        default_seed: str = "deterministic_sampler",
    ):
        super().__init__()
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode
        self.default_seed = default_seed
        # State is managed externally or created per-forward call
        self._state: Optional[DeterministicSamplingState] = None

    def set_state(self, state: DeterministicSamplingState) -> None:
        """Set the sampling state (RNGs) for this sampler."""
        self._state = state

    def create_state_from_prompts(
        self,
        prompts: List[str],
        base_seed: str = "",
    ) -> DeterministicSamplingState:
        """
        Create deterministic sampling state from prompt strings.
        Each prompt is hashed to create a unique RNG seed.
        """
        base = base_seed or self.default_seed
        seeds = {
            idx: f"{base}|{prompt}"
            for idx, prompt in enumerate(prompts)
        }
        return DeterministicSamplingState.from_seeds(seeds)

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
        seeds: Optional[Dict[int, str]] = None,
    ) -> SamplerOutput:
        """
        Sample next tokens deterministically.
        
        Args:
            logits: Tensor of shape [batch_size, vocab_size]
            sampling_metadata: Metadata containing sampling parameters
            predict_bonus_token: Whether this is for speculative decoding bonus token
            logprobs_mode_override: Override the default logprobs mode
            seeds: Optional per-request seeds for RNG (request_idx -> seed_string)
        
        Returns:
            SamplerOutput containing sampled token IDs and optional logprobs
        """
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        
        # Create or use existing state
        if seeds is not None:
            state = DeterministicSamplingState.from_seeds(seeds)
        elif self._state is not None:
            state = self._state
        else:
            # Create default state
            state = DeterministicSamplingState(rngs={})
        
        # Compute logprobs if requested (before any modifications)
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        # Use float32 for the logits
        logits = logits.to(torch.float32)
        
        # Apply logits processors
        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        
        # Sample the next token
        sampled, processed_logprobs = self.sample(logits, sampling_metadata, state)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
            
        # Convert to int64 for compatibility
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # Return full unsorted logprobs
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather topk and sampled token logprobs
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled,
                sampling_metadata=sampling_metadata
            )

        # Use int32 to reduce tensor size
        sampled = sampled.to(torch.int32)

        return SamplerOutput(
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """Greedy sampling (argmax)."""
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        state: DeterministicSamplingState,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from logits using deterministic categorical sampling.
        """
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_enforced)
        assert not (sampling_metadata.all_random and sampling_metadata.all_enforced)

        # Handle enforced tokens
        if sampling_metadata.enforced_token_ids:
            enforced_sampled = torch.empty(
                (len(sampling_metadata.enforced_req_ids),),
                dtype=torch.int64,
                device=logits.device
            )
            enforced_map = sampling_metadata.enforced_token_ids
            for i, req_index in enumerate(sampling_metadata.enforced_req_ids):
                seq = enforced_map[req_index]
                out = sampling_metadata.output_token_ids[req_index]
                step = len(out)
                if step < len(seq):
                    next_tok = seq[step]
                else:
                    next_tok = seq[-1]
                enforced_sampled[i] = next_tok
                
            if sampling_metadata.all_enforced:
                result = torch.empty(
                    (logits.size(0),), dtype=torch.int64, device=logits.device
                )
                for i, req_index in enumerate(sampling_metadata.enforced_req_ids):
                    result[req_index] = enforced_sampled[i]
                return result, None

        # Greedy sampling path
        if sampling_metadata.all_random or sampling_metadata.mixed_enforced:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature
        logits = self.apply_temperature(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.all_random or sampling_metadata.mixed_enforced
        )

        # Apply argmax-invariant logits processors
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and top_p masking
        logits = self.apply_top_k_top_p(
            logits,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # Deterministic sampling using cryptographic RNG
        random_sampled, processed_logprobs = self.deterministic_sample(
            logits, state, logprobs_mode
        )

        if greedy_sampled is None:
            if sampling_metadata.enforced_token_ids:
                for i, req_index in enumerate(sampling_metadata.enforced_req_ids):
                    random_sampled[int(req_index)] = enforced_sampled[i]
            return random_sampled, processed_logprobs

        # Mix greedy and random based on temperature
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )

        if sampling_metadata.enforced_token_ids:
            for i, req_index in enumerate(sampling_metadata.enforced_req_ids):
                sampled[int(req_index)] = enforced_sampled[i]

        return sampled, processed_logprobs

    def deterministic_sample(
        self,
        logits: torch.Tensor,
        state: DeterministicSamplingState,
        logprobs_mode: LogprobsMode,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample from logits using deterministic cryptographic RNG.
        
        This is the core difference from the standard sampler - we use
        SHA256-based RNG instead of torch random sampling.
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Compute softmax probabilities
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        
        # Prepare logprobs if needed
        processed_logprobs = None
        if logprobs_mode == "processed_logits":
            processed_logprobs = logits
        elif logprobs_mode == "processed_logprobs":
            processed_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        
        # Move probs to CPU for deterministic sampling
        probs_cpu = probs.cpu().numpy()
        
        # Sample each request deterministically
        sampled_tokens = []
        for i in range(batch_size):
            rng = state.get_rng(i, self.default_seed)
            token_id = sample_categorical(probs_cpu[i], rng)
            sampled_tokens.append(token_id)
        
        sampled = torch.tensor(sampled_tokens, dtype=torch.int64, device=device)
        return sampled, processed_logprobs

    @staticmethod
    def apply_top_k_top_p(
        logits: torch.Tensor,
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits."""
        if k is None and p is None:
            return logits

        if p is None:
            # Top-k only (without sorting full vocab)
            return DeterministicSampler._apply_top_k_only(logits, k)

        # Sort for top-p (and optionally top-k)
        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

        if k is not None:
            top_k_mask = logits_sort.size(1) - k.to(torch.long)
            top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
            top_k_mask = logits_sort < top_k_mask
            logits_sort.masked_fill_(top_k_mask, -float("inf"))

        if p is not None:
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False  # Keep at least one token
            logits_sort.masked_fill_(top_p_mask, -float("inf"))

        # Re-sort back to original order
        logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
        return logits

    @staticmethod
    def _apply_top_k_only(
        logits: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Apply top-k mask without full sorting."""
        no_top_k_mask = k == logits.shape[1]
        k = k.masked_fill(no_top_k_mask, 1)
        max_top_k = k.max()
        k_index = k.sub_(1).unsqueeze(1)
        top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
        top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
        logits.masked_fill_(logits < top_k_mask, -float("inf"))
        return logits

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities from logits."""
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.
        """
        assert token_ids.dtype == torch.int64
        
        if sampling_metadata.enforced_tokens:
            enforced_top_map = sampling_metadata.enforced_tokens
            enforced_map = sampling_metadata.enforced_token_ids
            
            topk_indices_enforced = torch.empty(
                (len(sampling_metadata.enforced_req_ids), num_logprobs),
                dtype=torch.int64,
                device=logprobs.device
            )

            for i, req_index in enumerate(sampling_metadata.enforced_req_ids):
                seq = enforced_map[req_index]
                seq_top_tokens = enforced_top_map[req_index]
                out = sampling_metadata.output_token_ids[req_index]
                step = len(out)

                if step < len(seq):
                    next_tok = seq[step]
                    next_top_tok = seq_top_tokens[step][next_tok]
                else:
                    next_tok = seq[-1]
                    step = len(seq) - 1
                    next_top_tok = seq_top_tokens[step][next_tok]

                topk_indices_enforced[i] = torch.tensor(
                    next_top_tok, device=logprobs.device
                )

            topk_logprobs_enforced = torch.gather(
                logprobs[sampling_metadata.enforced_req_ids],
                dim=1,
                index=topk_indices_enforced
            )

            if sampling_metadata.all_enforced:
                topk_indices = torch.empty(
                    (logprobs.size(0), num_logprobs),
                    dtype=torch.int64,
                    device=logprobs.device
                )
                topk_logprobs = torch.empty(
                    (logprobs.size(0), num_logprobs),
                    device=logprobs.device
                )
                for i, req_index in enumerate(sampling_metadata.enforced_req_ids):
                    topk_indices[req_index] = topk_indices_enforced[i]
                    topk_logprobs[req_index] = topk_logprobs_enforced[i]
            elif sampling_metadata.mixed_enforced:
                all_ids = torch.arange(logprobs.size(0), device=logprobs.device)
                mask_normal = torch.ones_like(all_ids, dtype=torch.bool)
                mask_normal[sampling_metadata.enforced_req_ids] = False

                topk_logprobs_normal, topk_indices_normal = torch.topk(
                    logprobs[mask_normal], num_logprobs, dim=-1
                )

                topk_indices = torch.empty_like(
                    logprobs, device=logprobs.device, dtype=torch.int64
                )[:, :num_logprobs]
                topk_logprobs = torch.empty_like(
                    logprobs, device=logprobs.device
                )[:, :num_logprobs]

                topk_indices[mask_normal] = topk_indices_normal
                topk_logprobs[mask_normal] = topk_logprobs_normal

                topk_indices[sampling_metadata.enforced_req_ids] = topk_indices_enforced
                topk_logprobs[sampling_metadata.enforced_req_ids] = topk_logprobs_enforced
        else:
            topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get the logprob of the prompt or sampled token
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute ranks
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate with topk
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce tensor size
        indices = indices.to(torch.int32)
        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: List[List[int]],
        spec_token_ids: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        if spec_token_ids is None:
            return output_token_ids
        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        """Apply all logits processors."""
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids

        if predict_bonus_token and any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Apply allowed token ids
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(
                sampling_metadata.allowed_token_ids_mask, float("-inf")
            )

        # Apply bad words exclusion
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply non-argmax-invariant logits processors
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: List[List[int]],
    ) -> torch.Tensor:
        """Apply repetition, frequency, and presence penalties."""
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )


# =============================================================================
# Standalone functions for sampling from logprobs/probs directly
# =============================================================================

def deterministic_sample_from_logprobs(
    logprobs: Sequence[Dict[int, float]],
    seed: str,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[int]:
    """
    Sample tokens deterministically from a sequence of logprob dictionaries.
    
    Args:
        logprobs: List of dictionaries mapping token_id -> logprob
        seed: Seed string for deterministic RNG
        temperature: Temperature for sampling (1.0 = no change)
        top_k: If set, only sample from top-k tokens
        top_p: If set, only sample from tokens with cumulative prob <= top_p
    
    Returns:
        List of sampled token IDs
    """
    rng = Sha256CounterRNG.from_seed_string(seed)
    sampled_tokens = []
    
    for step_logprobs in logprobs:
        if not step_logprobs:
            raise ValueError("Empty logprobs at step")
        
        # Convert logprobs to probabilities
        token_ids = list(step_logprobs.keys())
        log_values = [step_logprobs[tid] for tid in token_ids]
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            log_values = [lv / temperature for lv in log_values]
        
        # Convert to probabilities via softmax
        max_log = max(log_values)
        exp_values = [math.exp(lv - max_log) for lv in log_values]
        sum_exp = sum(exp_values)
        probs = [ev / sum_exp for ev in exp_values]
        
        # Apply top-k
        if top_k is not None and top_k < len(token_ids):
            # Sort by probability descending
            sorted_items = sorted(
                zip(token_ids, probs, log_values),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            token_ids = [x[0] for x in sorted_items]
            probs = [x[1] for x in sorted_items]
            # Renormalize
            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs]
        
        # Apply top-p (nucleus sampling)
        if top_p is not None and top_p < 1.0:
            # Sort by probability descending
            sorted_items = sorted(
                zip(token_ids, probs),
                key=lambda x: x[1],
                reverse=True
            )
            cumsum = 0.0
            filtered = []
            for tid, p in sorted_items:
                if cumsum >= top_p and filtered:
                    break
                filtered.append((tid, p))
                cumsum += p
            token_ids = [x[0] for x in filtered]
            probs = [x[1] for x in filtered]
            # Renormalize
            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs]
        
        # Sample using deterministic categorical sampler
        idx = sample_categorical(probs, rng)
        sampled_tokens.append(token_ids[idx])
    
    return sampled_tokens


def deterministic_sample_from_probs(
    probs_2d: Sequence[Sequence[float]],
    seed: str,
) -> List[int]:
    """
    Sample tokens deterministically from 2D probability array.
    
    Args:
        probs_2d: Shape [seq_len, vocab_size] probabilities
        seed: Seed string for deterministic RNG
    
    Returns:
        List of sampled token IDs
    """
    rng = Sha256CounterRNG.from_seed_string(seed)
    return [sample_categorical(step_probs, rng) for step_probs in probs_2d]


def resample_from_inference_logprobs(
    inference_results: List[Dict],
    prompt: str,
    temperature: float = 1.0,
    seed_prefix: str = "resample",
) -> List[int]:
    """
    Re-sample tokens from inference result logprobs.
    
    Args:
        inference_results: List of {token: int, logprobs: {token_id: logprob}}
        prompt: The original prompt (used as part of seed)
        temperature: Temperature for sampling
        seed_prefix: Prefix for the seed string
    
    Returns:
        List of sampled token IDs
    """
    seed = f"{seed_prefix}|{prompt}"
    
    # Extract logprobs from results
    logprobs_seq = []
    for result in inference_results:
        lp = result.get("logprobs", {})
        # Convert string keys to int if needed
        lp_int = {int(k): v for k, v in lp.items()}
        logprobs_seq.append(lp_int)
    
    return deterministic_sample_from_logprobs(
        logprobs_seq,
        seed=seed,
        temperature=temperature,
    )
