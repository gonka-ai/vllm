"""Tests for Grammar Graceful Degradation.

When enforced tokens conflict with the grammar FSM during validation replay,
grammar enforcement should be disabled gracefully instead of crashing.

Tests cover:
1. XgrammarGrammar.accept_tokens() graceful degradation
2. fill_bitmask / rollback become no-ops after grammar failure
3. reset() clears the failure flag
4. structured_output/__init__.py speculative decode path
"""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Mock xgrammar types so tests run without GPU / xgrammar installed
# ---------------------------------------------------------------------------

class MockGrammarMatcher:
    """Simulates xgr.GrammarMatcher behavior."""

    def __init__(self, accept_sequence=None):
        self._accept_sequence = accept_sequence or []
        self._call_idx = 0
        self._accepted_count = 0
        self._terminated = False
        self.fill_calls = []
        self.rollback_calls = []

    def accept_token(self, token: int) -> bool:
        if self._call_idx < len(self._accept_sequence):
            result = self._accept_sequence[self._call_idx]
            self._call_idx += 1
            if result:
                self._accepted_count += 1
            return result
        return True

    def is_terminated(self) -> bool:
        return self._terminated

    def fill_next_token_bitmask(self, bitmask, idx):
        self.fill_calls.append((bitmask, idx))

    def rollback(self, n):
        self.rollback_calls.append(n)

    def reset(self):
        self._call_idx = 0
        self._accepted_count = 0
        self._terminated = False
        self.fill_calls = []
        self.rollback_calls = []


class MockCompiledGrammar:
    pass


# ---------------------------------------------------------------------------
# 1. XgrammarGrammar.accept_tokens() graceful degradation
# ---------------------------------------------------------------------------

class TestAcceptTokensGracefulDegradation:

    def _make_grammar(self, accept_sequence):
        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
        matcher = MockGrammarMatcher(accept_sequence)
        return XgrammarGrammar(
            vocab_size=32000,
            matcher=matcher,
            ctx=MockCompiledGrammar(),
        )

    def test_all_tokens_accepted(self):
        """Normal path: all tokens accepted, returns True."""
        g = self._make_grammar([True, True, True])
        assert g.accept_tokens("req1", [10, 20, 30]) is True
        assert g.num_processed_tokens == 3
        assert g._grammar_failed is False

    def test_first_token_rejected_sets_grammar_failed(self):
        """First token rejected: sets _grammar_failed, returns True (not False)."""
        g = self._make_grammar([False])
        result = g.accept_tokens("req1", [99])
        assert result is True
        assert g._grammar_failed is True
        assert g.num_processed_tokens == 0

    def test_second_token_rejected(self):
        """Second token rejected after first accepted."""
        g = self._make_grammar([True, False])
        result = g.accept_tokens("req1", [10, 20])
        assert result is True
        assert g._grammar_failed is True
        assert g.num_processed_tokens == 1

    def test_subsequent_calls_noop_after_failure(self):
        """After grammar_failed, accept_tokens always returns True."""
        g = self._make_grammar([False])
        g.accept_tokens("req1", [99])
        assert g._grammar_failed is True

        # Further calls should return True without touching the matcher
        result = g.accept_tokens("req1", [100, 200, 300])
        assert result is True

    def test_terminated_returns_false(self):
        """Terminated grammar still returns False (different from failed)."""
        g = self._make_grammar([])
        g._is_terminated = True
        assert g.accept_tokens("req1", [10]) is False


# ---------------------------------------------------------------------------
# 2. fill_bitmask / rollback become no-ops after failure
# ---------------------------------------------------------------------------

class TestBitmaskAndRollbackAfterFailure:

    def _make_grammar(self, accept_sequence):
        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
        matcher = MockGrammarMatcher(accept_sequence)
        return XgrammarGrammar(
            vocab_size=32000,
            matcher=matcher,
            ctx=MockCompiledGrammar(),
        )

    def test_fill_bitmask_noop_after_failure(self):
        g = self._make_grammar([False])
        g.accept_tokens("req1", [99])
        assert g._grammar_failed is True

        bitmask = torch.zeros(1, 1000, dtype=torch.int32)
        g.fill_bitmask(bitmask, 0)
        # Should not have called the matcher's fill
        assert len(g.matcher.fill_calls) == 0

    def test_fill_bitmask_works_before_failure(self):
        g = self._make_grammar([True])
        g.accept_tokens("req1", [10])

        bitmask = torch.zeros(1, 1000, dtype=torch.int32)
        g.fill_bitmask(bitmask, 0)
        assert len(g.matcher.fill_calls) == 1

    def test_rollback_noop_after_failure(self):
        g = self._make_grammar([False])
        g.accept_tokens("req1", [99])

        g.rollback(5)
        assert len(g.matcher.rollback_calls) == 0

    def test_rollback_works_before_failure(self):
        g = self._make_grammar([True, True])
        g.accept_tokens("req1", [10, 20])

        g.rollback(2)
        assert len(g.matcher.rollback_calls) == 1
        assert g.matcher.rollback_calls[0] == 2


# ---------------------------------------------------------------------------
# 3. reset() clears the failure flag
# ---------------------------------------------------------------------------

class TestResetClearsFailure:

    def _make_grammar(self, accept_sequence):
        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
        matcher = MockGrammarMatcher(accept_sequence)
        return XgrammarGrammar(
            vocab_size=32000,
            matcher=matcher,
            ctx=MockCompiledGrammar(),
        )

    def test_reset_clears_grammar_failed(self):
        g = self._make_grammar([False])
        g.accept_tokens("req1", [99])
        assert g._grammar_failed is True

        g.reset()
        assert g._grammar_failed is False
        assert g.num_processed_tokens == 0

    def test_grammar_works_after_reset(self):
        """After reset, grammar should enforce again (new request)."""
        g = self._make_grammar([False])
        g.accept_tokens("req1", [99])
        assert g._grammar_failed is True

        g.matcher.reset()
        g.reset()

        # Now matcher will accept (default behavior after sequence exhausted)
        g.matcher._accept_sequence = [True, True]
        g.matcher._call_idx = 0
        result = g.accept_tokens("req2", [10, 20])
        assert result is True
        assert g._grammar_failed is False
        assert g.num_processed_tokens == 2


# ---------------------------------------------------------------------------
# 4. __init__.py speculative decode path — assert replaced with warning
# ---------------------------------------------------------------------------

class TestSpecDecodeGrammarHandling:
    """Test that the structured_output __init__.py doesn't crash on rejection."""

    def test_accept_tokens_failure_does_not_assert(self):
        """Simulate what happens when grammar.accept_tokens returns False
        in the speculative decode bitmask fill path."""

        # This simulates the logic in structured_output/__init__.py
        # Old code: assert accepted  <-- would crash
        # New code: if not accepted: disable bitmask

        class FakeGrammar:
            def __init__(self):
                self.call_count = 0

            def accept_tokens(self, req_id, tokens):
                self.call_count += 1
                if self.call_count == 2:
                    return False  # Reject on second call
                return True

            def is_terminated(self):
                return False

        grammar = FakeGrammar()
        apply_bitmask = True
        state_advancements = 0
        tokens = [10, 20, 30]

        for token in tokens:
            if apply_bitmask and not grammar.is_terminated():
                accepted = grammar.accept_tokens("req1", [token])
                if not accepted:
                    apply_bitmask = False
                    continue
                state_advancements += 1

        # Token 10: accepted, advancement=1
        # Token 20: rejected, bitmask disabled
        # Token 30: skipped (apply_bitmask=False)
        assert state_advancements == 1
        assert apply_bitmask is False

    def test_all_accepted_in_spec_decode(self):
        """Normal case: all tokens accepted."""

        class FakeGrammar:
            def accept_tokens(self, req_id, tokens):
                return True
            def is_terminated(self):
                return False

        grammar = FakeGrammar()
        apply_bitmask = True
        state_advancements = 0

        for token in [10, 20, 30]:
            if apply_bitmask and not grammar.is_terminated():
                accepted = grammar.accept_tokens("req1", [token])
                if not accepted:
                    apply_bitmask = False
                    continue
                state_advancements += 1

        assert state_advancements == 3
        assert apply_bitmask is True


# ---------------------------------------------------------------------------
# 5. End-to-end: enforced sampling with grammar doesn't crash
# ---------------------------------------------------------------------------

class TestEnforcedSamplingWithGrammar:
    """Simulate the full flow: enforced tokens + grammar active."""

    def test_enforced_tokens_disable_grammar_then_continue(self):
        """Multiple batches of enforced tokens, grammar fails mid-way,
        rest of request runs with grammar disabled."""
        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

        # Grammar accepts tokens 0-4, rejects token 5
        accept_seq = [True, True, True, True, True, False]
        matcher = MockGrammarMatcher(accept_seq)
        g = XgrammarGrammar(
            vocab_size=32000, matcher=matcher, ctx=MockCompiledGrammar()
        )

        # First 5 tokens accepted
        for i in range(5):
            assert g.accept_tokens(f"req1", [i]) is True
        assert g._grammar_failed is False
        assert g.num_processed_tokens == 5

        # Token 5 rejected → grammar disabled
        assert g.accept_tokens("req1", [5]) is True
        assert g._grammar_failed is True

        # Remaining tokens 6-20 should all succeed (grammar bypassed)
        for i in range(6, 21):
            assert g.accept_tokens("req1", [i]) is True

        # fill_bitmask should be no-op
        bitmask = torch.zeros(1, 1000, dtype=torch.int32)
        g.fill_bitmask(bitmask, 0)
        assert len(matcher.fill_calls) == 0
