# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal n-gram speculative-decode proposer.

The proposer wraps vLLM's pure-Python/Numba n-gram kernel, so these tests need no
model or engine: a ``SimpleNamespace`` ``vllm_config`` exercises the upstream
constructor (which reads only scalar config) and a hand-built ``ProposeContext``
drives ``propose``. They lock in the request filtering (greedy-only, skip
intermediate prefills) and the array marshalling into the upstream kernel.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.v1 import ngram_proposer as ngram_mod
from vllm_metal.v1.ngram_proposer import NgramProposer
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


def _proposer(
    *,
    prompt_lookup_min: int = 2,
    prompt_lookup_max: int = 3,
    num_speculative_tokens: int = 3,
    max_model_len: int = 512,
) -> NgramProposer:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            num_speculative_tokens=num_speculative_tokens,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return NgramProposer(
        vllm_config=vllm_config,
        controller=SpeculativeDecodeController(),
    )


def _request_state(
    token_ids: list[int], *, temperature: float = 0.0
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=temperature),
        generated_tokens=1,
    )


def _context(
    *,
    decode_reqs: list[tuple[str, SimpleNamespace]] | None = None,
    decode_token_ids: list[list[int]] | None = None,
    prefill_reqs: list[SimpleNamespace] | None = None,
    prefill_result_modes: list[str] | None = None,
    request_states: dict[str, SimpleNamespace] | None = None,
    num_speculative_tokens: int = 3,
) -> ProposeContext:
    decode_reqs = decode_reqs or []
    prefill_reqs = prefill_reqs or []
    if decode_token_ids is None:
        decode_token_ids = [[state.token_ids[-1]] for _, state in decode_reqs]
    if prefill_result_modes is None:
        prefill_result_modes = ["new_final"] * len(prefill_reqs)
    if request_states is None:
        request_states = dict(decode_reqs)
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=decode_reqs,
        decode_segments=[],
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=[0] * len(prefill_reqs),
        prefill_result_modes=prefill_result_modes,
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=len(decode_reqs),
        num_speculative_tokens=num_speculative_tokens,
        logitsprocs=None,
    )


class TestNgramProposerProtocol:
    def test_never_needs_target_hidden_states(self) -> None:
        proposer = _proposer()
        assert proposer.needs_target_hidden_states([], has_final_prefill=False) is False
        assert proposer.needs_target_hidden_states([], has_final_prefill=True) is False


class TestNgramProposePropose:
    def test_matches_repetitive_suffix_and_drafts_continuation(self) -> None:
        # Suffix [1, 2] recurs earlier; the tokens that followed it were 3, 1, 2.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(decode_reqs=[("r0", state)])

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3, 1, 2]]

    def test_uses_scheduler_selected_num_speculative_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            num_speculative_tokens=1,
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3]]

    def test_scheduler_selected_zero_tokens_returns_none(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            num_speculative_tokens=0,
        )

        assert proposer.propose(ctx) is None

    def test_no_match_returns_none(self) -> None:
        # Non-repeating context shorter than the n-gram window: no draft.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    def test_empty_context_returns_none(self) -> None:
        assert _proposer().propose(_context()) is None

    def test_skips_request_without_sampled_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        # An empty sampled-ids entry marks a row that did not decode this step.
        ctx = _context(decode_reqs=[("r0", state)], decode_token_ids=[[]])

        assert proposer.propose(ctx) is None

    def test_skips_non_greedy_request(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2], temperature=0.8)
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    def test_finalized_prefill_participates(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"p0": state},
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["p0"]
        assert drafts.draft_token_ids == [[6, 4, 5]]

    def test_skips_intermediate_prefill(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["intermediate"],
            request_states={"p0": state},
        )

        assert proposer.propose(ctx) is None

    def test_mixed_batch_drops_unmatched_keeps_matched(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        matched = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        unmatched = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", matched), ("r1", unmatched)])

        drafts = proposer.propose(ctx)

        # Only the matched request appears; the kernel returned [] for r1.
        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3, 1, 2]]

    def test_drops_prefill_already_seen_as_decode(self) -> None:
        # A request present in both decode and prefill lists must draft once.
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        prefill = SimpleNamespace(req_id="r0")
        ctx = _context(
            decode_reqs=[("r0", state)],
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"r0": state},
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]


class TestNgramMissThrottle:
    """The kernel scans a request's whole history every step whether or not
    it finds anything, so a request that never matches should stop being
    handed to the kernel after enough consecutive misses -- these tests
    mock the wrapped upstream call directly so they exercise only the
    throttle bookkeeping, independent of the installed vLLM's kernel
    signature."""

    def test_throttles_after_max_consecutive_misses(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                assert proposer.propose(ctx) is None
            calls_before_throttle = mock_propose.call_count

            # One more miss would be the (N+1)th in a row: by now the request
            # should be on cooldown and skipped before ever reaching the kernel.
            assert proposer.propose(ctx) is None
            assert mock_propose.call_count == calls_before_throttle

    def test_hit_resets_the_streak(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose") as mock_propose:
            mock_propose.return_value = [[]]
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES - 1):
                assert proposer.propose(ctx) is None

            # A hit right before the threshold must reset the streak, not
            # just delay the cutoff by one step.
            mock_propose.return_value = [[1]]
            assert proposer.propose(ctx) is not None

            mock_propose.return_value = [[]]
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES - 1):
                assert proposer.propose(ctx) is None
            calls_before = mock_propose.call_count

            # Still under threshold post-reset: the kernel must still be
            # reachable, not skipped.
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_before + 1

    def test_cooldown_expires_and_retries(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
            calls_at_throttle = mock_propose.call_count

            # Every call during cooldown must be skipped before the kernel.
            for _ in range(ngram_mod._COOLDOWN_STEPS - 1):
                proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle

            # The cooldown-th call is the last skipped one; the call after
            # that must reach the kernel again.
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle + 1

    def test_prune_finished_clears_throttle_state(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # r0 finishes: its bookkeeping must not survive to be misread by a
        # later, unrelated request that happens to reuse the same id.
        proposer._prune_finished({})
        assert "r0" not in proposer._cooldown
        assert "r0" not in proposer._miss_streak

        with patch.object(
            proposer._ngram, "propose", return_value=[[1]]
        ) as mock_propose:
            drafts = proposer.propose(ctx)
        assert drafts is not None
        mock_propose.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
