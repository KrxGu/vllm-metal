# SPDX-License-Identifier: Apache-2.0
"""N-gram (prompt-lookup) speculative decoding proposer for the Metal paged path.

An :class:`NgramProposer` drafts by matching the longest suffix n-gram of each
request's committed token history against an earlier occurrence and copying the
tokens that followed it (vLLM ``method="ngram"``). Unlike
:class:`vllm_metal.v1.draft_model_proposer.DraftModelProposer` it loads no model
and keeps no KV cache: the matching is the pure-Python + Numba KMP kernel that
vLLM ships in :mod:`vllm.v1.spec_decode.ngram_proposer`, which this class wraps.

The wrapper's only job is to translate the per-step :class:`ProposeContext` into
the runtime draft count and array arguments that upstream's stateless
``propose`` expects (``num_speculative_tokens``, ``sampled_token_ids``,
``num_tokens_no_spec``, ``token_ids_cpu``) and hand the result back as
:class:`DraftTokenIds`. The committed history lives in
``state.token_ids`` (already updated with this step's accepted/sampled tokens by
the time the runner builds the context).

The one piece of per-request bookkeeping this wrapper does keep: a consecutive-
miss streak per request, so a request with no exploitable repetition (free-form
prose, for instance) stops paying the match kernel's per-step scan cost after
a few misses in a row, with periodic retries in case the content turns
repetitive later. See ``_record_miss``/``_on_cooldown``.

The verify half is unchanged: drafts are handed back via ``take_draft_token_ids``
and verified next step by ``SpeculativeDecodeController.verify_greedy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer as VllmNgramProposer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm.config import VllmConfig

    from vllm_metal.v1.model_runner import RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)

# The match kernel scans a request's whole committed history every decode
# step whether or not it finds anything -- an O(history length) Numba scan
# plus a full-history copy into token_ids_cpu, paid regardless of outcome.
# A request whose content has no exploitable repetition (free-form prose,
# for instance) pays that tax every step for nothing. After this many
# consecutive misses, stop attempting a request for _COOLDOWN_STEPS steps
# rather than giving up on it forever -- generation can turn repetitive
# partway through (e.g. a response that starts as prose and then quotes
# earlier context), so a permanent cutoff would miss that.
_MAX_CONSECUTIVE_MISSES = 8
_COOLDOWN_STEPS = 8


class NgramProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by vLLM's n-gram kernel."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        # Per-request consecutive-miss count and, once throttled, remaining
        # cooldown steps before the next retry. Both pruned each step against
        # the live request set so they never grow past what's active.
        self._miss_streak: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        # Upstream reads only scalar config (prompt_lookup_min/max,
        # num_speculative_tokens, max_model_len, max_num_seqs) and runs a one-time
        # Numba JIT warmup in its constructor — keep that off the hot path.
        self._ngram = VllmNgramProposer(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None

        # Pre-allocate the int32 token-id buffer once. Upstream only reads
        # ``token_ids_cpu[i, :num_tokens_no_spec[i]]`` per row, so the buffer
        # just needs to be large enough to hold the longest any request's
        # committed history can ever be, across every simultaneously-scheduled
        # request. Reusing it removes a per-step ``np.zeros`` allocation.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        self._token_ids_cpu = np.zeros((max_num_seqs, max_model_len), dtype=np.int32)
        logger.info(
            "N-gram speculative decoding enabled "
            "(prompt_lookup=[%d, %d], num_speculative_tokens=%d, "
            "token_ids_cpu=(%d, %d) (%.2f MiB))",
            spec.prompt_lookup_min,
            spec.prompt_lookup_max,
            spec.num_speculative_tokens,
            max_num_seqs,
            max_model_len,
            self._token_ids_cpu.nbytes / (1024 * 1024),
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> NgramProposer:
        return cls(vllm_config=vllm_config, controller=controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # N-gram matches token ids only; it never reads the target's hidden states.
        return False

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        if ctx.num_speculative_tokens <= 0:
            return None

        self._prune_finished(ctx.request_states)

        drafting = list(
            self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_result_modes,
                ctx.request_states,
                logitsprocs=ctx.logitsprocs,
            )
        )
        if not drafting:
            return None

        drafting = [
            (req_id, state)
            for req_id, state in drafting
            if not self._on_cooldown(req_id)
        ]
        if not drafting:
            return None

        # Upstream marks a row "active" by a non-empty sampled-ids entry; the
        # match itself reads only token_ids_cpu[i, :num_tokens_no_spec[i]]. We
        # forward exactly the requests we have decided may draft, so every row is
        # active and num_tokens_no_spec is the committed history length.
        num_requests = len(drafting)
        num_tokens_no_spec = np.array(
            [len(state.token_ids) for _, state in drafting], dtype=np.int32
        )
        token_ids_cpu = self._token_ids_cpu[:num_requests]
        token_ids_cpu[:, :] = 0
        for i, (_, state) in enumerate(drafting):
            token_ids_cpu[i, : len(state.token_ids)] = state.token_ids
        sampled_token_ids: list[list[int]] = [[0]] * num_requests

        drafts = self._ngram.propose(
            ctx.num_speculative_tokens,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        for (req_id, _), draft in zip(drafting, drafts, strict=True):
            if not draft:
                self._record_miss(req_id)
                continue
            self._miss_streak.pop(req_id, None)
            req_ids.append(req_id)
            # Upstream already yields Python ints via ndarray.tolist() — the
            # old ``[int(t) for t in draft]`` was redundant.
            draft_token_ids.append(list(draft))

        if not req_ids:
            return None

        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)

    # -- miss-streak throttling ----------------------------------------------

    def _on_cooldown(self, req_id: str) -> bool:
        remaining = self._cooldown.get(req_id, 0)
        if remaining <= 0:
            return False
        if remaining == 1:
            del self._cooldown[req_id]
        else:
            self._cooldown[req_id] = remaining - 1
        return True

    def _record_miss(self, req_id: str) -> None:
        streak = self._miss_streak.get(req_id, 0) + 1
        if streak >= _MAX_CONSECUTIVE_MISSES:
            self._cooldown[req_id] = _COOLDOWN_STEPS
            self._miss_streak.pop(req_id, None)
        else:
            self._miss_streak[req_id] = streak

    def _prune_finished(self, request_states: Mapping[str, RequestState]) -> None:
        if not self._miss_streak and not self._cooldown:
            return
        for req_id in list(self._miss_streak):
            if req_id not in request_states:
                del self._miss_streak[req_id]
        for req_id in list(self._cooldown):
            if req_id not in request_states:
                del self._cooldown[req_id]
