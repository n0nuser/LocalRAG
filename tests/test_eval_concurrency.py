from __future__ import annotations

import asyncio
import math

import pytest

from evals.concurrency import ConcurrencyLimits, run_cases
from evals.run_evals import _score_rows_parallel


@pytest.mark.asyncio
async def test_cases_are_bounded_and_returned_in_input_order() -> None:
    active = 0
    peak = 0

    async def worker(value: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep((4 - value) * 0.001)
        active -= 1
        return value * 2

    outcomes = await run_cases([0, 1, 2, 3], worker, limit=2, timeout=1)
    assert [outcome.value for outcome in outcomes] == [0, 2, 4, 6]
    assert peak == 2


@pytest.mark.asyncio
async def test_timeout_and_provider_failure_are_row_outcomes() -> None:
    async def worker(value: str) -> str:
        if value == "slow":
            await asyncio.sleep(1)
        if value == "bad":
            raise RuntimeError("provider down")
        return value

    outcomes = await run_cases(["slow", "bad", "ok"], worker, limit=2, timeout=0.01)
    assert [outcome.status for outcome in outcomes] == ["timed_out", "failed", "completed"]
    assert "provider down" in (outcomes[1].error or "")


@pytest.mark.asyncio
async def test_cancellation_cleans_up_child_tasks() -> None:
    started = asyncio.Event()

    async def worker(_value: int) -> None:
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(run_cases([1, 2], worker, limit=1, timeout=20))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not [
        candidate
        for candidate in asyncio.all_tasks()
        if candidate is not asyncio.current_task() and not candidate.done()
    ]


def test_defaults_are_conservative_for_local_ollama() -> None:
    limits = ConcurrencyLimits()
    assert limits.generation == limits.judge == limits.embeddings == 1


@pytest.mark.asyncio
async def test_parallel_scoring_matches_sequential_deterministic_metrics() -> None:
    class Result:
        def __init__(self, value: float) -> None:
            self.value = value

    class Metric:
        async def ascore(self, **_kwargs: object) -> Result:
            await asyncio.sleep(0.001)
            return Result(1.0)

    rows = [
        {
            "record_id": record_id,
            "question": "q",
            "answer": answer,
            "contexts": ["context"],
            "ground_truth": answer,
            "ground_truths": [answer],
            "answer_citation_ids": None,
            "relevant_citation_ids": [],
        }
        for record_id, answer in (("b", "second"), ("a", "first"))
    ]
    scores, outcomes = await _score_rows_parallel(
        rows,
        faithfulness=Metric(),
        answer_relevancy=Metric(),
        context_precision=Metric(),
        context_recall=Metric(),
        limits=ConcurrencyLimits(metrics=2, total=2),
        timeout=1,
    )
    assert scores["exact_match"] == [1.0, 1.0]
    assert scores["f1"] == [1.0, 1.0]
    assert [outcome["record_id"] for outcome in outcomes] == ["b", "a"]
    assert all(outcome["status"] == "completed" for outcome in outcomes)


@pytest.mark.asyncio
async def test_judge_failure_keeps_partial_metric_values() -> None:
    class Result:
        value = 0.75

    class Metric:
        def __init__(self, *, fails: bool = False) -> None:
            self.fails = fails

        async def ascore(self, **_kwargs: object) -> Result:
            if self.fails:
                raise RuntimeError("judge unavailable")
            return Result()

    row = {
        "record_id": "r1",
        "question": "q",
        "answer": "answer",
        "contexts": ["context"],
        "ground_truth": "answer",
        "ground_truths": ["answer"],
        "answer_citation_ids": None,
        "relevant_citation_ids": [],
    }
    scores, outcomes = await _score_rows_parallel(
        [row],
        faithfulness=Metric(fails=True),
        answer_relevancy=Metric(),
        context_precision=Metric(),
        context_recall=Metric(),
        limits=ConcurrencyLimits(),
        timeout=1,
    )
    assert scores["exact_match"] == [1.0]
    assert scores["answer_relevancy"] == [0.75]
    assert math.isnan(scores["faithfulness"][0])
    assert outcomes[0]["status"] == "failed"
