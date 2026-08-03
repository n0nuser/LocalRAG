"""Bounded, ordered async case execution for evaluation workloads."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ConcurrencyLimits:
    """Conservative local-resource defaults for the evaluation stages."""

    retrieval: int = 2
    generation: int = 1
    judge: int = 1
    embeddings: int = 1
    metrics: int = 4
    total: int = 4

    def __post_init__(self) -> None:
        if any(value < 1 for value in self.__dict__.values()):
            raise ValueError("all concurrency limits must be at least 1")


@dataclass
class CaseOutcome[T]:
    """A stable-index outcome, including failures that did not produce a value."""

    index: int
    status: str
    value: T | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


async def run_cases[T, R](
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    limit: int,
    timeout: float,
) -> list[CaseOutcome[R]]:
    """Run cases with bounded concurrency and return outcomes in input order.

    Worker failures are represented as row failures.  Cancellation of the
    orchestration task cancels and awaits every child before propagating.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    semaphore = asyncio.Semaphore(limit)

    async def execute(index: int, item: T) -> CaseOutcome[R]:
        started = time.monotonic()
        try:
            async with semaphore:
                async with asyncio.timeout(timeout):
                    value = await worker(item)
        except TimeoutError:
            return CaseOutcome(
                index,
                "timed_out",
                error=f"case timed out after {timeout:g}s",
                elapsed_seconds=time.monotonic() - started,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # A failed case is a result.
            return CaseOutcome(
                index,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started,
            )
        return CaseOutcome(index, "completed", value=value,
                           elapsed_seconds=time.monotonic() - started)

    tasks = [asyncio.create_task(execute(index, item)) for index, item in enumerate(items)]
    try:
        return list(await asyncio.gather(*tasks))
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
