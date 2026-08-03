"""RAGAS evaluation runner.

Loads a registered dataset (see evals/dataset/), evaluates the RAG pipeline
(or uses pre-built answer/context if run in offline mode), and writes
results to evals/results/<timestamp>.json.

Usage:
    uv run python evals/run_evals.py [--api-url URL] [--api-key KEY] [--offline]

Options:
    --api-url       LocalRAG API base URL (default: http://localhost:8000)
    --api-key       X-API-Key header value (empty = no auth)
    --offline       Skip live API calls; use stored contexts/answers from the dataset only
    --judge-model   Ollama model used as the RAGAS LLM judge (default: gemma3:4b)
    --ollama-url    Ollama base URL for the judge/embeddings (default: http://localhost:11434)
    --seed          Seed for down-sampling and judge sampling. Precedence:
                    this flag > EVAL_SEED env var > built-in default (42).
    --sample        Evaluate only N examples, chosen deterministically from the seed
    --dataset       Registered dataset_id (default: localrag-core)
    --version       Dataset version (default: highest registered)
    --split         Named split within the dataset (default: default)

The RAGAS judge LLM and embeddings run on the same local Ollama instance
LocalRAG itself uses, via Ollama's OpenAI-compatible `/v1` endpoint and the
`openai` client already a core dependency of this project — no LangChain, no
new dependency, no external API key required. This matches LocalRAG's
offline-first positioning.

Runs are reproducible as far as the stack allows: the seed fixes down-sampling
and judge sampling, and each result file embeds dataset identity plus the
environment it was produced in — including capability flags for values that
may legitimately be unavailable (no GPU, Ollama unreachable) rather than
fabricated. This is an input/config reproducibility guarantee, not a
bit-for-bit model-output guarantee. See `docs/reproducibility.md`.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from evals.concurrency import CaseOutcome, ConcurrencyLimits, run_cases
from evals.dataset.checksum import manifest_checksum
from evals.dataset.errors import DatasetError, OfflineArtifactsMissingError
from evals.dataset.registry import load_dataset
from evals.dataset.schema import DatasetRecord
from evals.environment import capture_run_metadata, resolve_seed
from evals.metrics import exact_match, f1, score_citation_accuracy
from evals.results.schema import (
    EvaluationCaseResult,
    MetricCaseResult,
    MetricDescriptor,
    MetricResult,
    ResultFile,
)

RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_DATASET_ID = "localrag-core"
DEFAULT_SPLIT = "default"
JUDGE_EMBED_MODEL = "nomic-embed-text"

PASS_THRESHOLDS = {
    "exact_match": 1.0,
    "f1": 0.5,
    "hallucination_rate": 0.4,
    "citation_accuracy": 0.8,
    "faithfulness": 0.6,
    "answer_relevancy": 0.6,
    "context_precision": 0.5,
    "context_recall": 0.5,
}


def _query_api(question: str, api_url: str, api_key: str) -> tuple[str, list[str]]:
    """Call POST /query and return (answer, contexts)."""
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = httpx.post(
        f"{api_url.rstrip('/')}/query",
        json={"question": question},
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    answer = body.get("answer", "")
    sources = body.get("sources", [])
    contexts = [s.get("source", "") for s in sources]
    return answer, contexts


def _build_rows(
    records: list[DatasetRecord],
    api_url: str,
    api_key: str,
    offline: bool,
) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        if offline:
            if rec.offline_answer is None and not rec.reference_answer:
                raise OfflineArtifactsMissingError(rec.record_id, "answer")
            answer = rec.offline_answer if rec.offline_answer is not None else rec.reference_answer
            contexts = rec.offline_context_texts()
            if not contexts:
                raise OfflineArtifactsMissingError(rec.record_id, "contexts")
        else:
            print(f"  querying: {rec.question[:60]}...")
            answer, live_contexts = _query_api(rec.question, api_url, api_key)
            contexts = live_contexts or rec.offline_context_texts()

        rows.append(
            {
                "record_id": rec.record_id,
                "question": rec.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": rec.reference_answer,
                "ground_truths": rec.reference_answers_or_default(),
                "answer_citation_ids": rec.answer_citation_ids,
                "relevant_citation_ids": rec.relevant_citation_ids(),
            }
        )
    return rows


async def _build_rows_async(
    records: list[DatasetRecord],
    api_url: str,
    api_key: str,
    *,
    offline: bool,
    limits: ConcurrencyLimits,
    timeout: float,
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Build rows concurrently while retaining dataset order and row failures."""
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def build(record: DatasetRecord) -> dict:
            if offline:
                if record.offline_answer is None and not record.reference_answer:
                    raise OfflineArtifactsMissingError(record.record_id, "answer")
                answer = record.offline_answer or record.reference_answer
                contexts = record.offline_context_texts()
                if not contexts:
                    raise OfflineArtifactsMissingError(record.record_id, "contexts")
            else:
                response = await client.post(
                    f"{api_url.rstrip('/')}/query",
                    json={"question": record.question},
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
                answer = body.get("answer", "")
                contexts = [source.get("source", "") for source in body.get("sources", [])]
                contexts = contexts or record.offline_context_texts()
            return {
                "record_id": record.record_id,
                "question": record.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": record.reference_answer,
                "ground_truths": record.reference_answers_or_default(),
                "answer_citation_ids": record.answer_citation_ids,
                "relevant_citation_ids": record.relevant_citation_ids(),
            }

        outcomes = await run_cases(
            records,
            build,
            limit=min(limits.total, limits.retrieval, limits.generation),
            timeout=timeout,
        )
    rows: list[dict] = []
    execution: list[dict[str, Any]] = []
    for record, outcome in zip(records, outcomes, strict=True):
        execution.append(_execution_record(record.record_id, "retrieval", outcome))
        if outcome.value is not None:
            rows.append(outcome.value)
        else:
            rows.append(_empty_row(record))
    return rows, execution


def _empty_row(record: DatasetRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "question": record.question,
        "answer": "",
        "contexts": [],
        "ground_truth": record.reference_answer,
        "ground_truths": record.reference_answers_or_default(),
        "answer_citation_ids": record.answer_citation_ids,
        "relevant_citation_ids": record.relevant_citation_ids(),
    }


def _execution_record(record_id: str, stage: str, outcome: CaseOutcome[Any]) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "stage": stage,
        "status": outcome.status,
        "error": outcome.error,
        "elapsed_seconds": outcome.elapsed_seconds,
        "attempts": 1,
    }


def _select_records(
    records: list[DatasetRecord], *, seed: int, sample: int | None
) -> list[DatasetRecord]:
    """Order records by stable record_id, optionally down-sampling to ``sample`` rows.

    Ordering by record_id (not list position) makes selection independent of
    how records happen to be laid out in the manifest file, so two runs with
    the same seed evaluate the same records even after the file is reordered.
    """
    ordered = sorted(records, key=lambda rec: rec.record_id)
    if sample is None or sample >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, sample)  # noqa: S311 — benchmark sampling, not crypto


def _mean_score(values: list[float]) -> float:
    """Average a metric's per-row scores, ignoring NaNs."""
    clean = [v for v in values if not math.isnan(v)]
    return statistics.fmean(clean) if clean else math.nan


async def _score_rows(
    rows: list[dict],
    *,
    faithfulness: Faithfulness,
    answer_relevancy: AnswerRelevancy,
    context_precision: ContextPrecision,
    context_recall: ContextRecall,
) -> dict[str, list[float]]:
    scores, _ = await _score_rows_parallel(
        rows,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
        limits=ConcurrencyLimits(),
        timeout=120,
    )
    return scores


async def _score_rows_parallel(
    rows: list[dict],
    *,
    faithfulness: Faithfulness,
    answer_relevancy: AnswerRelevancy,
    context_precision: ContextPrecision,
    context_recall: ContextRecall,
    limits: ConcurrencyLimits,
    timeout: float,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Score independent rows with bounded judge/metric work."""
    per_metric: dict[str, list[float]] = {
        "exact_match": [],
        "f1": [],
        "hallucination_rate": [],
        "citation_accuracy": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    judge_limit = asyncio.Semaphore(min(limits.judge, limits.total))
    embedding_limit = asyncio.Semaphore(min(limits.embeddings, limits.total))

    async def judge_call(call: Any, *, embedding: bool = False) -> Any:
        semaphore = embedding_limit if embedding else judge_limit
        async with semaphore:
            return await call()

    async def score(row: dict) -> dict[str, Any]:
        user_input = row["question"]
        response = row["answer"]
        retrieved_contexts = row["contexts"]
        reference = row["ground_truth"]

        result: dict[str, Any] = {
            "exact_match": exact_match(response, row["ground_truths"]),
            "f1": f1(response, row["ground_truths"]),
        }
        citation = score_citation_accuracy(
            response, row["answer_citation_ids"], row["relevant_citation_ids"]
        )
        result["citation_accuracy"] = citation.value if citation.value is not None else math.nan

        judge_results = await asyncio.gather(
            judge_call(lambda: faithfulness.ascore(
                user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
            )),
            judge_call(
                lambda: answer_relevancy.ascore(user_input=user_input, response=response),
                embedding=True,
            ),
            judge_call(lambda: context_precision.ascore(
                user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
            )),
            judge_call(lambda: context_recall.ascore(
                user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference
            )),
            return_exceptions=True,
        )
        judge_names = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        errors: list[str] = []
        for name, judge_result in zip(judge_names, judge_results, strict=True):
            if isinstance(judge_result, BaseException):
                result[name] = math.nan
                errors.append(f"{name}: {type(judge_result).__name__}: {judge_result}")
            else:
                try:
                    result[name] = float(judge_result.value)
                except (AttributeError, TypeError, ValueError) as exc:
                    result[name] = math.nan
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
        result["hallucination_rate"] = (
            1.0 - result["faithfulness"]
            if math.isfinite(result["faithfulness"])
            else math.nan
        )
        if errors:
            result["_error"] = "; ".join(errors)
        return result

    outcomes = await run_cases(
        rows, score, limit=min(limits.metrics, limits.total), timeout=timeout
    )
    execution: list[dict[str, Any]] = []
    for row, outcome in zip(rows, outcomes, strict=True):
        execution_record = _execution_record(row["record_id"], "metrics", outcome)
        values = outcome.value or dict.fromkeys(per_metric, math.nan)
        if outcome.value and values.get("_error"):
            execution_record["status"] = "failed"
            execution_record["error"] = values["_error"]
        execution.append(execution_record)
        for name, metric_values in per_metric.items():
            value = values.get(name, math.nan)
            metric_values.append(value if math.isfinite(value) else math.nan)
    return per_metric, execution


def _print_summary(scores: dict[str, float]) -> bool:
    all_pass = True
    print("\n╔══════════════════════════════════╗")
    print("║       RAGAS Eval Results         ║")
    print("╠══════════════════════════════════╣")
    for metric, score in scores.items():
        threshold = PASS_THRESHOLDS.get(metric, 0.5)
        if not math.isfinite(score):
            status = "UNAVAILABLE"
        elif metric == "hallucination_rate":
            status = "PASS" if score <= threshold else "FAIL"
        else:
            status = "PASS" if score >= threshold else "FAIL"
        if status != "PASS":
            all_pass = False
        print(f"║  {metric:<22} {score:.3f}  {status} ║")
    print("╚══════════════════════════════════╝")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evals against the LocalRAG API.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--judge-model", default="gemma3:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for down-sampling and judge sampling. "
            "Precedence: this flag > EVAL_SEED env var > built-in default (42)."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate only N examples, chosen deterministically from the seed.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="Registered dataset_id.")
    parser.add_argument("--version", default=None, help="Dataset version (default: highest).")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Named split to evaluate.")
    parser.add_argument("--retrieval-concurrency", type=int, default=2)
    parser.add_argument("--generation-concurrency", type=int, default=1)
    parser.add_argument("--judge-concurrency", type=int, default=1)
    parser.add_argument("--embedding-concurrency", type=int, default=1)
    parser.add_argument("--metric-concurrency", type=int, default=4)
    parser.add_argument("--total-concurrency", type=int, default=4)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    args = parser.parse_args()

    try:
        seed, seed_source = resolve_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if seed < 0:
        parser.error(f"seed must be a non-negative integer, got {seed} (source: {seed_source})")

    random.seed(seed)

    try:
        manifest = load_dataset(args.dataset, args.version)
        all_records = manifest.split(args.split)
    except DatasetError as exc:
        parser.error(str(exc))

    records = _select_records(all_records, seed=seed, sample=args.sample)
    print(
        f"Loaded {len(all_records)} examples from {manifest.dataset_id} "
        f"v{manifest.dataset_version} split={args.split!r} "
        f"(evaluating {len(records)}, seed={seed})"
    )

    try:
        limits = ConcurrencyLimits(
            retrieval=args.retrieval_concurrency,
            generation=args.generation_concurrency,
            judge=args.judge_concurrency,
            embeddings=args.embedding_concurrency,
            metrics=args.metric_concurrency,
            total=args.total_concurrency,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print("Building dataset" + (" (offline mode)" if args.offline else " (live API)") + "...")
    rows, execution = asyncio.run(
        _build_rows_async(
            records,
            args.api_url,
            args.api_key,
            offline=args.offline,
            limits=limits,
            timeout=args.case_timeout,
        )
    )

    print(f"Running RAGAS evaluation (judge={args.judge_model} via {args.ollama_url})...")
    # AsyncOpenAI/OpenAIEmbeddings here are just clients for the OpenAI-shaped
    # wire protocol that Ollama's /v1 endpoint speaks. base_url points at the
    # local Ollama instance, api_key is a required-but-ignored dummy value —
    # no OpenAI account, key, or network call is ever involved.
    ollama_client = AsyncOpenAI(base_url=f"{args.ollama_url.rstrip('/')}/v1", api_key="ollama")
    # temperature=0 + a fixed seed make the judge's own verdicts repeatable;
    # without them the same answer can score differently run to run.
    judge_llm = llm_factory(
        args.judge_model,
        client=ollama_client,
        adapter="instructor",
        temperature=0.0,
        seed=seed,
    )
    judge_embeddings = OpenAIEmbeddings(client=ollama_client, model=JUDGE_EMBED_MODEL)
    per_metric, metric_execution = asyncio.run(
        _score_rows_parallel(
            rows,
            faithfulness=Faithfulness(llm=judge_llm),
            answer_relevancy=AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            context_precision=ContextPrecision(llm=judge_llm),
            context_recall=ContextRecall(llm=judge_llm),
            limits=limits,
            timeout=args.case_timeout,
        )
    )
    execution.extend(metric_execution)

    scores: dict[str, float] = {name: _mean_score(values) for name, values in per_metric.items()}

    metadata = capture_run_metadata(
        seed=seed,
        seed_source=seed_source,
        judge_model=args.judge_model,
        embedding_model=JUDGE_EMBED_MODEL,
        ollama_url=args.ollama_url,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{ts}.json"
    result = ResultFile.model_validate(
        {
            "run_id": ts,
            "timestamp": datetime.now(UTC),
            "dataset": {
                "dataset_id": manifest.dataset_id,
                "dataset_version": manifest.dataset_version,
                "split": args.split,
                "checksum": manifest_checksum(manifest),
            },
            "selected_ids": [row["record_id"] for row in rows],
            "metrics": [
                MetricResult(
                    descriptor=MetricDescriptor.model_validate(
                        {
                            "name": name,
                            "direction": "lower_is_better"
                            if name == "hallucination_rate"
                            else "higher_is_better",
                            "threshold": PASS_THRESHOLDS[name],
                            "missing_value": "not_applicable"
                            if name == "citation_accuracy"
                            else "missing",
                        }
                    ),
                    value=score,
                    cases={
                        row["record_id"]: value
                        for row, value in zip(rows, per_metric[name], strict=True)
                    },
                    non_finite_cases=[
                        row["record_id"]
                        for row, value in zip(rows, per_metric[name], strict=True)
                        if not math.isfinite(value)
                    ],
                    case_results={
                        row["record_id"]: MetricCaseResult.model_validate(
                            {
                                "value": value if math.isfinite(value) else None,
                                "threshold": PASS_THRESHOLDS[name],
                                "status": (
                                    "complete"
                                    if math.isfinite(value)
                                    else ("unavailable" if name == "citation_accuracy" else "error")
                                ),
                                "input_ids": row.get("relevant_citation_ids", []),
                                "warning": (
                                    "citation annotation is missing"
                                    if name == "citation_accuracy" and not math.isfinite(value)
                                    else None
                                ),
                                "error": (
                                    "judge returned a non-finite value"
                                    if name not in {"citation_accuracy", "exact_match", "f1"}
                                    and not math.isfinite(value)
                                    else None
                                ),
                            }
                        )
                        for row, value in zip(rows, per_metric[name], strict=True)
                    },
                    valid_count=sum(math.isfinite(value) for value in per_metric[name]),
                    missing_count=sum(not math.isfinite(value) for value in per_metric[name])
                    if name == "citation_accuracy"
                    else 0,
                    error_count=sum(not math.isfinite(value) for value in per_metric[name])
                    if name != "citation_accuracy"
                    else 0,
                )
                for name, score in scores.items()
            ],
            "provenance": {
                **metadata.to_dict(),
                "metric_contract_version": "1.0",
                "judge_prompt_version": "ragas-default-prompts@0.4.3",
                "judge_seed": seed,
                "judge_endpoint": args.ollama_url,
                "concurrency": {
                    "retrieval": limits.retrieval,
                    "generation": limits.generation,
                    "judge": limits.judge,
                    "embeddings": limits.embeddings,
                    "metrics": limits.metrics,
                    "total": limits.total,
                    "case_timeout_seconds": args.case_timeout,
                },
            },
            "cases": [EvaluationCaseResult.model_validate(item) for item in execution],
            "failure_counts": {
                status: sum(item["status"] == status for item in execution)
                for status in {item["status"] for item in execution if item["status"] != "completed"}
            },
            "status": "failed" if any(item["status"] != "completed" for item in execution) else "complete",
            "exit_code": 1 if any(item["status"] != "completed" for item in execution) else 0,
        }
    )
    out_path.write_text(result.model_dump_json_safe(), encoding="utf-8")
    if metadata.git_dirty.value:
        print("WARNING: working tree is dirty — these results are not tied to a clean commit.")
    print(f"\nResults written to {out_path}")

    all_pass = _print_summary(scores)
    sys.exit(0 if all_pass and not any(item["status"] != "completed" for item in execution) else 1)


if __name__ == "__main__":
    main()
