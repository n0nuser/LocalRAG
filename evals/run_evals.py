"""RAGAS evaluation runner.

Loads the dataset from evals/dataset.json, evaluates the RAG pipeline
(or uses pre-built answer/context if run in offline mode), and writes
results to evals/results/<timestamp>.json.

Usage:
    uv run python evals/run_evals.py [--api-url URL] [--api-key KEY] [--offline]

Options:
    --api-url       LocalRAG API base URL (default: http://localhost:8000)
    --api-key       X-API-Key header value (empty = no auth)
    --offline       Skip live API calls; use stored contexts from dataset only
                    (requires ground-truth answers to already be in the dataset)
    --judge-model   Ollama model used as the RAGAS LLM judge (default: gemma3:4b)
    --ollama-url    Ollama base URL for the judge/embeddings (default: http://localhost:11434)

The RAGAS judge LLM and embeddings run on the same local Ollama instance
LocalRAG itself uses, via Ollama's OpenAI-compatible `/v1` endpoint and the
`openai` client already a core dependency of this project — no LangChain, no
new dependency, no external API key required. This matches LocalRAG's
offline-first positioning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

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

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"

PASS_THRESHOLDS = {
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
    records: list[dict],
    api_url: str,
    api_key: str,
    offline: bool,
) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        question = rec["question"]
        ground_truth = rec.get("ground_truth", "")
        stored_contexts = rec.get("contexts", [])

        if offline:
            answer = rec.get("answer", ground_truth)
            contexts = stored_contexts
        else:
            print(f"  querying: {question[:60]}...")
            answer, live_contexts = _query_api(question, api_url, api_key)
            contexts = live_contexts or stored_contexts

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
    return rows


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
    per_metric: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    for i, row in enumerate(rows, start=1):
        print(f"  scoring {i}/{len(rows)}: {row['question'][:60]}...")
        user_input = row["question"]
        response = row["answer"]
        retrieved_contexts = row["contexts"]
        reference = row["ground_truth"]

        faithfulness_result = await faithfulness.ascore(
            user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
        )
        relevancy_result = await answer_relevancy.ascore(user_input=user_input, response=response)
        precision_result = await context_precision.ascore(
            user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
        )
        recall_result = await context_recall.ascore(
            user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference
        )

        per_metric["faithfulness"].append(float(faithfulness_result.value))
        per_metric["answer_relevancy"].append(float(relevancy_result.value))
        per_metric["context_precision"].append(float(precision_result.value))
        per_metric["context_recall"].append(float(recall_result.value))
    return per_metric


def _print_summary(scores: dict[str, float]) -> bool:
    all_pass = True
    print("\n╔══════════════════════════════════╗")
    print("║       RAGAS Eval Results         ║")
    print("╠══════════════════════════════════╣")
    for metric, score in scores.items():
        threshold = PASS_THRESHOLDS.get(metric, 0.5)
        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
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
    args = parser.parse_args()

    records: list[dict] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} evaluation examples from {DATASET_PATH}")

    print("Building dataset" + (" (offline mode)" if args.offline else " (live API)") + "...")
    rows = _build_rows(records, args.api_url, args.api_key, offline=args.offline)

    print(f"Running RAGAS evaluation (judge={args.judge_model} via {args.ollama_url})...")
    # AsyncOpenAI/OpenAIEmbeddings here are just clients for the OpenAI-shaped
    # wire protocol that Ollama's /v1 endpoint speaks. base_url points at the
    # local Ollama instance, api_key is a required-but-ignored dummy value —
    # no OpenAI account, key, or network call is ever involved.
    ollama_client = AsyncOpenAI(base_url=f"{args.ollama_url.rstrip('/')}/v1", api_key="ollama")
    judge_llm = llm_factory(args.judge_model, client=ollama_client, adapter="instructor")
    judge_embeddings = OpenAIEmbeddings(client=ollama_client, model="nomic-embed-text")
    per_metric = asyncio.run(
        _score_rows(
            rows,
            faithfulness=Faithfulness(llm=judge_llm),
            answer_relevancy=AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            context_precision=ContextPrecision(llm=judge_llm),
            context_recall=ContextRecall(llm=judge_llm),
        )
    )

    scores: dict[str, float] = {name: _mean_score(values) for name, values in per_metric.items()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{ts}.json"
    out_path.write_text(
        json.dumps({"timestamp": ts, "scores": scores}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults written to {out_path}")

    all_pass = _print_summary(scores)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
