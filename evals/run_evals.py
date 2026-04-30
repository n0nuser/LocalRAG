"""RAGAS evaluation runner.

Loads the dataset from evals/dataset.json, evaluates the RAG pipeline
(or uses pre-built answer/context if run in offline mode), and writes
results to evals/results/<timestamp>.json.

Usage:
    uv run python evals/run_evals.py [--api-url URL] [--api-key KEY] [--offline]

Options:
    --api-url   LocalRAG API base URL (default: http://localhost:8000)
    --api-key   X-API-Key header value (empty = no auth)
    --offline   Skip live API calls; use stored contexts from dataset only
                (requires ground-truth answers to already be in the dataset)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
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


def _build_hf_dataset(
    records: list[dict],
    api_url: str,
    api_key: str,
    offline: bool,
) -> Dataset:
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
    return Dataset.from_list(rows)


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
    args = parser.parse_args()

    records: list[dict] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} evaluation examples from {DATASET_PATH}")

    print("Building dataset" + (" (offline mode)" if args.offline else " (live API)") + "...")
    dataset = _build_hf_dataset(records, args.api_url, args.api_key, offline=args.offline)

    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores: dict[str, float] = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

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
