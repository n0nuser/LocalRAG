# Late-Interaction Feasibility Spike

Issue: #70

## Decision

**Reject for default adoption and defer a real-model benchmark.** The spike
validates the MaxSim/index mechanics and measurement contract, but does not
claim that ColBERT improves LocalRAG retrieval. The fixture is too small and
hand-authored to establish quality, and this CPU environment did not download
or execute a ColBERT model. This is a successful feasibility boundary, not a
failed implementation.

## Reproduction

From the repository root:

```bash
uv run python research/late_interaction_spike/run_fixture.py \
  --output evals/results/late-interaction-fixture
uv run pytest tests/test_late_interaction.py
```

The command writes `result.json` compatible with `evals.results.schema.ResultFile`,
`manifest.json` compatible with the matrix manifest contract, and the explicit
JSON token index. The fixture checksum and run metadata are in those artifacts.

## Fixed protocol

| Item | Specification |
| --- | --- |
| Model | Prototype: explicit deterministic vectors. Planned real model: `colbert-ir/colbertv2.0`, immutable Hugging Face revision recorded at execution |
| Tokenizer | Prototype: whitespace fixture tokens. Real run: the model's `AutoTokenizer` at the same revision |
| Artifact | JSON token matrices for the prototype; real run must record tokenizer/model revision, index format, precision, and checksum |
| Corpus/query set | `fixture.json`: 4 chunks, 4 queries, one relevant chunk per query, checksum `sha256:late-interaction-fixture-v1` |
| Baselines | Same fixture and queries for one dense vector, BM25 token overlap, and a deterministic dense+BM25 score sum |
| Hardware | Recorded at runtime in `result.json`; this run was WSL2 Linux, Python 3.13.13, CPU-only |
| Precision/batch | Prototype float64 Python values, batch size 1. Real run: float32, batch size recorded |
| Warm-up/repetitions | One warm-up plus 20 warm query repetitions per query; cold process startup is not mixed into warm timings |
| Quality | Recall@1 in the smoke fixture. Real run must add Recall@k and NDCG@10 against the same annotated dataset |
| Resources | Index build time, Python allocation peak, serialized index bytes, warm p50/p95. VRAM is explicitly unavailable in CPU-only mode; real run must sample RSS and VRAM |

## Observed fixture result

The checked-in result artifact reports Recall@1 of dense **1.00**, BM25
**1.00**, hybrid **1.00**, and late interaction **0.75**. This is a fixture
construction outcome, not a model comparison. The token index is **222 bytes**;
the observed build allocation peak is **1,960 bytes**. Warm late-interaction
latency was approximately **0.013 ms p50 / 0.013 ms p95** on this run. These
numbers are not extrapolations to production corpora.

## Offline and dependency assessment

The committed prototype has no new dependency and runs without network access.
The planned `colbert-ir/colbertv2.0` model card is MIT licensed, as is the
ColBERT repository. The planned runtime requires optional PyTorch,
Transformers, and ColBERT/PLAID components; their exact versions, transitive
licenses, model revision, and cache contents must be locked and recorded before
any real benchmark. The upstream documentation provides a CPU environment,
but notes GPU support is required for practical indexing; CPU behavior therefore
needs a first-class latency/resource measurement rather than an assumption.
First-run model/checkpoint/tokenizer downloads are preparation and must be
completed before starting the timed benchmark.

## Adoption matrix

Adopt only if every gate passes: NDCG@10 improves by at least 5 percentage
points over hybrid; warm p95 is no more than 2x hybrid; peak RAM is no more
than 2x hybrid; the serialized index is no more than 5x dense or the quality
gain justifies the excess; offline installation and licenses are acceptable;
and an isolated owner-supported integration plan exists. Missing evidence is
not a pass. Current status is **defer** because no gate can be evaluated on a
representative real-model run.

## Limitations and follow-up

The fixture has no long-tail vocabulary, distractor passages, multi-hop cases,
or learned embeddings. It measures no real RSS/VRAM, cold-start model loading,
tokenizer throughput, or production-scale index behavior. A follow-up issue
should first select a representative annotated LocalRAG corpus and hardware,
then run dense, BM25, hybrid, and late interaction through the same #73/#84
provenance contracts. This spike does not create the production plugin
architecture proposed by #81.

Sources: [ColBERT model card](https://huggingface.co/colbert-ir/colbertv2.0),
[ColBERT repository](https://github.com/stanford-futuredata/ColBERT).
