"""Generate deterministic, self-contained HTML reports from canonical artifacts."""

# The embedded document is intentionally formatted as readable HTML/JavaScript.
# ruff: noqa: E501

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.matrix import MatrixManifest
from evals.results.schema import ResultError, ResultFile, load_result


@dataclass(frozen=True)
class ReportResult:
    """Outcome of report generation, including non-fatal input errors."""

    output: Path
    run_count: int
    errors: list[str]
    incompatible_runs: list[str]


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _matrix_view(manifest: MatrixManifest, source: Path) -> dict[str, Any]:
    cases = []
    for case in sorted(manifest.cases, key=lambda item: item.case_id):
        cases.append(
            {
                "case_id": case.case_id,
                "run_id": case.run_id,
                "status": case.status,
                "config": case.effective_config,
                "metrics": {name: _finite(value) for name, value in sorted(case.metrics.items())},
                "latency": case.latency,
                "resources": case.resources,
                "error": case.error,
            }
        )
    return {
        "kind": "benchmark",
        "source": source.name,
        "run_id": manifest.run_id,
        "matrix_id": manifest.matrix_id,
        "status": manifest.status,
        "dataset": manifest.dataset.model_dump(mode="json"),
        "config": manifest.effective_config,
        "model": manifest.model,
        "cases": cases,
    }


def _result_view(result: ResultFile, source: Path) -> dict[str, Any]:
    metrics = []
    for metric in sorted(result.metrics, key=lambda item: item.descriptor.name):
        metrics.append(
            {
                "name": metric.descriptor.name,
                "value": _finite(metric.value),
                "threshold": _finite(metric.descriptor.threshold),
                "direction": metric.descriptor.direction,
                "status": "unavailable" if metric.value is None else "available",
                "outcome": (
                    "unavailable"
                    if metric.value is None
                    else (
                        "pass"
                        if metric.descriptor.threshold is None
                        or (
                            metric.value >= metric.descriptor.threshold
                            if metric.descriptor.direction == "higher_is_better"
                            else metric.value <= metric.descriptor.threshold
                        )
                        else "fail"
                    )
                ),
                "valid_count": metric.valid_count,
                "missing_count": metric.missing_count,
                "error_count": metric.error_count,
                "cases": {
                    key: {
                        "value": _finite(value),
                        "status": metric.case_results[key].status
                        if key in metric.case_results
                        else "available",
                    }
                    for key, value in sorted(metric.cases.items())
                },
            }
        )
    return {
        "kind": "evaluation",
        "source": source.name,
        "run_id": result.run_id,
        "status": result.status,
        "dataset": result.dataset.model_dump(mode="json"),
        "config": result.provenance.get("settings_snapshot", {}),
        "model": {
            key: result.provenance[key]
            for key in ("judge_model", "embedding_model")
            if key in result.provenance
        },
        "metrics": metrics,
        "cases": [case.model_dump(mode="json") for case in result.cases],
        "failure_analysis": result.failure_analysis,
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"{path}: could not read JSON: {exc}"
        raise ValueError(message) from exc
    try:
        manifest = MatrixManifest.model_validate(raw)
    except Exception:
        return _matrix_or_result_result(path)
    return _matrix_view(manifest, path)


def _matrix_or_result_result(path: Path) -> dict[str, Any]:
    try:
        result = load_result(path)
    except (ResultError, OSError, json.JSONDecodeError) as exc:
        message = f"{path}: unsupported or malformed result: {exc}"
        raise ValueError(message) from exc
    return _result_view(result, path)


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LocalRAG benchmark report</title><style>
body{font:15px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a;background:#f7f8fa}h1{margin-bottom:.25rem}section{background:white;border:1px solid #d9dee5;border-radius:8px;padding:1rem;margin:1rem 0;overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.5rem;border-bottom:1px solid #e5e7eb;vertical-align:top}th{background:#f0f3f6}.bad{color:#a61b1b}.ok{color:#176b3a}.muted{color:#667085}.pill{display:inline-block;padding:.15rem .45rem;border-radius:1rem;background:#e8edf2}.warning{border-left:4px solid #b7791f}</style></head>
<body><h1>LocalRAG benchmark report</h1><p class="muted">Generated from canonical result artifacts. Raw questions, answers, contexts, and source paths are omitted.</p>
<div id="app"></div><script id="report-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('report-data').textContent);const app=document.getElementById('app');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const val=v=>v===null||v===undefined?'Unavailable':esc(v);const row=(a,b)=>`<tr><th>${esc(a)}</th><td>${b}</td></tr>`;
function metricRows(run){return (run.metrics||[]).map(m=>`<tr><td>${esc(m.name)}</td><td>${val(m.value)}</td><td>${val(m.threshold)}</td><td>${esc(m.direction)}</td><td>${esc(m.outcome||m.status)}</td></tr>`).join('')||'<tr><td colspan="5" class="muted">No metrics recorded</td></tr>'}
 function runBlock(run){let cases=(run.cases||[]).map(c=>`<tr><td>${esc(c.case_id||c.record_id)}</td><td class="${c.status==='failed'?'bad':''}">${esc(c.status)}</td><td>${Object.entries(c.metrics||{}).map(([k,v])=>esc(k)+': '+val(v)).join('<br>')||'Unavailable'}</td><td>${Object.entries(c.latency||{}).map(([k,v])=>esc(k)+': '+val(v)).join('<br>')||'Unavailable'}</td><td>${Object.entries(c.resources||{}).map(([k,v])=>esc(k)+': '+val(v)).join('<br>')||'Unavailable'}</td><td>${c.error?esc(c.error.message||c.error):''}</td></tr>`).join('');let analysis=run.failure_analysis;if(analysis){analysis=`<h3>Failure analysis</h3><p>${esc(analysis.failed_count)} failed cases; counts: <code>${esc(JSON.stringify(analysis.counts||{}))}</code></p><table><tr><th>Case</th><th>Labels</th><th>Confidence</th></tr>${(analysis.cases||[]).map(c=>`<tr><td>${esc(c.case_id)}</td><td>${esc((c.labels||[]).join(', '))}</td><td>${val(c.confidence)}</td></tr>`).join('')}</table>`}else analysis='';return `<section><h2>${esc(run.run_id)} <span class="pill">${esc(run.kind)}</span></h2>${row('Dataset',val(run.dataset?.dataset_id)+' '+val(run.dataset?.split))}${row('Status',esc(run.status))}${row('Configuration',`<code>${esc(JSON.stringify(run.config||{}))}</code>`)}<h3>Metrics</h3><table><tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Direction</th><th>Availability</th></tr>${metricRows(run)}</table>${analysis}<h3>Cases and failures</h3><table><tr><th>Case</th><th>Status</th><th>Metrics</th><th>Latency</th><th>Resources</th><th>Failure</th></tr>${cases||'<tr><td colspan="6" class="muted">No cases recorded</td></tr>'}</table></section>`}
function configBlock(){const keys=[...new Set(data.runs.flatMap(r=>Object.keys(r.config||{})))].sort();if(!keys.length)return '';return `<section><h2>Configuration comparison</h2><table><tr><th>Setting</th>${data.runs.map(r=>`<th>${esc(r.run_id)}</th>`).join('')}</tr>${keys.map(k=>`<tr><th>${esc(k)}</th>${data.runs.map(r=>`<td>${val(r.config?.[k])}</td>`).join('')}</tr>`).join('')}</table></section>`}
if(!data.runs.length)app.innerHTML='<section><h2>No benchmark runs</h2><p>No valid artifacts were supplied.</p></section>';else{if(data.errors.length)app.innerHTML+=`<section class="warning"><h2>Input errors</h2><ul>${data.errors.map(esc).map(e=>`<li>${e}</li>`).join('')}</ul></section>`;if(data.incompatible.length)app.innerHTML+=`<section class="warning"><h2>Incompatible runs</h2><p>Runs use different dataset identities; score comparisons are not shown.</p><p>${data.incompatible.map(esc).join(', ')}</p></section>`;app.innerHTML+=configBlock()+data.runs.map(runBlock).join('')}
</script></body></html>"""


def generate_report(paths: list[Path], output: Path) -> ReportResult:
    """Write ``report.html`` from canonical result or matrix artifacts."""
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            runs.append(_load(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    runs.sort(key=lambda run: (run["run_id"], run["source"]))
    identities = {json.dumps(run["dataset"], sort_keys=True) for run in runs}
    incompatible = [run["run_id"] for run in runs] if len(identities) > 1 else []
    data = {"runs": runs, "errors": errors, "incompatible": incompatible}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_HTML.replace("__DATA__", _safe_json(data)), encoding="utf-8")
    return ReportResult(output, len(runs), errors, incompatible)
