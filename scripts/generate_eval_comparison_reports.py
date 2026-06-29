#!/usr/bin/env python3
"""Generate a combined cross-model evaluation markdown report."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANG = "hi"


def combined_report_path(lang: str = DEFAULT_LANG) -> str:
    return f"artifacts/results/{lang}/eval_summary_report.md"


def eval_artifacts_dir(repo_root: Path, lang: str = DEFAULT_LANG) -> Path:
    return repo_root / "artifacts/evals" / lang


def results_dir(repo_root: Path, lang: str = DEFAULT_LANG) -> Path:
    return repo_root / "artifacts/results" / lang

HEADLINE_METRICS: dict[str, tuple[str, ...]] = {
    "ner": ("f1", "precision", "recall", "accuracy"),
    "massive_intent": ("macro_f1", "accuracy"),
    "retrieval": (
        "mldr_hi.ndcg@10",
        "mldr_hi.recall@10",
        "mldr_hi.mrr@10",
        "mmarco_hindi.ndcg@10",
        "mmarco_hindi.recall@10",
        "mmarco_hindi.mrr@10",
    ),
}

TASK_TITLES: dict[str, str] = {
    "ner": "Naamapadam Hindi NER",
    "massive_intent": "MASSIVE Hindi Intent",
    "retrieval": "Retrieval (DPR fine-tuned)",
}

METRIC_LABELS: dict[str, str] = {
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "mldr_hi.ndcg@10": "MLDR nDCG@10",
    "mldr_hi.recall@10": "MLDR Recall@10",
    "mldr_hi.mrr@10": "MLDR MRR@10",
    "mmarco_hindi.ndcg@10": "mMARCO nDCG@10",
    "mmarco_hindi.recall@10": "mMARCO Recall@10",
    "mmarco_hindi.mrr@10": "mMARCO MRR@10",
}


def _metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " "))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "—"
    return str(value)


def _column_best_values(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    *,
    metrics_key: str = "metrics",
) -> dict[str, float]:
    bests: dict[str, float] = {}
    for name in metric_names:
        values = [
            float(row[metrics_key][name])
            for row in rows
            if isinstance(row.get(metrics_key, {}).get(name), (int, float))
        ]
        if values:
            bests[name] = max(values)
    return bests


def _format_metric_cell(
    value: Any,
    metric_name: str,
    best_values: dict[str, float],
) -> str:
    formatted = _format_value(value)
    if not isinstance(value, (int, float)):
        return formatted
    best = best_values.get(metric_name)
    if best is not None and value >= best - 1e-12:
        return f"**{formatted}**"
    return formatted


MODEL_LABEL_OVERRIDES: dict[str, str] = {
    "phase1": "hindi-modernBERT (phase 1)",
    "phase2_latest_ba1157": "hindi-modernBERT (phase 2)",
    "phase2_best_ba135": "hindi-modernBERT (ba135)",
    "phase2_latest_ba1157-DPR-0.00010972521281842244": "hindi-modernBERT (phase 2)",
}

HINDI_MODERNBERT_PROGRESSION_NOTE = (
    "Downstream evals improved from phase 1 to phase 2. We believe this was primarily "
    "due to continued pretraining on IndicCorp V2 Hindi (~4.85B tokens, reported as ~5B) "
    "during the 8192-context extension pass, after the initial Sangraha Hindi phase-1 run."
)

RETRIEVAL_FINETUNE_LR = 0.00010972521281842244
RETRIEVAL_FINETUNE_CONFIG = (
    "Config: local DPR fine-tune on 1.25M mMARCO Hindi pairs; "
    f"LR `{RETRIEVAL_FINETUNE_LR:.17g}` selected from the Optuna LR sweep "
    f"(`make retrieval-optuna`, log-uniform `1e-6`–`1e-2` on a 1k-query mMARCO Hindi subset)."
)

COLBERT_RETRIEVAL_NOTE = (
    "**DPR vs ColBERT:** This report covers dense DPR only (one embedding per query/document). "
    "In the upstream ModernBERT repo, the largest MLDR gains (~28 → ~80 nDCG@10 on English MLDR) "
    "come from **ColBERT MaxSim** multi-vector retrieval via PyLate, not from DPR. "
    "Our `mldr_hi` DPR scores (~0.26 nDCG@10) are expected for that setup — DPR compresses "
    "long documents into a single vector and hits a ceiling on MLDR. Do not compare these numbers "
    "to upstream Table 1 ColBERT headline results."
)

EVAL_TODO_ITEMS: tuple[str, ...] = (
    "**ColBERT / PyLate retrieval (Hindi)** — Port upstream multi-vector recipe "
    "(`_support_repo/ModernBERT/examples/train_pylate.py`, `evaluate_pylate.py`): "
    "Hindi mMARCO-style fine-tune (or KD), then evaluate `mldr_hi` with MaxSim. "
    "This is the main eval gap vs the ModernBERT paper and where upstream saw the biggest "
    "long-document retrieval jump; DPR alone cannot reproduce it.",
    "**Transfer tasks** — Run IndicXTREME-style sentiment, QA, and COPA "
    "(`configs/hi/evals/hindi_transfer.yaml`; `make run-evals-transfer`).",
    "**Report DPR vs ColBERT consistently** — Keep distinguishing short-passage DPR "
    "(mMARCO selection + eval) from long-document MLDR when citing upstream retrieval numbers.",
)


def _todo_section_lines() -> list[str]:
    lines = ["## TODO", ""]
    for item in EVAL_TODO_ITEMS:
        lines.append(f"- [ ] {item}")
    lines.append("")
    return lines


def _model_label(model_path: str) -> str:
    path = Path(model_path)
    name = path.name
    if name == "final":
        run_name = path.parent.name
        if run_name in MODEL_LABEL_OVERRIDES:
            return MODEL_LABEL_OVERRIDES[run_name]
        if "-DPR-" in run_name:
            base = run_name.split("-DPR-")[0]
            return MODEL_LABEL_OVERRIDES.get(base, base)
        if path.parent.parent.name not in {"hf_export", "model"}:
            return path.parent.parent.name
        return run_name
    return MODEL_LABEL_OVERRIDES.get(name, name)


def _load_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _task_result(run: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    for result in run.get("results", []):
        if result.get("name") == task_name:
            return result
    return None


def _comparison_table(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    *,
    sort_by: str | None = None,
    include_max_seq: bool = True,
    fixed_max_seq: int | None = None,
) -> str:
    if not rows:
        return "_No completed runs._\n"

    if sort_by:
        rows = sorted(
            rows,
            key=lambda row: (
                row["metrics"].get(sort_by) is None,
                -(row["metrics"].get(sort_by) or float("-inf")),
            ),
        )

    best_values = _column_best_values(rows, metric_names)

    header = ["Model"]
    if include_max_seq:
        header.append("Max Seq")
    header.extend(_metric_label(m) for m in metric_names)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        metric_cells = [
            _format_metric_cell(row["metrics"].get(name), name, best_values)
            for name in metric_names
        ]
        max_seq = fixed_max_seq if fixed_max_seq is not None else row["max_sequence_length"]
        cells = [row["model_label"]]
        if include_max_seq:
            cells.append(str(max_seq))
        cells.extend(metric_cells)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _dedupe_supervised_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per model, preferring model_max over common_128."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model_label"], []).append(row)

    deduped: list[dict[str, Any]] = []
    for group in by_model.values():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        group.sort(
            key=lambda row: (
                row["context_mode"] != "model_max",
                -(int(row["max_sequence_length"]) if str(row["max_sequence_length"]).isdigit() else 0),
            ),
        )
        deduped.append(group[0])
    return deduped


def _collect_task_rows(runs: list[dict[str, Any]], task_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        result = _task_result(run, task_name)
        if result is None:
            continue
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "model_label": _model_label(run.get("model_name_or_path", "unknown")),
                "model_path": run.get("model_name_or_path", ""),
                "context_mode": run.get("context_mode", ""),
                "max_sequence_length": run.get("max_sequence_length", ""),
                "status": result.get("status", "unknown"),
                "metrics": metrics,
                "error": result.get("error", ""),
            }
        )
    return rows


def _task_section_lines(
    *,
    task_name: str,
    runs: list[dict[str, Any]],
    heading_level: int = 3,
    dedupe_by_model: bool = False,
    include_max_seq: bool = True,
    fixed_max_seq: int | None = None,
    footnote: str | None = None,
) -> list[str]:
    title = TASK_TITLES.get(task_name, task_name)
    metrics = HEADLINE_METRICS.get(task_name, ())
    rows = _collect_task_rows(runs, task_name)
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] != "completed"]
    if dedupe_by_model:
        completed = _dedupe_supervised_rows(completed)
    sort_by = metrics[0] if metrics else None
    prefix = "#" * heading_level

    lines = [
        f"{prefix} {title}",
        "",
        f"Runs: `{len(completed)}` models shown"
        + (f" (deduplicated from `{len(rows)}` eval runs)" if dedupe_by_model and len(rows) != len(completed) else ""),
        "",
        _comparison_table(
            completed,
            metrics,
            sort_by=sort_by,
            include_max_seq=include_max_seq,
            fixed_max_seq=fixed_max_seq,
        ).rstrip(),
    ]
    if footnote:
        lines.extend(["", f"_{footnote}_"])
    if failed:
        lines.extend(["", f"{prefix}# Failures / Skipped", ""])
        for row in failed:
            detail = row["error"] or row["status"]
            if len(detail) > 160:
                detail = detail[:157] + "..."
            lines.append(f"- `{row['model_label']}` ({row['context_mode']}): {detail}")
    lines.append("")
    return lines


NER_CONTEXT_FOOTNOTE = (
    "Each model was evaluated at both 128 tokens (common_128) and at the model's "
    "maximum sequence length (model_max). NER scores were identical across both "
    "settings, so only the model_max row is shown."
)


def _parse_retrieval_log(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    model_match = re.search(r"Evaluating model \d+/\d+: ([^\s|]+)", text)
    context_match = re.search(r"context_mode=(\S+)", text)
    max_seq_match = re.search(r"max_sequence_length=(\d+)", text)
    metrics: dict[str, float] = {}
    for dataset in ("mldr_hi", "mmarco_hindi"):
        match = re.search(
            rf"Retrieval {dataset} \| nDCG@10: ([0-9.]+) \| recall@10: ([0-9.]+) \| MRR@10: ([0-9.]+)",
            text,
        )
        if match:
            metrics[f"{dataset}.ndcg@10"] = float(match.group(1))
            metrics[f"{dataset}.recall@10"] = float(match.group(2))
            metrics[f"{dataset}.mrr@10"] = float(match.group(3))
    if not model_match or not metrics:
        return None
    return {
        "model_name_or_path": model_match.group(1),
        "context_mode": context_match.group(1) if context_match else "model_max",
        "max_sequence_length": int(max_seq_match.group(1)) if max_seq_match else "",
        "results": [
            {
                "name": "retrieval",
                "status": "completed",
                "metrics": metrics,
            }
        ],
    }


def _parse_retrieval_finetune_selection(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Selection score \(avg nDCG@10\): ([0-9.]+)",
        text,
    )
    backbone_match = re.search(r'"backbone": "([^"]+)"', text)
    if not match:
        return None
    return {
        "model_label": _model_label(backbone_match.group(1) if backbone_match else log_path.stem),
        "selection_ndcg@10": float(match.group(1)),
    }


def _load_retrieval_runs(log_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for log_path in sorted(log_dir.glob("local_train_eval_*_eval_lr*.log")):
        parsed = _parse_retrieval_log(log_path)
        if parsed is not None:
            runs.append(parsed)
    return runs


def _retrieval_finetune_section_lines(log_dir: Path, heading_level: int = 3) -> list[str]:
    prefix = "#" * heading_level
    rows = []
    for log_path in sorted(log_dir.glob("local_train_eval_*_finetune_lr*.log")):
        parsed = _parse_retrieval_finetune_selection(log_path)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda item: item["selection_ndcg@10"], reverse=True)
    best_ndcg = max(row["selection_ndcg@10"] for row in sorted_rows)
    lines = [
        f"{prefix} Retrieval Fine-tune Selection (1k mMARCO Hindi)",
        "",
        "| Model | Selection nDCG@10 |",
        "| --- | ---: |",
    ]
    for row in sorted_rows:
        value = row["selection_ndcg@10"]
        cell = _format_metric_cell(value, "selection_ndcg@10", {"selection_ndcg@10": best_ndcg})
        lines.append(f"| {row['model_label']} | {cell} |")
    lines.append("")
    return lines


def _write_per_model_retrieval_reports(runs: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    written: list[Path] = []
    for run in runs:
        model_path = run["model_name_or_path"]
        slug = Path(model_path).parts[-2] if Path(model_path).name == "final" else Path(model_path).name
        context = run.get("context_mode", "model_max")
        max_seq = run.get("max_sequence_length", "")
        report_dir = output_dir / f"{slug}__{context}_{max_seq}"
        report_dir.mkdir(parents=True, exist_ok=True)

        result = _task_result(run, "retrieval")
        metrics = result.get("metrics", {}) if result else {}
        metric_text = ", ".join(f"{k}: {_format_value(v)}" for k, v in sorted(metrics.items()))
        report = "\n".join(
            [
                "# Hindi Evaluation Suite",
                "",
                f"- Model: `{model_path}`",
                f"- Tokenizer: `{model_path}`",
                f"- Context: `{context}` (max_sequence_length={max_seq})",
                "- Seed: `17`",
                "",
                "## Results",
                "",
                "| Task | Status | Key Metrics |",
                "|---|---|---|",
                f"| retrieval | completed | {metric_text} |",
                "",
            ]
        )
        report_path = report_dir / "suite_report.md"
        report_path.write_text(report, encoding="utf-8")
        written.append(report_path)

        summary = {
            "model_name_or_path": model_path,
            "tokenizer_name_or_path": model_path,
            "context_mode": context,
            "max_sequence_length": max_seq,
            "seed": 17,
            "results": run.get("results", []),
        }
        (report_dir / "suite_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return written


def _filter_phase1_export_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if str(run.get("model_name_or_path", "")).endswith("/phase1")
    ]


def _exclude_phase1_hindi_modernbert(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop phase-1 hindi-modernBERT; that checkpoint is covered in the progression table."""
    return [
        run
        for run in runs
        if not str(run.get("model_name_or_path", "")).endswith("/phase1")
    ]


def _filter_ba1157_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if "phase2_latest_ba1157" in str(run.get("model_name_or_path", ""))
    ]


def _merge_baseline_runs(
    phase1_runs: list[dict[str, Any]],
    phase2_ba1157_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [*phase1_runs, *phase2_ba1157_runs]


def _metric_delta(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "—"
    delta = current - previous
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _hindi_modernbert_progression_lines(
    phase1_runs: list[dict[str, Any]],
    phase2_ba1157_runs: list[dict[str, Any]],
) -> list[str]:
    phase1_export_runs = _filter_phase1_export_runs(phase1_runs)
    phase1_ner = _dedupe_supervised_rows(
        [row for row in _collect_task_rows(phase1_export_runs, "ner") if row["status"] == "completed"]
    )
    phase2_ner = _dedupe_supervised_rows(
        [row for row in _collect_task_rows(phase2_ba1157_runs, "ner") if row["status"] == "completed"]
    )
    phase1_intent = _dedupe_supervised_rows(
        [
            row
            for row in _collect_task_rows(phase1_export_runs, "massive_intent")
            if row["status"] == "completed"
        ]
    )
    phase2_intent = _dedupe_supervised_rows(
        [
            row
            for row in _collect_task_rows(phase2_ba1157_runs, "massive_intent")
            if row["status"] == "completed"
        ]
    )

    if not phase1_ner or not phase2_ner:
        return []

    p1_ner = phase1_ner[0]["metrics"]
    p2_ner = phase2_ner[0]["metrics"]
    p1_intent = phase1_intent[0]["metrics"] if phase1_intent else {}
    p2_intent = phase2_intent[0]["metrics"] if phase2_intent else {}

    progression_rows = [
        {"metrics": {"f1": p1_ner.get("f1"), "macro_f1": p1_intent.get("macro_f1")}},
        {"metrics": {"f1": p2_ner.get("f1"), "macro_f1": p2_intent.get("macro_f1")}},
    ]
    progression_bests = _column_best_values(progression_rows, ("f1", "macro_f1"))

    lines = [
        "### hindi-modernBERT: Phase 1 → Phase 2",
        "",
        HINDI_MODERNBERT_PROGRESSION_NOTE,
        "",
        "| Stage | Pretraining | Max Seq | NER F1 | MASSIVE Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: |",
        (
            "| Phase 1 | Sangraha Hindi (~23.6B tokens) | 1024 | "
            f"{_format_metric_cell(p1_ner.get('f1'), 'f1', progression_bests)} | "
            f"{_format_metric_cell(p1_intent.get('macro_f1'), 'macro_f1', progression_bests)} |"
        ),
        (
            "| Phase 2 | + IndicCorp V2 Hindi (~5B tokens), 8192 context | 8192 | "
            f"{_format_metric_cell(p2_ner.get('f1'), 'f1', progression_bests)} | "
            f"{_format_metric_cell(p2_intent.get('macro_f1'), 'macro_f1', progression_bests)} |"
        ),
        (
            "| Δ (phase 2 − phase 1) | | | "
            f"{_metric_delta(p2_ner.get('f1'), p1_ner.get('f1'))} | "
            f"{_metric_delta(p2_intent.get('macro_f1'), p1_intent.get('macro_f1'))} |"
        ),
        "",
        "Phase 1 numbers come from the phase-1 HF export eval gate; phase 2 numbers come from "
        "checkpoint **ba1157** (end of the IndicCorp context-extension run).",
        "",
    ]
    return lines


def _build_combined_report(
    *,
    lang: str,
    phase1_runs: list[dict[str, Any]],
    phase2_runs: list[dict[str, Any]],
    retrieval_runs: list[dict[str, Any]],
    retrieval_log_dir: Path,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Hindi Model Evaluation — Combined Report",
        "",
        f"- Generated: `{generated}`",
        "- Seed: `17`",
        "",
        "This report aggregates baseline models, hindi-modernBERT phase-1 and phase-2 downstream evals, and DPR retrieval fine-tunes.",
        "",
        "## Table of Contents",
        "",
        "1. [Baselines](#baselines)",
        "2. [Retrieval Fine-tuning](#retrieval-fine-tuning)",
        "3. [TODO](#todo)",
        "",
    ]

    phase2_ba1157_runs = _filter_ba1157_runs(phase2_runs)
    baseline_runs = _merge_baseline_runs(phase1_runs, phase2_ba1157_runs)
    supervised_comparison_runs = _merge_baseline_runs(
        _exclude_phase1_hindi_modernbert(phase1_runs),
        phase2_ba1157_runs,
    )

    if baseline_runs:
        lines.extend(
            [
                "## Baselines",
                "",
                "Config: `configs/hi/evals/hindi_phase1.yaml` (baselines) and `configs/hi/evals/hindi_phase2.yaml` (hindi-modernBERT phase 2).",
                "",
                "Cross-model tables show **hindi-modernBERT (phase 2)** with other Hindi encoders; "
                "phase 1 (1024 context) is summarized in the progression table above.",
                "",
            ]
        )
        lines.extend(_hindi_modernbert_progression_lines(phase1_runs, phase2_ba1157_runs))
        lines.extend(
            _task_section_lines(
                task_name="ner",
                runs=supervised_comparison_runs,
                dedupe_by_model=True,
                include_max_seq=False,
                footnote=NER_CONTEXT_FOOTNOTE,
            )
        )
        lines.extend(
            _task_section_lines(
                task_name="massive_intent",
                runs=supervised_comparison_runs,
                dedupe_by_model=True,
                include_max_seq=False,
            )
        )

    if retrieval_runs or retrieval_log_dir.is_dir():
        lines.extend(
            [
                "## Retrieval Fine-tuning",
                "",
                RETRIEVAL_FINETUNE_CONFIG,
                "",
                "**hindi-modernBERT (phase 2)** is the retrieval backbone (ba1157, 8192 context). "
                + HINDI_MODERNBERT_PROGRESSION_NOTE,
                "",
            ]
        )
        lines.extend(_retrieval_finetune_section_lines(retrieval_log_dir))
        if retrieval_runs:
            lines.extend(
                _task_section_lines(
                    task_name="retrieval",
                    runs=retrieval_runs,
                    dedupe_by_model=True,
                )
            )
        lines.extend(["", COLBERT_RETRIEVAL_NOTE, ""])

    lines.extend(_todo_section_lines())
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Bold** values mark the best score in each metric column (higher is better).")
    lines.append(f"- NER: {NER_CONTEXT_FOOTNOTE}")
    lines.append(f"- hindi-modernBERT: {HINDI_MODERNBERT_PROGRESSION_NOTE}")
    lines.append(f"- Per-model raw artifacts live under `artifacts/evals/{lang}/{{phase1,phase2,retrieval}}/`.")
    lines.append(f"- Combined cross-model reports live under `artifacts/results/{lang}/`.")
    lines.append("")
    return "\n".join(lines)


def generate_reports(repo_root: Path, lang: str = DEFAULT_LANG) -> list[Path]:
    created: list[Path] = []
    eval_dir = eval_artifacts_dir(repo_root, lang)

    phase1_runs = _load_summaries(eval_dir / "phase1/multi_model_summary.json")
    phase2_runs = _load_summaries(eval_dir / "phase2/multi_model_summary.json")
    retrieval_log_dir = repo_root / "logs" / lang / "retrieval" / "finetune"
    retrieval_runs = _load_retrieval_runs(retrieval_log_dir)

    combined_path = results_dir(repo_root, lang) / "eval_summary_report.md"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(
        _build_combined_report(
            lang=lang,
            phase1_runs=phase1_runs,
            phase2_runs=phase2_runs,
            retrieval_runs=retrieval_runs,
            retrieval_log_dir=retrieval_log_dir,
        ),
        encoding="utf-8",
    )
    created.append(combined_path)

    if retrieval_runs:
        created.extend(_write_per_model_retrieval_reports(retrieval_runs, eval_dir / "retrieval"))

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (default: auto-detected)",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"Language code for input/output paths (default: {DEFAULT_LANG})",
    )
    args = parser.parse_args()
    created = generate_reports(args.repo_root.resolve(), lang=args.lang)
    print(f"Wrote {len(created)} file(s)")
    for path in created:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
