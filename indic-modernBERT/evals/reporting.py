"""Report writers for evaluation suite outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evals.config import EvalSuiteConfig


def write_reports(cfg: EvalSuiteConfig, output_dir: Path, results: list[dict[str, Any]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    summary = {
        "model_name_or_path": cfg.model.model_name_or_path,
        "tokenizer_name_or_path": cfg.model.tokenizer_source,
        "context_mode": cfg.model.context_mode,
        "max_sequence_length": cfg.model.max_sequence_length,
        "seed": cfg.seed,
        "results": results,
    }

    if cfg.reporting.write_json:
        path = output_dir / "suite_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths["json"] = str(path)

    if cfg.reporting.write_csv:
        path = output_dir / "suite_metrics.csv"
        _write_csv(path, cfg, results)
        paths["csv"] = str(path)

    if cfg.reporting.write_markdown:
        path = output_dir / "suite_report.md"
        path.write_text(_markdown_report(cfg, results), encoding="utf-8")
        paths["markdown"] = str(path)

    return paths


def _write_csv(path: Path, cfg: EvalSuiteConfig, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        if isinstance(metrics, dict) and "lengths" in metrics:
            for row in metrics["lengths"]:
                for metric in (
                    "latency_mean_s",
                    "latency_std_s",
                    "examples_per_second",
                    "tokens_per_second",
                    "tokens_per_second_per_million_params",
                    "peak_cuda_memory_allocated_mb",
                    "peak_cuda_memory_reserved_mb",
                    "avg_power_watts",
                    "max_power_watts",
                    "num_parameters_m",
                ):
                    rows.append(
                        {
                            "model": cfg.model.model_name_or_path,
                            "context_mode": cfg.model.context_mode,
                            "max_sequence_length": cfg.model.max_sequence_length,
                            "task": result["name"],
                            "status": result["status"],
                            "metric": metric,
                            "value": row.get(metric),
                            "sequence_length": row["sequence_length"],
                        }
                    )
            continue
        for metric, value in metrics.items() if isinstance(metrics, dict) else []:
            rows.append(
                {
                    "model": cfg.model.model_name_or_path,
                    "context_mode": cfg.model.context_mode,
                    "max_sequence_length": cfg.model.max_sequence_length,
                    "task": result["name"],
                    "status": result["status"],
                    "metric": metric,
                    "value": value,
                    "sequence_length": "",
                }
            )
        if not metrics:
            rows.append(
                {
                    "model": cfg.model.model_name_or_path,
                    "context_mode": cfg.model.context_mode,
                    "max_sequence_length": cfg.model.max_sequence_length,
                    "task": result["name"],
                    "status": result["status"],
                    "metric": "",
                    "value": "",
                    "sequence_length": "",
                }
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "context_mode",
                "max_sequence_length",
                "task",
                "status",
                "metric",
                "value",
                "sequence_length",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(cfg: EvalSuiteConfig, results: list[dict[str, Any]]) -> str:
    lines = [
        "# Hindi Evaluation Suite",
        "",
        f"- Model: `{cfg.model.model_name_or_path}`",
        f"- Tokenizer: `{cfg.model.tokenizer_source}`",
        f"- Context: `{cfg.model.context_mode}` (max_sequence_length={cfg.model.max_sequence_length})",
        f"- Seed: `{cfg.seed}`",
        "",
        "## Results",
        "",
        "| Task | Status | Key Metrics |",
        "|---|---|---|",
    ]
    for result in results:
        metrics = result.get("metrics", {})
        if isinstance(metrics, dict) and "lengths" in metrics:
            key_metrics = ", ".join(
                f"{row['sequence_length']} tok: {row['tokens_per_second']:.1f} tok/s"
                f" ({_format_optional(row.get('tokens_per_second_per_million_params'))} tok/s/M)"
                for row in metrics["lengths"]
            )
        elif isinstance(metrics, dict) and metrics:
            key_metrics = ", ".join(
                f"{key}: {_format_value(value)}" for key, value in sorted(metrics.items())
            )
        else:
            key_metrics = result.get("error", "")
        lines.append(f"| {result['name']} | {result['status']} | {key_metrics} |")
    lines.append("")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return _format_value(value)
