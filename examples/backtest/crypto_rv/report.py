#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


try:
    from .schemas import load_prepared_catalog
    from .schemas import load_research_config
    from .schemas import load_snapshot
    from .schemas import save_artifact
except ImportError:  # pragma: no cover - script execution fallback
    from schemas import load_prepared_catalog
    from schemas import load_research_config
    from schemas import load_snapshot
    from schemas import save_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile crypto RV research summaries from run_backtest outputs.",
    )
    parser.add_argument("--config", required=True, help="Path to the research config JSON.")
    parser.add_argument("--snapshot", help="Optional override for the frozen universe snapshot.")
    parser.add_argument("--prepared-catalog", help="Optional override for the prepared catalog JSON.")
    parser.add_argument("--run-output-dir", help="Optional override for the run output directory.")
    parser.add_argument("--output", help="Optional override for the report JSON path.")
    return parser


def markdown_table(rows: list[dict[str, str]]) -> str:
    header = "| Variant | Status | Classification | Notes |"
    separator = "| --- | --- | --- | --- |"
    body = [
        f"| {row['variant']} | {row['status']} | {row['classification']} | {row['notes']} |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_research_config(config_path)
    paths = config.resolved_paths(config_path)

    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else paths["snapshot_output_path"]
    prepared_catalog_path = (
        Path(args.prepared_catalog).resolve()
        if args.prepared_catalog
        else paths["prepared_catalog_output_path"]
    )
    run_output_dir = Path(args.run_output_dir).resolve() if args.run_output_dir else paths["run_output_dir"]
    output_path = Path(args.output).resolve() if args.output else paths["report_output_path"]

    snapshot = load_snapshot(snapshot_path)
    prepared_catalog = load_prepared_catalog(prepared_catalog_path)
    index_path = run_output_dir / "index.json"
    index_payload = index_path.read_text(encoding="utf-8")

    import json

    index = json.loads(index_payload)
    variant_rows: list[dict[str, str]] = []
    variant_payloads: list[dict[str, object]] = []
    for item in index["variants"]:
        result_path = Path(item["result_path"])
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        variant_payloads.append(payload)
        variant_rows.append(
            {
                "variant": payload["variant_name"],
                "status": payload["status"],
                "classification": payload["classification"],
                "notes": "; ".join(payload["reasons"]) or "none",
            },
        )

    primary_variant = next(payload for payload in variant_payloads if payload["variant_name"] == "primary")
    lane_manifest_paths: list[str] = []
    for item, payload in zip(index["variants"], variant_payloads, strict=False):
        lane_manifest_path = item.get("lane_manifest_path") or payload.get("lane_manifest_path")
        if lane_manifest_path is not None:
            lane_manifest_paths.append(str(lane_manifest_path))
    report_payload = {
        "research_name": config.research_name,
        "snapshot_path": str(snapshot_path),
        "prepared_catalog_path": str(prepared_catalog_path),
        "run_output_dir": str(run_output_dir),
        "overall_status": (
            "planning-only"
            if primary_variant["status"] != "completed"
            else primary_variant["classification"]
        ),
        "primary_variant": primary_variant,
        "snapshot_summary": {
            "selected_short_count": snapshot.selected_short_count,
            "selected_short_funding_coverage_ratio": snapshot.selected_short_funding_coverage_ratio,
            "funding_coverage_classification": snapshot.funding_coverage_classification,
        },
        "prepared_catalog_summary": {
            "catalog_ready": prepared_catalog.catalog_ready,
            "mode": prepared_catalog.mode,
            "classification": prepared_catalog.classification,
        },
        "artifact_contract_summary": {
            "lane_manifests": lane_manifest_paths,
            "required_variant_artifacts": [
                "lane_manifest.json",
                "feature_panel_manifest.json",
                "feature_panel.parquet",
                "signal_panel.parquet",
                "signal_metrics.json",
                "portfolio_metrics.json",
                "portfolio_timeseries.parquet",
                "summary_report.md",
            ],
        },
        "variant_rows": variant_rows,
        "evidence_vs_inference": {
            "evidence": [
                "Frozen point-in-time universe snapshot",
                "Prepared catalog artifact with staged inputs",
                "BacktestNode-compatible run request JSONs",
                "Per-variant lane manifests and downstream machine-artifact placeholders",
            ],
            "inference": [
                "PnL, drawdown, fee, slippage, and funding decomposition remain placeholders until a "
                "strategy module and verified Nautilus catalog are connected.",
            ],
        },
        "required_metrics_placeholder": [
            "total_return",
            "max_drawdown",
            "long_basket_return",
            "short_basket_return",
            "fees",
            "slippage",
            "funding",
            "invalidation_review",
        ],
    }
    save_artifact(output_path, report_payload)

    markdown_path = output_path.with_suffix(".md")
    markdown = "\n".join(
        [
            f"# {config.research_name}",
            "",
            "## Status",
            "",
            f"- Overall status: `{report_payload['overall_status']}`",
            f"- Snapshot classification: `{snapshot.funding_coverage_classification}`",
            f"- Prepared catalog mode: `{prepared_catalog.mode}`",
            "",
            "## Variant Matrix",
            "",
            markdown_table(variant_rows),
            "",
            "## Evidence vs Inference",
            "",
            "- Evidence: frozen snapshot, prepared catalog artifact, run requests, and per-variant lane manifests.",
            "- Inference: realized PnL and decomposition still require the strategy module and verified "
            "catalog wiring outside this write scope.",
        ],
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")

    print(f"Wrote report JSON to {output_path}")
    print(f"Wrote report Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
