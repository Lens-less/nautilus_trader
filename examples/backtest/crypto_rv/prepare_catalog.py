#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


try:
    from .common import ensure_directory
    from .common import parse_utc_timestamp
    from .common import resolve_path
    from .schemas import SCHEMA_VERSION
    from .schemas import PreparedCatalogArtifact
    from .schemas import classify_funding_coverage
    from .schemas import load_history_manifest
    from .schemas import load_research_config
    from .schemas import load_snapshot
    from .schemas import save_artifact
except ImportError:  # pragma: no cover - script execution fallback
    from common import ensure_directory
    from common import parse_utc_timestamp
    from common import resolve_path
    from schemas import SCHEMA_VERSION
    from schemas import PreparedCatalogArtifact
    from schemas import classify_funding_coverage
    from schemas import load_history_manifest
    from schemas import load_research_config
    from schemas import load_snapshot
    from schemas import save_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and stage normalized historical inputs for crypto RV research.",
    )
    parser.add_argument("--config", required=True, help="Path to the research config JSON.")
    parser.add_argument("--snapshot", help="Optional override for the frozen universe snapshot.")
    parser.add_argument("--manifest", help="Optional override for the history manifest CSV/JSON.")
    parser.add_argument(
        "--output",
        help="Optional override for the prepared catalog artifact JSON path.",
    )
    parser.add_argument(
        "--catalog-path",
        help="Optional existing Nautilus catalog path. If present and exists, catalog_ready=true.",
    )
    parser.add_argument(
        "--stage-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to stage external inputs into the local artifact root.",
    )
    return parser


def classify_funding(
    funding_ratio: float,
    warning_threshold: float,
    partial_threshold: float,
) -> str:
    return classify_funding_coverage(
        funding_ratio,
        partial_threshold=partial_threshold,
        warning_threshold=warning_threshold,
    )


def stage_file(
    source: Path,
    destination: Path,
    stage_mode: str,
) -> str:
    ensure_directory(destination.parent)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if stage_mode == "none":
        return "not_staged"
    if stage_mode == "copy":
        shutil.copy2(source, destination)
        return "copied"

    destination.symlink_to(source)
    return "symlinked"


def resolve_required_symbols(config, snapshot) -> set[str]:
    required_symbols = {member.raw_symbol for member in config.long_leg}
    ranked_eligible = snapshot.ranked_eligible_candidates()
    max_variant_shorts = min(config.max_short_count, len(ranked_eligible))
    required_shorts = ranked_eligible[:max_variant_shorts] or snapshot.selected_candidates()
    required_symbols.update(candidate.raw_symbol for candidate in required_shorts)
    return required_symbols


def validate_and_stage_entries(
    *,
    required_symbols: set[str],
    manifest,
    manifest_base_dir: Path,
    snapshot,
    config,
    staged_root: Path,
    stage_mode: str,
) -> tuple[dict[str, object], list[dict[str, str]], list[str]]:
    manifest_by_symbol = {entry.raw_symbol: entry for entry in manifest}
    missing_symbols = sorted(required_symbols - set(manifest_by_symbol))
    if missing_symbols:
        raise ValueError(f"History manifest is missing symbols: {', '.join(missing_symbols)}")

    staged_files: list[dict[str, str]] = []
    validated_symbols: list[str] = []
    window_start = parse_utc_timestamp(snapshot.test_window_start)
    window_end = parse_utc_timestamp(snapshot.test_window_end)

    for symbol in sorted(required_symbols):
        entry = manifest_by_symbol[symbol]
        entry.validate()

        price_path = resolve_path(entry.price_path, manifest_base_dir)
        if price_path is None or not price_path.exists():
            raise FileNotFoundError(f"{symbol} is missing price input: {entry.price_path}")

        funding_path = resolve_path(entry.funding_path, manifest_base_dir)
        if funding_path is not None and not funding_path.exists():
            raise FileNotFoundError(f"{symbol} is missing funding input: {entry.funding_path}")

        if parse_utc_timestamp(entry.start_ts) > window_start:
            raise ValueError(f"{symbol} history starts after the test window start")
        if parse_utc_timestamp(entry.end_ts) < window_end:
            raise ValueError(f"{symbol} history ends before the test window end")
        if entry.price_coverage_ratio < config.thresholds.min_price_coverage_ratio:
            raise ValueError(f"{symbol} price coverage ratio is below configured floor")

        validated_symbols.append(symbol)
        staged_files.extend(stage_symbol_files(symbol, price_path, funding_path, staged_root, stage_mode))

    return manifest_by_symbol, staged_files, validated_symbols


def stage_symbol_files(
    symbol: str,
    price_path: Path,
    funding_path: Path | None,
    staged_root: Path,
    stage_mode: str,
) -> list[dict[str, str]]:
    staged_price = staged_root / f"{symbol}_price{price_path.suffix}"
    price_stage_status = stage_file(price_path, staged_price, stage_mode)
    staged_files = [
        {
            "symbol": symbol,
            "kind": "price",
            "source": str(price_path),
            "staged_path": str(staged_price),
            "stage_status": price_stage_status,
        },
    ]

    if funding_path is not None:
        staged_funding = staged_root / f"{symbol}_funding{funding_path.suffix}"
        funding_stage_status = stage_file(funding_path, staged_funding, stage_mode)
        staged_files.append(
            {
                "symbol": symbol,
                "kind": "funding",
                "source": str(funding_path),
                "staged_path": str(staged_funding),
                "stage_status": funding_stage_status,
            },
        )

    return staged_files


def resolve_catalog_path(args_catalog_path: str | None, manifest, manifest_base_dir: Path) -> Path | None:
    if args_catalog_path:
        return Path(args_catalog_path).resolve()

    candidate_catalog_paths = {
        resolve_path(entry.catalog_path, manifest_base_dir)
        for entry in manifest
        if entry.catalog_path is not None
    }
    candidate_catalog_paths.discard(None)
    if len(candidate_catalog_paths) == 1:
        return candidate_catalog_paths.pop()
    return None


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_research_config(config_path)
    paths = config.resolved_paths(config_path)

    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else paths["snapshot_output_path"]
    manifest_path = Path(args.manifest).resolve() if args.manifest else paths["history_manifest_path"]
    output_path = Path(args.output).resolve() if args.output else paths["prepared_catalog_output_path"]

    snapshot = load_snapshot(snapshot_path)
    manifest = load_history_manifest(manifest_path)
    manifest_base_dir = manifest_path.parent
    artifact_root = output_path.parent / output_path.stem
    staged_root = ensure_directory(artifact_root / "staged_inputs")
    required_symbols = resolve_required_symbols(config, snapshot)

    manifest_by_symbol, staged_files, validated_symbols = validate_and_stage_entries(
        required_symbols=required_symbols,
        manifest=manifest,
        manifest_base_dir=manifest_base_dir,
        snapshot=snapshot,
        config=config,
        staged_root=staged_root,
        stage_mode=args.stage_mode,
    )

    primary_selected = snapshot.selected_candidates()
    manifest_funding_ratio = sum(
        manifest_by_symbol[candidate.raw_symbol].funding_coverage_ratio for candidate in primary_selected
    ) / len(primary_selected)
    classification = classify_funding(
        manifest_funding_ratio,
        config.thresholds.funding_warning_threshold,
        config.thresholds.funding_partial_evidence_threshold,
    )

    catalog_path = resolve_catalog_path(args.catalog_path, manifest, manifest_base_dir)
    catalog_ready = bool(catalog_path and catalog_path.exists())
    notes = (
        ["Catalog path exists and can be used for BacktestNode wiring."]
        if catalog_ready
        else [
            "No verified Nautilus catalog path was supplied; run_backtest.py will remain in "
            "plan-only mode while still emitting BacktestNode-compatible configs.",
        ]
    )

    artifact = PreparedCatalogArtifact(
        schema_version=SCHEMA_VERSION,
        research_name=config.research_name,
        snapshot_path=str(snapshot_path),
        history_manifest_path=str(manifest_path),
        artifact_root=str(artifact_root),
        catalog_path=str(catalog_path) if catalog_path else None,
        catalog_ready=catalog_ready,
        mode="nautilus-ready" if catalog_ready else "externalized-bundle",
        selected_short_funding_coverage_ratio=manifest_funding_ratio,
        classification=classification,
        notes=notes,
        validated_symbols=validated_symbols,
        staged_files=staged_files,
        manifest_entries=[manifest_by_symbol[symbol] for symbol in sorted(required_symbols)],
    )
    save_artifact(output_path, artifact.to_dict())

    print(
        f"Wrote prepared catalog artifact to {output_path}. "
        f"Validated {len(validated_symbols)} symbols; catalog_ready={artifact.catalog_ready}.",
    )
    print(
        "Selected short funding coverage ratio "
        f"{artifact.selected_short_funding_coverage_ratio:.2%} -> {artifact.classification}",
    )


if __name__ == "__main__":
    main()
