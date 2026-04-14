#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any


try:
    from .common import format_utc_timestamp
    from .common import parse_utc_timestamp
    from .common import stable_payload_hash
    from .common import write_parquet_records
    from .schemas import ARTIFACT_SCHEMA_VERSION
    from .schemas import FeaturePanelManifest
    from .schemas import LaneArtifactManifest
    from .schemas import LongLegMember
    from .schemas import ResearchConfig
    from .schemas import classify_funding_coverage
    from .schemas import load_prepared_catalog
    from .schemas import load_research_config
    from .schemas import load_snapshot
    from .schemas import renormalize_long_weights
    from .schemas import save_artifact
    from .schemas import snapshot_identity
except ImportError:  # pragma: no cover - script execution fallback
    from common import format_utc_timestamp
    from common import parse_utc_timestamp
    from common import stable_payload_hash
    from common import write_parquet_records
    from schemas import ARTIFACT_SCHEMA_VERSION
    from schemas import FeaturePanelManifest
    from schemas import LaneArtifactManifest
    from schemas import LongLegMember
    from schemas import ResearchConfig
    from schemas import classify_funding_coverage
    from schemas import load_prepared_catalog
    from schemas import load_research_config
    from schemas import load_snapshot
    from schemas import renormalize_long_weights
    from schemas import save_artifact
    from schemas import snapshot_identity

from nautilus_trader.backtest.config import BacktestDataConfig
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.config import BacktestRunConfig
from nautilus_trader.backtest.config import BacktestVenueConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig


@dataclass(slots=True)
class VariantSpec:
    name: str
    cost_model_name: str
    rebalance_cadence: str
    include_hype: bool
    short_count: int
    long_leg: list[LongLegMember]
    short_leg: list[dict[str, Any]]
    funding_coverage_ratio: float
    classification: str
    reasons: list[str]


FEATURE_COLUMNS = [
    "ts",
    "instrument_id",
    "raw_symbol",
    "leg",
    "target_weight",
    "funding_coverage_ratio",
    "frozen_rank",
    "selected",
    "eligible",
    "placeholder",
]

SIGNAL_COLUMNS = [
    "ts",
    "instrument_id",
    "raw_symbol",
    "leg",
    "score",
    "rank",
    "selected",
    "eligible",
    "target_weight",
    "placeholder",
]

PORTFOLIO_TIMESERIES_COLUMNS = [
    "ts",
    "gross_return",
    "net_return",
    "long_return",
    "short_return",
    "turnover",
    "gross_exposure",
    "net_exposure",
    "active_names",
    "placeholder",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit BacktestNode-compatible crypto RV run configs. "
            "Execution stays plan-only unless a real strategy module and catalog are present."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the research config JSON.")
    parser.add_argument("--snapshot", help="Optional override for the frozen universe snapshot.")
    parser.add_argument("--prepared-catalog", help="Optional override for the prepared catalog JSON.")
    parser.add_argument("--output-dir", help="Optional override for the run output directory.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Attempt a real BacktestNode run when strategy import and catalog are both available.",
    )
    return parser


def renormalize_long_leg(
    long_leg: list[LongLegMember],
    include_hype: bool,
) -> list[LongLegMember]:
    if include_hype:
        return [LongLegMember(**asdict(member)) for member in long_leg]

    filtered = [member for member in long_leg if member.asset != "HYPE"]
    normalized_weights = renormalize_long_weights(
        {member.asset: member.weight for member in long_leg},
        excluded_symbol="HYPE",
    )
    return [
        LongLegMember(
            asset=member.asset,
            raw_symbol=member.raw_symbol,
            instrument_id=member.instrument_id,
            weight=normalized_weights[member.asset],
        )
        for member in filtered
    ]


def classify_variant(
    funding_ratio: float,
    config: ResearchConfig,
) -> str:
    return classify_funding_coverage(
        funding_ratio,
        partial_threshold=config.thresholds.funding_partial_evidence_threshold,
        warning_threshold=config.thresholds.funding_warning_threshold,
    )


def build_bar_type(instrument_id: str, bar_spec: str) -> str:
    normalized_spec = bar_spec.strip()
    if normalized_spec.endswith(("-EXTERNAL", "-INTERNAL")):
        return f"{instrument_id}-{normalized_spec}"
    return f"{instrument_id}-{normalized_spec}-EXTERNAL"


def select_ranked_eligible(snapshot) -> list[dict[str, object]]:
    eligible = snapshot.ranked_eligible_candidates()
    return eligible or snapshot.selected_candidates()


def take_shorts(eligible, count: int) -> list[dict[str, Any]]:
    selected = eligible[:count]
    if not selected:
        return []
    weight = 1.0 / len(selected)
    return [
        {
            "raw_symbol": candidate.raw_symbol,
            "instrument_id": candidate.instrument_id,
            "weight": weight,
            "funding_coverage_ratio": candidate.funding_coverage_ratio,
        }
        for candidate in selected
    ]


def funding_ratio_for(shorts: list[dict[str, Any]]) -> float:
    if not shorts:
        return 0.0
    return sum(item["funding_coverage_ratio"] for item in shorts) / len(shorts)


def make_variant(
    *,
    name: str,
    config: ResearchConfig,
    cost_model_name: str,
    rebalance_cadence: str,
    include_hype: bool,
    short_count: int,
    eligible,
    reasons: list[str] | None = None,
) -> VariantSpec:
    short_leg = take_shorts(eligible, short_count)
    funding_ratio = funding_ratio_for(short_leg)
    return VariantSpec(
        name=name,
        cost_model_name=cost_model_name,
        rebalance_cadence=rebalance_cadence,
        include_hype=include_hype,
        short_count=len(short_leg),
        long_leg=renormalize_long_leg(config.long_leg, include_hype=include_hype),
        short_leg=short_leg,
        funding_coverage_ratio=funding_ratio,
        classification=classify_variant(funding_ratio, config),
        reasons=reasons or [],
    )


def build_variants(config: ResearchConfig, snapshot) -> list[VariantSpec]:
    eligible = select_ranked_eligible(snapshot)
    variants: list[VariantSpec] = []
    variants.append(
        make_variant(
            name="primary",
            config=config,
            cost_model_name="base",
            rebalance_cadence=config.rebalance_cadence,
            include_hype=True,
            short_count=config.primary_short_count,
            eligible=eligible,
        ),
    )
    variants.append(
        make_variant(
            name="primary-stress",
            config=config,
            cost_model_name="stress",
            rebalance_cadence=config.rebalance_cadence,
            include_hype=True,
            short_count=config.primary_short_count,
            eligible=eligible,
        ),
    )

    if config.sensitivity.enable_hype_out:
        variants.append(
            make_variant(
                name="no-hype",
                config=config,
                cost_model_name="base",
                rebalance_cadence=config.rebalance_cadence,
                include_hype=False,
                short_count=config.primary_short_count,
                eligible=eligible,
                reasons=["HYPE removed and remaining long weights re-normalized to preserve gross."],
            ),
        )

    for count in config.sensitivity.short_counts:
        reasons: list[str] = []
        if len(eligible) < count:
            reasons.append(
                f"Snapshot only has {len(eligible)} eligible shorts, below requested {count}.",
            )
        variants.append(
            make_variant(
                name=f"short-{count}",
                config=config,
                cost_model_name="base",
                rebalance_cadence=config.rebalance_cadence,
                include_hype=True,
                short_count=count,
                eligible=eligible,
                reasons=reasons,
            ),
        )

    for cadence in config.sensitivity.rebalance_cadences:
        if cadence == config.rebalance_cadence:
            continue
        variants.append(
            make_variant(
                name=cadence,
                config=config,
                cost_model_name="base",
                rebalance_cadence=cadence,
                include_hype=True,
                short_count=config.primary_short_count,
                eligible=eligible,
            ),
        )

    return variants


def strategy_path_exists(strategy_path: str) -> bool:
    module_name, _, object_name = strategy_path.partition(":")
    module = importlib.import_module(module_name)
    return hasattr(module, object_name)


def config_path_exists(config_path: str) -> bool:
    module_name, _, object_name = config_path.partition(":")
    module = importlib.import_module(module_name)
    return hasattr(module, object_name)


def build_backtest_request(
    *,
    config: ResearchConfig,
    prepared_catalog,
    variant: VariantSpec,
) -> tuple[BacktestRunConfig, dict[str, Any]]:
    manifest_by_instrument = {
        entry.instrument_id: entry for entry in prepared_catalog.manifest_entries
    }
    required_instrument_ids = [member.instrument_id for member in variant.long_leg]
    required_instrument_ids.extend(item["instrument_id"] for item in variant.short_leg)
    strategy_payload = {
        "long_instrument_ids": [member.instrument_id for member in variant.long_leg],
        "short_instrument_ids": [item["instrument_id"] for item in variant.short_leg],
        "bar_types": [
            build_bar_type(
                instrument_id,
                manifest_by_instrument[instrument_id].bar_spec,
            )
            for instrument_id in required_instrument_ids
        ],
        "leg_notional_usd": 500_000.0,
        "rebalance_cadence": variant.rebalance_cadence,
        "excluded_instrument_ids": [],
        "min_order_notional_usd": 25.0,
        "long_fee_bps": config.cost_models[variant.cost_model_name].fee_bps,
        "short_fee_bps": config.cost_models[variant.cost_model_name].fee_bps,
        "long_slippage_bps": config.cost_models[variant.cost_model_name].slippage_bps,
        "short_slippage_bps": config.cost_models[variant.cost_model_name].slippage_bps,
        "long_funding_bps_per_day": 0.0,
        "short_funding_bps_per_day": 0.0,
        "short_borrow_bps_per_day": (
            config.cost_models[variant.cost_model_name].short_borrow_bps_annual / 365.0
        ),
    }

    strategy_config = ImportableStrategyConfig(
        strategy_path=config.strategy.strategy_path,
        config_path=config.strategy.config_path,
        config=strategy_payload,
    )
    engine_config = BacktestEngineConfig(
        strategies=[strategy_config],
        logging=LoggingConfig(log_level="INFO"),
        run_analysis=True,
    )
    venue_config = BacktestVenueConfig(
        name=config.venue,
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency=config.base_currency,
        starting_balances=[f"1000000 {config.base_currency}"],
    )

    data_configs = [
        BacktestDataConfig(
            catalog_path=prepared_catalog.catalog_path or prepared_catalog.artifact_root,
            data_cls=manifest_by_instrument[instrument_id].data_cls,
            instrument_id=instrument_id,
            start_time=config.test_window_start,
            end_time=config.test_window_end,
            bar_spec=manifest_by_instrument[instrument_id].bar_spec,
        )
        for instrument_id in required_instrument_ids
    ]
    run_config = BacktestRunConfig(
        engine=engine_config,
        venues=[venue_config],
        data=data_configs,
        start=config.test_window_start,
        end=config.test_window_end,
    )

    request_payload = {
        "variant_name": variant.name,
        "classification": variant.classification,
        "cost_model_name": variant.cost_model_name,
        "rebalance_cadence": variant.rebalance_cadence,
        "include_hype": variant.include_hype,
        "short_count": variant.short_count,
        "funding_coverage_ratio": variant.funding_coverage_ratio,
        "reasons": variant.reasons,
        "engine": {
            "trader_id": str(engine_config.trader_id),
            "run_analysis": engine_config.run_analysis,
            "strategies": [
                {
                    "strategy_path": config.strategy.strategy_path,
                    "config_path": config.strategy.config_path,
                    "config": strategy_payload,
                },
            ],
        },
        "venues": [
            {
                "name": venue_config.name,
                "oms_type": venue_config.oms_type,
                "account_type": venue_config.account_type,
                "base_currency": venue_config.base_currency,
                "starting_balances": venue_config.starting_balances,
            },
        ],
        "data": [
            {
                "catalog_path": data_config.catalog_path,
                "data_cls": data_config.data_cls,
                "instrument_id": str(data_config.instrument_id),
                "start_time": data_config.start_time,
                "end_time": data_config.end_time,
                "bar_spec": data_config.bar_spec,
            }
            for data_config in data_configs
        ],
    }
    return run_config, request_payload


def build_feature_panel_rows(
    *,
    config: ResearchConfig,
    snapshot,
    variant: VariantSpec,
) -> list[dict[str, Any]]:
    ts = config.formation_ts
    candidate_by_instrument = {candidate.instrument_id: candidate for candidate in snapshot.candidates}
    rows: list[dict[str, Any]] = []

    for member in variant.long_leg:
        rows.append(
            {
                "ts": ts,
                "instrument_id": member.instrument_id,
                "raw_symbol": member.raw_symbol,
                "leg": "long",
                "target_weight": member.weight,
                "funding_coverage_ratio": 1.0,
                "frozen_rank": None,
                "selected": True,
                "eligible": True,
                "placeholder": True,
            },
        )

    for item in variant.short_leg:
        candidate = candidate_by_instrument[item["instrument_id"]]
        rows.append(
            {
                "ts": ts,
                "instrument_id": item["instrument_id"],
                "raw_symbol": item["raw_symbol"],
                "leg": "short",
                "target_weight": -float(item["weight"]),
                "funding_coverage_ratio": float(item["funding_coverage_ratio"]),
                "frozen_rank": candidate.frozen_rank,
                "selected": candidate.selected,
                "eligible": candidate.eligible,
                "placeholder": True,
            },
        )

    return rows


def build_signal_panel_rows(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    long_rank = 0
    short_rank = 0
    for row in feature_rows:
        if row["leg"] == "long":
            long_rank += 1
            rank = long_rank
            score = float(row["target_weight"])
        else:
            short_rank += 1
            rank = short_rank
            score = -float(row["funding_coverage_ratio"])
        rows.append(
            {
                "ts": row["ts"],
                "instrument_id": row["instrument_id"],
                "raw_symbol": row["raw_symbol"],
                "leg": row["leg"],
                "score": score,
                "rank": rank,
                "selected": row["selected"],
                "eligible": row["eligible"],
                "target_weight": row["target_weight"],
                "placeholder": True,
            },
        )
    return rows


def build_portfolio_timeseries_rows(
    *,
    config: ResearchConfig,
    variant: VariantSpec,
) -> list[dict[str, Any]]:
    start = parse_utc_timestamp(config.test_window_start)
    end = parse_utc_timestamp(config.test_window_end)
    total_names = len(variant.long_leg) + len(variant.short_leg)
    rows: list[dict[str, Any]] = []
    current = start
    while current <= end:
        rows.append(
            {
                "ts": format_utc_timestamp(current),
                "gross_return": 0.0,
                "net_return": 0.0,
                "long_return": 0.0,
                "short_return": 0.0,
                "turnover": 0.0,
                "gross_exposure": 1.0,
                "net_exposure": 0.0,
                "active_names": total_names,
                "placeholder": True,
            },
        )
        current += timedelta(days=1)
    return rows


def write_downstream_artifacts(
    *,
    config: ResearchConfig,
    snapshot,
    prepared_catalog,
    snapshot_path: Path,
    prepared_catalog_path: Path,
    variant: VariantSpec,
    variant_dir: Path,
    request_path: Path,
    result_path: Path,
    status: str,
) -> dict[str, str]:
    feature_rows = build_feature_panel_rows(config=config, snapshot=snapshot, variant=variant)
    signal_rows = build_signal_panel_rows(feature_rows)
    portfolio_timeseries_rows = build_portfolio_timeseries_rows(config=config, variant=variant)

    feature_panel_path = variant_dir / "feature_panel.parquet"
    signal_panel_path = variant_dir / "signal_panel.parquet"
    portfolio_timeseries_path = variant_dir / "portfolio_timeseries.parquet"
    feature_panel_manifest_path = variant_dir / "feature_panel_manifest.json"
    signal_metrics_path = variant_dir / "signal_metrics.json"
    portfolio_metrics_path = variant_dir / "portfolio_metrics.json"
    summary_report_path = variant_dir / "summary_report.md"
    lane_manifest_path = variant_dir / "lane_manifest.json"

    write_parquet_records(feature_panel_path, feature_rows)
    write_parquet_records(signal_panel_path, signal_rows)
    write_parquet_records(portfolio_timeseries_path, portfolio_timeseries_rows)

    feature_manifest = FeaturePanelManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        variant_name=variant.name,
        placeholder=True,
        feature_schema_version=ARTIFACT_SCHEMA_VERSION,
        feature_columns=FEATURE_COLUMNS,
        date_range={"start": config.formation_ts, "end": config.formation_ts},
        instrument_count=len(feature_rows),
        row_count=len(feature_rows),
        artifact_path=str(feature_panel_path),
        feature_panel_path=str(feature_panel_path),
        signal_artifact_path=str(signal_panel_path),
        signal_panel_path=str(signal_panel_path),
    )
    save_artifact(feature_panel_manifest_path, feature_manifest.to_dict())

    snapshot_id = snapshot_identity(snapshot)
    config_hash = stable_payload_hash(config.to_dict())[:12]
    signal_metrics_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "variant_name": variant.name,
        "snapshot_id": snapshot_id,
        "config_hash": config_hash,
        "mode": "plan-only",
        "placeholder": True,
        "row_count": len(signal_rows),
        "long_count": len([row for row in signal_rows if row["leg"] == "long"]),
        "short_count": len([row for row in signal_rows if row["leg"] == "short"]),
        "feature_columns": FEATURE_COLUMNS,
        "signal_columns": SIGNAL_COLUMNS,
    }
    save_artifact(signal_metrics_path, signal_metrics_payload)

    portfolio_metrics_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "variant_name": variant.name,
        "snapshot_id": snapshot_id,
        "config_hash": config_hash,
        "mode": "plan-only",
        "placeholder": True,
        "classification": variant.classification,
        "funding_coverage_ratio": variant.funding_coverage_ratio,
        "active_names": len(variant.long_leg) + len(variant.short_leg),
        "timeseries_columns": PORTFOLIO_TIMESERIES_COLUMNS,
        "notes": [
            "Kernel contract placeholder output.",
            "Realized portfolio metrics are deferred until a verified catalog and lane-specific signal logic are connected.",
        ],
    }
    save_artifact(portfolio_metrics_path, portfolio_metrics_payload)

    summary_lines = [
        f"# {variant.name}",
        "",
        "## Contract Status",
        "",
        f"- Mode: `plan-only`",
        f"- Status: `{status}`",
        f"- Classification: `{variant.classification}`",
        f"- Snapshot ID: `{snapshot_id}`",
        f"- Config hash: `{config_hash}`",
        "",
        "## Machine Artifacts",
        "",
        f"- Feature panel: `{feature_panel_path.name}`",
        f"- Signal panel: `{signal_panel_path.name}`",
        f"- Portfolio timeseries: `{portfolio_timeseries_path.name}`",
        "",
        "## Notes",
        "",
        "- This is a kernel placeholder artifact set intended to freeze downstream schemas.",
        "- Realized signal and portfolio values will be produced by later alpha lanes.",
    ]
    summary_report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    lane_manifest = LaneArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        research_name=config.research_name,
        variant_name=variant.name,
        mode="plan-only",
        status=status,
        placeholder=True,
        snapshot_id=snapshot_id,
        config_hash=config_hash,
        snapshot_path=str(snapshot_path),
        prepared_catalog_path=str(prepared_catalog_path),
        run_request_path=str(request_path),
        result_path=str(result_path),
        feature_panel_manifest_path=str(feature_panel_manifest_path),
        feature_panel_path=str(feature_panel_path),
        signal_panel_path=str(signal_panel_path),
        signal_metrics_path=str(signal_metrics_path),
        portfolio_metrics_path=str(portfolio_metrics_path),
        portfolio_timeseries_path=str(portfolio_timeseries_path),
        summary_report_path=str(summary_report_path),
    )
    lane_manifest.validate()
    save_artifact(lane_manifest_path, lane_manifest.to_dict())
    return {
        "lane_manifest_path": str(lane_manifest_path),
        "feature_panel_manifest_path": str(feature_panel_manifest_path),
        "feature_panel_path": str(feature_panel_path),
        "signal_panel_path": str(signal_panel_path),
        "signal_metrics_path": str(signal_metrics_path),
        "portfolio_metrics_path": str(portfolio_metrics_path),
        "portfolio_timeseries_path": str(portfolio_timeseries_path),
        "summary_report_path": str(summary_report_path),
        "snapshot_id": snapshot_id,
        "config_hash": config_hash,
    }


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
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["run_output_dir"]

    snapshot = load_snapshot(snapshot_path)
    prepared_catalog = load_prepared_catalog(prepared_catalog_path)
    variants = build_variants(config, snapshot)

    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_import_ok = False
    config_import_ok = False
    import_errors: list[str] = []
    try:
        strategy_import_ok = strategy_path_exists(config.strategy.strategy_path)
        config_import_ok = config_path_exists(config.strategy.config_path)
    except Exception as exc:  # pragma: no cover - bounded fallback
        import_errors.append(str(exc))

    index_payload = {
        "research_name": config.research_name,
        "snapshot_path": str(snapshot_path),
        "prepared_catalog_path": str(prepared_catalog_path),
        "variants": [],
    }

    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        run_config, request_payload = build_backtest_request(
            config=config,
            prepared_catalog=prepared_catalog,
            variant=variant,
        )
        request_path = variant_dir / "run_request.json"
        save_artifact(request_path, request_payload)

        status = "planned"
        status_reasons = list(variant.reasons)
        result_payload: dict[str, Any] = {
            "variant_name": variant.name,
            "classification": variant.classification,
            "status": status,
            "funding_coverage_ratio": variant.funding_coverage_ratio,
            "cost_model_name": variant.cost_model_name,
            "rebalance_cadence": variant.rebalance_cadence,
            "include_hype": variant.include_hype,
            "short_count": variant.short_count,
            "reasons": status_reasons,
            "run_request_path": str(request_path),
        }

        if args.execute:
            if not prepared_catalog.catalog_ready:
                status = "blocked"
                status_reasons.append("catalog_ready=false")
            elif not (strategy_import_ok and config_import_ok):
                status = "blocked"
                status_reasons.append("strategy import path is not yet available")
                status_reasons.extend(import_errors)
            else:
                node = BacktestNode(configs=[run_config])
                [result] = node.run()
                status = "completed"
                result_payload["raw_result_repr"] = repr(result)

        if not args.execute and not prepared_catalog.catalog_ready:
            status_reasons.append("plan-only: prepared catalog is not a verified Nautilus catalog")
        if not args.execute and not (strategy_import_ok and config_import_ok):
            status_reasons.append(
                "plan-only: strategy module/config path is outside this write scope and not yet importable",
            )

        result_payload["status"] = status
        result_payload["reasons"] = status_reasons
        result_payload["strategy_import_ok"] = strategy_import_ok
        result_payload["config_import_ok"] = config_import_ok
        result_path = variant_dir / "result.json"
        save_artifact(result_path, result_payload)

        artifact_paths = write_downstream_artifacts(
            config=config,
            snapshot=snapshot,
            prepared_catalog=prepared_catalog,
            snapshot_path=snapshot_path,
            prepared_catalog_path=prepared_catalog_path,
            variant=variant,
            variant_dir=variant_dir,
            request_path=request_path,
            result_path=result_path,
            status=status,
        )
        result_payload["artifact_paths"] = artifact_paths
        save_artifact(result_path, result_payload)

        index_payload["variants"].append(
            {
                "variant_name": variant.name,
                "result_path": str(result_path),
                "request_path": str(request_path),
                "lane_manifest_path": artifact_paths["lane_manifest_path"],
                "status": status,
                "classification": variant.classification,
            },
        )

    index_path = output_dir / "index.json"
    save_artifact(index_path, index_payload)
    print(f"Wrote {len(variants)} variant plans to {output_dir}")


if __name__ == "__main__":
    main()
