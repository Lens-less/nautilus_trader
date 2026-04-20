#!/usr/bin/env python3
# mypy: disable-error-code=no-redef

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog


try:
    from .common import load_json
    from .common import write_json
    from .run_ctrend_backtest import VariantSpec
    from .run_ctrend_backtest import build_variants
    from .run_ctrend_backtest import simulate_variant
    from .signals.vol_managed_trend import TrendSignalSpec
except ImportError:  # pragma: no cover - script execution fallback
    from common import load_json
    from common import write_json
    from run_ctrend_backtest import VariantSpec
    from run_ctrend_backtest import build_variants
    from run_ctrend_backtest import simulate_variant
    from signals.vol_managed_trend import TrendSignalSpec


@dataclass(slots=True)
class RealPublicConfig:
    research_root: Path
    selected_universe_path: Path
    catalog_path: Path
    baseline_metrics_path: Path
    top_n_candidates: int
    output_json: Path
    output_md: Path
    signal: TrendSignalSpec
    leg_notional_usd: float
    long_bucket_frac: float
    short_bucket_frac: float
    rebalance_cadence: str
    cost_models: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CTREND on the real_public_2026q1 dataset.")
    parser.add_argument(
        "--config",
        default="examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json",
        help="Path to the real-public CTREND config JSON.",
    )
    return parser


def load_config(path: Path) -> RealPublicConfig:
    payload = load_json(path)
    base_dir = path.parent
    repo_root = Path(__file__).resolve().parents[3]

    def _resolve(value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            return (base_dir / candidate).resolve()
        if candidate.exists():
            return candidate
        if "examples" in candidate.parts:
            suffix = Path(*candidate.parts[candidate.parts.index("examples") :])
            fallback = (repo_root / suffix).resolve()
            if fallback.exists():
                return fallback
        return candidate

    return RealPublicConfig(
        research_root=_resolve(payload["research_root"]),
        selected_universe_path=_resolve(payload["selected_universe_path"]),
        catalog_path=_resolve(payload["catalog_path"]),
        baseline_metrics_path=_resolve(payload["baseline_metrics_path"]),
        top_n_candidates=int(payload.get("top_n_candidates", 50)),
        output_json=_resolve(payload["output_json"]),
        output_md=_resolve(payload["output_md"]),
        signal=TrendSignalSpec(**payload["signal"]),
        leg_notional_usd=float(payload["portfolio"]["leg_notional_usd"]),
        long_bucket_frac=float(payload["portfolio"]["long_bucket_frac"]),
        short_bucket_frac=float(payload["portfolio"]["short_bucket_frac"]),
        rebalance_cadence=str(payload["portfolio"]["rebalance_cadence"]),
        cost_models=payload["cost_models"],
    )


def load_real_public_universe(path: Path, top_n_candidates: int) -> tuple[list[str], list[str], list[str]]:
    payload = load_json(path)
    longs = [f"{pair}-PERP.BINANCE" for pair in payload["longs"]]
    ranked_pairs = [item["pair"] for item in payload["all_ranked_candidates"][:top_n_candidates]]
    top_candidates = [f"{pair}-PERP.BINANCE" for pair in ranked_pairs]
    baseline_shorts = top_candidates[:20]
    return longs, top_candidates, baseline_shorts


def load_catalog_frames(
    catalog_path: Path,
    instrument_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    catalog = ParquetDataCatalog(catalog_path)
    close_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    raw_symbols: dict[str, str] = {}

    for instrument_id in instrument_ids:
        bars = catalog.bars([BarType.from_str(f"{instrument_id}-1-DAY-LAST-EXTERNAL")])
        if not bars:
            continue
        timestamps = [pd.Timestamp(bar.ts_event, unit="ns", tz="UTC") for bar in bars]
        closes = [float(bar.close) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        close_series[instrument_id] = pd.Series(closes, index=timestamps)
        volume_series[instrument_id] = pd.Series(volumes, index=timestamps)
        raw_symbols[instrument_id] = instrument_id.split("-PERP.")[0]

    if not close_series:
        raise ValueError(f"No bar series loaded from {catalog_path}")

    union_index = sorted(set().union(*(series.index for series in close_series.values())))
    close_frame = pd.DataFrame(
        {instrument_id: series.reindex(union_index) for instrument_id, series in close_series.items()},
        index=pd.Index(union_index),
    ).sort_index()
    volume_frame = pd.DataFrame(
        {instrument_id: series.reindex(union_index) for instrument_id, series in volume_series.items()},
        index=pd.Index(union_index),
    ).sort_index()
    return close_frame, volume_frame, raw_symbols


def build_runtime_config(config: RealPublicConfig, baseline_longs: list[str]) -> Any:
    return SimpleNamespace(
        research_name="ctrend-real-public-q1",
        signal=SimpleNamespace(
            fast_window=config.signal.fast_window,
            slow_window=config.signal.slow_window,
            sma_window=config.signal.sma_window,
            vol_window=config.signal.vol_window,
            volume_window=config.signal.volume_window,
            min_history_bars=config.signal.min_history_bars,
        ),
        portfolio=SimpleNamespace(
            leg_notional_usd=config.leg_notional_usd,
            long_bucket_frac=config.long_bucket_frac,
            short_bucket_frac=config.short_bucket_frac,
            rebalance_cadence=config.rebalance_cadence,
        ),
        cost_models={
            name: SimpleNamespace(**payload)
            for name, payload in config.cost_models.items()
        },
        baseline_long_instrument_ids=baseline_longs,
    )


def summarize_markdown(report_payload: dict[str, Any]) -> str:
    winner = report_payload["winner"]
    lines = [
        "# CTREND Real Public Q1 Report",
        "",
        f"- Window: {report_payload['window']['start']} to {report_payload['window']['end']}",
        f"- Universe size: {report_payload['universe']['instrument_count']}",
        f"- Long bucket fraction: {report_payload['signal_and_portfolio']['long_bucket_frac']:.0%}",
        f"- Short bucket fraction: {report_payload['signal_and_portfolio']['short_bucket_frac']:.0%}",
        f"- Winner: `{winner['variant_name']}`",
        f"- Winner net return: {winner['net_return']:.2%}",
        f"- Winner max drawdown: {winner['max_drawdown']:.2%}",
        f"- Reconstructed RV baseline net return: {winner['baseline_net_return']:.2%}",
        f"- Reconstructed RV baseline max drawdown: {winner['baseline_max_drawdown']:.2%}",
        f"- Provisional replace-mainline verdict: `{winner['replace_mainline']}`",
        "",
        "## Variants",
        "",
    ]
    for variant in report_payload["variants"]:
        lines.extend(
            [
                f"### {variant['variant_name']}",
                f"- Net return: {variant['net_return']:.2%}",
                f"- Max drawdown: {variant['max_drawdown']:.2%}",
                f"- Rebalance cadence: `{variant['rebalance_cadence']}`",
                f"- Volatility managed: `{variant['volatility_managed']}`",
                "",
            ],
        )

    lines.extend(
        [
            "## Legacy RV Context",
            "",
            f"- Primary base return: {report_payload['legacy_rv_context']['primary_base_return_pct']:.2f}%",
            f"- Primary stress return: {report_payload['legacy_rv_context']['primary_stress_return_pct']:.2f}%",
            f"- Short50 base return: {report_payload['legacy_rv_context']['short50_base_return_pct']:.2f}%",
            "",
            "## Caveats",
            "",
        ]
        + [f"- {note}" for note in report_payload["notes"]]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    baseline_longs, top_candidates, baseline_shorts = load_real_public_universe(
        config.selected_universe_path,
        config.top_n_candidates,
    )
    universe_ids = baseline_longs + [instrument_id for instrument_id in top_candidates if instrument_id not in baseline_longs]
    close_frame, volume_frame, raw_symbols = load_catalog_frames(config.catalog_path, universe_ids)
    runtime_config = build_runtime_config(config, baseline_longs)
    variants: list[VariantSpec] = build_variants(runtime_config)
    snapshot = SimpleNamespace(funding_coverage_classification="funding_not_integrated_in_real_public_runner")
    prepared_catalog = SimpleNamespace()

    results: list[dict[str, Any]] = []
    for variant in variants:
        _feature_rows, _signal_rows, _portfolio_rows, result = simulate_variant(
            config=runtime_config,
            snapshot=snapshot,
            prepared_catalog=prepared_catalog,
            close_frame=close_frame,
            volume_frame=volume_frame,
            index=close_frame.index,
            raw_symbols=raw_symbols,
            universe_ids=list(close_frame.columns),
            baseline_shorts=baseline_shorts,
            signal_spec=config.signal,
            variant=variant,
        )
        results.append(result)

    winner = max(results, key=lambda item: item["net_return"])
    legacy_metrics = load_json(config.baseline_metrics_path)
    report_payload = {
        "window": legacy_metrics["window"],
        "signal_and_portfolio": {
            "fast_window": config.signal.fast_window,
            "slow_window": config.signal.slow_window,
            "sma_window": config.signal.sma_window,
            "vol_window": config.signal.vol_window,
            "volume_window": config.signal.volume_window,
            "min_history_bars": config.signal.min_history_bars,
            "long_bucket_frac": config.long_bucket_frac,
            "short_bucket_frac": config.short_bucket_frac,
            "rebalance_cadence": config.rebalance_cadence,
        },
        "universe": {
            "instrument_count": len(close_frame.columns),
            "top_candidate_count": len(top_candidates),
            "baseline_short_count": len(baseline_shorts),
            "longs": baseline_longs,
        },
        "winner": winner,
        "variants": results,
        "legacy_rv_context": {
            "primary_base_return_pct": legacy_metrics["variants"]["primary_base"]["total_return_pct"],
            "primary_stress_return_pct": legacy_metrics["variants"]["primary_stress"]["total_return_pct"],
            "short50_base_return_pct": legacy_metrics["variants"]["short50_base"]["total_return_pct"],
            "selection_method": legacy_metrics["selection_method"],
        },
        "notes": [
            "CTREND Q1 report is computed from the real_public_2026q1 catalog_50 daily bars.",
            "The CTREND runner does not currently integrate funding; legacy RV context does.",
            "Replace-mainline verdict is provisional against the reconstructed price-only RV baseline, not a funding-complete apples-to-apples comparison.",
        ],
    }

    write_json(config.output_json, report_payload)
    config.output_md.write_text(summarize_markdown(report_payload), encoding="utf-8")
    print(f"Wrote CTREND real-public report JSON to {config.output_json}")
    print(f"Wrote CTREND real-public report Markdown to {config.output_md}")


if __name__ == "__main__":
    main()
