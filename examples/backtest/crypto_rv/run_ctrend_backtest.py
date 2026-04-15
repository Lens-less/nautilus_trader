#!/usr/bin/env python3
# mypy: disable-error-code=no-redef

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


try:
    from .common import format_utc_timestamp
    from .common import load_records
    from .common import parse_utc_timestamp
    from .common import resolve_path
    from .common import stable_payload_hash
    from .common import write_parquet_records
    from .ctrend_schemas import load_ctrend_config
    from .ctrend_schemas import load_ctrend_prepared_catalog
    from .ctrend_schemas import load_ctrend_snapshot
    from .ctrend_schemas import save_artifact
    from .schemas import ARTIFACT_SCHEMA_VERSION
    from .schemas import FeaturePanelManifest
    from .schemas import HistoryManifestEntry
    from .schemas import LaneArtifactManifest
    from .schemas import snapshot_identity
    from .signals.vol_managed_trend import TrendSignalSpec
    from .signals.vol_managed_trend import build_cross_sectional_scores
    from .signals.vol_managed_trend import build_target_weights
    from .signals.vol_managed_trend import compute_trend_components
except ImportError:  # pragma: no cover - script execution fallback
    from common import format_utc_timestamp
    from common import load_records
    from common import parse_utc_timestamp
    from common import resolve_path
    from common import stable_payload_hash
    from common import write_parquet_records
    from ctrend_schemas import load_ctrend_config
    from ctrend_schemas import load_ctrend_prepared_catalog
    from ctrend_schemas import load_ctrend_snapshot
    from ctrend_schemas import save_artifact
    from schemas import ARTIFACT_SCHEMA_VERSION
    from schemas import FeaturePanelManifest
    from schemas import HistoryManifestEntry
    from schemas import LaneArtifactManifest
    from schemas import snapshot_identity
    from signals.vol_managed_trend import TrendSignalSpec
    from signals.vol_managed_trend import build_cross_sectional_scores
    from signals.vol_managed_trend import build_target_weights
    from signals.vol_managed_trend import compute_trend_components

from nautilus_trader.backtest.config import BacktestDataConfig
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.config import BacktestRunConfig
from nautilus_trader.backtest.config import BacktestVenueConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig


FEATURE_COLUMNS = [
    "ts",
    "instrument_id",
    "raw_symbol",
    "fast_momentum",
    "slow_momentum",
    "ma_gap",
    "volume_confirmation",
    "realized_vol",
    "composite_score",
    "target_weight",
    "selected_long",
    "selected_short",
]

SIGNAL_COLUMNS = [
    "ts",
    "instrument_id",
    "raw_symbol",
    "composite_score",
    "target_weight",
    "selected_long",
    "selected_short",
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
]


@dataclass(slots=True)
class VariantSpec:
    name: str
    cost_model_name: str
    rebalance_cadence: str
    volatility_managed: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CTREND cross-sectional research lane.",
    )
    parser.add_argument("--config", required=True, help="Path to the CTREND research config JSON.")
    parser.add_argument("--snapshot", help="Optional override for the frozen universe snapshot.")
    parser.add_argument(
        "--prepared-catalog",
        help="Optional override for the prepared catalog artifact JSON.",
    )
    parser.add_argument("--output-dir", help="Optional override for the run output directory.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Attempt a real BacktestNode execution when catalog + strategy import are ready.",
    )
    return parser


def build_bar_type(instrument_id: str, bar_spec: str) -> str:
    normalized_spec = bar_spec.strip()
    if normalized_spec.endswith(("-EXTERNAL", "-INTERNAL")):
        return f"{instrument_id}-{normalized_spec}"
    return f"{instrument_id}-{normalized_spec}-EXTERNAL"


def strategy_path_exists(path: str) -> bool:
    module_name, attr_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return hasattr(module, attr_name)


def config_path_exists(path: str) -> bool:
    module_name, attr_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return hasattr(module, attr_name)


def build_variants(config: Any) -> list[VariantSpec]:
    cadence_sensitivity = VariantSpec(
        name=(
            "vol-managed-biweekly-base"
            if config.portfolio.rebalance_cadence == "weekly"
            else "vol-managed-weekly-base"
        ),
        cost_model_name="base",
        rebalance_cadence=(
            "biweekly" if config.portfolio.rebalance_cadence == "weekly" else "weekly"
        ),
        volatility_managed=True,
    )
    return [
        VariantSpec(
            name="raw-base",
            cost_model_name="base",
            rebalance_cadence=config.portfolio.rebalance_cadence,
            volatility_managed=False,
        ),
        VariantSpec(
            name="vol-managed-base",
            cost_model_name="base",
            rebalance_cadence=config.portfolio.rebalance_cadence,
            volatility_managed=True,
        ),
        VariantSpec(
            name="vol-managed-stress",
            cost_model_name="stress",
            rebalance_cadence=config.portfolio.rebalance_cadence,
            volatility_managed=True,
        ),
        cadence_sensitivity,
    ]


def _load_price_frame(entry: HistoryManifestEntry, manifest_base_dir: Path) -> pd.DataFrame:
    price_path = resolve_path(entry.price_path, manifest_base_dir)
    if price_path is None or not price_path.exists():
        raise FileNotFoundError(f"Missing price history for {entry.instrument_id}: {entry.price_path}")

    frame = pd.DataFrame(load_records(price_path))
    required_columns = {"ts_event", "close", "volume"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"{price_path} is missing required columns: {required_columns}")

    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame["close"] = frame["close"].astype(float)
    frame["volume"] = frame["volume"].astype(float)
    return frame.set_index("ts_event")[["close", "volume"]].sort_index()


def build_price_matrices(prepared_catalog: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    manifest_base_dir = Path(prepared_catalog.history_manifest_path).resolve().parent
    frames: dict[str, pd.DataFrame] = {}
    raw_symbols: dict[str, str] = {}

    for entry in prepared_catalog.manifest_entries:
        frames[entry.instrument_id] = _load_price_frame(entry, manifest_base_dir)
        raw_symbols[entry.instrument_id] = entry.raw_symbol

    combined_index = sorted(set().union(*(frame.index for frame in frames.values())))
    if len(combined_index) < 2:
        raise ValueError("Combined price history must contain at least two timestamps")

    close_frame = pd.DataFrame(
        {
            instrument_id: frame.reindex(combined_index)["close"].astype(float)
            for instrument_id, frame in frames.items()
        },
        index=pd.Index(combined_index),
    ).sort_index()
    volume_frame = pd.DataFrame(
        {
            instrument_id: frame.reindex(combined_index)["volume"].astype(float)
            for instrument_id, frame in frames.items()
        },
        index=pd.Index(combined_index),
    ).sort_index()
    return close_frame, volume_frame, raw_symbols


def iter_rebalance_windows(
    index: pd.Index,
    *,
    min_history_bars: int,
    step: int,
) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    for entry in range(min_history_bars, len(index) - 1, step):
        signal_position = entry - 1
        end = min(entry + step, len(index) - 1)
        windows.append((signal_position, entry, end))
    return windows


def bars_per_rebalance(index: pd.Index, cadence: str) -> int:
    if len(index) < 2:
        return 1

    delta = index[1] - index[0]
    target_days = 7 if cadence == "weekly" else 14
    target_seconds = target_days * 24 * 60 * 60
    step = max(1, round(target_seconds / delta.total_seconds()))
    return step


def select_active_instruments(
    close_frame: pd.DataFrame,
    position: int,
    *,
    min_history_bars: int,
) -> list[str]:
    active: list[str] = []
    window = close_frame.iloc[position - min_history_bars + 1 : position + 1]
    for instrument_id in close_frame.columns:
        series = window[instrument_id]
        if series.isna().any():
            continue
        if float(series.iloc[-1]) <= 0:
            continue
        active.append(str(instrument_id))
    return active


def baseline_weights(
    close_frame: pd.DataFrame,
    *,
    baseline_longs: list[str],
    baseline_shorts: list[str],
) -> dict[str, float]:
    weights = dict.fromkeys(close_frame.columns, 0.0)
    active_longs = [instrument_id for instrument_id in baseline_longs if instrument_id in close_frame.columns]
    active_shorts = [instrument_id for instrument_id in baseline_shorts if instrument_id in close_frame.columns]
    if active_longs:
        long_weight = 0.5 / len(active_longs)
        for instrument_id in active_longs:
            weights[instrument_id] = long_weight
    if active_shorts:
        short_weight = -0.5 / len(active_shorts)
        for instrument_id in active_shorts:
            weights[instrument_id] = short_weight
    return weights


def compute_period_return(
    weights: dict[str, float],
    period_returns: pd.Series,
    *,
    turnover: float,
    fee_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    period_days: float,
) -> dict[str, float]:
    long_return = 0.0
    short_return = 0.0
    short_gross = 0.0

    for instrument_id, weight in weights.items():
        asset_return = float(period_returns.get(instrument_id, 0.0))
        if weight > 0:
            long_return += weight * asset_return
        elif weight < 0:
            short_return += weight * asset_return
            short_gross += abs(weight)

    gross_return = long_return + short_return
    trade_cost = turnover * ((fee_bps + slippage_bps) / 10_000.0)
    borrow_cost = short_gross * (short_borrow_bps_per_day / 10_000.0) * period_days
    net_return = gross_return - trade_cost - borrow_cost
    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "long_return": long_return,
        "short_return": short_return,
        "turnover": turnover,
        "trade_cost": trade_cost,
        "borrow_cost": borrow_cost,
        "gross_exposure": sum(abs(weight) for weight in weights.values()),
        "net_exposure": sum(weights.values()),
        "active_names": len([weight for weight in weights.values() if weight != 0]),
    }


def max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    return drawdown


def evaluate_replacement(
    *,
    net_return: float,
    baseline_net_return: float,
    max_drawdown_value: float,
    baseline_max_drawdown: float,
) -> bool:
    return net_return > baseline_net_return and max_drawdown_value >= baseline_max_drawdown


def compute_baseline_metrics(
    *,
    close_frame: pd.DataFrame,
    index: pd.Index,
    baseline_longs: list[str],
    baseline_shorts: list[str],
    variant: VariantSpec,
    config: Any,
) -> dict[str, float]:
    step = bars_per_rebalance(index, variant.rebalance_cadence)
    weights = baseline_weights(
        close_frame,
        baseline_longs=baseline_longs,
        baseline_shorts=baseline_shorts,
    )
    previous_weights = dict.fromkeys(close_frame.columns, 0.0)
    net_returns: list[float] = []
    first_interval = True

    for entry in range(1, len(index) - 1, step):
        end = min(entry + step, len(index) - 1)
        period_returns = close_frame.iloc[end] / close_frame.iloc[entry] - 1.0
        turnover = (
            sum(abs(weights[instrument_id] - previous_weights[instrument_id]) for instrument_id in weights)
            if first_interval
            else 0.0
        )
        first_interval = False
        previous_weights = dict(weights)
        period_days = (index[end] - index[entry]).total_seconds() / (24 * 60 * 60)
        metrics = compute_period_return(
            weights,
            period_returns,
            turnover=turnover,
            fee_bps=config.cost_models[variant.cost_model_name].fee_bps,
            slippage_bps=config.cost_models[variant.cost_model_name].slippage_bps,
            short_borrow_bps_per_day=(
                config.cost_models[variant.cost_model_name].short_borrow_bps_annual / 365.0
            ),
            period_days=period_days,
        )
        net_returns.append(metrics["net_return"])

    total_return = 1.0
    for value in net_returns:
        total_return *= 1.0 + value

    return {
        "net_return": total_return - 1.0,
        "max_drawdown": max_drawdown(net_returns),
    }


def simulate_variant(
    *,
    config: Any,
    snapshot: Any,
    prepared_catalog: Any,
    close_frame: pd.DataFrame,
    volume_frame: pd.DataFrame,
    index: pd.Index,
    raw_symbols: dict[str, str],
    universe_ids: list[str],
    baseline_shorts: list[str],
    signal_spec: TrendSignalSpec,
    variant: VariantSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    step = bars_per_rebalance(index, variant.rebalance_cadence)
    feature_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    previous_weights = dict.fromkeys(universe_ids, 0.0)
    net_returns: list[float] = []
    gross_returns: list[float] = []

    for signal_position, entry, end in iter_rebalance_windows(
        index,
        min_history_bars=config.signal.min_history_bars,
        step=step,
    ):
        active_ids = select_active_instruments(
            close_frame,
            signal_position,
            min_history_bars=config.signal.min_history_bars,
        )
        if len(active_ids) < 2:
            continue

        components_by_instrument = {
            instrument_id: compute_trend_components(
                close_frame[instrument_id].iloc[: signal_position + 1].tolist(),
                volume_frame[instrument_id].iloc[: signal_position + 1].tolist(),
                signal_spec,
            )
            for instrument_id in active_ids
        }
        score_payload = build_cross_sectional_scores(
            components_by_instrument,
            volatility_managed=variant.volatility_managed,
        )
        weights = build_target_weights(
            {
                instrument_id: payload["composite_score"]
                for instrument_id, payload in score_payload.items()
            },
            long_bucket_frac=config.portfolio.long_bucket_frac,
            short_bucket_frac=config.portfolio.short_bucket_frac,
        )
        period_returns = close_frame.iloc[end] / close_frame.iloc[entry] - 1.0
        turnover = sum(
            abs(weights.get(instrument_id, 0.0) - previous_weights.get(instrument_id, 0.0))
            for instrument_id in universe_ids
        )
        previous_weights = {instrument_id: weights.get(instrument_id, 0.0) for instrument_id in universe_ids}
        period_days = (index[end] - index[entry]).total_seconds() / (24 * 60 * 60)

        metrics = compute_period_return(
            previous_weights,
            period_returns,
            turnover=turnover,
            fee_bps=config.cost_models[variant.cost_model_name].fee_bps,
            slippage_bps=config.cost_models[variant.cost_model_name].slippage_bps,
            short_borrow_bps_per_day=(
                config.cost_models[variant.cost_model_name].short_borrow_bps_annual / 365.0
            ),
            period_days=period_days,
        )
        gross_returns.append(metrics["gross_return"])
        net_returns.append(metrics["net_return"])
        longs = {instrument_id for instrument_id, weight in previous_weights.items() if weight > 0}
        shorts = {instrument_id for instrument_id, weight in previous_weights.items() if weight < 0}

        ts_value = format_utc_timestamp(index[signal_position].to_pydatetime())
        for instrument_id in active_ids:
            components = components_by_instrument[instrument_id]
            score = score_payload[instrument_id]["composite_score"]
            target_weight = previous_weights.get(instrument_id, 0.0)
            feature_rows.append(
                {
                    "ts": ts_value,
                    "instrument_id": instrument_id,
                    "raw_symbol": raw_symbols[instrument_id],
                    "fast_momentum": components["fast_momentum"],
                    "slow_momentum": components["slow_momentum"],
                    "ma_gap": components["ma_gap"],
                    "volume_confirmation": components["volume_confirmation"],
                    "realized_vol": components["realized_vol"],
                    "composite_score": score,
                    "target_weight": target_weight,
                    "selected_long": instrument_id in longs,
                    "selected_short": instrument_id in shorts,
                },
            )
            signal_rows.append(
                {
                    "ts": ts_value,
                    "instrument_id": instrument_id,
                    "raw_symbol": raw_symbols[instrument_id],
                    "composite_score": score,
                    "target_weight": target_weight,
                    "selected_long": instrument_id in longs,
                    "selected_short": instrument_id in shorts,
                },
            )

        portfolio_rows.append(
            {
                "ts": format_utc_timestamp(index[end].to_pydatetime()),
                "gross_return": metrics["gross_return"],
                "net_return": metrics["net_return"],
                "long_return": metrics["long_return"],
                "short_return": metrics["short_return"],
                "turnover": metrics["turnover"],
                "gross_exposure": metrics["gross_exposure"],
                "net_exposure": metrics["net_exposure"],
                "active_names": metrics["active_names"],
            },
        )

    if not portfolio_rows:
        raise ValueError("CTREND runner did not produce any portfolio intervals")

    total_return = 1.0
    for value in net_returns:
        total_return *= 1.0 + value
    gross_total_return = 1.0
    for value in gross_returns:
        gross_total_return *= 1.0 + value

    baseline_metrics = compute_baseline_metrics(
        close_frame=close_frame,
        index=index,
        baseline_longs=config.baseline_long_instrument_ids,
        baseline_shorts=baseline_shorts,
        variant=variant,
        config=config,
    )
    net_return = total_return - 1.0
    gross_return = gross_total_return - 1.0
    max_drawdown_value = max_drawdown(net_returns)
    result_payload: dict[str, Any] = {
        "variant_name": variant.name,
        "status": "completed",
        "mode": "offline-simulated",
        "cost_model_name": variant.cost_model_name,
        "volatility_managed": variant.volatility_managed,
        "rebalance_cadence": variant.rebalance_cadence,
        "gross_return": gross_return,
        "net_return": net_return,
        "max_drawdown": max_drawdown_value,
        "baseline_net_return": baseline_metrics["net_return"],
        "baseline_max_drawdown": baseline_metrics["max_drawdown"],
        "replace_mainline": evaluate_replacement(
            net_return=net_return,
            baseline_net_return=baseline_metrics["net_return"],
            max_drawdown_value=max_drawdown_value,
            baseline_max_drawdown=baseline_metrics["max_drawdown"],
        ),
        "coverage_classification": snapshot.funding_coverage_classification,
        "notes": [
            "Metrics are computed by the CTREND research runner from historical price paths.",
            "Baseline comparison uses the preserved fixed majors-vs-selected-shorts basket.",
        ],
    }
    return feature_rows, signal_rows, portfolio_rows, result_payload


def build_backtest_request(
    config: Any,
    prepared_catalog: Any,
    variant: VariantSpec,
    universe_ids: list[str],
) -> tuple[BacktestRunConfig, dict[str, Any]]:
    manifest_by_instrument = {
        entry.instrument_id: entry for entry in prepared_catalog.manifest_entries
    }
    strategy_payload = {
        "universe_instrument_ids": universe_ids,
        "bar_types": [
            build_bar_type(instrument_id, manifest_by_instrument[instrument_id].bar_spec)
            for instrument_id in universe_ids
        ],
        "leg_notional_usd": config.portfolio.leg_notional_usd,
        "rebalance_cadence": variant.rebalance_cadence,
        "long_bucket_frac": config.portfolio.long_bucket_frac,
        "short_bucket_frac": config.portfolio.short_bucket_frac,
        "fast_window": config.signal.fast_window,
        "slow_window": config.signal.slow_window,
        "sma_window": config.signal.sma_window,
        "vol_window": config.signal.vol_window,
        "volume_window": config.signal.volume_window,
        "min_history_bars": config.signal.min_history_bars,
        "volatility_managed": variant.volatility_managed,
        "min_order_notional_usd": config.portfolio.min_order_notional_usd,
        "long_fee_bps": config.cost_models[variant.cost_model_name].fee_bps,
        "short_fee_bps": config.cost_models[variant.cost_model_name].fee_bps,
        "long_slippage_bps": config.cost_models[variant.cost_model_name].slippage_bps,
        "short_slippage_bps": config.cost_models[variant.cost_model_name].slippage_bps,
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
        name="BINANCE",
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency="USDT",
        starting_balances=["1000000 USDT"],
    )
    data_configs = [
        BacktestDataConfig(
            catalog_path=prepared_catalog.catalog_path or prepared_catalog.artifact_root,
            data_cls=manifest_by_instrument[instrument_id].data_cls,
            instrument_id=instrument_id,
            start_time=str(parse_utc_timestamp(manifest_by_instrument[instrument_id].start_ts)),
            end_time=str(parse_utc_timestamp(manifest_by_instrument[instrument_id].end_ts)),
            bar_spec=manifest_by_instrument[instrument_id].bar_spec,
        )
        for instrument_id in universe_ids
    ]
    run_config = BacktestRunConfig(
        engine=engine_config,
        venues=[venue_config],
        data=data_configs,
        start=str(parse_utc_timestamp(manifest_by_instrument[universe_ids[0]].start_ts)),
        end=str(parse_utc_timestamp(manifest_by_instrument[universe_ids[0]].end_ts)),
    )
    request_payload = {
        "variant_name": variant.name,
        "cost_model_name": variant.cost_model_name,
        "volatility_managed": variant.volatility_managed,
        "rebalance_cadence": variant.rebalance_cadence,
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
    }
    return run_config, request_payload


def summarize_result(result_payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {result_payload['variant_name']}",
            "",
            "## CTREND Result",
            "",
            f"- Net return: `{result_payload['net_return']:.4%}`",
            f"- Max drawdown: `{result_payload['max_drawdown']:.4%}`",
            f"- Baseline net return: `{result_payload['baseline_net_return']:.4%}`",
            f"- Baseline max drawdown: `{result_payload['baseline_max_drawdown']:.4%}`",
            f"- Replace mainline: `{result_payload['replace_mainline']}`",
        ],
    )


def write_artifacts(
    *,
    config: Any,
    snapshot: Any,
    prepared_catalog: Any,
    snapshot_path: Path,
    prepared_catalog_path: Path,
    variant: VariantSpec,
    variant_dir: Path,
    request_path: Path,
    feature_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    portfolio_rows: list[dict[str, Any]],
    result_payload: dict[str, Any],
    status: str,
) -> None:
    feature_panel_path = variant_dir / "feature_panel.parquet"
    signal_panel_path = variant_dir / "signal_panel.parquet"
    portfolio_timeseries_path = variant_dir / "portfolio_timeseries.parquet"
    feature_panel_manifest_path = variant_dir / "feature_panel_manifest.json"
    signal_metrics_path = variant_dir / "signal_metrics.json"
    portfolio_metrics_path = variant_dir / "portfolio_metrics.json"
    summary_report_path = variant_dir / "summary_report.md"
    lane_manifest_path = variant_dir / "lane_manifest.json"
    result_path = variant_dir / "result.json"

    write_parquet_records(feature_panel_path, feature_rows)
    write_parquet_records(signal_panel_path, signal_rows)
    write_parquet_records(portfolio_timeseries_path, portfolio_rows)

    feature_manifest = FeaturePanelManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        variant_name=variant.name,
        placeholder=False,
        feature_schema_version=ARTIFACT_SCHEMA_VERSION,
        feature_columns=FEATURE_COLUMNS,
        date_range={
            "start": feature_rows[0]["ts"],
            "end": feature_rows[-1]["ts"],
        },
        instrument_count=len({row["instrument_id"] for row in feature_rows}),
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
        "mode": "offline-simulated",
        "placeholder": False,
        "feature_columns": FEATURE_COLUMNS,
        "signal_columns": SIGNAL_COLUMNS,
        "row_count": len(signal_rows),
        "selected_long_count": len([row for row in signal_rows if row["selected_long"]]),
        "selected_short_count": len([row for row in signal_rows if row["selected_short"]]),
    }
    save_artifact(signal_metrics_path, signal_metrics_payload)

    portfolio_metrics_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "variant_name": variant.name,
        "snapshot_id": snapshot_id,
        "config_hash": config_hash,
        "mode": "offline-simulated",
        "placeholder": False,
        **result_payload,
        "timeseries_columns": PORTFOLIO_TIMESERIES_COLUMNS,
    }
    save_artifact(portfolio_metrics_path, portfolio_metrics_payload)
    save_artifact(result_path, result_payload)
    summary_report_path.write_text(summarize_result(result_payload) + "\n", encoding="utf-8")

    lane_manifest = LaneArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        research_name=config.research_name,
        variant_name=variant.name,
        mode="offline-simulated",
        status=status,
        placeholder=False,
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


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_ctrend_config(config_path)
    paths = config.resolved_paths(config_path)

    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else paths["snapshot_path"]
    prepared_catalog_path = (
        Path(args.prepared_catalog).resolve()
        if args.prepared_catalog
        else paths["prepared_catalog_path"]
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["output_dir"]

    snapshot = load_ctrend_snapshot(snapshot_path)
    prepared_catalog = load_ctrend_prepared_catalog(prepared_catalog_path)
    close_frame, volume_frame, raw_symbols = build_price_matrices(prepared_catalog)
    variants = build_variants(config)
    index = close_frame.index
    universe_ids = list(close_frame.columns)
    baseline_shorts = [
        instrument_id
        for instrument_id in snapshot.selected_short_instrument_ids
        if instrument_id in close_frame.columns
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    index_payload: dict[str, Any] = {
        "research_name": config.research_name,
        "snapshot_path": str(snapshot_path),
        "prepared_catalog_path": str(prepared_catalog_path),
        "variants": [],
    }
    signal_spec = TrendSignalSpec(
        fast_window=config.signal.fast_window,
        slow_window=config.signal.slow_window,
        sma_window=config.signal.sma_window,
        vol_window=config.signal.vol_window,
        volume_window=config.signal.volume_window,
        min_history_bars=config.signal.min_history_bars,
    )

    strategy_import_ok = False
    config_import_ok = False
    import_errors: list[str] = []
    try:
        strategy_import_ok = strategy_path_exists(config.strategy.strategy_path)
        config_import_ok = config_path_exists(config.strategy.config_path)
    except Exception as exc:  # pragma: no cover - bounded fallback
        import_errors.append(str(exc))

    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        run_config, request_payload = build_backtest_request(
            config=config,
            prepared_catalog=prepared_catalog,
            variant=variant,
            universe_ids=universe_ids,
        )
        request_path = variant_dir / "run_request.json"
        save_artifact(request_path, request_payload)

        feature_rows, signal_rows, portfolio_rows, result_payload = simulate_variant(
            config=config,
            snapshot=snapshot,
            prepared_catalog=prepared_catalog,
            close_frame=close_frame,
            volume_frame=volume_frame,
            index=index,
            raw_symbols=raw_symbols,
            universe_ids=universe_ids,
            baseline_shorts=baseline_shorts,
            signal_spec=signal_spec,
            variant=variant,
        )

        if args.execute:
            if not prepared_catalog.catalog_ready:
                result_payload["status"] = "blocked"
                result_payload["notes"].append("catalog_ready=false")
            elif not (strategy_import_ok and config_import_ok):
                result_payload["status"] = "blocked"
                result_payload["notes"].append("strategy import path is not available")
                result_payload["notes"].extend(import_errors)
            else:
                node = BacktestNode(configs=[run_config])
                [result] = node.run()
                result_payload["raw_result_repr"] = repr(result)

        write_artifacts(
            config=config,
            snapshot=snapshot,
            prepared_catalog=prepared_catalog,
            snapshot_path=snapshot_path,
            prepared_catalog_path=prepared_catalog_path,
            variant=variant,
            variant_dir=variant_dir,
            request_path=request_path,
            feature_rows=feature_rows,
            signal_rows=signal_rows,
            portfolio_rows=portfolio_rows,
            result_payload=result_payload,
            status=result_payload["status"],
        )

        index_payload["variants"].append(
            {
                "variant_name": variant.name,
                "result_path": str(variant_dir / "result.json"),
                "request_path": str(request_path),
                "status": result_payload["status"],
            },
        )

    save_artifact(output_dir / "index.json", index_payload)
    print(f"Wrote {len(index_payload['variants'])} CTREND variant runs to {output_dir}")


if __name__ == "__main__":
    main()
