#!/usr/bin/env python3
# mypy: disable-error-code=no-redef

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd


try:
    from .common import load_json
    from .common import write_json
    from .run_ctrend_backtest import VariantSpec
    from .run_ctrend_backtest import bars_per_rebalance
    from .run_ctrend_backtest import compute_period_return
    from .run_ctrend_backtest import evaluate_replacement
    from .run_ctrend_backtest import iter_rebalance_windows
    from .run_ctrend_backtest import max_drawdown
    from .run_ctrend_backtest import select_active_instruments
    from .run_ctrend_real_public import build_runtime_config
    from .run_ctrend_real_public import load_catalog_frames
    from .run_ctrend_real_public import load_config as load_real_public_config
    from .run_ctrend_real_public import load_real_public_universe
    from .signals.vol_managed_trend import build_cross_sectional_scores
    from .signals.vol_managed_trend import compute_trend_components
    from .signals.vol_managed_trend import select_bucket_members
except ImportError:  # pragma: no cover - script execution fallback
    from common import load_json
    from common import write_json
    from run_ctrend_backtest import VariantSpec
    from run_ctrend_backtest import bars_per_rebalance
    from run_ctrend_backtest import compute_period_return
    from run_ctrend_backtest import evaluate_replacement
    from run_ctrend_backtest import iter_rebalance_windows
    from run_ctrend_backtest import max_drawdown
    from run_ctrend_backtest import select_active_instruments
    from run_ctrend_real_public import build_runtime_config
    from run_ctrend_real_public import load_catalog_frames
    from run_ctrend_real_public import load_config as load_real_public_config
    from run_ctrend_real_public import load_real_public_universe
    from signals.vol_managed_trend import build_cross_sectional_scores
    from signals.vol_managed_trend import compute_trend_components
    from signals.vol_managed_trend import select_bucket_members


COMBO_MODES = ("baseline", "hard_filter", "post_rank_filter", "short_tilt")
BAND_NAMES = ("rank_1_20", "rank_1_30", "rank_1_50", "rank_21_30", "rank_21_50", "rank_31_50")
FOCUS_BAND_NAMES = ("rank_1_20", "rank_1_30", "rank_1_50")


@dataclass(slots=True)
class FundingCoverage:
    requested_months: tuple[str, ...]
    loaded_months: tuple[str, ...]
    loaded_month_count: int
    requested_month_count: int
    data_source: str
    coverage_end: str | None
    symbols_with_any_data: int
    symbols_missing_all_data: list[str]


@dataclass(slots=True)
class ComboConfig:
    real_public_config_path: Path
    liquidity_reversal_report_path: Path
    tilt_multiplier: float
    output_json: Path
    output_md: Path


class BinanceVisionFundingLoader:
    def __init__(self, *, start_ms: int, end_ms: int) -> None:
        self._start_ms = start_ms
        self._end_ms = end_ms
        self._months = month_labels(start_ms, end_ms)
        self._events_for_month_cache: dict[tuple[str, str], tuple[tuple[int, float], ...]] = {}
        self._funding_events_cache: dict[str, tuple[tuple[int, float], ...]] = {}
        self._loaded_months_cache: dict[str, tuple[str, ...]] = {}
        self._rate_sum_cache: dict[tuple[str, int, int], float] = {}

    @property
    def months(self) -> tuple[str, ...]:
        return self._months

    @staticmethod
    def _url(symbol: str, month: str) -> str:
        return (
            "https://data.binance.vision/data/futures/um/monthly/"
            f"fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
        )

    @staticmethod
    def _parse_rows(
        rows: list[list[str]],
        *,
        start_ms: int,
        end_ms: int,
    ) -> tuple[tuple[int, float], ...]:
        events: list[tuple[int, float]] = []
        for row in rows[1:]:
            calc_time = int(row[0])
            interval_hours = int(row[1])
            rate = float(row[2])
            interval_ms = interval_hours * 60 * 60 * 1000
            bucket = (calc_time // interval_ms) * interval_ms
            if start_ms <= bucket < end_ms:
                events.append((bucket, rate))
        return tuple(events)

    def _events_for_month(self, symbol: str, month: str) -> tuple[tuple[int, float], ...]:
        cache_key = (symbol, month)
        if cache_key in self._events_for_month_cache:
            return self._events_for_month_cache[cache_key]
        url = self._url(symbol, month)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                payload = response.read()
        except Exception:
            self._events_for_month_cache[cache_key] = ()
            return ()
        archive = zipfile.ZipFile(io.BytesIO(payload))
        rows = list(csv.reader(io.StringIO(archive.read(archive.namelist()[0]).decode("utf-8"))))
        events = self._parse_rows(
            rows,
            start_ms=self._start_ms,
            end_ms=self._end_ms,
        )
        self._events_for_month_cache[cache_key] = events
        return events

    def funding_events(self, symbol: str) -> tuple[tuple[int, float], ...]:
        if symbol in self._funding_events_cache:
            return self._funding_events_cache[symbol]
        events: list[tuple[int, float]] = []
        for month in self._months:
            events.extend(self._events_for_month(symbol, month))
        payload = tuple(sorted(events))
        self._funding_events_cache[symbol] = payload
        return payload

    def loaded_months(self, symbol: str) -> tuple[str, ...]:
        if symbol in self._loaded_months_cache:
            return self._loaded_months_cache[symbol]
        payload = tuple(month for month in self._months if self._events_for_month(symbol, month))
        self._loaded_months_cache[symbol] = payload
        return payload

    def rate_sum(self, symbol: str, start_ms: int, end_ms: int) -> float:
        cache_key = (symbol, start_ms, end_ms)
        if cache_key in self._rate_sum_cache:
            return self._rate_sum_cache[cache_key]
        total = sum(rate for ts_ms, rate in self.funding_events(symbol) if start_ms <= ts_ms < end_ms)
        self._rate_sum_cache[cache_key] = total
        return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CTREND x liquidity/reversal conditional combos on real-public Q1 data.")
    parser.add_argument(
        "--config",
        default="examples/backtest/crypto_rv/configs/ctrend.liq_combo.real_public_q1.json",
        help="Path to the combo config JSON.",
    )
    return parser


def load_combo_config(path: Path) -> ComboConfig:
    payload = load_json(path)
    base_dir = path.parent

    def _resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base_dir / value).resolve()

    return ComboConfig(
        real_public_config_path=_resolve(payload["real_public_config_path"]),
        liquidity_reversal_report_path=_resolve(payload["liquidity_reversal_report_path"]),
        tilt_multiplier=float(payload.get("tilt_multiplier", 2.0)),
        output_json=_resolve(payload["output_json"]),
        output_md=_resolve(payload["output_md"]),
    )


def month_labels(start_ms: int, end_ms: int) -> tuple[str, ...]:
    current = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc).replace(  # noqa: UP017
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end_month = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc).replace(  # noqa: UP017
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    labels: list[str] = []
    while current <= end_month:
        labels.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return tuple(labels)


def timestamp_ms(value: Any) -> int:
    if isinstance(value, pd.Timestamp):
        return int(value.value // 1_000_000)
    return int(pd.Timestamp(value).value // 1_000_000)


def instrument_symbol(instrument_id: str) -> str:
    return instrument_id.split("-PERP.")[0]


def period_funding_return(
    weights: dict[str, float],
    *,
    period_start: Any,
    period_end: Any,
    funding_loader: Any,
) -> float:
    start_ms = timestamp_ms(period_start)
    end_ms = timestamp_ms(period_end)
    total = 0.0
    for instrument_id, weight in weights.items():
        if weight == 0.0:
            continue
        total += -weight * float(funding_loader.rate_sum(instrument_symbol(instrument_id), start_ms, end_ms))
    return total


def matched_legacy_baseline_variant(variant: VariantSpec) -> str:
    if variant.cost_model_name == "stress":
        return "primary_stress"
    if variant.rebalance_cadence == "biweekly":
        return "primary_base_biweekly"
    return "primary_base"


def funding_coverage_summary(
    funding_loader: BinanceVisionFundingLoader,
    instrument_ids: list[str],
) -> FundingCoverage:
    symbols = sorted({instrument_symbol(instrument_id) for instrument_id in instrument_ids})
    loaded_by_symbol = {symbol: funding_loader.loaded_months(symbol) for symbol in symbols}
    loaded_months = sorted({month for months in loaded_by_symbol.values() for month in months})
    last_event_ms = max(
        (
            events[-1][0]
            for symbol in symbols
            if (events := funding_loader.funding_events(symbol))
        ),
        default=None,
    )
    coverage_end = (
        datetime.fromtimestamp(last_event_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017
        if last_event_ms is not None
        else None
    )
    missing_all = [symbol for symbol, months in loaded_by_symbol.items() if not months]
    return FundingCoverage(
        requested_months=funding_loader.months,
        loaded_months=tuple(loaded_months),
        loaded_month_count=len(loaded_months),
        requested_month_count=len(funding_loader.months),
        data_source="binance.vision monthly fundingRate archives",
        coverage_end=coverage_end,
        symbols_with_any_data=len(symbols) - len(missing_all),
        symbols_missing_all_data=missing_all,
    )


def build_variants(runtime_config: Any) -> list[VariantSpec]:
    cadence_sensitivity = VariantSpec(
        name=(
            "vol-managed-biweekly-base"
            if runtime_config.portfolio.rebalance_cadence == "weekly"
            else "vol-managed-weekly-base"
        ),
        cost_model_name="base",
        rebalance_cadence=(
            "biweekly" if runtime_config.portfolio.rebalance_cadence == "weekly" else "weekly"
        ),
        volatility_managed=True,
    )
    return [
        VariantSpec(
            name="raw-base",
            cost_model_name="base",
            rebalance_cadence=runtime_config.portfolio.rebalance_cadence,
            volatility_managed=False,
        ),
        VariantSpec(
            name="vol-managed-base",
            cost_model_name="base",
            rebalance_cadence=runtime_config.portfolio.rebalance_cadence,
            volatility_managed=True,
        ),
        VariantSpec(
            name="vol-managed-stress",
            cost_model_name="stress",
            rebalance_cadence=runtime_config.portfolio.rebalance_cadence,
            volatility_managed=True,
        ),
        cadence_sensitivity,
    ]


def build_rank_bands(top_candidates: list[str]) -> dict[str, set[str]]:
    return {
        "rank_1_20": set(top_candidates[:20]),
        "rank_1_30": set(top_candidates[:30]),
        "rank_1_50": set(top_candidates[:50]),
        "rank_21_30": set(top_candidates[20:30]),
        "rank_21_50": set(top_candidates[20:50]),
        "rank_31_50": set(top_candidates[30:50]),
    }


def _target_short_count(active_ids: list[str], short_bucket_frac: float) -> int:
    return max(1, round(len(active_ids) * short_bucket_frac))


def _baseline_long_short_sets(
    composite_scores: dict[str, float],
    *,
    long_bucket_frac: float,
    short_bucket_frac: float,
) -> tuple[list[str], list[str]]:
    return select_bucket_members(
        composite_scores,
        long_bucket_frac=long_bucket_frac,
        short_bucket_frac=short_bucket_frac,
    )


def _apply_short_weights(weights: dict[str, float], selected_shorts: list[str]) -> dict[str, float]:
    if selected_shorts:
        short_weight = -0.5 / len(selected_shorts)
        for instrument_id in selected_shorts:
            weights[instrument_id] = short_weight
    return weights


def _apply_short_tilt(
    weights: dict[str, float],
    base_shorts: list[str],
    *,
    band_set: set[str],
    tilt_multiplier: float,
) -> dict[str, float]:
    if not base_shorts:
        return weights
    base = dict.fromkeys(base_shorts, 1.0)
    for instrument_id in list(base):
        if instrument_id in band_set:
            base[instrument_id] *= tilt_multiplier
    gross = sum(base.values())
    for instrument_id, scale in base.items():
        weights[instrument_id] = -0.5 * (scale / gross)
    return weights


def conditioned_short_weights(
    composite_scores: dict[str, float],
    active_ids: list[str],
    *,
    band_set: set[str],
    mode: str,
    long_bucket_frac: float,
    short_bucket_frac: float,
    tilt_multiplier: float,
) -> dict[str, float]:
    longs, base_shorts = _baseline_long_short_sets(
        composite_scores,
        long_bucket_frac=long_bucket_frac,
        short_bucket_frac=short_bucket_frac,
    )
    weights = dict.fromkeys(composite_scores, 0.0)
    if longs:
        long_weight = 0.5 / len(longs)
        for instrument_id in longs:
            weights[instrument_id] = long_weight

    if mode == "baseline":
        return _apply_short_weights(weights, base_shorts)

    eligible_band = [instrument_id for instrument_id in active_ids if instrument_id in band_set]
    ordered_all = [instrument_id for instrument_id, _ in sorted(composite_scores.items(), key=lambda item: (item[1], item[0]))]
    target_short_count = _target_short_count(active_ids, short_bucket_frac)

    if mode == "hard_filter":
        selected_shorts = [instrument_id for instrument_id in ordered_all if instrument_id in eligible_band][:target_short_count]
        return _apply_short_weights(weights, selected_shorts)

    if mode == "post_rank_filter":
        selected_shorts = [instrument_id for instrument_id in base_shorts if instrument_id in band_set]
        return _apply_short_weights(weights, selected_shorts)

    if mode == "short_tilt":
        return _apply_short_tilt(
            weights,
            base_shorts,
            band_set=band_set,
            tilt_multiplier=tilt_multiplier,
        )

    raise ValueError(f"Unsupported combo mode: {mode}")


def compound_total_return(period_returns: list[float]) -> float:
    total = 1.0
    for value in period_returns:
        total *= 1.0 + value
    return total - 1.0


def compute_combo_baseline_metrics(
    *,
    close_frame: Any,
    index: Any,
    baseline_longs: list[str],
    baseline_shorts: list[str],
    variant: VariantSpec,
    config: Any,
    funding_loader: Any,
) -> dict[str, float]:
    step = bars_per_rebalance(index, variant.rebalance_cadence)
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

    price_only_net_returns: list[float] = []
    funding_aware_net_returns: list[float] = []
    first_interval = True
    previous_weights = dict.fromkeys(close_frame.columns, 0.0)

    for _signal_position, entry, end in iter_rebalance_windows(
        index,
        min_history_bars=config.signal.min_history_bars,
        step=step,
    ):
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
        funding_return = period_funding_return(
            weights,
            period_start=index[entry],
            period_end=index[end],
            funding_loader=funding_loader,
        )
        price_only_net_returns.append(metrics["net_return"])
        funding_aware_net_returns.append(metrics["net_return"] + funding_return)

    if not price_only_net_returns:
        raise ValueError("Legacy RV baseline helper did not produce any aligned portfolio intervals")

    price_only_net_return = compound_total_return(price_only_net_returns)
    funding_aware_net_return = compound_total_return(funding_aware_net_returns)
    return {
        "price_only_net_return": price_only_net_return,
        "price_only_max_drawdown": max_drawdown(price_only_net_returns),
        "net_return": funding_aware_net_return,
        "max_drawdown": max_drawdown(funding_aware_net_returns),
        "funding_return_lift": funding_aware_net_return - price_only_net_return,
    }


def simulate_combo_variant(
    *,
    runtime_config: Any,
    close_frame: Any,
    volume_frame: Any,
    index: Any,
    universe_ids: list[str],
    baseline_shorts: list[str],
    signal_spec: Any,
    variant: VariantSpec,
    band_name: str,
    band_set: set[str],
    mode: str,
    tilt_multiplier: float,
    funding_loader: Any,
    legacy_metrics: dict[str, Any],
) -> dict[str, Any]:
    step = bars_per_rebalance(index, variant.rebalance_cadence)
    previous_weights = dict.fromkeys(universe_ids, 0.0)
    price_only_net_returns: list[float] = []
    price_only_gross_returns: list[float] = []
    funding_aware_net_returns: list[float] = []
    funding_aware_gross_returns: list[float] = []

    for signal_position, entry, end in iter_rebalance_windows(
        index,
        min_history_bars=runtime_config.signal.min_history_bars,
        step=step,
    ):
        active_ids = select_active_instruments(
            close_frame,
            signal_position,
            min_history_bars=runtime_config.signal.min_history_bars,
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
        composite_scores = {
            instrument_id: payload["composite_score"]
            for instrument_id, payload in score_payload.items()
        }
        weights = conditioned_short_weights(
            composite_scores,
            active_ids,
            band_set=band_set,
            mode=mode,
            long_bucket_frac=runtime_config.portfolio.long_bucket_frac,
            short_bucket_frac=runtime_config.portfolio.short_bucket_frac,
            tilt_multiplier=tilt_multiplier,
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
            fee_bps=runtime_config.cost_models[variant.cost_model_name].fee_bps,
            slippage_bps=runtime_config.cost_models[variant.cost_model_name].slippage_bps,
            short_borrow_bps_per_day=(
                runtime_config.cost_models[variant.cost_model_name].short_borrow_bps_annual / 365.0
            ),
            period_days=period_days,
        )
        funding_return = period_funding_return(
            previous_weights,
            period_start=index[entry],
            period_end=index[end],
            funding_loader=funding_loader,
        )
        price_only_gross_returns.append(metrics["gross_return"])
        price_only_net_returns.append(metrics["net_return"])
        funding_aware_gross_returns.append(metrics["gross_return"] + funding_return)
        funding_aware_net_returns.append(metrics["net_return"] + funding_return)

    if not price_only_net_returns:
        raise ValueError("CTREND combo runner did not produce any portfolio intervals")

    price_only_net_return = compound_total_return(price_only_net_returns)
    price_only_gross_return = compound_total_return(price_only_gross_returns)
    funding_aware_net_return = compound_total_return(funding_aware_net_returns)
    funding_aware_gross_return = compound_total_return(funding_aware_gross_returns)

    baseline_metrics = compute_combo_baseline_metrics(
        close_frame=close_frame,
        index=index,
        baseline_longs=runtime_config.baseline_long_instrument_ids,
        baseline_shorts=baseline_shorts,
        variant=variant,
        config=runtime_config,
        funding_loader=funding_loader,
    )
    net_return = funding_aware_net_return
    gross_return = funding_aware_gross_return
    max_drawdown_value = max_drawdown(funding_aware_net_returns)
    active_short_count = len([weight for weight in previous_weights.values() if weight < 0])
    legacy_variant_name = matched_legacy_baseline_variant(variant)
    legacy_variant_metrics = legacy_metrics["variants"][legacy_variant_name]
    legacy_engine_net_return = float(legacy_variant_metrics["total_return_pct"]) / 100.0
    legacy_engine_max_drawdown = float(legacy_variant_metrics["max_drawdown_pct"]) / 100.0
    return {
        "variant_name": variant.name,
        "combo_mode": mode,
        "band_name": band_name,
        "rebalance_cadence": variant.rebalance_cadence,
        "cost_model_name": variant.cost_model_name,
        "volatility_managed": variant.volatility_managed,
        "gross_return": gross_return,
        "net_return": net_return,
        "max_drawdown": max_drawdown_value,
        "price_only_gross_return": price_only_gross_return,
        "price_only_net_return": price_only_net_return,
        "price_only_max_drawdown": max_drawdown(price_only_net_returns),
        "funding_return_lift": funding_aware_net_return - price_only_net_return,
        "baseline_net_return": baseline_metrics["net_return"],
        "baseline_max_drawdown": baseline_metrics["max_drawdown"],
        "baseline_price_only_net_return": baseline_metrics["price_only_net_return"],
        "baseline_price_only_max_drawdown": baseline_metrics["price_only_max_drawdown"],
        "baseline_funding_return_lift": baseline_metrics["funding_return_lift"],
        "legacy_rv_engine_baseline_variant": legacy_variant_name,
        "legacy_rv_engine_baseline_net_return": legacy_engine_net_return,
        "legacy_rv_engine_baseline_max_drawdown": legacy_engine_max_drawdown,
        "legacy_rv_engine_alpha": funding_aware_net_return - legacy_engine_net_return,
        "replace_mainline": evaluate_replacement(
            net_return=net_return,
            baseline_net_return=baseline_metrics["net_return"],
            max_drawdown_value=max_drawdown_value,
            baseline_max_drawdown=baseline_metrics["max_drawdown"],
        ),
        "active_short_count": active_short_count,
    }


def summarize_markdown(report_payload: dict[str, Any]) -> str:
    winner = report_payload["winner"]
    coverage = report_payload["coverage"]
    lines = [
        f"# CTREND x Liquidity/Reversal Conditional Q1 Report ({report_payload['funding_overlay_status'].title()} Funding Overlay)",
        "",
        f"- Report version: `{report_payload['report_version']}`",
        f"- Generated at: {report_payload['generated_at']}",
        f"- Window: {report_payload['window']['start']} to {report_payload['window']['end']}",
        f"- Universe size: {report_payload['universe']['instrument_count']}",
        f"- Funding overlay status: `{report_payload['funding_overlay_status']}`",
        f"- Funding source: {coverage['data_source']}",
        f"- Funding months loaded: {', '.join(coverage['loaded_months']) or 'none'}",
        f"- Funding coverage end: {coverage['coverage_end'] or 'none'}",
        f"- Winner combo: `{winner['variant_name']} | {winner['combo_mode']} | {winner['band_name']}`",
        f"- Winner net return: {winner['net_return']:.2%}",
        f"- Winner max drawdown: {winner['max_drawdown']:.2%}",
        f"- Winner price-only net return: {winner['price_only_net_return']:.2%}",
        f"- Winner funding lift: {winner['funding_return_lift']:.2%}",
        f"- Winner baseline net return: {winner['baseline_net_return']:.2%}",
        f"- Winner baseline max drawdown: {winner['baseline_max_drawdown']:.2%}",
        (
            "- Winner legacy RV engine baseline: "
            f"`{winner['legacy_rv_engine_baseline_variant']}` / "
            f"{winner['legacy_rv_engine_baseline_net_return']:.2%} / "
            f"{winner['legacy_rv_engine_baseline_max_drawdown']:.2%}"
        ),
        "",
        "## Focus Post-Rank Filter",
        "",
    ]
    for item in report_payload["focus_post_rank_results"]:
        lines.extend(
            [
                f"### {item['variant_name']} | {item['band_name']}",
                f"- Funding-aware net return: {item['net_return']:.2%}",
                f"- Max drawdown: {item['max_drawdown']:.2%}",
                f"- Price-only net return: {item['price_only_net_return']:.2%}",
                f"- Apples-to-apples legacy basket baseline: {item['baseline_net_return']:.2%}",
                "",
            ],
        )
    lines.extend(
        [
        "## Top Combos",
        "",
        ],
    )
    for item in report_payload["top_combos"]:
        lines.extend(
            [
                f"### {item['variant_name']} | {item['combo_mode']} | {item['band_name']}",
                f"- Net return: {item['net_return']:.2%}",
                f"- Max drawdown: {item['max_drawdown']:.2%}",
                f"- Funding lift vs price-only: {item['funding_return_lift']:.2%}",
                f"- Active short count: {item['active_short_count']}",
                "",
            ],
        )
    lines.extend(
        [
            "## Caveats",
            "",
        ]
        + [f"- {note}" for note in report_payload["notes"]]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_combo_config(config_path)
    real_public_config = load_real_public_config(config.real_public_config_path)
    runtime_config = build_runtime_config(
        real_public_config,
        load_real_public_universe(real_public_config.selected_universe_path, real_public_config.top_n_candidates)[0],
    )
    baseline_longs, top_candidates, baseline_shorts = load_real_public_universe(
        real_public_config.selected_universe_path,
        real_public_config.top_n_candidates,
    )
    universe_ids = baseline_longs + [instrument_id for instrument_id in top_candidates if instrument_id not in baseline_longs]
    close_frame, volume_frame, _raw_symbols = load_catalog_frames(real_public_config.catalog_path, universe_ids)
    band_sets = build_rank_bands(top_candidates)
    variants = build_variants(runtime_config)
    legacy_metrics = load_json(real_public_config.baseline_metrics_path)
    funding_loader = BinanceVisionFundingLoader(
        start_ms=timestamp_ms(close_frame.index[0]),
        end_ms=timestamp_ms(close_frame.index[-1]),
    )
    coverage = funding_coverage_summary(funding_loader, list(close_frame.columns))

    results: list[dict[str, Any]] = []
    for variant in variants:
        results.append(
            simulate_combo_variant(
                runtime_config=runtime_config,
                close_frame=close_frame,
                volume_frame=volume_frame,
                index=close_frame.index,
                universe_ids=list(close_frame.columns),
                baseline_shorts=baseline_shorts,
                signal_spec=real_public_config.signal,
                variant=variant,
                band_name="none",
                band_set=set(),
                mode="baseline",
                tilt_multiplier=config.tilt_multiplier,
                funding_loader=funding_loader,
                legacy_metrics=legacy_metrics,
            ),
        )
        for mode in COMBO_MODES[1:]:
            for band_name in BAND_NAMES:
                results.append(
                    simulate_combo_variant(
                        runtime_config=runtime_config,
                        close_frame=close_frame,
                        volume_frame=volume_frame,
                        index=close_frame.index,
                        universe_ids=list(close_frame.columns),
                        baseline_shorts=baseline_shorts,
                        signal_spec=real_public_config.signal,
                        variant=variant,
                        band_name=band_name,
                        band_set=band_sets[band_name],
                        mode=mode,
                        tilt_multiplier=config.tilt_multiplier,
                        funding_loader=funding_loader,
                        legacy_metrics=legacy_metrics,
                    ),
                )

    winner = max(results, key=lambda item: (item["net_return"], item["max_drawdown"]))
    top_combos = sorted(results, key=lambda item: item["net_return"], reverse=True)[:15]
    focus_post_rank_results = sorted(
        [
            item
            for item in results
            if item["combo_mode"] == "post_rank_filter" and item["band_name"] in FOCUS_BAND_NAMES
        ],
        key=lambda item: item["net_return"],
        reverse=True,
    )
    liq_report = load_json(config.liquidity_reversal_report_path)
    funding_overlay_status = (
        "complete"
        if coverage.loaded_month_count == coverage.requested_month_count
        else "partial"
    )
    report_payload = {
        "report_version": (
            "funding-aware-v1"
            if funding_overlay_status == "complete"
            else "partial-funding-overlay-v1"
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
        "funding_overlay_status": funding_overlay_status,
        "window": legacy_metrics["window"],
        "universe": {
            "instrument_count": len(close_frame.columns),
            "top_candidate_count": len(top_candidates),
            "band_names": list(BAND_NAMES),
            "combo_modes": list(COMBO_MODES),
            "focus_band_names": list(FOCUS_BAND_NAMES),
        },
        "coverage": {
            "requested_months": list(coverage.requested_months),
            "loaded_months": list(coverage.loaded_months),
            "loaded_month_count": coverage.loaded_month_count,
            "requested_month_count": coverage.requested_month_count,
            "data_source": coverage.data_source,
            "coverage_end": coverage.coverage_end,
            "full_window_complete": funding_overlay_status == "complete",
            "symbols_with_any_data": coverage.symbols_with_any_data,
            "symbols_missing_all_data": coverage.symbols_missing_all_data,
        },
        "winner": winner,
        "top_summary": {
            "winner_combo": f"{winner['variant_name']} | {winner['combo_mode']} | {winner['band_name']}",
            "winner_net_return": winner["net_return"],
            "winner_max_drawdown": winner["max_drawdown"],
            "winner_apples_to_apples_alpha": winner["net_return"] - winner["baseline_net_return"],
            "winner_legacy_rv_engine_alpha": winner["legacy_rv_engine_alpha"],
        },
        "top_combos": top_combos,
        "focus_post_rank_results": focus_post_rank_results,
        "all_results": results,
        "liquidity_reversal_context": {
            "winner": liq_report["winner"],
            "winner_proxy_stress_return_pct": liq_report["winner_metrics"]["proxy_stress_return_pct"],
            "baseline_short50_proxy_base_return_pct": liq_report["baseline_short50"]["proxy_base_return_pct"],
            "baseline_short50_proxy_stress_return_pct": liq_report["baseline_short50"]["proxy_stress_return_pct"],
        },
        "legacy_rv_context": {
            "primary_base": legacy_metrics["variants"]["primary_base"],
            "primary_stress": legacy_metrics["variants"]["primary_stress"],
            "primary_base_biweekly": legacy_metrics["variants"]["primary_base_biweekly"],
        },
        "notes": [
            "This report enumerates the main short-side conditional combination families between CTREND and liquidity/reversal.",
            "Funding-aware net returns are computed by applying monthly Binance funding-rate events to the held CTREND combo weights after each rebalance decision.",
            "Funding is a report-layer overlay only; it is not fed back into the ranking or rebalance-time signal construction.",
            "The apples-to-apples baseline now uses the same CTREND warm-up gate, rebalance windows, and funding overlay as the combo variants.",
            "The preserved legacy RV engine variants remain in the report as a separate control because their fixed-notional engine accounting does not exactly match the combo return-space reconstruction.",
            "This report is a partial funding overlay when the requested Q1 window extends beyond the funding archive coverage end.",
            "Binance monthly funding archives are only loaded for the months that exist under the requested window; coverage therefore ends at the last observed funding event in the archive set.",
            "Hard filter ranks only within the band; post-rank filter intersects with the CTREND short bucket; short-tilt keeps CTREND shorts but overweights names inside the target band.",
        ],
    }

    write_json(config.output_json, report_payload)
    config.output_md.write_text(summarize_markdown(report_payload), encoding="utf-8")
    print(f"Wrote CTREND x liquidity/reversal combo report JSON to {config.output_json}")
    print(f"Wrote CTREND x liquidity/reversal combo report Markdown to {config.output_md}")


if __name__ == "__main__":
    main()
