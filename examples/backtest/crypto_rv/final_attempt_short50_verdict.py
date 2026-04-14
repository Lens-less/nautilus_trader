#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any


try:
    from .common import ensure_directory
    from .common import load_json
    from .common import write_json
except ImportError:  # pragma: no cover - script execution fallback
    from common import ensure_directory
    from common import load_json
    from common import write_json

from nautilus_trader.backtest.config import BacktestDataConfig
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.config import BacktestRunConfig
from nautilus_trader.backtest.config import BacktestVenueConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog


LONG_LEG = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "HYPEUSDT"]
LEG_NOTIONAL_USD = 500_000.0
BASE_COST_BPS = 12.0
STRESS_COST_BPS = 33.0
STRESS_BORROW_BPS_ANNUAL = 300.0
FOLLOWUPS_CATALOG = "followups/catalog/catalog_50"
DATA_MONTHS = ("2026-01", "2026-02", "2026-03")


@dataclass(slots=True)
class CandidateVerdict:
    name: str
    count: int
    pairs: list[str]
    engine_gross_return_pct: float
    engine_gross_sharpe_252: float
    short_pnl_usdt: float
    funding_models: dict[str, dict[str, float]]
    base_return_proxy_pct: float
    stress_return_proxy_pct: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GridBenchmark:
    instrument: str
    assumptions: dict[str, Any]
    best_config: dict[str, Any]
    median_return_pct: float
    buy_and_hold_full_pct: float
    buy_and_hold_50_50_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final keep-or-kill short50 verdict.")
    parser.add_argument(
        "--research-root",
        default="output/real_public_2026q1",
        help="Path to the real_public_2026q1 artifact root, relative to this script.",
    )
    parser.add_argument(
        "--output-json",
        default="output/real_public_2026q1/edge_search/final_attempt_short50_verdict.json",
        help="Path to write the machine-readable verdict.",
    )
    parser.add_argument(
        "--output-md",
        default="output/real_public_2026q1/edge_search/final_attempt_short50_verdict.md",
        help="Path to write the markdown verdict.",
    )
    return parser


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _round_trip_cost(turnover_usdt: float, cost_bps: float) -> float:
    return turnover_usdt * (cost_bps / 10_000.0)


def _borrow_cost(notional_days: float) -> float:
    return notional_days * (STRESS_BORROW_BPS_ANNUAL / 10_000.0) / 365.0


def _load_ranked_top50(research_root: Path) -> list[str]:
    approx_payload = load_json(research_root / "selected_universe.json")
    true_pit_payload = load_json(research_root / "selected_universe_true_pit_80.json")
    approx_top50 = [item["pair"] for item in approx_payload["all_ranked_candidates"][:50]]
    true_top50 = [item["pair"] for item in true_pit_payload["selected_true_pit_50"]]
    if approx_top50 != true_top50:
        raise ValueError("Approx and true-PIT top50 differ; final verdict cannot use trusted decomposition.")
    return true_top50


def _known_short50_metrics(research_root: Path) -> dict[str, float]:
    payload = load_json(research_root / "followups/followup_metrics.json")["short50"]
    return {
        "funding_pnl_usdt": float(payload["base"]["funding_pnl_usdt"]),
        "base_return_pct": float(payload["base"]["total_return_pct"]),
        "stress_return_pct": float(payload["stress"]["total_return_pct"]),
    }


def _load_price_matrix(catalog: ParquetDataCatalog, symbols: list[str]) -> dict[str, list[float]]:
    matrix: dict[str, list[float]] = {}
    for symbol in symbols:
        bars = catalog.bars([BarType.from_str(f"{symbol}-PERP.BINANCE-1-DAY-LAST-EXTERNAL")])
        matrix[symbol] = [float(bar.close) for bar in bars]
    return matrix


def _simulate_leg(
    price_matrix: dict[str, list[float]],
    symbols: list[str],
    *,
    is_long: bool,
    rebalance_every: int = 7,
) -> tuple[float, float, float]:
    quantities = {symbol: 0.0 for symbol in symbols}
    cash = 0.0
    turnover = 0.0
    notional_days = 0.0
    n_bars = len(price_matrix[symbols[0]])

    for index in range(n_bars):
        prices = {symbol: price_matrix[symbol][index] for symbol in symbols}
        if index % rebalance_every == 0:
            per_symbol_notional = LEG_NOTIONAL_USD / len(symbols)
            for symbol in symbols:
                target_quantity = per_symbol_notional / prices[symbol]
                if not is_long:
                    target_quantity *= -1.0
                delta_quantity = target_quantity - quantities[symbol]
                cash -= delta_quantity * prices[symbol]
                turnover += abs(delta_quantity) * prices[symbol]
                quantities[symbol] = target_quantity

        position_value = sum(quantities[symbol] * prices[symbol] for symbol in symbols)
        if index < n_bars - 1:
            notional_days += sum(abs(quantities[symbol] * prices[symbol]) for symbol in symbols)

    return cash + position_value, turnover, notional_days


def _make_run_config_with_longs(
    research_root: Path,
    long_symbols: list[str],
    band_symbols: list[str],
) -> BacktestRunConfig:
    long_ids = [f"{symbol}-PERP.BINANCE" for symbol in long_symbols]
    short_ids = [f"{symbol}-PERP.BINANCE" for symbol in band_symbols]
    all_ids = long_ids + short_ids

    strategy_config = ImportableStrategyConfig(
        strategy_path="nautilus_trader.examples.strategies.crypto_rv_basket:CryptoRVBasketStrategy",
        config_path="nautilus_trader.examples.strategies.crypto_rv_basket:CryptoRVBasketConfig",
        config={
            "long_instrument_ids": long_ids,
            "short_instrument_ids": short_ids,
            "bar_types": [f"{instrument_id}-1-DAY-LAST-EXTERNAL" for instrument_id in all_ids],
            "leg_notional_usd": LEG_NOTIONAL_USD,
            "rebalance_cadence": "weekly",
            "excluded_instrument_ids": [],
            "min_order_notional_usd": 25.0,
            "long_fee_bps": 4.0,
            "short_fee_bps": 4.0,
            "long_slippage_bps": 8.0,
            "short_slippage_bps": 8.0,
            "long_funding_bps_per_day": 0.0,
            "short_funding_bps_per_day": 0.0,
            "short_borrow_bps_per_day": 0.0,
        },
    )

    catalog_path = str(research_root / FOLLOWUPS_CATALOG)
    data_configs = [
        BacktestDataConfig(
            catalog_path=catalog_path,
            data_cls="nautilus_trader.model.data:Bar",
            instrument_id=instrument_id,
            start_time="2026-01-13T00:00:00Z",
            end_time="2026-04-13T00:00:00Z",
            bar_spec="1-DAY-LAST-EXTERNAL",
        )
        for instrument_id in all_ids
    ]

    return BacktestRunConfig(
        engine=BacktestEngineConfig(
            strategies=[strategy_config],
            logging=LoggingConfig(log_level="ERROR"),
            run_analysis=True,
        ),
        venues=[
            BacktestVenueConfig(
                name="BINANCE",
                oms_type="NETTING",
                account_type="MARGIN",
                base_currency="USDT",
                starting_balances=["1000000 USDT"],
            ),
        ],
        data=data_configs,
        start="2026-01-13T00:00:00Z",
        end="2026-04-13T00:00:00Z",
    )


def _make_run_config(research_root: Path, band_symbols: list[str]) -> BacktestRunConfig:
    return _make_run_config_with_longs(research_root, LONG_LEG, band_symbols)


def run_simple_grid_benchmark(research_root: Path) -> GridBenchmark:
    catalog = ParquetDataCatalog(research_root / FOLLOWUPS_CATALOG)
    bars = catalog.bars([BarType.from_str("BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL")])
    closes = [float(bar.close) for bar in bars]
    initial_capital = 1_000_000.0
    cost_rate = BASE_COST_BPS / 10_000.0

    def simulate(step_pct: float, tranche_frac: float) -> dict[str, Any]:
        trade_notional = initial_capital * tranche_frac
        cash = initial_capital / 2.0
        btc_qty = (initial_capital / 2.0) / closes[0]
        reference_price = closes[0]
        turnover = 0.0
        trades = 0

        for close in closes[1:]:
            while close <= reference_price * (1.0 - step_pct) and cash > trade_notional * (1.0 + cost_rate):
                execution_price = reference_price * (1.0 - step_pct)
                quantity = trade_notional / execution_price
                cash -= trade_notional * (1.0 + cost_rate)
                btc_qty += quantity
                turnover += trade_notional
                trades += 1
                reference_price = execution_price

            while close >= reference_price * (1.0 + step_pct) and btc_qty * (reference_price * (1.0 + step_pct)) >= trade_notional:
                execution_price = reference_price * (1.0 + step_pct)
                quantity = trade_notional / execution_price
                btc_qty -= quantity
                cash += trade_notional * (1.0 - cost_rate)
                turnover += trade_notional
                trades += 1
                reference_price = execution_price

        ending_equity = cash + btc_qty * closes[-1]
        return {
            "step_pct": step_pct,
            "tranche_frac": tranche_frac,
            "trades": trades,
            "turnover_usdt": turnover,
            "ending_equity_usdt": ending_equity,
            "return_pct": (ending_equity / initial_capital - 1.0) * 100.0,
        }

    configs = [
        simulate(step_pct=step_pct, tranche_frac=tranche_frac)
        for step_pct in (0.01, 0.015, 0.02, 0.03, 0.05, 0.08)
        for tranche_frac in (0.05, 0.10, 0.15, 0.20)
    ]
    configs.sort(key=lambda item: item["return_pct"], reverse=True)
    best = configs[0]
    buy_hold_full_pct = (closes[-1] / closes[0] - 1.0) * 100.0
    buy_hold_50_50_pct = ((0.5 + 0.5 * closes[-1] / closes[0]) - 1.0) * 100.0
    median_return_pct = mean(sorted(item["return_pct"] for item in configs)[11:13])

    return GridBenchmark(
        instrument="BTCUSDT-PERP.BINANCE",
        assumptions={
            "window": {"start": "2026-01-13", "end": "2026-04-12"},
            "model": "single-asset close-based spot-style grid",
            "initial_capital_usdt": initial_capital,
            "starting_mix": "50% cash / 50% BTC",
            "trade_cost_bps_per_fill": BASE_COST_BPS,
            "step_grid": [0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
            "tranche_grid": [0.05, 0.10, 0.15, 0.20],
        },
        best_config=best,
        median_return_pct=median_return_pct,
        buy_and_hold_full_pct=buy_hold_full_pct,
        buy_and_hold_50_50_pct=buy_hold_50_50_pct,
    )


def _parse_month_start_ms(month: str) -> int:
    year, month_num = month.split("-")
    return int(datetime(int(year), int(month_num), 1, tzinfo=timezone.utc).timestamp() * 1000)


class FundingLoader:
    def __init__(self, research_root: Path, ranked_top50: list[str]) -> None:
        self._research_root = research_root
        self._ranked_top50 = ranked_top50
        self._catalog = ParquetDataCatalog(research_root / FOLLOWUPS_CATALOG)
        self._start_ms = int(datetime(2026, 1, 13, tzinfo=timezone.utc).timestamp() * 1000)
        self._end_ms = int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)

    @staticmethod
    def _binance_url(kind: str, symbol: str, month: str, interval: str | None = None) -> str:
        if kind == "fundingRate":
            return (
                "https://data.binance.vision/data/futures/um/monthly/"
                f"fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
            )
        if interval is None:
            raise ValueError("interval is required for markPriceKlines")
        return (
            "https://data.binance.vision/data/futures/um/monthly/"
            f"{kind}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
        )

    @lru_cache(maxsize=None)
    def load_zip_csv(
        self,
        kind: str,
        symbol: str,
        month: str,
        interval: str | None = None,
    ) -> list[list[str]] | None:
        url = self._binance_url(kind, symbol, month, interval)
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
        except Exception:
            return None
        archive = zipfile.ZipFile(io.BytesIO(data))
        return list(csv.reader(io.StringIO(archive.read(archive.namelist()[0]).decode("utf-8"))))

    @lru_cache(maxsize=None)
    def funding_events(
        self,
        symbol: str,
        *,
        impute_missing: bool,
    ) -> tuple[tuple[int, int, float], ...]:
        events: list[tuple[int, int, float]] = []
        observed_rates: list[float] = []
        observed_hours: list[int] = []

        for month in DATA_MONTHS:
            rows = self.load_zip_csv("fundingRate", symbol, month)
            if rows is None:
                if impute_missing and observed_rates and observed_hours:
                    dominant_hours = Counter(observed_hours).most_common(1)[0][0]
                    interval_ms = dominant_hours * 60 * 60 * 1000
                    avg_rate = mean(observed_rates)
                    month_start_ms = _parse_month_start_ms(month)
                    next_month_ms = (
                        int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)
                        if month == "2026-03"
                        else self._end_ms
                    )
                    current = max(self._start_ms, month_start_ms)
                    while current < min(self._end_ms, next_month_ms):
                        events.append((current, dominant_hours, avg_rate))
                        current += interval_ms
                continue

            for row in rows[1:]:
                calc_time = int(row[0])
                interval_hours = int(row[1])
                rate = float(row[2])
                interval_ms = interval_hours * 60 * 60 * 1000
                bucket = (calc_time // interval_ms) * interval_ms
                if self._start_ms <= bucket < self._end_ms:
                    events.append((bucket, interval_hours, rate))
                    observed_rates.append(rate)
                    observed_hours.append(interval_hours)

        return tuple(events)

    @lru_cache(maxsize=None)
    def mark_map(self, symbol: str, interval_hours: int) -> dict[int, float]:
        interval = f"{interval_hours}h"
        marks: dict[int, float] = {}
        for month in DATA_MONTHS:
            rows = self.load_zip_csv("markPriceKlines", symbol, month, interval=interval)
            if rows is None:
                continue
            for row in rows[1:]:
                open_time = int(row[0])
                if self._start_ms <= open_time < self._end_ms:
                    marks[open_time] = float(row[4])
        return marks

    @lru_cache(maxsize=None)
    def quantity_schedule(self, symbols_key: tuple[str, ...]) -> tuple[tuple[int, dict[str, float]], ...]:
        bars_map: dict[str, list[tuple[int, float]]] = {}
        symbols = list(symbols_key)
        for symbol in symbols:
            bars = self._catalog.bars([BarType.from_str(f"{symbol}-PERP.BINANCE-1-DAY-LAST-EXTERNAL")])
            bars_map[symbol] = [(int(bar.ts_event / 1_000_000), float(bar.close)) for bar in bars]

        schedule: list[tuple[int, dict[str, float]]] = []
        for index, (ts_event_ms, _) in enumerate(bars_map[symbols[0]]):
            if ts_event_ms >= self._end_ms:
                break
            if index % 7 == 0:
                per_symbol_notional = LEG_NOTIONAL_USD / len(symbols)
                schedule.append(
                    (
                        ts_event_ms,
                        {
                            symbol: per_symbol_notional / bars_map[symbol][index][1]
                            for symbol in symbols
                        },
                    ),
                )
        return tuple(schedule)

    @staticmethod
    def quantity_at(
        schedule: tuple[tuple[int, dict[str, float]], ...],
        symbol: str,
        event_ms: int,
    ) -> float:
        quantity = 0.0
        for ts_event_ms, basket in schedule:
            if ts_event_ms <= event_ms:
                quantity = basket[symbol]
            else:
                break
        return quantity

    def funding_by_model(
        self,
        symbols: list[str],
        *,
        impute_missing: bool,
        constant_notional: bool,
    ) -> float:
        total = 0.0
        long_schedule = self.quantity_schedule(tuple(LONG_LEG))
        short_schedule = self.quantity_schedule(tuple(symbols))
        long_constant = LEG_NOTIONAL_USD / len(LONG_LEG)
        short_constant = LEG_NOTIONAL_USD / len(symbols)

        for symbol in LONG_LEG:
            for event_ms, interval_hours, rate in self.funding_events(symbol, impute_missing=impute_missing):
                price = self.mark_map(symbol, interval_hours).get(event_ms)
                if price is None:
                    continue
                notional = long_constant if constant_notional else self.quantity_at(long_schedule, symbol, event_ms) * price
                total += -notional * rate

        for symbol in symbols:
            for event_ms, interval_hours, rate in self.funding_events(symbol, impute_missing=impute_missing):
                price = self.mark_map(symbol, interval_hours).get(event_ms)
                if price is None:
                    continue
                notional = short_constant if constant_notional else self.quantity_at(short_schedule, symbol, event_ms) * price
                total += notional * rate

        return total

    def missing_funding_symbols(self, symbols: list[str]) -> list[str]:
        missing: list[str] = []
        for symbol in symbols:
            for month in DATA_MONTHS:
                if self.load_zip_csv("fundingRate", symbol, month) is None:
                    missing.append(symbol)
                    break
        return sorted(set(missing))


def run_verdict(research_root: Path) -> dict[str, Any]:
    ranked_top50 = _load_ranked_top50(research_root)
    short50_known = _known_short50_metrics(research_root)
    price_catalog = ParquetDataCatalog(research_root / FOLLOWUPS_CATALOG)
    price_matrix = _load_price_matrix(price_catalog, list(dict.fromkeys(LONG_LEG + ranked_top50)))
    long_pnl, long_turnover, _ = _simulate_leg(price_matrix, LONG_LEG, is_long=True)

    candidates = {
        "rank_1_50": ranked_top50[:50],
        "rank_21_50": ranked_top50[20:50],
        "rank_31_50": ranked_top50[30:50],
    }

    run_order = [
        ("rank_1_50", _make_run_config(research_root, candidates["rank_1_50"])),
        ("rank_21_50", _make_run_config(research_root, candidates["rank_21_50"])),
        ("rank_31_50", _make_run_config(research_root, candidates["rank_31_50"])),
        (
            "rank_31_50_hype_out",
            _make_run_config_with_longs(
                research_root,
                ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
                candidates["rank_31_50"],
            ),
        ),
    ]
    node = BacktestNode(configs=[config for _, config in run_order])
    engine_results = {
        name: result
        for (name, _), result in zip(run_order, node.run(), strict=True)
    }
    hype_out_run = engine_results["rank_31_50_hype_out"]

    funding_loader = FundingLoader(research_root, ranked_top50)
    funding_models = {
        "raw_position_net_drop_missing": {"impute_missing": False, "constant_notional": False},
        "raw_position_net_impute_missing": {"impute_missing": True, "constant_notional": False},
        "raw_constant_notional_drop_missing": {"impute_missing": False, "constant_notional": True},
        "raw_constant_notional_impute_missing": {"impute_missing": True, "constant_notional": True},
    }

    baseline_raw = {
        name: funding_loader.funding_by_model(
            candidates["rank_1_50"],
            impute_missing=params["impute_missing"],
            constant_notional=params["constant_notional"],
        )
        for name, params in funding_models.items()
    }

    verdicts: list[CandidateVerdict] = []
    for name, pairs in candidates.items():
        short_pnl, short_turnover, short_notional_days = _simulate_leg(price_matrix, pairs, is_long=False)
        trade_base = _round_trip_cost(long_turnover + short_turnover, BASE_COST_BPS)
        trade_stress = _round_trip_cost(long_turnover + short_turnover, STRESS_COST_BPS)
        borrow_stress = _borrow_cost(short_notional_days)

        model_payload: dict[str, dict[str, float]] = {}
        scaled_base_values: list[float] = []
        scaled_stress_values: list[float] = []
        for model_name, params in funding_models.items():
            raw_value = funding_loader.funding_by_model(
                pairs,
                impute_missing=params["impute_missing"],
                constant_notional=params["constant_notional"],
            )
            raw_baseline = baseline_raw[model_name]
            scale = short50_known["funding_pnl_usdt"] / raw_baseline if raw_baseline else 1.0
            scaled_funding = raw_value * scale
            base_return_pct = (long_pnl + short_pnl + scaled_funding - trade_base) / 10_000.0
            stress_return_pct = (
                long_pnl + short_pnl + scaled_funding - trade_stress - borrow_stress
            ) / 10_000.0
            scaled_base_values.append(base_return_pct)
            scaled_stress_values.append(stress_return_pct)
            model_payload[model_name] = {
                "raw_funding_usdt_jan_to_mar": raw_value,
                "scaled_funding_usdt": scaled_funding,
                "scale_to_known_short50": scale,
                "base_return_pct": base_return_pct,
                "stress_return_pct": stress_return_pct,
            }

        verdicts.append(
            CandidateVerdict(
                name=name,
                count=len(pairs),
                pairs=pairs,
                engine_gross_return_pct=float(engine_results[name].stats_pnls["USDT"]["PnL% (total)"]),
                engine_gross_sharpe_252=float(engine_results[name].stats_returns["Sharpe Ratio (252 days)"]),
                short_pnl_usdt=short_pnl,
                funding_models=model_payload,
                base_return_proxy_pct=min(scaled_base_values),
                stress_return_proxy_pct=min(scaled_stress_values),
                notes=[
                    (
                        "base_return_proxy_pct / stress_return_proxy_pct are the conservative minima "
                        "across the four raw-funding calibration models"
                    ),
                ],
            ),
        )

    baseline = next(item for item in verdicts if item.name == "rank_1_50")
    rank21 = next(item for item in verdicts if item.name == "rank_21_50")
    rank31 = next(item for item in verdicts if item.name == "rank_31_50")

    rank31_wins_all_models = all(
        rank31.funding_models[model_name]["stress_return_pct"]
        > baseline.funding_models[model_name]["stress_return_pct"]
        for model_name in funding_models
    )
    rank21_wins_all_models = all(
        rank21.funding_models[model_name]["stress_return_pct"]
        > baseline.funding_models[model_name]["stress_return_pct"]
        for model_name in funding_models
    )

    keep_testing = rank31_wins_all_models and rank31.engine_gross_return_pct > baseline.engine_gross_return_pct
    grid_benchmark = run_simple_grid_benchmark(research_root)
    live_ready = False
    summary = {
        "keep_testing": keep_testing,
        "live_trading_ready": live_ready,
        "live_trading_decision": "NO",
        "winner_under_all_models": "rank_31_50" if rank31_wins_all_models else "none",
        "rank_21_50_wins_all_models": rank21_wins_all_models,
        "baseline_short50_known_funding_pnl_usdt": short50_known["funding_pnl_usdt"],
        "missing_funding_symbols_top50": funding_loader.missing_funding_symbols(ranked_top50),
        "rank_31_50_vs_best_simple_grid_return_pct_gap": (
            rank31.stress_return_proxy_pct - grid_benchmark.best_config["return_pct"]
        ),
        "rank_31_50_hype_out_engine_gross_return_pct": float(
            hype_out_run.stats_pnls["USDT"]["PnL% (total)"],
        ),
        "rank_31_50_hype_out_engine_gross_sharpe_252": float(
            hype_out_run.stats_returns["Sharpe Ratio (252 days)"],
        ),
        "hard_stop_datetime_utc": "2026-06-30T23:59:59Z",
        "earliest_live_datetime_utc_if_all_gates_pass": "2026-07-01T00:00:00Z",
        "required_before_live": [
            "two additional out-of-sample windows with raw funding and positive stress",
            "rank_31_50 must survive HYPE-out or an explicitly accepted HYPE dependency",
            "live borrow / slippage / execution checks on target venue",
        ],
    }
    return {
        "window": {
            "start": "2026-01-13",
            "end": "2026-04-12",
            "raw_funding_data_end": "2026-03-31",
        },
        "summary": summary,
        "models": list(funding_models.keys()),
        "candidates": [item.to_dict() for item in verdicts],
        "simple_grid_benchmark": grid_benchmark.to_dict(),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    by_name = {item["name"]: item for item in payload["candidates"]}
    baseline = by_name["rank_1_50"]
    rank21 = by_name["rank_21_50"]
    rank31 = by_name["rank_31_50"]
    grid = payload["simple_grid_benchmark"]
    lines = [
        "# Final Short50 Verdict",
        "",
        f"- Window: {payload['window']['start']} to {payload['window']['end']}",
        f"- Raw funding data coverage used here ends at {payload['window']['raw_funding_data_end']}.",
        (
            f"- Final verdict: `{'KEEP TESTING' if summary['keep_testing'] else 'KILL'}` "
            "based on engine gross plus four raw-funding calibration models."
        ),
        f"- Live trading decision right now: `{summary['live_trading_decision']}`.",
        f"- Hard stop: `{summary['hard_stop_datetime_utc']}`. Earliest possible live date, if every gate passes: `{summary['earliest_live_datetime_utc_if_all_gates_pass']}`.",
        "",
        "| Candidate | Engine Gross | Worst-Case Base | Worst-Case Stress |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| rank_1_50 | {baseline['engine_gross_return_pct']:.2f}% | "
            f"{baseline['base_return_proxy_pct']:.2f}% | {baseline['stress_return_proxy_pct']:.2f}% |"
        ),
        (
            f"| rank_21_50 | {rank21['engine_gross_return_pct']:.2f}% | "
            f"{rank21['base_return_proxy_pct']:.2f}% | {rank21['stress_return_proxy_pct']:.2f}% |"
        ),
        (
            f"| rank_31_50 | {rank31['engine_gross_return_pct']:.2f}% | "
            f"{rank31['base_return_proxy_pct']:.2f}% | {rank31['stress_return_proxy_pct']:.2f}% |"
        ),
        "",
        "## Simple Grid Benchmark",
        (
            f"- Instrument: `{grid['instrument']}`; best close-based simple grid return "
            f"{grid['best_config']['return_pct']:.2f}% "
            f"(step={grid['best_config']['step_pct']:.3f}, tranche={grid['best_config']['tranche_frac']:.2f})."
        ),
        (
            f"- Median simple-grid return across the sweep: {grid['median_return_pct']:.2f}%. "
            f"BTC buy-and-hold full notional: {grid['buy_and_hold_full_pct']:.2f}%. "
            f"BTC 50/50 hold: {grid['buy_and_hold_50_50_pct']:.2f}%."
        ),
        (
            f"- `rank_31_50` worst-case stress beats the best simple-grid run by "
            f"{summary['rank_31_50_vs_best_simple_grid_return_pct_gap']:.2f} percentage points."
        ),
        "",
        "## Read-Through",
        (
            f"- `rank_31_50` beats baseline `rank_1_50` under all raw-funding models: "
            f"{summary['winner_under_all_models'] == 'rank_31_50'}."
        ),
        (
            f"- `rank_21_50` beats baseline under all raw-funding models: "
            f"{summary['rank_21_50_wins_all_models']}."
        ),
        (
            f"- Missing raw funding inside top50 is limited to: "
            f"{', '.join(summary['missing_funding_symbols_top50']) or 'none'}."
        ),
        (
            f"- `rank_31_50` fails the immediate live gate on HYPE dependence: HYPE-out engine gross = "
            f"{summary['rank_31_50_hype_out_engine_gross_return_pct']:.2f}%."
        ),
        (
            "- If you only keep one line alive, keep `rank_31_50`. "
            "If that one fails in additional windows, this thesis is done."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    research_root = _resolve(base_dir, args.research_root)
    output_json = _resolve(base_dir, args.output_json)
    output_md = _resolve(base_dir, args.output_md)

    payload = run_verdict(research_root)
    ensure_directory(output_json.parent)
    write_json(output_json, payload)
    output_md.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Wrote final verdict JSON to {output_json}")
    print(f"Wrote final verdict markdown to {output_md}")


if __name__ == "__main__":
    main()
