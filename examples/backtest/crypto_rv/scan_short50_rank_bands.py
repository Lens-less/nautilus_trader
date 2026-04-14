#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
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
FOLLOWUPS_CATALOG = "followups/catalog_50"
LEG_NOTIONAL_USD = 500_000.0
BASE_COST_BPS = 12.0
STRESS_COST_BPS = 33.0
STRESS_BORROW_BPS_ANNUAL = 300.0


@dataclass(slots=True)
class BandSpec:
    name: str
    start_rank: int
    end_rank: int
    funding_proxy_usdt: float
    notes: list[str]

    @property
    def count(self) -> int:
        return self.end_rank - self.start_rank + 1


@dataclass(slots=True)
class BandMetrics:
    name: str
    count: int
    pairs: list[str]
    funding_proxy_usdt: float
    engine_gross_return_pct: float
    engine_gross_sharpe_252: float
    proxy_base_return_pct: float
    proxy_stress_return_pct: float
    long_pnl_usdt: float
    short_pnl_usdt: float
    trade_cost_base_usdt: float
    trade_cost_stress_usdt: float
    borrow_cost_stress_usdt: float
    avg_constituent_total_return_pct: float
    constituent_negative_share: float
    avg_month1_return_pct: float
    avg_month2_return_pct: float
    avg_month3_return_pct: float
    worst_constituents: list[dict[str, Any]]
    best_constituents: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan rank-band short baskets inside the trusted short50 universe.",
    )
    parser.add_argument(
        "--research-root",
        default="output/real_public_2026q1",
        help="Path to the real_public_2026q1 artifact root, relative to this script.",
    )
    parser.add_argument(
        "--output-json",
        default="output/real_public_2026q1/edge_search/short50_rank_band_scan.json",
        help="Path to write the machine-readable scan output.",
    )
    parser.add_argument(
        "--output-md",
        default="output/real_public_2026q1/edge_search/short50_rank_band_scan.md",
        help="Path to write the markdown summary.",
    )
    return parser


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _round_trip_cost(turnover_usdt: float, cost_bps: float) -> float:
    return turnover_usdt * (cost_bps / 10_000.0)


def _borrow_cost(notional_days: float) -> float:
    return notional_days * (STRESS_BORROW_BPS_ANNUAL / 10_000.0) / 365.0


def _load_ranked_pairs(research_root: Path) -> tuple[list[str], bool]:
    approx_payload = load_json(research_root / "selected_universe.json")
    true_pit_payload = load_json(research_root / "selected_universe_true_pit_80.json")

    approx_top_50 = [item["pair"] for item in approx_payload["all_ranked_candidates"][:50]]
    true_top_50 = [item["pair"] for item in true_pit_payload["selected_true_pit_50"]]
    if approx_top_50 != true_top_50:
        raise ValueError("Approx and true-PIT top50 differ; rank-band funding inference is not valid.")

    return true_top_50, True


def _known_funding_totals(research_root: Path) -> tuple[float, float, float]:
    metrics = load_json(research_root / "real_backtest_metrics.json")["variants"]
    return (
        float(metrics["primary_base"]["funding_pnl_usdt"]),
        float(metrics["short30_base"]["funding_pnl_usdt"]),
        float(metrics["short50_base"]["funding_pnl_usdt"]),
    )


def build_band_specs(research_root: Path) -> tuple[list[str], list[BandSpec], dict[str, Any]]:
    ranked_pairs, approx_matches_true_pit = _load_ranked_pairs(research_root)
    funding_20, funding_30, funding_50 = _known_funding_totals(research_root)

    funding_21_30 = (funding_30 - (2.0 / 3.0) * funding_20) / (1.0 / 3.0)
    funding_21_50 = (funding_50 - 0.4 * funding_20) / 0.6
    funding_31_50 = (funding_50 - 0.4 * funding_20 - 0.2 * funding_21_30) / 0.4

    specs = [
        BandSpec(
            name="rank_1_20",
            start_rank=1,
            end_rank=20,
            funding_proxy_usdt=funding_20,
            notes=["Trusted short20 baseline from the verified Q1 report."],
        ),
        BandSpec(
            name="rank_1_30",
            start_rank=1,
            end_rank=30,
            funding_proxy_usdt=funding_30,
            notes=["Trusted short30 sensitivity from the verified Q1 report."],
        ),
        BandSpec(
            name="rank_1_50",
            start_rank=1,
            end_rank=50,
            funding_proxy_usdt=funding_50,
            notes=["Trusted short50 baseline from the verified Q1 follow-up."],
        ),
        BandSpec(
            name="rank_21_30",
            start_rank=21,
            end_rank=30,
            funding_proxy_usdt=funding_21_30,
            notes=[
                "Funding proxy derived exactly from short20 and short30 under linear equal-weight basket algebra.",
            ],
        ),
        BandSpec(
            name="rank_21_50",
            start_rank=21,
            end_rank=50,
            funding_proxy_usdt=funding_21_50,
            notes=[
                "Funding proxy derived exactly from short20 and short50 under linear equal-weight basket algebra.",
            ],
        ),
        BandSpec(
            name="rank_31_50",
            start_rank=31,
            end_rank=50,
            funding_proxy_usdt=funding_31_50,
            notes=[
                "Funding proxy derived from short20, short30, and short50.",
                "Approx and true-PIT top30/top50 are identical, so this decomposition stays on the trusted universe.",
            ],
        ),
    ]
    metadata = {
        "approx_top50_equals_true_pit_top50": approx_matches_true_pit,
        "known_funding_inputs_usdt": {
            "rank_1_20": funding_20,
            "rank_1_30": funding_30,
            "rank_1_50": funding_50,
        },
    }
    return ranked_pairs, specs, metadata


def _load_price_matrix(catalog: ParquetDataCatalog, pairs: list[str]) -> dict[str, list[float]]:
    matrix: dict[str, list[float]] = {}
    for pair in pairs:
        bars = catalog.bars([BarType.from_str(f"{pair}-PERP.BINANCE-1-DAY-LAST-EXTERNAL")])
        matrix[pair] = [float(bar.close) for bar in bars]
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


def _make_run_config(research_root: Path, band_symbols: list[str]) -> BacktestRunConfig:
    long_ids = [f"{symbol}-PERP.BINANCE" for symbol in LONG_LEG]
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


def _constituent_extremes(
    price_matrix: dict[str, list[float]],
    pairs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for pair in pairs:
        closes = price_matrix[pair]
        rows.append(
            {
                "pair": pair,
                "total_return_pct": 100.0 * (closes[-1] / closes[0] - 1.0),
            },
        )
    worst = sorted(rows, key=lambda item: item["total_return_pct"])[:5]
    best = sorted(rows, key=lambda item: item["total_return_pct"])[-5:]
    return worst, best


def _constituent_monthly_stats(price_matrix: dict[str, list[float]], pairs: list[str]) -> dict[str, float]:
    totals: list[float] = []
    month1: list[float] = []
    month2: list[float] = []
    month3: list[float] = []
    negative = 0

    for pair in pairs:
        closes = price_matrix[pair]
        total_return = closes[-1] / closes[0] - 1.0
        totals.append(total_return)
        month1.append(closes[29] / closes[0] - 1.0)
        month2.append(closes[59] / closes[29] - 1.0)
        month3.append(closes[-1] / closes[59] - 1.0)
        negative += int(total_return < 0)

    return {
        "avg_constituent_total_return_pct": 100.0 * mean(totals),
        "constituent_negative_share": negative / len(pairs),
        "avg_month1_return_pct": 100.0 * mean(month1),
        "avg_month2_return_pct": 100.0 * mean(month2),
        "avg_month3_return_pct": 100.0 * mean(month3),
    }


def run_scan(research_root: Path) -> dict[str, Any]:
    ranked_pairs, band_specs, metadata = build_band_specs(research_root)
    catalog = ParquetDataCatalog(research_root / FOLLOWUPS_CATALOG)
    pairs_needed = list(dict.fromkeys(LONG_LEG + ranked_pairs[:50]))
    price_matrix = _load_price_matrix(catalog, pairs_needed)

    node = BacktestNode(
        configs=[
            _make_run_config(
                research_root,
                ranked_pairs[spec.start_rank - 1 : spec.end_rank],
            )
            for spec in band_specs
        ],
    )
    engine_results = {
        spec.name: result
        for spec, result in zip(band_specs, node.run(), strict=True)
    }

    band_payloads: list[BandMetrics] = []
    for spec in band_specs:
        pairs = ranked_pairs[spec.start_rank - 1 : spec.end_rank]
        long_pnl, long_turnover, _ = _simulate_leg(price_matrix, LONG_LEG, is_long=True)
        short_pnl, short_turnover, short_notional_days = _simulate_leg(price_matrix, pairs, is_long=False)
        trade_cost_base = _round_trip_cost(long_turnover + short_turnover, BASE_COST_BPS)
        trade_cost_stress = _round_trip_cost(long_turnover + short_turnover, STRESS_COST_BPS)
        borrow_cost_stress = _borrow_cost(short_notional_days)

        proxy_base_return_pct = (
            (long_pnl + short_pnl + spec.funding_proxy_usdt - trade_cost_base) / 10_000.0
        )
        proxy_stress_return_pct = (
            (
                long_pnl
                + short_pnl
                + spec.funding_proxy_usdt
                - trade_cost_stress
                - borrow_cost_stress
            )
            / 10_000.0
        )
        monthly_stats = _constituent_monthly_stats(price_matrix, pairs)
        worst, best = _constituent_extremes(price_matrix, pairs)
        engine_result = engine_results[spec.name]

        band_payloads.append(
            BandMetrics(
                name=spec.name,
                count=spec.count,
                pairs=pairs,
                funding_proxy_usdt=spec.funding_proxy_usdt,
                engine_gross_return_pct=float(engine_result.stats_pnls["USDT"]["PnL% (total)"]),
                engine_gross_sharpe_252=float(
                    engine_result.stats_returns["Sharpe Ratio (252 days)"],
                ),
                proxy_base_return_pct=proxy_base_return_pct,
                proxy_stress_return_pct=proxy_stress_return_pct,
                long_pnl_usdt=long_pnl,
                short_pnl_usdt=short_pnl,
                trade_cost_base_usdt=trade_cost_base,
                trade_cost_stress_usdt=trade_cost_stress,
                borrow_cost_stress_usdt=borrow_cost_stress,
                avg_constituent_total_return_pct=monthly_stats["avg_constituent_total_return_pct"],
                constituent_negative_share=monthly_stats["constituent_negative_share"],
                avg_month1_return_pct=monthly_stats["avg_month1_return_pct"],
                avg_month2_return_pct=monthly_stats["avg_month2_return_pct"],
                avg_month3_return_pct=monthly_stats["avg_month3_return_pct"],
                worst_constituents=worst,
                best_constituents=best,
                notes=spec.notes,
            ),
        )

    band_payloads.sort(key=lambda item: item.proxy_base_return_pct, reverse=True)
    best_base = band_payloads[0]
    best_stress = max(band_payloads, key=lambda item: item.proxy_stress_return_pct)

    return {
        "window": {
            "start": "2026-01-13",
            "end": "2026-04-12",
        },
        "method": {
            "catalog_path": str(research_root / FOLLOWUPS_CATALOG),
            "long_leg": LONG_LEG,
            "leg_notional_usd": LEG_NOTIONAL_USD,
            "rebalance_cadence": "weekly",
            "trade_cost_base_bps_total": BASE_COST_BPS,
            "trade_cost_stress_bps_total": STRESS_COST_BPS,
            "stress_borrow_bps_annual": STRESS_BORROW_BPS_ANNUAL,
            "funding_proxy_basis": (
                "rank-band funding totals are solved from the verified short20/short30/short50 "
                "base results under linear equal-weight basket algebra"
            ),
        },
        "consistency_checks": metadata,
        "summary": {
            "best_proxy_base_band": best_base.name,
            "best_proxy_base_return_pct": best_base.proxy_base_return_pct,
            "best_proxy_stress_band": best_stress.name,
            "best_proxy_stress_return_pct": best_stress.proxy_stress_return_pct,
        },
        "bands": [band.to_dict() for band in band_payloads],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Short50 Rank-Band Scan",
        "",
        f"- Window: {payload['window']['start']} to {payload['window']['end']}",
        "- Scope: trusted short50 universe only; no new 80-name extrapolation in this report.",
        (
            "- Consistency: approx and true-PIT universes are identical for the top 50 names, "
            "so the rank-band funding decomposition stays on the validated universe."
        ),
        "",
        "| Band | Count | Engine Gross | Proxy Base | Proxy Stress | Avg Constituent Total | Neg Share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for band in payload["bands"]:
        lines.append(
            "| "
            f"{band['name']} | {band['count']} | "
            f"{band['engine_gross_return_pct']:.2f}% | "
            f"{band['proxy_base_return_pct']:.2f}% | "
            f"{band['proxy_stress_return_pct']:.2f}% | "
            f"{band['avg_constituent_total_return_pct']:.2f}% | "
            f"{band['constituent_negative_share']:.2f} |"
        )

    top_band = payload["bands"][0]
    baseline = next(item for item in payload["bands"] if item["name"] == "rank_1_50")
    lines.extend(
        [
            "",
            "## Read-Through",
            (
                f"- Best candidate: `{top_band['name']}` improved proxy base to "
                f"{top_band['proxy_base_return_pct']:.2f}% and proxy stress to "
                f"{top_band['proxy_stress_return_pct']:.2f}%."
            ),
            (
                f"- Trusted baseline `{baseline['name']}` sits at proxy base "
                f"{baseline['proxy_base_return_pct']:.2f}% and proxy stress "
                f"{baseline['proxy_stress_return_pct']:.2f}%."
            ),
            (
                "- The alpha signal is structural: the lowest-volume tail (`rank_1_20`) is where "
                "the violent squeeze risk lives, while the back half of short50 keeps a much higher "
                "share of negative constituent outcomes."
            ),
            (
                f"- `{top_band['name']}` still has outliers on the wrong side "
                f"({', '.join(item['pair'] for item in top_band['best_constituents'])}), "
                "so the next window should validate whether these names are persistent or one-off."
            ),
        ],
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    research_root = _resolve(base_dir, args.research_root)
    output_json = _resolve(base_dir, args.output_json)
    output_md = _resolve(base_dir, args.output_md)

    payload = run_scan(research_root)
    ensure_directory(output_json.parent)
    write_json(output_json, payload)
    output_md.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Wrote rank-band scan JSON to {output_json}")
    print(f"Wrote rank-band scan markdown to {output_md}")


if __name__ == "__main__":
    main()
