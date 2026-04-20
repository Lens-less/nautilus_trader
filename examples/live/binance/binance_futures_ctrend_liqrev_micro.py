#!/usr/bin/env python3
# mypy: disable-error-code=no-redef
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import threading
from pathlib import Path

from nautilus_trader.adapters.binance import BINANCE
from nautilus_trader.adapters.binance import BinanceAccountType
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceExecClientConfig
from nautilus_trader.adapters.binance import BinanceInstrumentProviderConfig
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesMarginType
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.examples.strategies.crypto_xsec_trend_conditional import (
    CryptoXSecTrendConditionalConfig,
)
from nautilus_trader.examples.strategies.crypto_xsec_trend_conditional import (
    CryptoXSecTrendConditionalStrategy,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId


try:
    from examples.live.binance.binance_futures_ctrend_liqrev_ops import build_namespace
    from examples.live.binance.binance_futures_ctrend_liqrev_ops import command_killswitch
    from examples.live.binance.binance_futures_ctrend_liqrev_ops import command_status
    from examples.live.binance.binance_futures_ctrend_liqrev_ops import write_json_report
except ImportError:  # pragma: no cover - script execution fallback
    from binance_futures_ctrend_liqrev_ops import build_namespace
    from binance_futures_ctrend_liqrev_ops import command_killswitch
    from binance_futures_ctrend_liqrev_ops import command_status
    from binance_futures_ctrend_liqrev_ops import write_json_report


try:
    from examples.backtest.crypto_rv.run_ctrend_real_public import (
        load_config as _load_research_config,
    )
    from examples.backtest.crypto_rv.run_ctrend_real_public import (
        load_real_public_universe as _load_real_public_universe,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from run_ctrend_real_public import load_config as _load_research_config
    from run_ctrend_real_public import load_real_public_universe as _load_real_public_universe


load_research_config = _load_research_config
load_real_public_universe = _load_real_public_universe


SHORT_BAND_CHOICES = ("rank_1_20", "rank_1_30", "rank_1_50")


def build_parser() -> argparse.ArgumentParser:
    default_research_config = (
        Path(__file__).resolve().parents[2] / "backtest/crypto_rv/configs/ctrend.real_public_q1.json"
    )
    parser = argparse.ArgumentParser(
        description="Run the accepted CTREND x liquidity/reversal micro-live lane on Binance futures.",
    )
    parser.add_argument(
        "--research-config",
        default=str(default_research_config),
        help="Path to the accepted CTREND real-public research config JSON.",
    )
    parser.add_argument(
        "--binance-environment",
        choices=("demo", "testnet", "live"),
        default="demo",
        help="Binance futures environment.",
    )
    parser.add_argument(
        "--trader-id",
        default="CTREND-LIVE-001",
        help="Trader identifier for the live node.",
    )
    parser.add_argument(
        "--leg-notional-usd",
        type=float,
        default=75.0,
        help="Target USD notional per leg. 75 => 150 gross on a 200U account.",
    )
    parser.add_argument(
        "--short-band",
        choices=SHORT_BAND_CHOICES,
        default="rank_1_20",
        help="Frozen short-eligibility band from the accepted research universe.",
    )
    parser.add_argument(
        "--min-active-shorts",
        type=int,
        default=4,
        help="Fail-close threshold when conditioned shorts shrink below this count.",
    )
    parser.add_argument(
        "--short-fallback-bands",
        default="rank_1_30,rank_1_50",
        help="Comma-separated short bands used when the primary post-rank-filter intersection is too small.",
    )
    parser.add_argument(
        "--history-lookback-days",
        type=int,
        default=90,
        help="Historical daily bar lookback used to seed live signal state.",
    )
    parser.add_argument(
        "--order-expire-seconds",
        type=int,
        default=900,
        help="Passive limit order expiry in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Trading node log level.",
    )
    parser.add_argument(
        "--initial-leverage",
        type=int,
        default=2,
        help="Target initial symbol leverage for the micro-live lane.",
    )
    parser.add_argument(
        "--starting-equity-usdt",
        type=float,
        default=200.0,
        help="Starting equity used by the automatic kill switch.",
    )
    parser.add_argument(
        "--max-total-drawdown-usdt",
        type=float,
        default=20.0,
        help="Absolute drawdown stop used by the automatic kill switch.",
    )
    parser.add_argument(
        "--max-daily-loss-usdt",
        type=float,
        default=6.0,
        help="UTC-day trading loss stop used by the automatic kill switch.",
    )
    parser.add_argument(
        "--min-available-balance-usdt",
        type=float,
        default=50.0,
        help="Minimum free balance before the automatic kill switch stops the lane.",
    )
    parser.add_argument(
        "--kill-switch-check-interval-secs",
        type=int,
        default=60,
        help="Polling interval for the kill-switch sidecar.",
    )
    return parser


def build_rank_bands(top_candidates: list[str]) -> dict[str, list[str]]:
    return {
        "rank_1_20": top_candidates[:20],
        "rank_1_30": top_candidates[:30],
        "rank_1_50": top_candidates[:50],
    }


def parse_short_fallback_bands(raw_value: str, *, primary_short_band: str) -> tuple[str, ...]:
    fallback_bands: list[str] = []
    for raw_band in raw_value.split(","):
        band_name = raw_band.strip()
        if not band_name or band_name == primary_short_band:
            continue
        if band_name not in SHORT_BAND_CHOICES:
            raise ValueError(
                f"Unsupported fallback short band {band_name!r}; expected one of {SHORT_BAND_CHOICES}",
            )
        if band_name not in fallback_bands:
            fallback_bands.append(band_name)
    return tuple(fallback_bands)


def parse_environment(value: str) -> BinanceEnvironment:
    if value == "demo":
        return BinanceEnvironment.DEMO
    if value == "testnet":
        return BinanceEnvironment.TESTNET
    return BinanceEnvironment.LIVE


def build_ops_namespace(args: argparse.Namespace, *, command: str, force: bool = False) -> argparse.Namespace:
    runtime_report = (
        Path(__file__).resolve().parent / "runtime" / "ctrend_liqrev_micro_status.json"
    )
    return build_namespace(
        research_config=args.research_config,
        binance_environment=args.binance_environment,
        short_band=args.short_band,
        starting_equity_usdt=args.starting_equity_usdt,
        max_total_drawdown_usdt=args.max_total_drawdown_usdt,
        max_daily_loss_usdt=args.max_daily_loss_usdt,
        min_available_balance_usdt=args.min_available_balance_usdt,
        leverage=args.initial_leverage,
        output_json=str(runtime_report),
        allow_shared_account=False,
        command=command,
        force=force,
    )


def fetch_status_payload(status_args: argparse.Namespace) -> dict[str, object]:
    status_payload = asyncio.run(command_status(status_args))
    write_json_report(Path(status_args.output_json).resolve(), status_payload)
    return status_payload


def monitor_kill_switch(args: argparse.Namespace, stop_event: threading.Event) -> None:
    status_args = build_ops_namespace(args, command="status")
    kill_args = build_ops_namespace(args, command="killswitch")
    while not stop_event.wait(args.kill_switch_check_interval_secs):
        status_payload = fetch_status_payload(status_args)
        if not status_payload["dedicated_account_ready"]:
            os.kill(os.getpid(), signal.SIGINT)
            return
        if status_payload["kill_switch_triggered"]:
            asyncio.run(command_killswitch(kill_args))
            os.kill(os.getpid(), signal.SIGINT)
            return


def main() -> None:
    args = build_parser().parse_args()
    status_namespace = build_ops_namespace(args, command="status")
    preflight_payload = fetch_status_payload(status_namespace)
    if not preflight_payload["dedicated_account_ready"]:
        raise RuntimeError(
            "Refusing to start: unrelated non-zero Binance futures positions detected in the account",
        )
    research_config = load_research_config(Path(args.research_config).resolve())
    baseline_longs, top_candidates, _baseline_shorts = load_real_public_universe(
        research_config.selected_universe_path,
        research_config.top_n_candidates,
    )
    rank_bands = build_rank_bands(top_candidates)
    eligible_short_ids = rank_bands[args.short_band]
    fallback_short_band_names = parse_short_fallback_bands(
        args.short_fallback_bands,
        primary_short_band=args.short_band,
    )
    universe_ids = baseline_longs + [
        instrument_id for instrument_id in top_candidates if instrument_id not in baseline_longs
    ]
    instrument_ids = tuple(InstrumentId.from_str(instrument_id) for instrument_id in universe_ids)
    eligible_short_instrument_ids = tuple(
        InstrumentId.from_str(instrument_id) for instrument_id in eligible_short_ids
    )
    fallback_short_instrument_ids = tuple(
        tuple(InstrumentId.from_str(instrument_id) for instrument_id in rank_bands[band_name])
        for band_name in fallback_short_band_names
    )
    raw_symbols = [instrument_id.split("-PERP.")[0] for instrument_id in universe_ids]
    bar_types = tuple(
        BarType.from_str(f"{instrument_id}-1-DAY-LAST-EXTERNAL") for instrument_id in instrument_ids
    )
    environment = parse_environment(args.binance_environment)
    symbol_leverages = {
        BinanceSymbol(symbol): args.initial_leverage
        for symbol in raw_symbols
    }
    symbol_margin_types = {
        BinanceSymbol(symbol): BinanceFuturesMarginType.CROSS
        for symbol in raw_symbols
    }

    node = TradingNode(
        config=TradingNodeConfig(
            trader_id=TraderId(args.trader_id),
            logging=LoggingConfig(log_level=args.log_level, use_pyo3=True),
            data_engine=LiveDataEngineConfig(external_clients=[ClientId(BINANCE)]),
            exec_engine=LiveExecEngineConfig(
                reconciliation=True,
                open_check_open_only=False,
                purge_closed_orders_interval_mins=1,
                purge_closed_orders_buffer_mins=0,
                purge_closed_positions_interval_mins=1,
                purge_closed_positions_buffer_mins=0,
                purge_account_events_interval_mins=1,
                purge_account_events_lookback_mins=0,
                purge_from_database=True,
                graceful_shutdown_on_exception=True,
            ),
            cache=CacheConfig(
                timestamps_as_iso8601=True,
                flush_on_start=False,
            ),
            data_clients={
                BINANCE: BinanceDataClientConfig(
                    account_type=BinanceAccountType.USDT_FUTURES,
                    environment=environment,
                    instrument_provider=BinanceInstrumentProviderConfig(
                        load_ids=frozenset(instrument_ids),
                        query_commission_rates=True,
                    ),
                ),
            },
            exec_clients={
                BINANCE: BinanceExecClientConfig(
                    account_type=BinanceAccountType.USDT_FUTURES,
                    environment=environment,
                    instrument_provider=BinanceInstrumentProviderConfig(
                        load_ids=frozenset(instrument_ids),
                        query_commission_rates=True,
                    ),
                    max_retries=3,
                    futures_leverages=symbol_leverages,
                    futures_margin_types=symbol_margin_types,
                    log_rejected_due_post_only_as_warning=False,
                ),
            },
            timeout_connection=30.0,
            timeout_reconciliation=10.0,
            timeout_portfolio=10.0,
            timeout_disconnection=10.0,
            timeout_post_stop=5.0,
        ),
    )

    strategy = CryptoXSecTrendConditionalStrategy(
        config=CryptoXSecTrendConditionalConfig(
            universe_instrument_ids=instrument_ids,
            eligible_short_instrument_ids=eligible_short_instrument_ids,
            primary_short_band_name=args.short_band,
            fallback_short_band_names=fallback_short_band_names,
            fallback_short_instrument_ids=fallback_short_instrument_ids,
            external_order_claims=list(instrument_ids),
            bar_types=bar_types,
            leg_notional_usd=args.leg_notional_usd,
            rebalance_cadence="biweekly",
            long_bucket_frac=research_config.long_bucket_frac,
            short_bucket_frac=research_config.short_bucket_frac,
            fast_window=research_config.signal.fast_window,
            slow_window=research_config.signal.slow_window,
            sma_window=research_config.signal.sma_window,
            vol_window=research_config.signal.vol_window,
            volume_window=research_config.signal.volume_window,
            min_history_bars=research_config.signal.min_history_bars,
            volatility_managed=True,
            conditional_mode="post_rank_filter",
            tilt_multiplier=2.0,
            min_active_shorts=args.min_active_shorts,
            request_bars=True,
            history_lookback_days=args.history_lookback_days,
            subscribe_quote_ticks=True,
            use_passive_limits=True,
            allow_market_fallback=False,
            order_time_in_force=TimeInForce.GTD,
            order_expire_seconds=args.order_expire_seconds,
            rebalance_execution_window_secs=3600.0,
            retry_rebalance_after_target_expiry=True,
            min_order_notional_usd=5.0,
            manual_approval_rebalances=2,
            approval_artifact_dir=str(Path(__file__).resolve().parent / "runtime" / "approvals"),
            close_positions_on_stop=False,
        ),
    )

    node.trader.add_strategy(strategy)
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
    node.build()

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_kill_switch,
        args=(args, stop_event),
        daemon=True,
        name="ctrend-liqrev-killswitch",
    )
    monitor_thread.start()

    try:
        node.run()
    finally:
        stop_event.set()
        node.dispose()


if __name__ == "__main__":
    main()
