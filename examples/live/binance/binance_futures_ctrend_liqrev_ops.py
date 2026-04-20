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
import json
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import msgspec

from nautilus_trader.adapters.binance.common.credentials import get_api_key
from nautilus_trader.adapters.binance.common.credentials import get_api_secret
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from nautilus_trader.adapters.binance.common.enums import BinanceOrderSide
from nautilus_trader.adapters.binance.common.enums import BinanceOrderType
from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
from nautilus_trader.adapters.binance.common.urls import get_http_base_url
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesMarginType
from nautilus_trader.adapters.binance.futures.http.account import BinanceFuturesAccountHttpAPI
from nautilus_trader.common.component import LiveClock
from nautilus_trader.core.nautilus_pyo3 import HttpMethod


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
TRADING_INCOME_TYPES = {"REALIZED_PNL", "COMMISSION", "FUNDING_FEE"}


def parse_environment(value: str) -> BinanceEnvironment:
    if value == "demo":
        return BinanceEnvironment.DEMO
    if value == "testnet":
        return BinanceEnvironment.TESTNET
    return BinanceEnvironment.LIVE


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def utc_day_start_ms() -> int:
    now = datetime.now(timezone.utc)  # noqa: UP017
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(day_start.timestamp() * 1000)


def build_rank_bands(top_candidates: list[str]) -> dict[str, list[str]]:
    return {
        "rank_1_20": top_candidates[:20],
        "rank_1_30": top_candidates[:30],
        "rank_1_50": top_candidates[:50],
    }


def resolve_research_config_path(value: str) -> Path:
    return Path(value).resolve()


def summarize_income_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        income_type = str(row["incomeType"])
        if income_type not in TRADING_INCOME_TYPES:
            continue
        totals[income_type] += float(row["income"])
    return dict(sorted(totals.items()))


def normalize_account_snapshot(payload: dict[str, Any]) -> dict[str, float]:
    assets = payload.get("assets", [])
    usdt_asset = next((row for row in assets if row.get("asset") == "USDT"), None)
    if usdt_asset is not None:
        return {
            "wallet_balance": float(usdt_asset["walletBalance"]),
            "margin_balance": float(usdt_asset["marginBalance"]),
            "available_balance": float(usdt_asset["availableBalance"]),
            "unrealized_profit": float(usdt_asset["unrealizedProfit"]),
            "initial_margin": float(usdt_asset["initialMargin"]),
            "maint_margin": float(usdt_asset["maintMargin"]),
        }

    return {
        "wallet_balance": float(payload["totalWalletBalance"]),
        "margin_balance": float(payload["totalMarginBalance"]),
        "available_balance": float(payload["availableBalance"]),
        "unrealized_profit": float(payload["totalUnrealizedProfit"]),
        "initial_margin": float(payload["totalInitialMargin"]),
        "maint_margin": float(payload["totalMaintMargin"]),
    }


def evaluate_kill_switch(
    snapshot: dict[str, float],
    *,
    starting_equity_usdt: float,
    max_total_drawdown_usdt: float,
    max_daily_loss_usdt: float,
    daily_income_summary: dict[str, float],
    min_available_balance_usdt: float,
) -> list[str]:
    reasons: list[str] = []
    total_drawdown = starting_equity_usdt - snapshot["margin_balance"]
    daily_pnl = sum(daily_income_summary.values())

    if total_drawdown >= max_total_drawdown_usdt:
        reasons.append(
            f"total_drawdown {total_drawdown:.2f} >= max_total_drawdown {max_total_drawdown_usdt:.2f}",
        )
    if -daily_pnl >= max_daily_loss_usdt:
        reasons.append(
            f"daily_loss {-daily_pnl:.2f} >= max_daily_loss {max_daily_loss_usdt:.2f}",
        )
    if snapshot["available_balance"] <= min_available_balance_usdt:
        reasons.append(
            "available_balance "
            f"{snapshot['available_balance']:.2f} <= min_available_balance {min_available_balance_usdt:.2f}",
        )
    return reasons


def build_parser() -> argparse.ArgumentParser:
    default_research_config = (
        Path(__file__).resolve().parents[2] / "backtest/crypto_rv/configs/ctrend.real_public_q1.json"
    )
    parser = argparse.ArgumentParser(
        description="Operational helper for the CTREND conditional Binance micro-live lane.",
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
        "--short-band",
        choices=("rank_1_20", "rank_1_30", "rank_1_50"),
        default="rank_1_20",
        help="Frozen short-eligibility band for micro-live operations.",
    )
    parser.add_argument(
        "--starting-equity-usdt",
        type=float,
        default=200.0,
        help="Starting account equity used for total drawdown checks.",
    )
    parser.add_argument(
        "--max-total-drawdown-usdt",
        type=float,
        default=20.0,
        help="Account-level absolute drawdown stop.",
    )
    parser.add_argument(
        "--max-daily-loss-usdt",
        type=float,
        default=6.0,
        help="UTC-day realized loss stop using income history.",
    )
    parser.add_argument(
        "--min-available-balance-usdt",
        type=float,
        default=50.0,
        help="Minimum free balance required before kill switch triggers.",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=2,
        help="Target initial leverage for prepare.",
    )
    parser.add_argument(
        "--output-json",
        default=str(
            Path(__file__).resolve().parent / "runtime" / "ctrend_liqrev_micro_status.json",
        ),
        help="Path to write status or kill-switch reports.",
    )
    parser.add_argument(
        "--allow-shared-account",
        action="store_true",
        help="Allow account checks even if unrelated futures positions are open.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Prepare account mode, margin type, and leverage.")
    subparsers.add_parser("status", help="Fetch account status and income summary.")
    kill_parser = subparsers.add_parser("killswitch", help="Evaluate and optionally flatten.")
    kill_parser.add_argument(
        "--force",
        action="store_true",
        help="Trigger flatten regardless of threshold evaluation.",
    )
    return parser


async def fetch_income_rows(
    client: Any,
    *,
    start_time_ms: int,
    income_type: str | None = None,
) -> list[dict[str, Any]]:
    payload = {
        "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),  # noqa: UP017
        "startTime": str(start_time_ms),
        "limit": "1000",
    }
    if income_type is not None:
        payload["incomeType"] = income_type
    raw = await client.sign_request(
        HttpMethod.GET,
        "/fapi/v1/income",
        payload=payload,
        ratelimiter_keys=["binance:/fapi/v1/income"],
    )
    return json.loads(raw)


def selected_symbols(research_config_path: Path, short_band: str) -> tuple[list[str], list[str]]:
    research_config = load_research_config(research_config_path)
    _baseline_longs, top_candidates, _baseline_shorts = load_real_public_universe(
        research_config.selected_universe_path,
        research_config.top_n_candidates,
    )
    long_symbols = [instrument_id.split("-PERP.")[0] for instrument_id in _baseline_longs]
    rank_bands = build_rank_bands([instrument_id.split("-PERP.")[0] for instrument_id in top_candidates])
    return long_symbols, rank_bands[short_band]


def tradable_symbols(research_config_path: Path) -> list[str]:
    research_config = load_research_config(research_config_path)
    baseline_longs, top_candidates, _baseline_shorts = load_real_public_universe(
        research_config.selected_universe_path,
        research_config.top_n_candidates,
    )
    return sorted(
        {
            instrument_id.split("-PERP.")[0]
            for instrument_id in baseline_longs + top_candidates
        },
    )


def strategy_symbols(research_config_path: Path, short_band: str) -> list[str]:
    _long_symbols, short_symbols = selected_symbols(research_config_path, short_band)
    return sorted(set(tradable_symbols(research_config_path) + short_symbols))


def foreign_open_positions(
    positions_payload: list[dict[str, Any]],
    strategy_symbol_set: set[str],
) -> list[dict[str, Any]]:
    foreign: list[dict[str, Any]] = []
    for position in positions_payload:
        symbol = str(position["symbol"])
        if symbol in strategy_symbol_set:
            continue
        if float(position["positionAmt"]) == 0.0:
            continue
        foreign.append(
            {
                "symbol": symbol,
                "position_amt": float(position["positionAmt"]),
                "unrealized_profit": float(position.get("unRealizedProfit", position.get("unrealizedProfit", 0.0))),
            },
        )
    return foreign


def build_namespace(
    *,
    research_config: str,
    binance_environment: str,
    short_band: str,
    starting_equity_usdt: float,
    max_total_drawdown_usdt: float,
    max_daily_loss_usdt: float,
    min_available_balance_usdt: float,
    leverage: int,
    output_json: str,
    allow_shared_account: bool,
    command: str,
    force: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        research_config=research_config,
        binance_environment=binance_environment,
        short_band=short_band,
        starting_equity_usdt=starting_equity_usdt,
        max_total_drawdown_usdt=max_total_drawdown_usdt,
        max_daily_loss_usdt=max_daily_loss_usdt,
        min_available_balance_usdt=min_available_balance_usdt,
        leverage=leverage,
        output_json=output_json,
        allow_shared_account=allow_shared_account,
        command=command,
        force=force,
    )


async def build_account_clients_for(environment_name: str) -> tuple[Any, BinanceFuturesAccountHttpAPI]:
    environment = parse_environment(environment_name)
    clock = LiveClock()
    api_key = get_api_key(BinanceAccountType.USDT_FUTURES, environment)
    api_secret = get_api_secret(BinanceAccountType.USDT_FUTURES, environment)
    client = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.USDT_FUTURES,
        api_key=api_key,
        api_secret=api_secret,
        environment=environment,
        base_url=get_http_base_url(BinanceAccountType.USDT_FUTURES, environment, False),
    )
    account_api = BinanceFuturesAccountHttpAPI(
        clock=clock,
        client=client,
        account_type=BinanceAccountType.USDT_FUTURES,
    )
    return client, account_api


async def build_account_clients(args: argparse.Namespace) -> tuple[Any, BinanceFuturesAccountHttpAPI]:
    return await build_account_clients_for(args.binance_environment)


async def command_prepare(args: argparse.Namespace) -> dict[str, Any]:
    client, account_api = await build_account_clients(args)
    symbols = tradable_symbols(resolve_research_config_path(args.research_config))

    result: dict[str, Any] = {
        "prepared_at": utc_now_iso(),
        "environment": args.binance_environment,
        "symbols": symbols,
        "hedge_mode": "one_way",
        "margin_type": "CROSSED",
        "leverage": args.leverage,
        "steps": [],
    }

    try:
        await account_api.set_futures_hedge_mode(False)
        result["steps"].append("set hedge mode -> one_way")
    except Exception as exc:  # pragma: no cover - exchange may report already-set state
        result["steps"].append(f"hedge mode unchanged ({exc})")

    for symbol in symbols:
        await account_api.set_leverage(symbol=BinanceSymbol(symbol), leverage=args.leverage)
        result["steps"].append(f"set leverage {symbol} -> {args.leverage}")
        try:
            await account_api.set_margin_type(
                symbol=BinanceSymbol(symbol),
                margin_type=BinanceFuturesMarginType.CROSS,
            )
            result["steps"].append(f"set margin type {symbol} -> CROSSED")
        except Exception as exc:  # pragma: no cover - exchange may report already-set state
            result["steps"].append(f"margin type {symbol} unchanged ({exc})")

    return result


async def command_status(args: argparse.Namespace) -> dict[str, Any]:
    client, account_api = await build_account_clients(args)
    symbols = strategy_symbols(resolve_research_config_path(args.research_config), args.short_band)
    strategy_symbol_set = set(symbols)
    account_info = await account_api.query_futures_account_info(recv_window="5000")
    account_payload = msgspec.json.decode(msgspec.json.encode(account_info))
    snapshot = normalize_account_snapshot(account_payload)
    income_rows = await fetch_income_rows(client, start_time_ms=utc_day_start_ms())
    strategy_income_rows = [row for row in income_rows if str(row.get("symbol", "")) in strategy_symbol_set]
    income_summary = summarize_income_rows(strategy_income_rows)
    positions = await account_api.query_futures_position_risk(recv_window="5000")
    positions_payload = msgspec.json.decode(msgspec.json.encode(positions))
    foreign_positions = foreign_open_positions(positions_payload, strategy_symbol_set)
    kill_reasons = evaluate_kill_switch(
        snapshot,
        starting_equity_usdt=args.starting_equity_usdt,
        max_total_drawdown_usdt=args.max_total_drawdown_usdt,
        max_daily_loss_usdt=args.max_daily_loss_usdt,
        daily_income_summary=income_summary,
        min_available_balance_usdt=args.min_available_balance_usdt,
    )
    dedicated_account_ready = not foreign_positions
    if foreign_positions and not args.allow_shared_account:
        kill_reasons.append("foreign_open_positions detected outside strategy symbol set")

    result = {
        "generated_at": utc_now_iso(),
        "environment": args.binance_environment,
        "strategy_symbols": symbols,
        "account_snapshot": snapshot,
        "daily_income_summary": income_summary,
        "daily_income_rows": len(strategy_income_rows),
        "foreign_open_positions": foreign_positions,
        "dedicated_account_ready": dedicated_account_ready,
        "kill_switch_reasons": kill_reasons,
        "kill_switch_triggered": bool(kill_reasons),
    }
    return result


async def command_killswitch(args: argparse.Namespace) -> dict[str, Any]:
    status_result = await command_status(args)
    client, account_api = await build_account_clients(args)
    symbols = strategy_symbols(resolve_research_config_path(args.research_config), args.short_band)
    strategy_symbol_set = set(symbols)
    trigger = args.force or status_result["kill_switch_triggered"]

    result = {
        **status_result,
        "forced": args.force,
        "executed": False,
        "flattened_symbols": [],
    }
    if not trigger:
        return result

    positions = await account_api.query_futures_position_risk(recv_window="5000")
    positions_payload = msgspec.json.decode(msgspec.json.encode(positions))
    for position in positions_payload:
        symbol = str(position["symbol"])
        if symbol not in strategy_symbol_set:
            continue
        if float(position["positionAmt"]) == 0.0:
            continue
        await account_api.cancel_all_open_orders(symbol=symbol, recv_window="5000")
        side = (
            BinanceOrderSide.SELL
            if float(position["positionAmt"]) > 0
            else BinanceOrderSide.BUY
        )
        await account_api.new_order(
            symbol=symbol,
            side=side,
            order_type=BinanceOrderType.MARKET,
            quantity=str(abs(float(position["positionAmt"]))),
            reduce_only="true",
            recv_window="5000",
        )
        result["flattened_symbols"].append(symbol)

    for symbol in symbols:
        await account_api.cancel_all_open_orders(symbol=symbol, recv_window="5000")

    result["executed"] = True
    return result


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prepare":
        return await command_prepare(args)
    if args.command == "status":
        return await command_status(args)
    return await command_killswitch(args)


def main() -> None:
    args = build_parser().parse_args()
    payload = asyncio.run(async_main(args))
    output_path = Path(args.output_json).resolve()
    write_json_report(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote ops report to {output_path}")


if __name__ == "__main__":
    main()
