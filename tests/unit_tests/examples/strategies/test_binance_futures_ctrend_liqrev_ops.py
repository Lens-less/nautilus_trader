import json
from pathlib import Path

from examples.live.binance.binance_futures_ctrend_liqrev_ops import evaluate_kill_switch
from examples.live.binance.binance_futures_ctrend_liqrev_ops import normalize_account_snapshot
from examples.live.binance.binance_futures_ctrend_liqrev_ops import strategy_symbols
from examples.live.binance.binance_futures_ctrend_liqrev_ops import summarize_income_rows
from examples.live.binance.binance_futures_ctrend_liqrev_ops import tradable_symbols


def test_normalize_account_snapshot_prefers_usdt_asset_row() -> None:
    payload = {
        "assets": [
            {
                "asset": "USDT",
                "walletBalance": "190.00",
                "marginBalance": "188.50",
                "availableBalance": "92.00",
                "unrealizedProfit": "-1.50",
                "initialMargin": "60.00",
                "maintMargin": "12.00",
            },
        ],
    }

    snapshot = normalize_account_snapshot(payload)

    assert snapshot == {
        "wallet_balance": 190.0,
        "margin_balance": 188.5,
        "available_balance": 92.0,
        "unrealized_profit": -1.5,
        "initial_margin": 60.0,
        "maint_margin": 12.0,
    }


def test_summarize_income_rows_groups_by_income_type() -> None:
    rows = [
        {"incomeType": "REALIZED_PNL", "income": "-3.50"},
        {"incomeType": "COMMISSION", "income": "-0.40"},
        {"incomeType": "TRANSFER", "income": "12.00"},
        {"incomeType": "REALIZED_PNL", "income": "1.25"},
    ]

    summary = summarize_income_rows(rows)

    assert summary == {"COMMISSION": -0.4, "REALIZED_PNL": -2.25}


def test_tradable_symbols_matches_live_runner_universe_scope() -> None:
    research_config = (
        Path(__file__).resolve().parents[4]
        / "examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json"
    )

    symbols = tradable_symbols(research_config)

    assert len(symbols) == 55
    assert "AINUSDT" in symbols
    assert "BTCUSDT" in symbols


def test_strategy_symbols_scope_matches_tradable_universe_for_rank_1_20() -> None:
    research_config = (
        Path(__file__).resolve().parents[4]
        / "examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json"
    )

    assert strategy_symbols(research_config, "rank_1_20") == tradable_symbols(research_config)


def test_evaluate_kill_switch_returns_reasons_for_breaches() -> None:
    reasons = evaluate_kill_switch(
        {
            "wallet_balance": 190.0,
            "margin_balance": 178.0,
            "available_balance": 45.0,
            "unrealized_profit": -2.0,
            "initial_margin": 60.0,
            "maint_margin": 12.0,
        },
        starting_equity_usdt=200.0,
        max_total_drawdown_usdt=20.0,
        max_daily_loss_usdt=6.0,
        daily_income_summary={"REALIZED_PNL": -5.0, "COMMISSION": -1.5},
        min_available_balance_usdt=50.0,
    )

    assert len(reasons) == 3
    assert "total_drawdown" in json.dumps(reasons)
    assert "daily_loss" in json.dumps(reasons)
    assert "available_balance" in json.dumps(reasons)
