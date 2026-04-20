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

from decimal import Decimal
from types import SimpleNamespace

import pytest

import nautilus_trader.examples.strategies.crypto_xsec_trend_conditional as conditional_module
from nautilus_trader.examples.strategies.crypto_xsec_trend_conditional import (
    CryptoXSecTrendConditionalStrategy,
)
from nautilus_trader.examples.strategies.crypto_xsec_trend_conditional import (
    build_conditioned_weight_plan,
)
from nautilus_trader.examples.strategies.crypto_xsec_trend_conditional import (
    conditioned_plan_tradeable,
)


def test_post_rank_filter_keeps_only_cttrend_short_intersection() -> None:
    composite_scores = {
        "LONG_A": 0.8,
        "LONG_B": 0.7,
        "SHORT_A": -0.1,
        "SHORT_B": -0.2,
        "SHORT_C": -0.3,
        "SHORT_D": -0.4,
    }

    plan = build_conditioned_weight_plan(
        composite_scores,
        list(composite_scores),
        eligible_short_ids={"SHORT_D"},
        mode="post_rank_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
        tilt_multiplier=2.0,
    )

    assert plan.longs == ["LONG_B", "LONG_A"]
    assert plan.shorts == ["SHORT_D"]
    assert plan.weights["SHORT_D"] == -0.5
    assert plan.weights["SHORT_C"] == 0.0


def test_hard_filter_chooses_shorts_inside_frozen_band() -> None:
    composite_scores = {
        "LONG_A": 0.8,
        "LONG_B": 0.7,
        "SHORT_A": -0.1,
        "SHORT_B": -0.2,
        "SHORT_C": -0.3,
        "SHORT_D": -0.4,
    }

    plan = build_conditioned_weight_plan(
        composite_scores,
        list(composite_scores),
        eligible_short_ids={"SHORT_C", "SHORT_D"},
        mode="hard_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
        tilt_multiplier=2.0,
    )

    assert set(plan.shorts) == {"SHORT_C", "SHORT_D"}
    assert plan.weights["SHORT_A"] == 0.0
    assert plan.weights["SHORT_B"] == 0.0


def test_short_tilt_preserves_short_gross_and_overweights_frozen_names() -> None:
    composite_scores = {
        "LONG_A": 0.8,
        "LONG_B": 0.7,
        "SHORT_A": -0.1,
        "SHORT_B": -0.2,
        "SHORT_C": -0.3,
        "SHORT_D": -0.4,
    }

    plan = build_conditioned_weight_plan(
        composite_scores,
        list(composite_scores),
        eligible_short_ids={"SHORT_D"},
        mode="short_tilt",
        long_bucket_frac=0.25,
        short_bucket_frac=0.5,
        tilt_multiplier=2.0,
    )

    short_weights = [weight for weight in plan.weights.values() if weight < 0]
    assert abs(sum(short_weights) + 0.5) < 1e-12
    assert abs(plan.weights["SHORT_D"]) > abs(plan.weights["SHORT_C"])


def test_conditioned_plan_tradeable_uses_minimum_short_threshold() -> None:
    plan = build_conditioned_weight_plan(
        {
            "LONG_A": 0.8,
            "LONG_B": 0.7,
            "SHORT_A": -0.1,
            "SHORT_B": -0.2,
        },
        ["LONG_A", "LONG_B", "SHORT_A", "SHORT_B"],
        eligible_short_ids={"SHORT_B"},
        mode="post_rank_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
        tilt_multiplier=2.0,
    )

    assert conditioned_plan_tradeable(plan, 1) is True
    assert conditioned_plan_tradeable(plan, 2) is False


def test_post_rank_filter_falls_back_to_wider_band_when_primary_band_is_too_small() -> None:
    composite_scores = {
        "LONG_A": 0.8,
        "LONG_B": 0.7,
        "SHORT_A": -0.1,
        "SHORT_B": -0.2,
        "SHORT_C": -0.3,
        "SHORT_D": -0.4,
    }

    plan = build_conditioned_weight_plan(
        composite_scores,
        list(composite_scores),
        eligible_short_ids={"SHORT_D"},
        primary_short_band_name="rank_1_20",
        fallback_short_bands=(("rank_1_30", {"SHORT_C", "SHORT_D"}),),
        min_active_shorts=2,
        mode="post_rank_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.5,
        tilt_multiplier=2.0,
    )

    assert plan.selected_short_band_name == "rank_1_30"
    assert plan.fallback_used is True
    assert plan.shorts == ["SHORT_D", "SHORT_C"]


def test_invalid_conditional_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported conditional mode"):
        build_conditioned_weight_plan(
            {"A": 0.1, "B": -0.1},
            ["A", "B"],
            eligible_short_ids={"B"},
            mode="bad-mode",  # type: ignore[arg-type]
            long_bucket_frac=0.5,
            short_bucket_frac=0.5,
            tilt_multiplier=2.0,
        )


def test_activate_rebalance_targets_anchors_deadline_to_current_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._target_quantities = {}
    strategy._cancel_all_open_orders = lambda: None
    strategy._work_pending_target = lambda instrument_id: False
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "config",
        property(lambda self: SimpleNamespace(rebalance_execution_window_secs=3600.0)),
        raising=False,
    )
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "clock",
        property(lambda self: SimpleNamespace(timestamp_ns=lambda: 999_000_000_000)),
        raising=False,
    )
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "log",
        property(lambda self: SimpleNamespace(info=lambda *args, **kwargs: None)),
        raising=False,
    )

    strategy._activate_rebalance_targets(
        123,
        {"BTCUSDT-PERP.BINANCE": Decimal(1)},
    )

    assert strategy._last_rebalance_ts == 123
    assert strategy._execution_deadline_ns == 999_000_000_000 + 3_600_000_000_000


def test_latest_complete_snapshot_ts_requires_full_synchronized_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._latest_bars = {
        "BTC": SimpleNamespace(ts_event=100, is_single_price=lambda: False, close=SimpleNamespace(as_decimal=lambda: Decimal(1))),
        "ETH": SimpleNamespace(ts_event=100, is_single_price=lambda: False, close=SimpleNamespace(as_decimal=lambda: Decimal(1))),
    }
    strategy._close_history = {
        "BTC": [1.0] * 25,
        "ETH": [1.0] * 25,
    }
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "config",
        property(
            lambda self: SimpleNamespace(
                universe_instrument_ids=("BTC", "ETH"),
                min_history_bars=25,
            ),
        ),
        raising=False,
    )

    assert strategy._latest_complete_snapshot_ts() == 100

    strategy._latest_bars["ETH"] = SimpleNamespace(
        ts_event=101,
        is_single_price=lambda: False,
        close=SimpleNamespace(as_decimal=lambda: Decimal(1)),
    )

    assert strategy._latest_complete_snapshot_ts() is None


def test_latest_complete_snapshot_ts_allows_single_price_daily_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._latest_bars = {
        "BTC": SimpleNamespace(ts_event=100, is_single_price=lambda: True, close=SimpleNamespace(as_decimal=lambda: Decimal(1))),
        "ETH": SimpleNamespace(ts_event=100, is_single_price=lambda: True, close=SimpleNamespace(as_decimal=lambda: Decimal(1))),
    }
    strategy._close_history = {
        "BTC": [1.0] * 25,
        "ETH": [1.0] * 25,
    }
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "config",
        property(
            lambda self: SimpleNamespace(
                universe_instrument_ids=("BTC", "ETH"),
                min_history_bars=25,
            ),
        ),
        raising=False,
    )

    assert strategy._latest_complete_snapshot_ts() == 100


def test_quote_tick_triggers_initial_rebalance_from_historical_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._instruments = {"BTC": object()}
    strategy._latest_quotes = {}
    strategy._pending_approval_targets = {}
    strategy._target_quantities = {}
    strategy._last_rebalance_ts = None
    triggered: list[int] = []

    monkeypatch.setattr(strategy, "_try_execute_pending_approval", lambda: None)
    monkeypatch.setattr(strategy, "_work_pending_target", lambda instrument_id: None)
    monkeypatch.setattr(strategy, "_latest_complete_snapshot_ts", lambda: 123)
    monkeypatch.setattr(strategy, "_rebalance_due", lambda ts_event: True)
    monkeypatch.setattr(strategy, "_rebalance", lambda ts_event: triggered.append(ts_event))
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "log",
        property(lambda self: SimpleNamespace(info=lambda *args, **kwargs: None)),
        raising=False,
    )

    strategy.on_quote_tick(SimpleNamespace(instrument_id="BTC"))

    assert triggered == [123]
    assert strategy._latest_quotes["BTC"].instrument_id == "BTC"


def test_historical_data_updates_latest_bar_and_attempts_initial_rebalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._close_history = {"BTC": []}
    strategy._latest_bars = {}
    triggered: list[str] = []

    class DummyBar:
        def __init__(self) -> None:
            self.bar_type = SimpleNamespace(instrument_id="BTC")

    bar = DummyBar()

    monkeypatch.setattr(conditional_module, "Bar", DummyBar)
    monkeypatch.setattr(strategy, "_ingest_bar", lambda data: triggered.append("ingested"))
    monkeypatch.setattr(strategy, "_maybe_rebalance_from_latest_snapshot", lambda: triggered.append("rebalanced"))

    strategy.on_historical_data(bar)

    assert strategy._latest_bars["BTC"] is bar
    assert triggered == ["ingested", "rebalanced"]


def test_work_pending_target_resets_rebalance_after_execution_window_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._target_quantities = {"BTC": Decimal(1)}
    strategy._execution_deadline_ns = 100
    strategy._last_rebalance_ts = 123
    strategy._approval_retry_bypass_active = False
    strategy._cancel_all_open_orders = lambda: None
    retried: list[int] = []
    monkeypatch.setattr(strategy, "_latest_complete_snapshot_ts", lambda: 222)
    monkeypatch.setattr(strategy, "_rebalance", lambda ts_event: retried.append(ts_event))

    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "clock",
        property(lambda self: SimpleNamespace(timestamp_ns=lambda: 101)),
        raising=False,
    )
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "config",
        property(lambda self: SimpleNamespace(retry_rebalance_after_target_expiry=True)),
        raising=False,
    )
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "log",
        property(lambda self: SimpleNamespace(warning=lambda *args, **kwargs: None)),
        raising=False,
    )

    assert strategy._work_pending_target("BTC") is False
    assert strategy._target_quantities == {}
    assert strategy._execution_deadline_ns is None
    assert strategy._last_rebalance_ts is None
    assert strategy._approval_retry_bypass_active is True
    assert retried == [222]


def test_approval_required_bypasses_manual_gate_for_expiry_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._approval_retry_bypass_active = True
    strategy._rebalance_count = 0
    strategy._last_rebalance_ts = None

    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "config",
        property(
            lambda self: SimpleNamespace(
                approval_artifact_dir="approvals",
                manual_approval_rebalances=2,
            ),
        ),
        raising=False,
    )

    assert strategy._approval_required(124) is False


def test_submit_delta_order_drops_untradeable_zero_quantity_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = CryptoXSecTrendConditionalStrategy.__new__(CryptoXSecTrendConditionalStrategy)
    strategy._instruments = {
        "BTC": SimpleNamespace(
            multiplier=SimpleNamespace(as_decimal=lambda: Decimal(1)),
            make_qty=lambda value, round_down=True: SimpleNamespace(as_decimal=lambda: Decimal(0)),
        ),
    }
    strategy._latest_bars = {
        "BTC": SimpleNamespace(close=SimpleNamespace(as_decimal=lambda: Decimal(41000))),
    }
    strategy._target_quantities = {"BTC": Decimal("0.00009")}

    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "portfolio",
        property(lambda self: SimpleNamespace(net_position=lambda instrument_id: Decimal(0))),
        raising=False,
    )
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "config",
        property(
            lambda self: SimpleNamespace(
                min_order_notional_usd=5.0,
                use_passive_limits=False,
                allow_market_fallback=False,
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        CryptoXSecTrendConditionalStrategy,
        "log",
        property(lambda self: SimpleNamespace(warning=lambda *args, **kwargs: None)),
        raising=False,
    )

    assert strategy._submit_delta_order("BTC", Decimal("0.00009")) is False
    assert strategy._target_quantities == {}
