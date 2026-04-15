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
