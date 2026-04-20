from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from examples.backtest.crypto_rv.run_ctrend_backtest import VariantSpec
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import (
    build_conditioned_weight_plan,
)
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import build_rank_bands
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import (
    compute_combo_baseline_metrics,
)
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import conditioned_short_weights
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import (
    evaluate_optimization_candidate,
)
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import (
    matched_legacy_baseline_variant,
)
from examples.backtest.crypto_rv.run_ctrend_liq_combo_real_public import period_funding_return


@pytest.fixture
def event_loop(session_event_loop):
    return session_event_loop


def test_build_rank_bands_slices_expected_ranges() -> None:
    top_candidates = [f"ALT{i:02d}-PERP.BINANCE" for i in range(1, 51)]

    bands = build_rank_bands(top_candidates)

    assert len(bands["rank_1_20"]) == 20
    assert len(bands["rank_1_30"]) == 30
    assert len(bands["rank_1_50"]) == 50
    assert bands["rank_21_30"] == set(top_candidates[20:30])
    assert bands["rank_21_50"] == set(top_candidates[20:50])
    assert bands["rank_31_50"] == set(top_candidates[30:50])


def test_hard_filter_limits_shorts_to_band_and_preserves_long_leg() -> None:
    composite_scores = {
        "LONG_A": 0.6,
        "LONG_B": 0.5,
        "SHORT_1": -0.1,
        "SHORT_2": -0.2,
        "SHORT_3": -0.3,
        "SHORT_4": -0.4,
    }
    weights = conditioned_short_weights(
        composite_scores,
        list(composite_scores),
        band_set={"SHORT_3", "SHORT_4"},
        mode="hard_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
        tilt_multiplier=2.0,
    )

    assert weights["LONG_A"] > 0
    assert weights["LONG_B"] > 0
    assert weights["SHORT_4"] < 0
    assert weights["SHORT_3"] < 0
    assert weights["SHORT_1"] == 0.0
    assert weights["SHORT_2"] == 0.0


def test_post_rank_filter_only_keeps_intersection_with_cttrend_short_bucket() -> None:
    composite_scores = {
        "LONG_A": 0.6,
        "LONG_B": 0.5,
        "SHORT_1": -0.1,
        "SHORT_2": -0.2,
        "SHORT_3": -0.3,
        "SHORT_4": -0.4,
    }
    weights = conditioned_short_weights(
        composite_scores,
        list(composite_scores),
        band_set={"SHORT_4"},
        mode="post_rank_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
        tilt_multiplier=2.0,
    )

    assert weights["LONG_A"] > 0
    assert weights["SHORT_4"] < 0
    assert weights["SHORT_3"] == 0.0


def test_short_tilt_overweights_target_band_and_preserves_short_gross() -> None:
    composite_scores = {
        "LONG_A": 0.6,
        "LONG_B": 0.5,
        "SHORT_1": -0.1,
        "SHORT_2": -0.2,
        "SHORT_3": -0.3,
        "SHORT_4": -0.4,
    }
    weights = conditioned_short_weights(
        composite_scores,
        list(composite_scores),
        band_set={"SHORT_4"},
        mode="short_tilt",
        long_bucket_frac=0.25,
        short_bucket_frac=0.5,
        tilt_multiplier=2.0,
    )

    short_weights = {key: value for key, value in weights.items() if value < 0}
    assert abs(sum(short_weights.values()) + 0.5) < 1e-9
    assert abs(weights["LONG_A"] - 0.25) < 1e-9
    assert abs(weights["LONG_B"] - 0.25) < 1e-9
    assert abs(weights["SHORT_4"]) > abs(weights["SHORT_3"])


def test_post_rank_filter_falls_back_to_wider_band_when_threshold_not_met() -> None:
    composite_scores = {
        "LONG_A": 0.8,
        "LONG_B": 0.7,
        "SHORT_1": -0.1,
        "SHORT_2": -0.2,
        "SHORT_3": -0.3,
        "SHORT_4": -0.4,
    }

    plan = build_conditioned_weight_plan(
        composite_scores,
        list(composite_scores),
        band_name="rank_1_20",
        band_set={"SHORT_4"},
        mode="post_rank_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.5,
        tilt_multiplier=2.0,
        min_active_shorts=2,
        fallback_band_sequence=(("rank_1_30", {"SHORT_3", "SHORT_4"}),),
        flatten_on_breach=True,
    )

    assert plan.selected_band_name == "rank_1_30"
    assert plan.fallback_used is True
    assert plan.flattened_due_to_min_shorts is False
    assert plan.shorts == ["SHORT_4", "SHORT_3"]
    assert plan.weights["SHORT_4"] < 0.0
    assert plan.weights["SHORT_3"] < 0.0


def test_post_rank_filter_flattens_when_no_band_meets_threshold() -> None:
    composite_scores = {
        "LONG_A": 0.8,
        "LONG_B": 0.7,
        "SHORT_1": -0.1,
        "SHORT_2": -0.2,
        "SHORT_3": -0.3,
        "SHORT_4": -0.4,
    }

    plan = build_conditioned_weight_plan(
        composite_scores,
        list(composite_scores),
        band_name="rank_1_20",
        band_set={"SHORT_4"},
        mode="post_rank_filter",
        long_bucket_frac=0.25,
        short_bucket_frac=0.5,
        tilt_multiplier=2.0,
        min_active_shorts=3,
        fallback_band_sequence=(("rank_1_30", {"SHORT_3", "SHORT_4"}),),
        flatten_on_breach=True,
    )

    assert plan.selected_band_name == "rank_1_20"
    assert plan.fallback_used is False
    assert plan.flattened_due_to_min_shorts is True
    assert plan.shorts == []
    assert all(weight >= 0.0 for instrument_id, weight in plan.weights.items() if instrument_id.startswith("SHORT_"))


@dataclass
class StubFundingLoader:
    rate_sums: dict[str, float]

    def rate_sum(self, symbol: str, start_ms: int, end_ms: int) -> float:
        assert end_ms > start_ms
        return self.rate_sums.get(symbol, 0.0)


def test_period_funding_return_applies_long_short_signs_without_touching_signal_logic() -> None:
    weights = {
        "BTCUSDT-PERP.BINANCE": 0.25,
        "ETHUSDT-PERP.BINANCE": 0.25,
        "FORTHUSDT-PERP.BINANCE": -0.25,
        "SIRENUSDT-PERP.BINANCE": -0.25,
    }
    funding_loader = StubFundingLoader(
        rate_sums={
            "BTCUSDT": 0.0010,
            "ETHUSDT": -0.0020,
            "FORTHUSDT": 0.0040,
        },
    )

    funding_return = period_funding_return(
        weights,
        period_start="2026-01-13T00:00:00Z",
        period_end="2026-01-20T00:00:00Z",
        funding_loader=funding_loader,
    )

    assert abs(funding_return - 0.00125) < 1e-12


def test_evaluate_optimization_candidate_requires_all_guardrails() -> None:
    summary = evaluate_optimization_candidate(
        parity_result={
            "net_return": 0.20,
            "max_drawdown": -0.05,
            "total_turnover": 1.0,
            "funding_return_lift": 0.02,
        },
        experimental_result={
            "net_return": 0.21,
            "max_drawdown": -0.04,
            "total_turnover": 1.2,
            "funding_return_lift": 0.018,
        },
    )

    assert summary["credible_optimization_candidate"] is True

    rejected = evaluate_optimization_candidate(
        parity_result={
            "net_return": 0.20,
            "max_drawdown": -0.05,
            "total_turnover": 1.0,
            "funding_return_lift": 0.02,
        },
        experimental_result={
            "net_return": 0.205,
            "max_drawdown": -0.06,
            "total_turnover": 1.3,
            "funding_return_lift": 0.01,
        },
    )

    assert rejected["credible_optimization_candidate"] is False


def test_matched_legacy_baseline_variant_uses_cost_and_cadence() -> None:
    assert (
        matched_legacy_baseline_variant(
            VariantSpec(
                name="vol-managed-stress",
                cost_model_name="stress",
                rebalance_cadence="weekly",
                volatility_managed=True,
            ),
        )
        == "primary_stress"
    )
    assert (
        matched_legacy_baseline_variant(
            VariantSpec(
                name="vol-managed-biweekly-base",
                cost_model_name="base",
                rebalance_cadence="biweekly",
                volatility_managed=True,
            ),
        )
        == "primary_base_biweekly"
    )


def test_combo_baseline_metrics_respects_ctrend_warmup_window() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    close_frame = pd.DataFrame(
        {
            "LONG-PERP.BINANCE": [1.0, 2.0, 4.0, 8.0, 8.0, 8.0],
            "SHORT-PERP.BINANCE": [1.0, 0.5, 0.25, 0.125, 0.125, 0.125],
        },
        index=index,
    )
    variant = VariantSpec(
        name="raw-base",
        cost_model_name="base",
        rebalance_cadence="weekly",
        volatility_managed=False,
    )
    config = SimpleNamespace(
        signal=SimpleNamespace(min_history_bars=4),
        cost_models={
            "base": SimpleNamespace(
                fee_bps=0.0,
                slippage_bps=0.0,
                short_borrow_bps_annual=0.0,
            ),
        },
    )

    metrics = compute_combo_baseline_metrics(
        close_frame=close_frame,
        index=index,
        baseline_longs=["LONG-PERP.BINANCE"],
        baseline_shorts=["SHORT-PERP.BINANCE"],
        variant=variant,
        config=config,
        funding_loader=StubFundingLoader(rate_sums={}),
    )

    assert metrics["price_only_net_return"] == 0.0
    assert metrics["net_return"] == 0.0
