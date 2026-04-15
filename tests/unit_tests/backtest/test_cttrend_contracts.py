from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.backtest.crypto_rv.ctrend_schemas import load_ctrend_config
from examples.backtest.crypto_rv.run_ctrend_backtest import build_price_matrices
from examples.backtest.crypto_rv.run_ctrend_backtest import build_variants
from examples.backtest.crypto_rv.run_ctrend_real_public import load_real_public_universe
from examples.backtest.crypto_rv.schemas import HistoryManifestEntry
from examples.backtest.crypto_rv.signals.vol_managed_trend import TrendSignalSpec
from examples.backtest.crypto_rv.signals.vol_managed_trend import build_cross_sectional_scores
from examples.backtest.crypto_rv.signals.vol_managed_trend import build_target_weights
from examples.backtest.crypto_rv.signals.vol_managed_trend import centered_rank
from examples.backtest.crypto_rv.signals.vol_managed_trend import compute_trend_components
from examples.backtest.crypto_rv.signals.vol_managed_trend import select_bucket_members


REPO_ROOT = Path(__file__).resolve().parents[3]
CTREND_CONFIG = REPO_ROOT / "examples" / "backtest" / "crypto_rv" / "configs" / "ctrend.example.json"


@pytest.fixture
def event_loop(session_event_loop):
    return session_event_loop


def test_centered_rank_is_symmetric_and_stable() -> None:
    ranked = centered_rank({"AAA": 1.0, "BBB": 3.0, "CCC": 2.0})

    assert ranked == {
        "AAA": pytest.approx(-0.5),
        "CCC": pytest.approx(0.0),
        "BBB": pytest.approx(0.5),
    }


def test_compute_trend_components_requires_history_and_emits_expected_fields() -> None:
    spec = TrendSignalSpec(
        fast_window=2,
        slow_window=4,
        sma_window=4,
        vol_window=4,
        volume_window=2,
        min_history_bars=5,
    )
    components = compute_trend_components(
        [100.0, 101.0, 102.0, 104.0, 105.0, 107.0],
        [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0],
        spec,
    )

    assert set(components) == {
        "fast_momentum",
        "slow_momentum",
        "ma_gap",
        "volume_confirmation",
        "realized_vol",
    }
    assert components["slow_momentum"] > 0
    assert components["ma_gap"] > 0


def test_build_cross_sectional_scores_and_bucket_weights_use_extremes() -> None:
    components = {
        "AAA": {
            "fast_momentum": 0.10,
            "slow_momentum": 0.12,
            "ma_gap": 0.08,
            "volume_confirmation": 0.01,
            "realized_vol": 0.02,
        },
        "BBB": {
            "fast_momentum": 0.02,
            "slow_momentum": 0.03,
            "ma_gap": 0.01,
            "volume_confirmation": 0.00,
            "realized_vol": 0.03,
        },
        "CCC": {
            "fast_momentum": -0.03,
            "slow_momentum": -0.02,
            "ma_gap": -0.01,
            "volume_confirmation": -0.01,
            "realized_vol": 0.02,
        },
        "DDD": {
            "fast_momentum": -0.09,
            "slow_momentum": -0.08,
            "ma_gap": -0.07,
            "volume_confirmation": -0.02,
            "realized_vol": 0.04,
        },
    }

    scores = build_cross_sectional_scores(components, volatility_managed=True)
    composite_scores = {
        instrument_id: payload["composite_score"]
        for instrument_id, payload in scores.items()
    }

    longs, shorts = select_bucket_members(
        composite_scores,
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
    )
    weights = build_target_weights(
        composite_scores,
        long_bucket_frac=0.25,
        short_bucket_frac=0.25,
    )

    assert longs == ["AAA"]
    assert shorts == ["DDD"]
    assert weights["AAA"] == pytest.approx(0.5)
    assert weights["DDD"] == pytest.approx(-0.5)


def test_ctrend_example_config_loads() -> None:
    config = load_ctrend_config(CTREND_CONFIG)

    assert config.research_name == "crypto-ctrend-example"
    assert config.strategy.strategy_path.endswith("CryptoXSecTrendStrategy")
    assert config.portfolio.rebalance_cadence == "weekly"
    assert config.signal.min_history_bars == 5


def test_build_variants_respects_config_cadence_and_emits_sensitivity() -> None:
    config = load_ctrend_config(CTREND_CONFIG)

    variants = build_variants(config)

    assert variants[0].rebalance_cadence == "weekly"
    assert any(variant.rebalance_cadence == "biweekly" for variant in variants)


def test_build_price_matrices_uses_union_index_not_strict_intersection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    alpha_path = tmp_path / "alpha.csv"
    beta_path = tmp_path / "beta.csv"

    alpha_path.write_text(
        (
            "ts_event,open,high,low,close,volume\n"
            "2026-01-13T00:00:00Z,1,1,1,1,100\n"
            "2026-01-13T01:00:00Z,1,1,1,2,100\n"
            "2026-01-13T02:00:00Z,1,1,1,3,100\n"
        ),
        encoding="utf-8",
    )
    beta_path.write_text(
        (
            "ts_event,open,high,low,close,volume\n"
            "2026-01-13T01:00:00Z,1,1,1,2,100\n"
            "2026-01-13T02:00:00Z,1,1,1,3,100\n"
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        "raw_symbol,instrument_id,price_path,funding_path,start_ts,end_ts,price_coverage_ratio,funding_coverage_ratio,data_cls,bar_spec,catalog_path\n",
        encoding="utf-8",
    )

    prepared_catalog = SimpleNamespace(
        history_manifest_path=str(manifest_path),
        manifest_entries=[
            HistoryManifestEntry(
                raw_symbol="ALPHAUSDT-PERP",
                instrument_id="ALPHAUSDT-PERP.BINANCE",
                price_path=str(alpha_path),
                funding_path="",
                start_ts="2026-01-13T00:00:00Z",
                end_ts="2026-01-13T02:00:00Z",
                price_coverage_ratio=1.0,
                funding_coverage_ratio=1.0,
            ),
            HistoryManifestEntry(
                raw_symbol="BETAUSDT-PERP",
                instrument_id="BETAUSDT-PERP.BINANCE",
                price_path=str(beta_path),
                funding_path="",
                start_ts="2026-01-13T01:00:00Z",
                end_ts="2026-01-13T02:00:00Z",
                price_coverage_ratio=1.0,
                funding_coverage_ratio=1.0,
            ),
        ],
    )

    close_frame, _volume_frame, _raw_symbols = build_price_matrices(prepared_catalog)

    assert len(close_frame.index) == 3
    assert json.loads(close_frame.iloc[0].to_json())["BETAUSDT-PERP.BINANCE"] is None


def test_load_real_public_universe_uses_ranked_top_n_and_fixed_top20_baseline(tmp_path: Path) -> None:
    path = tmp_path / "selected_universe.json"
    path.write_text(
        json.dumps(
            {
                "longs": ["BTCUSDT", "ETHUSDT"],
                "all_ranked_candidates": [
                    {"pair": "AAAUSDT"},
                    {"pair": "BBBUSDT"},
                    {"pair": "CCCUSDT"},
                ],
            },
        ),
        encoding="utf-8",
    )

    longs, top_candidates, baseline_shorts = load_real_public_universe(path, top_n_candidates=2)

    assert longs == ["BTCUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE"]
    assert top_candidates == ["AAAUSDT-PERP.BINANCE", "BBBUSDT-PERP.BINANCE"]
    assert baseline_shorts == ["AAAUSDT-PERP.BINANCE", "BBBUSDT-PERP.BINANCE"]
