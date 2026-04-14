from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from examples.backtest.crypto_rv.run_backtest import select_ranked_eligible
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import StrategyFactory


FIXTURES = Path(__file__).resolve().parents[2] / "test_data" / "crypto_rv"
FORMATION_TS = "2026-01-01T00:00:00Z"


@pytest.fixture
def event_loop(session_event_loop):
    return session_event_loop


def _load_fixture(name: str) -> dict:
    with (FIXTURES / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _validate_snapshot(snapshot: dict) -> None:
    required_top_level = {
        "formation_ts",
        "screening_ts",
        "source",
        "thresholds",
        "long_basket",
        "selected_short_symbols",
        "candidates",
    }
    assert required_top_level.issubset(snapshot)

    formation_ts = _parse_utc(snapshot["formation_ts"])
    screening_ts = _parse_utc(snapshot["screening_ts"])
    assert screening_ts <= formation_ts
    assert snapshot["formation_ts"] == FORMATION_TS
    assert snapshot["long_basket"] == ["BTC", "ETH", "SOL", "BNB", "HYPE"]

    candidates = snapshot["candidates"]
    assert candidates

    candidate_by_symbol = {candidate["raw_symbol"]: candidate for candidate in candidates}
    assert len(candidate_by_symbol) == len(candidates)
    assert set(snapshot["selected_short_symbols"]).issubset(candidate_by_symbol)

    selected_from_candidates = sorted(
        candidate["raw_symbol"] for candidate in candidates if candidate["included"]
    )
    assert sorted(snapshot["selected_short_symbols"]) == selected_from_candidates

    for candidate in candidates:
        assert candidate["raw_symbol"]
        assert candidate["instrument_id"].endswith(".BINANCE")
        assert "listed_at" in candidate
        assert "first_price_ts" in candidate
        assert "pre_window_history_days" in candidate
        assert "funding_coverage_ratio" in candidate
        assert "funding_missing" in candidate
        assert "included" in candidate
        assert "exclusion_reason" in candidate
        assert isinstance(candidate["market_cap_usdt"], int)
        assert isinstance(candidate["volume_24h_usdt"], int)

        listed_at = _parse_utc(candidate["listed_at"])
        first_price_ts = _parse_utc(candidate["first_price_ts"])
        if listed_at > formation_ts:
            assert candidate["included"] is False
            assert candidate["exclusion_reason"] == "listed_after_formation"
        if candidate["included"]:
            assert listed_at <= formation_ts
            assert first_price_ts <= formation_ts
            assert candidate["pre_window_history_days"] >= 14
            assert candidate["exclusion_reason"] is None
            assert candidate["funding_missing"] is False
            assert isinstance(candidate["funding_coverage_ratio"], float)
        else:
            assert candidate["exclusion_reason"]


def _classify_short_funding_coverage(coverage_ratio: float) -> str:
    if coverage_ratio < 0.80:
        return "partial evidence only"
    if coverage_ratio < 0.95:
        return "primary_with_residual_risk"
    return "primary"


def _renormalize_long_weights(
    weights: dict[str, Decimal],
    excluded_symbols: set[str],
) -> dict[str, Decimal]:
    remaining = {symbol: weight for symbol, weight in weights.items() if symbol not in excluded_symbols}
    total = sum(remaining.values(), start=Decimal(0))
    if total == 0:
        raise ValueError("Cannot renormalize an empty long basket")
    return {symbol: weight / total for symbol, weight in remaining.items()}


def test_universe_snapshot_fixture_has_required_pit_fields_and_exclusion_handling():
    snapshot = _load_fixture("universe_snapshot.json")

    _validate_snapshot(snapshot)


def test_universe_snapshot_validator_rejects_future_listing_and_missing_exclusion_reason():
    snapshot = _load_fixture("universe_snapshot.json")

    future_listing = deepcopy(snapshot)
    future_listing["candidates"][0]["listed_at"] = "2026-01-01T00:01:00Z"

    with pytest.raises(AssertionError):
        _validate_snapshot(future_listing)

    missing_reason = deepcopy(snapshot)
    missing_reason["candidates"][4]["exclusion_reason"] = None

    with pytest.raises(AssertionError):
        _validate_snapshot(missing_reason)


@pytest.mark.parametrize(
    ("coverage_ratio", "expected"),
    [
        (0.79, "partial evidence only"),
        (0.80, "primary_with_residual_risk"),
        (0.949, "primary_with_residual_risk"),
        (0.95, "primary"),
    ],
)
def test_short_funding_coverage_classification_boundaries(coverage_ratio: float, expected: str):
    assert _classify_short_funding_coverage(coverage_ratio) == expected


def test_hype_out_renormalizes_remaining_long_weights_without_changing_gross_exposure():
    weights = {
        "BTC": Decimal("0.2"),
        "ETH": Decimal("0.2"),
        "SOL": Decimal("0.2"),
        "BNB": Decimal("0.2"),
        "HYPE": Decimal("0.2"),
    }

    renormalized = _renormalize_long_weights(weights, {"HYPE"})

    assert set(renormalized) == {"BTC", "ETH", "SOL", "BNB"}
    assert sum(renormalized.values(), start=Decimal(0)) == Decimal(1)
    assert all(weight == Decimal("0.25") for weight in renormalized.values())


@dataclass
class _FakeCandidate:
    raw_symbol: str
    frozen_rank: int | None
    selection_rank: int | None


class _FakeSnapshot:
    def __init__(self) -> None:
        self._eligible = [
            _FakeCandidate("ZZZUSDT", frozen_rank=3, selection_rank=None),
            _FakeCandidate("AAAUSDT", frozen_rank=1, selection_rank=1),
            _FakeCandidate("MMMUSDT", frozen_rank=2, selection_rank=None),
        ]

    def eligible_candidates(self):
        return list(self._eligible)

    def ranked_eligible_candidates(self):
        return sorted(self._eligible, key=lambda candidate: candidate.frozen_rank or 10_000_000)

    def selected_candidates(self):
        return [self._eligible[1]]


def test_select_ranked_eligible_uses_frozen_rank_order_for_sensitivity_baskets():
    snapshot = _FakeSnapshot()

    ranked = select_ranked_eligible(snapshot)

    assert [candidate.raw_symbol for candidate in ranked] == [
        "AAAUSDT",
        "MMMUSDT",
        "ZZZUSDT",
    ]


def test_crypto_rv_strategy_importable_from_planned_example_path():
    fixture = _load_fixture("importable_strategy_config.json")

    raw = json.dumps(fixture).encode("utf-8")
    importable = ImportableStrategyConfig.parse(raw)

    strategy = StrategyFactory.create(importable)

    assert strategy.__class__.__name__ == "CryptoRVBasketStrategy"
    assert [str(instrument_id) for instrument_id in strategy.config.long_instrument_ids] == [
        "BTCUSDT-PERP.BINANCE",
        "ETHUSDT-PERP.BINANCE",
        "SOLUSDT-PERP.BINANCE",
        "BNBUSDT-PERP.BINANCE",
        "HYPEUSDT-PERP.BINANCE",
    ]
    assert [str(instrument_id) for instrument_id in strategy.config.short_instrument_ids] == [
        "ALPHAUSDT-PERP.BINANCE",
        "GAMMAUSDT-PERP.BINANCE",
    ]
    assert strategy.config.leg_notional_usd == 1_000.0
