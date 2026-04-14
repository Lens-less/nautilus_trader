from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from examples.backtest.crypto_rv import run_backtest as crypto_rv_run_backtest
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


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


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


def _build_run_contract_config(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = tmp_path / "research.json"
    snapshot_path = tmp_path / "snapshot.json"
    prepared_catalog_path = tmp_path / "prepared_catalog.json"
    run_output_dir = tmp_path / "run-output"
    artifact_root = tmp_path / "artifact-root"

    long_members = [
        ("BTC", "BTCUSDT-PERP.BINANCE"),
        ("ETH", "ETHUSDT-PERP.BINANCE"),
        ("SOL", "SOLUSDT-PERP.BINANCE"),
        ("BNB", "BNBUSDT-PERP.BINANCE"),
        ("HYPE", "HYPEUSDT-PERP.BINANCE"),
    ]
    short_members = [
        ("ALPHAUSDT", "ALPHAUSDT-PERP.BINANCE", 1, True, 0.97),
        ("GAMMAUSDT", "GAMMAUSDT-PERP.BINANCE", 2, True, 0.88),
    ]

    _write_json(
        config_path,
        {
            "schema_version": "1.0",
            "research_name": "crypto-rv-kernel-contract",
            "formation_ts": "2026-01-01T00:00:00Z",
            "test_window_start": "2026-01-02T00:00:00Z",
            "test_window_end": "2026-02-01T00:00:00Z",
            "universe_input_path": str(tmp_path / "universe.csv"),
            "history_manifest_path": str(tmp_path / "history.csv"),
            "snapshot_output_path": str(snapshot_path),
            "prepared_catalog_output_path": str(prepared_catalog_path),
            "run_output_dir": str(run_output_dir),
            "report_output_path": str(tmp_path / "report.json"),
            "long_leg": [
                {
                    "asset": asset,
                    "raw_symbol": f"{asset}USDT-PERP",
                    "instrument_id": instrument_id,
                    "weight": 0.2,
                }
                for asset, instrument_id in long_members
            ],
            "thresholds": {
                "min_market_cap_usd": 10_000_000,
                "max_market_cap_usd": 100_000_000,
                "max_volume_24h_usd": 10_000_000,
                "pre_window_history_days": 14,
                "min_price_coverage_ratio": 1.0,
                "funding_partial_evidence_threshold": 0.8,
                "funding_warning_threshold": 0.95,
            },
            "cost_models": {
                "base": {"fee_bps": 4.0, "slippage_bps": 8.0, "short_borrow_bps_annual": 0.0},
                "stress": {
                    "fee_bps": 8.0,
                    "slippage_bps": 25.0,
                    "short_borrow_bps_annual": 300.0,
                },
            },
            "strategy": {
                "strategy_path": "nautilus_trader.examples.strategies.crypto_rv_basket:CryptoRVBasketStrategy",
                "config_path": "nautilus_trader.examples.strategies.crypto_rv_basket:CryptoRVBasketConfig",
                "allow_placeholder_runner": True,
            },
            "sensitivity": {
                "enable_hype_out": True,
                "short_counts": [2],
                "rebalance_cadences": ["weekly"],
            },
            "primary_short_count": 2,
            "min_short_count": 2,
            "max_short_count": 2,
            "rebalance_cadence": "weekly",
            "neutrality_tolerance_pct": 1.0,
            "venue": "BINANCE",
            "base_currency": "USDT",
        },
    )

    _write_json(
        snapshot_path,
        {
            "schema_version": "1.0",
            "research_name": "crypto-rv-kernel-contract",
            "formation_ts": "2026-01-01T00:00:00Z",
            "screening_ts": "2025-12-31T12:00:00Z",
            "test_window_start": "2026-01-02T00:00:00Z",
            "test_window_end": "2026-02-01T00:00:00Z",
            "thresholds": {
                "market_cap_min_usdt": 10_000_000,
                "market_cap_max_usdt": 100_000_000,
                "volume_24h_max_usdt": 10_000_000,
            },
            "sources": {"market_cap": "synthetic", "volume_24h": "synthetic"},
            "long_leg_assets": [asset for asset, _ in long_members],
            "selected_short_count": 2,
            "selected_short_symbols": [raw_symbol for raw_symbol, _, _, _, _ in short_members],
            "selected_short_instrument_ids": [instrument_id for _, instrument_id, _, _, _ in short_members],
            "selected_short_funding_coverage_ratio": 0.925,
            "funding_coverage_classification": "credible hypothesis test with funding warning",
            "funding_coverage_basis": "eligible-short average",
            "candidates": [
                {
                    "raw_symbol": raw_symbol,
                    "instrument_id": instrument_id,
                    "market_cap_usd": 25_000_000 + (rank * 5_000_000),
                    "volume_24h_usd": 5_000_000 - (rank * 500_000),
                    "listed_ts": "2025-10-01T00:00:00Z",
                    "delisted_ts": None,
                    "first_price_ts": "2025-10-01T00:00:00Z",
                    "price_history_start_ts": "2025-10-01T00:00:00Z",
                    "price_history_end_ts": "2026-02-01T00:00:00Z",
                    "price_coverage_ratio": 1.0,
                    "funding_coverage_ratio": funding_coverage_ratio,
                    "market_cap_source": "synthetic",
                    "volume_24h_source": "synthetic",
                    "metadata_source": "synthetic",
                    "screen_rank": rank,
                    "frozen_rank": rank,
                    "eligible": True,
                    "selected": selected,
                    "selection_rank": rank if selected else None,
                    "exclusion_reasons": [],
                }
                for raw_symbol, instrument_id, rank, selected, funding_coverage_ratio in short_members
            ],
        },
    )

    manifest_entries = [
        {
            "raw_symbol": raw_symbol,
            "instrument_id": instrument_id,
            "price_path": str(artifact_root / f"{raw_symbol.lower()}_bars.parquet"),
            "funding_path": str(artifact_root / f"{raw_symbol.lower()}_funding.parquet"),
            "start_ts": "2026-01-02T00:00:00Z",
            "end_ts": "2026-02-01T00:00:00Z",
            "price_coverage_ratio": 1.0,
            "funding_coverage_ratio": 1.0,
            "data_cls": "nautilus_trader.model.data:Bar",
            "bar_spec": "60-MINUTE-LAST",
            "catalog_path": None,
        }
        for raw_symbol, instrument_id in (
            [(f"{asset}USDT", instrument_id) for asset, instrument_id in long_members]
            + [(raw_symbol, instrument_id) for raw_symbol, instrument_id, *_ in short_members]
        )
    ]

    _write_json(
        prepared_catalog_path,
        {
            "schema_version": "1.0",
            "research_name": "crypto-rv-kernel-contract",
            "snapshot_path": str(snapshot_path),
            "history_manifest_path": str(tmp_path / "history.csv"),
            "artifact_root": str(artifact_root),
            "catalog_path": None,
            "catalog_ready": False,
            "mode": "plan-only",
            "selected_short_funding_coverage_ratio": 0.925,
            "classification": "credible hypothesis test with funding warning",
            "notes": ["Synthetic contract fixture"],
            "validated_symbols": [raw_symbol for raw_symbol, _ in [(f"{asset}USDT", instrument_id) for asset, instrument_id in long_members]],
            "staged_files": [],
            "manifest_entries": manifest_entries,
        },
    )

    return config_path, snapshot_path, prepared_catalog_path, run_output_dir


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


def test_plan_only_run_backtest_emits_lane_artifact_contract_per_variant(tmp_path, monkeypatch):
    config_path, snapshot_path, prepared_catalog_path, run_output_dir = _build_run_contract_config(
        tmp_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest.py",
            "--config",
            str(config_path),
            "--snapshot",
            str(snapshot_path),
            "--prepared-catalog",
            str(prepared_catalog_path),
            "--output-dir",
            str(run_output_dir),
        ],
    )

    crypto_rv_run_backtest.main()

    index_payload = json.loads((run_output_dir / "index.json").read_text(encoding="utf-8"))
    assert {item["variant_name"] for item in index_payload["variants"]} == {
        "primary",
        "primary-stress",
        "no-hype",
        "short-2",
    }

    for item in index_payload["variants"]:
        result_path = Path(item["result_path"])
        request_path = Path(item["request_path"])
        variant_dir = result_path.parent

        assert result_path.exists()
        assert request_path.exists()

        lane_manifest_path = variant_dir / "lane_manifest.json"
        feature_panel_manifest_path = variant_dir / "feature_panel_manifest.json"
        signal_metrics_path = variant_dir / "signal_metrics.json"
        portfolio_metrics_path = variant_dir / "portfolio_metrics.json"
        summary_report_path = variant_dir / "summary_report.md"

        assert lane_manifest_path.exists()
        assert feature_panel_manifest_path.exists()
        assert signal_metrics_path.exists()
        assert portfolio_metrics_path.exists()
        assert summary_report_path.exists()

        lane_manifest = json.loads(lane_manifest_path.read_text(encoding="utf-8"))
        feature_panel_manifest = json.loads(feature_panel_manifest_path.read_text(encoding="utf-8"))

        assert lane_manifest["variant_name"] == item["variant_name"]
        assert lane_manifest["status"] == item["status"]
        assert Path(lane_manifest["run_request_path"]) == request_path
        assert Path(lane_manifest["result_path"]) == result_path

        assert Path(lane_manifest["feature_panel_manifest_path"]).name == "feature_panel_manifest.json"
        assert Path(lane_manifest["signal_panel_path"]).name == "signal_panel.parquet"
        assert Path(lane_manifest["portfolio_timeseries_path"]).name == "portfolio_timeseries.parquet"
        assert Path(lane_manifest["signal_metrics_path"]).name == "signal_metrics.json"
        assert Path(lane_manifest["portfolio_metrics_path"]).name == "portfolio_metrics.json"
        assert Path(lane_manifest["summary_report_path"]).name == "summary_report.md"

        assert feature_panel_manifest["variant_name"] == item["variant_name"]
        assert Path(feature_panel_manifest["feature_panel_path"]).suffix == ".parquet"
