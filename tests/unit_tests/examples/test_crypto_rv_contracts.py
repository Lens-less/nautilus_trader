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

import asyncio
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import msgspec
import pandas as pd
import pytest

from examples.backtest.crypto_rv import report as crypto_rv_report
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import StrategyFactory


TEST_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data" / "crypto_rv"
VALID_SNAPSHOT_PATH = TEST_DATA_DIR / "universe_snapshot_valid.json"
INVALID_SNAPSHOT_PATH = TEST_DATA_DIR / "universe_snapshot_missing_formation_ts.json"
IMPORTABLE_CONFIG_PATH = TEST_DATA_DIR / "importable_strategy_config.json"

STRATEGY_MODULE = "nautilus_trader.examples.strategies.crypto_rv_basket"
SNAPSHOT_MODULE_CANDIDATES = (
    "examples.backtest.crypto_rv.prepare_universe",
    "examples.backtest.crypto_rv.schemas",
    "examples.backtest.crypto_rv.snapshot",
)
FUNDING_FN_CANDIDATES = (
    "classify_funding_coverage",
    "classify_funding_coverage_ratio",
    "classify_short_leg_funding_coverage",
)
RENORMALIZE_FN_CANDIDATES = (
    "renormalize_long_weights",
    "renormalize_long_leg_weights",
    "build_hype_out_weights",
)
SNAPSHOT_TYPE_CANDIDATES = (
    "UniverseSnapshot",
    "CryptoRVUniverseSnapshot",
)
ARTIFACT_MODULE_CANDIDATES = ("examples.backtest.crypto_rv.schemas",)
STRATEGY_TYPE_CANDIDATES = (
    "CryptoRVBasket",
    "CryptoRVBasketStrategy",
)
CONFIG_TYPE_CANDIDATES = (
    "CryptoRVBasketConfig",
    "CryptoRVConfig",
)
FEATURE_PANEL_MANIFEST_CANDIDATES = ("load_feature_panel_manifest",)
LANE_ARTIFACT_MANIFEST_CANDIDATES = ("load_lane_artifact_manifest",)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _import_module_or_xfail(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.xfail(f"crypto RV implementation module not available yet: {module_name} ({exc})")


def _resolve_first_module_or_xfail(module_names: tuple[str, ...]):
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    pytest.xfail(
        "crypto RV universe implementation module not available yet: "
        + ", ".join(module_names),
    )


def _resolve_attr_or_xfail(module, candidates: tuple[str, ...], label: str):
    for name in candidates:
        value = getattr(module, name, None)
        if value is not None:
            return value
    pytest.xfail(
        f"crypto RV implementation does not yet expose {label}: "
        + ", ".join(f"{module.__name__}:{name}" for name in candidates),
    )


def _normalize_classification(result) -> str:
    if hasattr(result, "value"):
        result = result.value
    return str(result).strip().lower().replace("-", " ").replace("_", " ")


def _extract_importable_payload_or_xfail() -> dict:
    if not IMPORTABLE_CONFIG_PATH.exists():
        pytest.xfail(f"crypto RV importable config fixture missing: {IMPORTABLE_CONFIG_PATH}")
    return _load_json(IMPORTABLE_CONFIG_PATH)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def test_crypto_rv_snapshot_fixture_covers_pit_and_exclusions_contract() -> None:
    payload = _load_json(VALID_SNAPSHOT_PATH)

    assert payload["formation_ts"] == "2026-01-13T00:00:00Z"
    assert payload["screening_ts"] == "2026-01-13T00:00:00Z"
    assert payload["thresholds"]["max_volume_24h_usd"] == 10_000_000
    assert len(payload["candidates"]) == 4
    assert payload["selected_short_count"] == 2
    assert payload["selected_short_symbols"] == ["ALPHAUSDT", "GAMMAUSDT"]
    assert payload["selected_short_instrument_ids"] == [
        "ALPHAUSDT-PERP.BINANCE",
        "GAMMAUSDT-PERP.BINANCE",
    ]
    assert {
        tuple(item["exclusion_reasons"])
        for item in payload["candidates"]
        if item["eligible"] is False
    } == {
        ("insufficient_pre_window_history",),
        ("history_ends_before_window",),
    }
    assert payload["selected_short_funding_coverage_ratio"] == pytest.approx(0.925)


def test_crypto_rv_snapshot_schema_accepts_valid_fixture_and_rejects_missing_formation_ts() -> None:
    module = _resolve_first_module_or_xfail(ARTIFACT_MODULE_CANDIDATES)
    load_snapshot = _resolve_attr_or_xfail(module, ("load_snapshot",), "snapshot loader")

    decoded = load_snapshot(VALID_SNAPSHOT_PATH)
    assert decoded is not None

    with pytest.raises((msgspec.ValidationError, TypeError, ValueError, KeyError)):
        load_snapshot(INVALID_SNAPSHOT_PATH)


def test_crypto_rv_funding_coverage_thresholds_match_prd_classification() -> None:
    module = _resolve_first_module_or_xfail(ARTIFACT_MODULE_CANDIDATES)
    classify = _resolve_attr_or_xfail(module, FUNDING_FN_CANDIDATES, "funding coverage classifier")

    below_threshold = _normalize_classification(classify(0.79))
    warning_band = _normalize_classification(classify(0.80))
    fully_covered = _normalize_classification(classify(0.95))

    assert "partial" in below_threshold
    assert ("warning" in warning_band) or ("residual" in warning_band) or ("disclosure" in warning_band)
    assert ("primary" in fully_covered) or ("credible" in fully_covered) or ("full" in fully_covered)


def test_crypto_rv_hype_out_renormalization_preserves_long_gross_notional() -> None:
    module = _resolve_first_module_or_xfail(ARTIFACT_MODULE_CANDIDATES)
    renormalize = _resolve_attr_or_xfail(module, RENORMALIZE_FN_CANDIDATES, "HYPE-out renormalizer")

    input_weights = {
        "BTC": 0.20,
        "ETH": 0.20,
        "SOL": 0.20,
        "BNB": 0.20,
        "HYPE": 0.20,
    }

    output_weights = renormalize(input_weights, excluded_symbol="HYPE")

    assert "HYPE" not in output_weights
    assert pytest.approx(sum(output_weights.values()), rel=1e-9) == 1.0
    assert output_weights == {
        "BTC": pytest.approx(0.25),
        "ETH": pytest.approx(0.25),
        "SOL": pytest.approx(0.25),
        "BNB": pytest.approx(0.25),
    }


def test_crypto_rv_strategy_is_factory_importable_from_example_config() -> None:
    _import_module_or_xfail(STRATEGY_MODULE)
    payload = _extract_importable_payload_or_xfail()

    importable = ImportableStrategyConfig.parse(msgspec.json.encode(payload))
    strategy = StrategyFactory.create(importable)

    strategy_module = importlib.import_module(importable.strategy_path.split(":")[0])
    strategy_type = _resolve_attr_or_xfail(strategy_module, STRATEGY_TYPE_CANDIDATES, "strategy type")
    _resolve_attr_or_xfail(strategy_module, CONFIG_TYPE_CANDIDATES, "strategy config type")

    assert isinstance(strategy, strategy_type)


def test_crypto_rv_strategy_module_exports_expected_types() -> None:
    module = _import_module_or_xfail(STRATEGY_MODULE)
    strategy_type = _resolve_attr_or_xfail(module, STRATEGY_TYPE_CANDIDATES, "strategy type")
    config_type = _resolve_attr_or_xfail(module, CONFIG_TYPE_CANDIDATES, "strategy config type")

    assert inspect.isclass(strategy_type)
    assert inspect.isclass(config_type)


def test_crypto_rv_report_keeps_index_compatibility_with_lane_manifests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "research.json"
    snapshot_path = tmp_path / "snapshot.json"
    prepared_catalog_path = tmp_path / "prepared_catalog.json"
    run_output_dir = tmp_path / "run-output"
    report_output_path = tmp_path / "report.json"
    primary_dir = run_output_dir / "primary"
    stress_dir = run_output_dir / "primary-stress"

    _write_json(
        config_path,
        {
            "schema_version": "1.0",
            "research_name": "crypto-rv-report-contract",
            "formation_ts": "2026-01-01T00:00:00Z",
            "test_window_start": "2026-01-02T00:00:00Z",
            "test_window_end": "2026-02-01T00:00:00Z",
            "universe_input_path": str(tmp_path / "universe.csv"),
            "history_manifest_path": str(tmp_path / "history.csv"),
            "snapshot_output_path": str(snapshot_path),
            "prepared_catalog_output_path": str(prepared_catalog_path),
            "run_output_dir": str(run_output_dir),
            "report_output_path": str(report_output_path),
            "long_leg": [
                {
                    "asset": asset,
                    "raw_symbol": f"{asset}USDT-PERP",
                    "instrument_id": f"{asset}USDT-PERP.BINANCE",
                    "weight": 0.2,
                }
                for asset in ("BTC", "ETH", "SOL", "BNB", "HYPE")
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
            "research_name": "crypto-rv-report-contract",
            "formation_ts": "2026-01-01T00:00:00Z",
            "screening_ts": "2025-12-31T12:00:00Z",
            "test_window_start": "2026-01-02T00:00:00Z",
            "test_window_end": "2026-02-01T00:00:00Z",
            "thresholds": {"market_cap_min_usdt": 10_000_000},
            "sources": {"market_cap": "synthetic", "volume_24h": "synthetic"},
            "long_leg_assets": ["BTC", "ETH", "SOL", "BNB", "HYPE"],
            "selected_short_count": 2,
            "selected_short_symbols": ["ALPHAUSDT", "GAMMAUSDT"],
            "selected_short_instrument_ids": [
                "ALPHAUSDT-PERP.BINANCE",
                "GAMMAUSDT-PERP.BINANCE",
            ],
            "selected_short_funding_coverage_ratio": 0.925,
            "funding_coverage_classification": "credible hypothesis test with funding warning",
            "funding_coverage_basis": "eligible-short average",
            "candidates": [
                {
                    "raw_symbol": "ALPHAUSDT",
                    "instrument_id": "ALPHAUSDT-PERP.BINANCE",
                    "market_cap_usd": 30_000_000,
                    "volume_24h_usd": 4_500_000,
                    "listed_ts": "2025-10-01T00:00:00Z",
                    "delisted_ts": None,
                    "first_price_ts": "2025-10-01T00:00:00Z",
                    "price_history_start_ts": "2025-10-01T00:00:00Z",
                    "price_history_end_ts": "2026-02-01T00:00:00Z",
                    "price_coverage_ratio": 1.0,
                    "funding_coverage_ratio": 0.97,
                    "market_cap_source": "synthetic",
                    "volume_24h_source": "synthetic",
                    "metadata_source": "synthetic",
                    "screen_rank": 1,
                    "frozen_rank": 1,
                    "eligible": True,
                    "selected": True,
                    "selection_rank": 1,
                    "exclusion_reasons": [],
                },
                {
                    "raw_symbol": "GAMMAUSDT",
                    "instrument_id": "GAMMAUSDT-PERP.BINANCE",
                    "market_cap_usd": 35_000_000,
                    "volume_24h_usd": 4_000_000,
                    "listed_ts": "2025-10-02T00:00:00Z",
                    "delisted_ts": None,
                    "first_price_ts": "2025-10-02T00:00:00Z",
                    "price_history_start_ts": "2025-10-02T00:00:00Z",
                    "price_history_end_ts": "2026-02-01T00:00:00Z",
                    "price_coverage_ratio": 1.0,
                    "funding_coverage_ratio": 0.88,
                    "market_cap_source": "synthetic",
                    "volume_24h_source": "synthetic",
                    "metadata_source": "synthetic",
                    "screen_rank": 2,
                    "frozen_rank": 2,
                    "eligible": True,
                    "selected": True,
                    "selection_rank": 2,
                    "exclusion_reasons": [],
                },
            ],
        },
    )

    _write_json(
        prepared_catalog_path,
        {
            "schema_version": "1.0",
            "research_name": "crypto-rv-report-contract",
            "snapshot_path": str(snapshot_path),
            "history_manifest_path": str(tmp_path / "history.csv"),
            "artifact_root": str(tmp_path / "artifact-root"),
            "catalog_path": None,
            "catalog_ready": False,
            "mode": "plan-only",
            "selected_short_funding_coverage_ratio": 0.925,
            "classification": "credible hypothesis test with funding warning",
            "notes": ["Synthetic report fixture"],
            "validated_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "HYPEUSDT"],
            "staged_files": [],
            "manifest_entries": [
                {
                    "raw_symbol": symbol,
                    "instrument_id": instrument_id,
                    "price_path": str(tmp_path / "artifact-root" / f"{symbol.lower()}_bars.parquet"),
                    "funding_path": str(tmp_path / "artifact-root" / f"{symbol.lower()}_funding.parquet"),
                    "start_ts": "2026-01-02T00:00:00Z",
                    "end_ts": "2026-02-01T00:00:00Z",
                    "price_coverage_ratio": 1.0,
                    "funding_coverage_ratio": 1.0,
                    "data_cls": "nautilus_trader.model.data:Bar",
                    "bar_spec": "60-MINUTE-LAST",
                    "catalog_path": None,
                }
                for symbol, instrument_id in (
                    ("BTCUSDT", "BTCUSDT-PERP.BINANCE"),
                    ("ETHUSDT", "ETHUSDT-PERP.BINANCE"),
                    ("SOLUSDT", "SOLUSDT-PERP.BINANCE"),
                    ("BNBUSDT", "BNBUSDT-PERP.BINANCE"),
                    ("HYPEUSDT", "HYPEUSDT-PERP.BINANCE"),
                    ("ALPHAUSDT", "ALPHAUSDT-PERP.BINANCE"),
                    ("GAMMAUSDT", "GAMMAUSDT-PERP.BINANCE"),
                )
            ],
        },
    )

    _write_json(primary_dir / "result.json", {
        "variant_name": "primary",
        "classification": "credible hypothesis test with funding warning",
        "status": "planned",
        "reasons": ["plan-only fixture"],
        "run_request_path": str(primary_dir / "run_request.json"),
        "lane_manifest_path": str(primary_dir / "lane_manifest.json"),
    })
    _write_json(stress_dir / "result.json", {
        "variant_name": "primary-stress",
        "classification": "credible hypothesis test with funding warning",
        "status": "planned",
        "reasons": ["plan-only fixture"],
        "run_request_path": str(stress_dir / "run_request.json"),
        "lane_manifest_path": str(stress_dir / "lane_manifest.json"),
    })
    _write_json(
        run_output_dir / "index.json",
        {
            "research_name": "crypto-rv-report-contract",
            "snapshot_path": str(snapshot_path),
            "prepared_catalog_path": str(prepared_catalog_path),
            "variants": [
                {
                    "variant_name": "primary",
                    "result_path": str(primary_dir / "result.json"),
                    "request_path": str(primary_dir / "run_request.json"),
                    "status": "planned",
                    "classification": "credible hypothesis test with funding warning",
                },
                {
                    "variant_name": "primary-stress",
                    "result_path": str(stress_dir / "result.json"),
                    "request_path": str(stress_dir / "run_request.json"),
                    "status": "planned",
                    "classification": "credible hypothesis test with funding warning",
                },
            ],
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report.py",
            "--config",
            str(config_path),
            "--snapshot",
            str(snapshot_path),
            "--prepared-catalog",
            str(prepared_catalog_path),
            "--run-output-dir",
            str(run_output_dir),
            "--output",
            str(report_output_path),
        ],
    )

    crypto_rv_report.main()

    report_payload = _load_json(report_output_path)
    assert report_payload["run_output_dir"] == str(run_output_dir)
    assert report_payload["overall_status"] == "planning-only"
    assert report_payload["primary_variant"]["variant_name"] == "primary"
    assert report_payload["primary_variant"]["lane_manifest_path"] == str(primary_dir / "lane_manifest.json")
    assert [row["variant"] for row in report_payload["variant_rows"]] == [
        "primary",
        "primary-stress",
    ]
    assert report_output_path.with_suffix(".md").exists()


def test_crypto_rv_run_backtest_emits_machine_artifact_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    template_path = repo_root / "examples" / "backtest" / "crypto_rv" / "configs" / "research.example.json"
    config_payload = _load_json(template_path)
    config_dir = template_path.parent

    config_payload["universe_input_path"] = str(config_dir / config_payload["universe_input_path"])
    config_payload["history_manifest_path"] = str(config_dir / config_payload["history_manifest_path"])
    config_payload["snapshot_output_path"] = str(tmp_path / "output" / "example_snapshot.json")
    config_payload["prepared_catalog_output_path"] = str(tmp_path / "output" / "example_prepared_catalog.json")
    config_payload["run_output_dir"] = str(tmp_path / "output" / "runs")
    config_payload["report_output_path"] = str(tmp_path / "output" / "example_report.json")

    config_path = _write_json(tmp_path / "research.runtime.json", config_payload)

    for script_name in (
        "prepare_universe.py",
        "prepare_catalog.py",
        "run_backtest.py",
        "report.py",
    ):
        subprocess.run(
            [
                sys.executable,
                f"examples/backtest/crypto_rv/{script_name}",
                "--config",
                str(config_path),
            ],
            cwd=repo_root,
            check=True,
        )

    artifact_module = _resolve_first_module_or_xfail(ARTIFACT_MODULE_CANDIDATES)
    load_feature_panel_manifest = _resolve_attr_or_xfail(
        artifact_module,
        FEATURE_PANEL_MANIFEST_CANDIDATES,
        "feature panel manifest loader",
    )
    load_lane_manifest = _resolve_attr_or_xfail(
        artifact_module,
        LANE_ARTIFACT_MANIFEST_CANDIDATES,
        "lane artifact manifest loader",
    )

    run_root = tmp_path / "output" / "runs"
    index_payload = _load_json(run_root / "index.json")
    primary_payload = next(item for item in index_payload["variants"] if item["variant_name"] == "primary")
    lane_manifest = load_lane_manifest(Path(primary_payload["lane_manifest_path"]))
    feature_manifest = load_feature_panel_manifest(Path(lane_manifest.feature_panel_manifest_path))

    required_paths = [
        Path(lane_manifest.feature_panel_path),
        Path(lane_manifest.signal_panel_path),
        Path(lane_manifest.signal_metrics_path),
        Path(lane_manifest.portfolio_metrics_path),
        Path(lane_manifest.portfolio_timeseries_path),
        Path(lane_manifest.summary_report_path),
    ]
    assert all(path.exists() for path in required_paths)

    feature_frame = pd.read_parquet(lane_manifest.feature_panel_path)
    signal_frame = pd.read_parquet(lane_manifest.signal_panel_path)
    portfolio_frame = pd.read_parquet(lane_manifest.portfolio_timeseries_path)

    assert feature_manifest.variant_name == "primary"
    assert feature_manifest.artifact_path == lane_manifest.feature_panel_path
    assert set(feature_manifest.feature_columns).issubset(feature_frame.columns)
    assert {"score", "rank", "target_weight", "placeholder"}.issubset(signal_frame.columns)
    assert {
        "gross_return",
        "net_return",
        "long_return",
        "short_return",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "active_names",
        "placeholder",
    }.issubset(portfolio_frame.columns)
    assert len(feature_frame) > 0
    assert len(signal_frame) > 0
    assert len(portfolio_frame) > 1
