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
from pathlib import Path

import msgspec
import pytest

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
STRATEGY_TYPE_CANDIDATES = (
    "CryptoRVBasket",
    "CryptoRVBasketStrategy",
)
CONFIG_TYPE_CANDIDATES = (
    "CryptoRVBasketConfig",
    "CryptoRVConfig",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


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
    module = _resolve_first_module_or_xfail(SNAPSHOT_MODULE_CANDIDATES)
    load_snapshot = _resolve_attr_or_xfail(module, ("load_snapshot",), "snapshot loader")

    decoded = load_snapshot(VALID_SNAPSHOT_PATH)
    assert decoded is not None

    with pytest.raises((msgspec.ValidationError, TypeError, ValueError, KeyError)):
        load_snapshot(INVALID_SNAPSHOT_PATH)


def test_crypto_rv_funding_coverage_thresholds_match_prd_classification() -> None:
    module = _resolve_first_module_or_xfail(SNAPSHOT_MODULE_CANDIDATES)
    classify = _resolve_attr_or_xfail(module, FUNDING_FN_CANDIDATES, "funding coverage classifier")

    below_threshold = _normalize_classification(classify(0.79))
    warning_band = _normalize_classification(classify(0.80))
    fully_covered = _normalize_classification(classify(0.95))

    assert "partial" in below_threshold
    assert ("warning" in warning_band) or ("residual" in warning_band) or ("disclosure" in warning_band)
    assert ("primary" in fully_covered) or ("credible" in fully_covered) or ("full" in fully_covered)


def test_crypto_rv_hype_out_renormalization_preserves_long_gross_notional() -> None:
    module = _resolve_first_module_or_xfail(SNAPSHOT_MODULE_CANDIDATES)
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
