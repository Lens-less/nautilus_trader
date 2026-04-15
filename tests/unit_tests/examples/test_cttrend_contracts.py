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

import importlib
import inspect
import json
import sys
from pathlib import Path

import msgspec
import pytest

import nautilus_trader
import nautilus_trader.examples
import nautilus_trader.examples.strategies
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import StrategyFactory


TEST_DATA_DIR = Path(__file__).resolve().parents[2] / "test_data" / "crypto_rv"
WORKTREE_ROOT = Path(__file__).resolve().parents[3]
IMPORTABLE_CONFIG_PATH = TEST_DATA_DIR / "ctrend_importable_strategy_config.json"
STRATEGY_MODULE = "nautilus_trader.examples.strategies.crypto_xsec_trend"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def event_loop(session_event_loop):
    return session_event_loop


def _extend_worktree_package_paths() -> None:
    worktree_root = str(WORKTREE_ROOT)
    strategy_root = str(WORKTREE_ROOT / "nautilus_trader" / "examples" / "strategies")
    examples_root = str(WORKTREE_ROOT / "nautilus_trader" / "examples")
    package_root = str(WORKTREE_ROOT / "nautilus_trader")

    if worktree_root not in sys.path:
        sys.path.append(worktree_root)
    if package_root not in nautilus_trader.__path__:
        nautilus_trader.__path__.append(package_root)
    if examples_root not in nautilus_trader.examples.__path__:
        nautilus_trader.examples.__path__.append(examples_root)
    if strategy_root not in nautilus_trader.examples.strategies.__path__:
        nautilus_trader.examples.strategies.__path__.append(strategy_root)
    importlib.invalidate_caches()


def test_crypto_xsec_trend_strategy_is_factory_importable_from_example_config() -> None:
    _extend_worktree_package_paths()
    payload = _load_json(IMPORTABLE_CONFIG_PATH)

    importable = ImportableStrategyConfig.parse(msgspec.json.encode(payload))
    strategy = StrategyFactory.create(importable)

    assert strategy.__class__.__name__ == "CryptoXSecTrendStrategy"
    assert strategy.config.volatility_managed is True
    assert len(strategy.config.universe_instrument_ids) == 4


def test_crypto_xsec_trend_strategy_module_exports_expected_types() -> None:
    _extend_worktree_package_paths()
    module = importlib.import_module(STRATEGY_MODULE)

    strategy_type = module.CryptoXSecTrendStrategy
    config_type = module.CryptoXSecTrendConfig

    assert inspect.isclass(strategy_type)
    assert inspect.isclass(config_type)
