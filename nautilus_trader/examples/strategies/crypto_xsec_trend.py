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

import sys
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Literal


try:
    from examples.backtest.crypto_rv.signals.vol_managed_trend import TrendSignalSpec
    from examples.backtest.crypto_rv.signals.vol_managed_trend import build_cross_sectional_scores
    from examples.backtest.crypto_rv.signals.vol_managed_trend import build_target_weights
    from examples.backtest.crypto_rv.signals.vol_managed_trend import compute_trend_components
except ModuleNotFoundError:  # pragma: no cover - worktree/runtime fallback
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from examples.backtest.crypto_rv.signals.vol_managed_trend import TrendSignalSpec
    from examples.backtest.crypto_rv.signals.vol_managed_trend import build_cross_sectional_scores
    from examples.backtest.crypto_rv.signals.vol_managed_trend import build_target_weights
    from examples.backtest.crypto_rv.signals.vol_managed_trend import compute_trend_components
from nautilus_trader.config import NonNegativeFloat
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


RebalanceCadence = Literal["weekly", "biweekly"]


class CryptoXSecTrendConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``CryptoXSecTrendStrategy`` instances.
    """

    universe_instrument_ids: tuple[InstrumentId, ...]
    bar_types: tuple[BarType, ...]
    leg_notional_usd: PositiveFloat
    rebalance_cadence: RebalanceCadence = "weekly"
    long_bucket_frac: PositiveFloat = 0.2
    short_bucket_frac: PositiveFloat = 0.2
    fast_window: int = 2
    slow_window: int = 4
    sma_window: int = 4
    vol_window: int = 4
    volume_window: int = 2
    min_history_bars: int = 5
    volatility_managed: bool = True
    min_order_notional_usd: PositiveFloat = 25.0
    order_time_in_force: TimeInForce | None = None
    close_positions_on_stop: bool = True
    long_fee_bps: NonNegativeFloat = 0.0
    short_fee_bps: NonNegativeFloat = 0.0
    long_slippage_bps: NonNegativeFloat = 0.0
    short_slippage_bps: NonNegativeFloat = 0.0
    short_borrow_bps_per_day: NonNegativeFloat = 0.0


class CryptoXSecTrendStrategy(Strategy):
    """
    Cross-sectional trend strategy using multi-indicator ranking over a broad universe.
    """

    def __init__(self, config: CryptoXSecTrendConfig) -> None:
        self._validate_config(config)
        super().__init__(config)

        self._bar_types = self._build_bar_type_index(config.bar_types)
        self._instruments: dict[InstrumentId, Instrument] = {}
        self._latest_bars: dict[InstrumentId, Bar] = {}
        self._close_history: dict[InstrumentId, deque[float]] = {}
        self._volume_history: dict[InstrumentId, deque[float]] = {}
        self._last_rebalance_ts: int | None = None
        self._rebalance_count = 0
        self._rebalance_interval_ns = self._cadence_to_ns(config.rebalance_cadence)
        self._signal_spec = TrendSignalSpec(
            fast_window=config.fast_window,
            slow_window=config.slow_window,
            sma_window=config.sma_window,
            vol_window=config.vol_window,
            volume_window=config.volume_window,
            min_history_bars=config.min_history_bars,
        )

    @staticmethod
    def _cadence_to_ns(cadence: RebalanceCadence) -> int:
        days = 7 if cadence == "weekly" else 14
        return days * 24 * 60 * 60 * 1_000_000_000

    def _validate_config(self, config: CryptoXSecTrendConfig) -> None:
        PyCondition.is_true(
            bool(config.universe_instrument_ids),
            "universe_instrument_ids must not be empty",
        )
        PyCondition.is_true(
            len(config.universe_instrument_ids) == len(set(config.universe_instrument_ids)),
            "duplicate universe instrument IDs are not allowed",
        )
        bar_ids = [bar_type.instrument_id for bar_type in config.bar_types]
        PyCondition.is_true(
            set(bar_ids) == set(config.universe_instrument_ids),
            "bar_types must exactly match universe_instrument_ids",
        )
        PyCondition.is_true(
            0.0 < config.long_bucket_frac < 1.0,
            "long_bucket_frac must be within (0, 1)",
        )
        PyCondition.is_true(
            0.0 < config.short_bucket_frac < 1.0,
            "short_bucket_frac must be within (0, 1)",
        )
        PyCondition.is_true(config.fast_window > 0, "fast_window must be positive")
        PyCondition.is_true(config.slow_window > 0, "slow_window must be positive")
        PyCondition.is_true(config.sma_window > 0, "sma_window must be positive")
        PyCondition.is_true(config.vol_window > 0, "vol_window must be positive")
        PyCondition.is_true(config.volume_window > 0, "volume_window must be positive")
        PyCondition.is_true(config.min_history_bars > 0, "min_history_bars must be positive")

    def _build_bar_type_index(self, bar_types: tuple[BarType, ...]) -> dict[InstrumentId, BarType]:
        return {bar_type.instrument_id: bar_type for bar_type in bar_types}

    def on_start(self) -> None:
        max_history = self.config.min_history_bars + 1
        for instrument_id in self.config.universe_instrument_ids:
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                self.log.error(f"Could not find instrument for {instrument_id}")
                self.stop()
                return

            self._instruments[instrument_id] = instrument
            self._close_history[instrument_id] = deque(maxlen=max_history)
            self._volume_history[instrument_id] = deque(maxlen=max_history)
            self.subscribe_bars(self._bar_types[instrument_id])

        self.log.info(
            "Started CryptoXSecTrendStrategy "
            f"with {len(self.config.universe_instrument_ids)} instruments, "
            f"cadence={self.config.rebalance_cadence}, "
            f"bucket_frac=({self.config.long_bucket_frac}, {self.config.short_bucket_frac}), "
            f"vol_managed={self.config.volatility_managed}",
        )

    def on_bar(self, bar: Bar) -> None:
        instrument_id = bar.bar_type.instrument_id
        if instrument_id not in self._instruments:
            return

        self._latest_bars[instrument_id] = bar
        self._close_history[instrument_id].append(float(bar.close.as_decimal()))
        self._volume_history[instrument_id].append(float(bar.volume.as_decimal()))

        if self._last_rebalance_ts is not None:
            elapsed_ns = bar.ts_event - self._last_rebalance_ts
            if elapsed_ns < self._rebalance_interval_ns:
                return

        if not self._has_signal_ready_snapshot(bar.ts_event):
            return

        self._rebalance(bar.ts_event)

    def on_stop(self) -> None:
        for instrument_id in self.config.universe_instrument_ids:
            self.cancel_all_orders(instrument_id)
            self.unsubscribe_bars(self._bar_types[instrument_id])
            if self.config.close_positions_on_stop:
                self.close_all_positions(instrument_id)

    def on_save(self) -> dict[str, bytes]:
        return {
            "last_rebalance_ts": str(self._last_rebalance_ts or 0).encode(),
            "rebalance_count": str(self._rebalance_count).encode(),
        }

    def on_load(self, state: dict[str, bytes]) -> None:
        last_rebalance_ts = state.get("last_rebalance_ts")
        rebalance_count = state.get("rebalance_count")
        if last_rebalance_ts is not None:
            value = int(last_rebalance_ts.decode())
            self._last_rebalance_ts = value or None
        if rebalance_count is not None:
            self._rebalance_count = int(rebalance_count.decode())

    def _has_signal_ready_snapshot(self, ts_event: int) -> bool:
        return len(self._active_signal_ids(ts_event)) >= 2

    def _active_signal_ids(self, ts_event: int) -> list[InstrumentId]:
        active_ids: list[InstrumentId] = []
        for instrument_id in self.config.universe_instrument_ids:
            bar = self._latest_bars.get(instrument_id)
            if bar is None or bar.ts_event != ts_event or bar.is_single_price():
                continue
            if bar.close.as_decimal() <= 0:
                continue
            if len(self._close_history[instrument_id]) < self.config.min_history_bars:
                continue
            active_ids.append(instrument_id)
        return active_ids

    def _rebalance(self, ts_event: int) -> None:
        active_ids = self._active_signal_ids(ts_event)
        if len(active_ids) < 2:
            self.log.warning("Skipping rebalance: not enough active instruments")
            return

        components_by_instrument = {
            str(instrument_id): compute_trend_components(
                list(self._close_history[instrument_id]),
                list(self._volume_history[instrument_id]),
                self._signal_spec,
            )
            for instrument_id in active_ids
        }
        score_payload = build_cross_sectional_scores(
            components_by_instrument,
            volatility_managed=self.config.volatility_managed,
        )
        weights = build_target_weights(
            {
                instrument_id: payload["composite_score"]
                for instrument_id, payload in score_payload.items()
            },
            long_bucket_frac=self.config.long_bucket_frac,
            short_bucket_frac=self.config.short_bucket_frac,
        )

        gross_notional = Decimal(str(self.config.leg_notional_usd)) * 2
        targets: dict[InstrumentId, Decimal] = {}
        for instrument_id in active_ids:
            weight = Decimal(str(weights[str(instrument_id)]))
            if weight == 0:
                continue
            target_quantity = self._target_quantity_for_notional(
                instrument_id,
                gross_notional * abs(weight),
            )
            targets[instrument_id] = target_quantity if weight > 0 else -target_quantity

        self._close_untracked_positions(targets)

        submitted = 0
        for instrument_id, target_quantity in targets.items():
            submitted += int(self._submit_delta_order(instrument_id, target_quantity))

        long_count = len([weight for weight in weights.values() if weight > 0])
        short_count = len([weight for weight in weights.values() if weight < 0])
        self._last_rebalance_ts = ts_event
        self._rebalance_count += 1
        self.log.info(
            "Rebalanced crypto cross-sectional trend basket "
            f"#{self._rebalance_count}: longs={long_count} shorts={short_count} "
            f"submitted_orders={submitted}",
        )

    def _target_quantity_for_notional(self, instrument_id: InstrumentId, target_notional_usd: Decimal) -> Decimal:
        instrument = self._instruments[instrument_id]
        bar = self._latest_bars[instrument_id]
        price = Decimal(bar.close.as_decimal())
        multiplier = Decimal(instrument.multiplier.as_decimal())
        return target_notional_usd / (price * multiplier)

    def _close_untracked_positions(self, targets: dict[InstrumentId, Decimal]) -> None:
        for instrument_id in self.config.universe_instrument_ids:
            if instrument_id in targets:
                continue

            current_quantity = Decimal(self.portfolio.net_position(instrument_id))
            if current_quantity != 0:
                self.close_all_positions(instrument_id)

    def _submit_delta_order(self, instrument_id: InstrumentId, target_quantity: Decimal) -> bool:
        current_quantity = Decimal(self.portfolio.net_position(instrument_id))
        delta_quantity = target_quantity - current_quantity
        if delta_quantity == 0:
            return False

        instrument = self._instruments[instrument_id]
        price = Decimal(self._latest_bars[instrument_id].close.as_decimal())
        multiplier = Decimal(instrument.multiplier.as_decimal())
        delta_notional = abs(delta_quantity) * price * multiplier
        if delta_notional < Decimal(str(self.config.min_order_notional_usd)):
            return False

        order_side = OrderSide.BUY if delta_quantity > 0 else OrderSide.SELL
        quantity = instrument.make_qty(abs(delta_quantity), round_down=True)
        if quantity.as_decimal() <= 0:
            return False

        order: MarketOrder = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            time_in_force=self.config.order_time_in_force or TimeInForce.GTC,
        )
        self.submit_order(order)
        return True


CryptoXSecTrend = CryptoXSecTrendStrategy
