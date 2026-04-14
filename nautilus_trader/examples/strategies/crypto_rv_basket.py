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

from decimal import Decimal
from typing import Literal

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


class CryptoRVBasketConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``CryptoRVBasketStrategy`` instances.

    Parameters
    ----------
    long_instrument_ids : tuple[InstrumentId, ...]
        The long-leg instruments.
    short_instrument_ids : tuple[InstrumentId, ...]
        The short-leg instruments.
    bar_types : tuple[BarType, ...]
        One bar type for each tracked instrument.
    leg_notional_usd : float
        The target USD notional per leg.
    rebalance_cadence : {"weekly", "biweekly"}, default "weekly"
        The rebalance interval.
    excluded_instrument_ids : tuple[InstrumentId, ...], optional
        Instruments to ignore while keeping the remaining leg equal-weighted.
    min_order_notional_usd : float, default 25.0
        Minimum notional delta required before a rebalance order is submitted.
    order_time_in_force : TimeInForce, optional
        Optional market-order time in force.
    close_positions_on_stop : bool, default True
        If all tracked positions should be closed when the strategy stops.
    long_fee_bps : float, default 0.0
        Research knob for long-leg fee assumptions.
    short_fee_bps : float, default 0.0
        Research knob for short-leg fee assumptions.
    long_slippage_bps : float, default 0.0
        Research knob for long-leg slippage assumptions.
    short_slippage_bps : float, default 0.0
        Research knob for short-leg slippage assumptions.
    long_funding_bps_per_day : float, default 0.0
        Research knob for long-leg funding assumptions.
    short_funding_bps_per_day : float, default 0.0
        Research knob for short-leg funding assumptions.
    short_borrow_bps_per_day : float, default 0.0
        Research knob for short-leg borrow/carry assumptions.

    """

    long_instrument_ids: tuple[InstrumentId, ...]
    short_instrument_ids: tuple[InstrumentId, ...]
    bar_types: tuple[BarType, ...]
    leg_notional_usd: PositiveFloat
    rebalance_cadence: RebalanceCadence = "weekly"
    excluded_instrument_ids: tuple[InstrumentId, ...] = ()
    min_order_notional_usd: PositiveFloat = 25.0
    order_time_in_force: TimeInForce | None = None
    close_positions_on_stop: bool = True
    long_fee_bps: NonNegativeFloat = 0.0
    short_fee_bps: NonNegativeFloat = 0.0
    long_slippage_bps: NonNegativeFloat = 0.0
    short_slippage_bps: NonNegativeFloat = 0.0
    long_funding_bps_per_day: NonNegativeFloat = 0.0
    short_funding_bps_per_day: NonNegativeFloat = 0.0
    short_borrow_bps_per_day: NonNegativeFloat = 0.0


class CryptoRVBasketStrategy(Strategy):
    """
    Lightweight research-first basket strategy for long-major/short-alt RV tests.

    The strategy assumes the universe and data contract were prepared upstream.
    It only consumes explicit instrument lists, tracks synchronized bars, and
    rebalances both legs to equal-weight USD notional targets.

    """

    def __init__(self, config: CryptoRVBasketConfig) -> None:
        self._validate_config(config)
        super().__init__(config)

        self._all_instrument_ids = config.long_instrument_ids + config.short_instrument_ids
        self._excluded_ids = set(config.excluded_instrument_ids)
        self._bar_types = self._build_bar_type_index(config.bar_types)
        self._instruments: dict[InstrumentId, Instrument] = {}
        self._latest_bars: dict[InstrumentId, Bar] = {}
        self._last_rebalance_ts: int | None = None
        self._rebalance_count = 0
        self._rebalance_interval_ns = self._cadence_to_ns(config.rebalance_cadence)

    @staticmethod
    def _cadence_to_ns(cadence: RebalanceCadence) -> int:
        days = 7 if cadence == "weekly" else 14
        return days * 24 * 60 * 60 * 1_000_000_000

    def _validate_config(self, config: CryptoRVBasketConfig) -> None:
        PyCondition.is_true(bool(config.long_instrument_ids), "long_instrument_ids must not be empty")
        PyCondition.is_true(bool(config.short_instrument_ids), "short_instrument_ids must not be empty")

        long_set = set(config.long_instrument_ids)
        short_set = set(config.short_instrument_ids)
        overlap = long_set.intersection(short_set)
        PyCondition.is_true(not overlap, f"long/short universe overlap detected: {sorted(map(str, overlap))}")

        tracked_ids = config.long_instrument_ids + config.short_instrument_ids
        PyCondition.is_true(
            len(tracked_ids) == len(set(tracked_ids)),
            "duplicate instrument IDs are not allowed",
        )

        bar_ids = [bar_type.instrument_id for bar_type in config.bar_types]
        PyCondition.is_true(
            len(bar_ids) == len(set(bar_ids)),
            "bar_types must contain one entry per instrument",
        )
        PyCondition.is_true(
            set(bar_ids) == set(tracked_ids),
            "bar_types must exactly match the configured long and short instruments",
        )
        PyCondition.is_true(
            set(config.excluded_instrument_ids).issubset(set(tracked_ids)),
            "excluded_instrument_ids must be a subset of tracked instruments",
        )

    def _build_bar_type_index(self, bar_types: tuple[BarType, ...]) -> dict[InstrumentId, BarType]:
        return {bar_type.instrument_id: bar_type for bar_type in bar_types}

    def on_start(self) -> None:
        """
        Actions to be performed on strategy start.
        """
        for instrument_id in self._all_instrument_ids:
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                self.log.error(f"Could not find instrument for {instrument_id}")
                self.stop()
                return

            self._instruments[instrument_id] = instrument
            self.subscribe_bars(self._bar_types[instrument_id])

        self.log.info(
            "Started CryptoRVBasketStrategy "
            f"with {len(self.config.long_instrument_ids)} longs, "
            f"{len(self.config.short_instrument_ids)} shorts, "
            f"cadence={self.config.rebalance_cadence}, "
            f"leg_notional_usd={self.config.leg_notional_usd}",
        )
        self.log.info(
            "Research cost knobs "
            f"(long_fee_bps={self.config.long_fee_bps}, short_fee_bps={self.config.short_fee_bps}, "
            f"long_slippage_bps={self.config.long_slippage_bps}, short_slippage_bps={self.config.short_slippage_bps}, "
            f"long_funding_bps_per_day={self.config.long_funding_bps_per_day}, "
            f"short_funding_bps_per_day={self.config.short_funding_bps_per_day}, "
            f"short_borrow_bps_per_day={self.config.short_borrow_bps_per_day})",
        )

    def on_bar(self, bar: Bar) -> None:
        """
        Actions to be performed when the strategy receives a bar.
        """
        instrument_id = bar.bar_type.instrument_id
        if instrument_id not in self._instruments:
            return

        self._latest_bars[instrument_id] = bar

        if not self._has_synchronized_snapshot(bar.ts_event):
            return

        if self._last_rebalance_ts is not None:
            elapsed_ns = bar.ts_event - self._last_rebalance_ts
            if elapsed_ns < self._rebalance_interval_ns:
                return

        self._rebalance(bar.ts_event)

    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy stops.
        """
        for instrument_id in self._all_instrument_ids:
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

    def _has_synchronized_snapshot(self, ts_event: int) -> bool:
        active_ids = [instrument_id for instrument_id in self._all_instrument_ids if instrument_id not in self._excluded_ids]
        return all(
            (bar := self._latest_bars.get(instrument_id)) is not None and bar.ts_event == ts_event
            for instrument_id in active_ids
        )

    def _rebalance(self, ts_event: int) -> None:
        targets: dict[InstrumentId, Decimal] = {}

        long_ids = self._select_active_ids(self.config.long_instrument_ids)
        short_ids = self._select_active_ids(self.config.short_instrument_ids)

        if not long_ids or not short_ids:
            self.log.warning(
                f"Skipping rebalance at {ts_event}: active longs={len(long_ids)}, active shorts={len(short_ids)}",
            )
            return

        per_long_notional = Decimal(str(self.config.leg_notional_usd)) / Decimal(len(long_ids))
        per_short_notional = Decimal(str(self.config.leg_notional_usd)) / Decimal(len(short_ids))

        for instrument_id in long_ids:
            targets[instrument_id] = self._target_quantity_for_notional(instrument_id, per_long_notional)
        for instrument_id in short_ids:
            targets[instrument_id] = -self._target_quantity_for_notional(instrument_id, per_short_notional)

        self._close_untracked_positions(targets)

        submitted = 0
        for instrument_id, target_quantity in targets.items():
            submitted += int(self._submit_delta_order(instrument_id, target_quantity))

        self._last_rebalance_ts = ts_event
        self._rebalance_count += 1
        self.log.info(
            f"Rebalanced crypto RV basket #{self._rebalance_count}: "
            f"longs={len(long_ids)} shorts={len(short_ids)} submitted_orders={submitted}",
        )

    def _select_active_ids(self, instrument_ids: tuple[InstrumentId, ...]) -> list[InstrumentId]:
        active_ids: list[InstrumentId] = []
        for instrument_id in instrument_ids:
            if instrument_id in self._excluded_ids:
                continue

            bar = self._latest_bars.get(instrument_id)
            if bar is None or bar.is_single_price():
                continue

            if bar.close.as_decimal() <= 0:
                continue

            active_ids.append(instrument_id)

        return active_ids

    def _target_quantity_for_notional(self, instrument_id: InstrumentId, target_notional_usd: Decimal) -> Decimal:
        instrument = self._instruments[instrument_id]
        bar = self._latest_bars[instrument_id]
        price = Decimal(bar.close.as_decimal())
        multiplier = Decimal(instrument.multiplier.as_decimal())
        return target_notional_usd / (price * multiplier)

    def _close_untracked_positions(self, targets: dict[InstrumentId, Decimal]) -> None:
        for instrument_id in self._all_instrument_ids:
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


CryptoRVBasket = CryptoRVBasketStrategy
