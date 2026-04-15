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

import json
import sys
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from typing import Literal

import pandas as pd


try:
    from examples.backtest.crypto_rv.signals.vol_managed_trend import TrendSignalSpec
    from examples.backtest.crypto_rv.signals.vol_managed_trend import build_cross_sectional_scores
    from examples.backtest.crypto_rv.signals.vol_managed_trend import compute_trend_components
    from examples.backtest.crypto_rv.signals.vol_managed_trend import select_bucket_members
except ModuleNotFoundError:  # pragma: no cover - worktree/runtime fallback
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from examples.backtest.crypto_rv.signals.vol_managed_trend import TrendSignalSpec
    from examples.backtest.crypto_rv.signals.vol_managed_trend import build_cross_sectional_scores
    from examples.backtest.crypto_rv.signals.vol_managed_trend import compute_trend_components
    from examples.backtest.crypto_rv.signals.vol_managed_trend import select_bucket_members
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderCanceled
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.events import OrderRejected
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


RebalanceCadence = Literal["weekly", "biweekly"]
ConditionalMode = Literal["baseline", "hard_filter", "post_rank_filter", "short_tilt"]


@dataclass(frozen=True, slots=True)
class ConditionalWeightPlan:
    weights: dict[str, float]
    longs: list[str]
    shorts: list[str]


def conditioned_plan_tradeable(plan: ConditionalWeightPlan, min_active_shorts: int) -> bool:
    return len(plan.shorts) >= min_active_shorts


def _apply_short_weights(weights: dict[str, float], selected_shorts: list[str]) -> dict[str, float]:
    if selected_shorts:
        short_weight = -0.5 / len(selected_shorts)
        for instrument_id in selected_shorts:
            weights[instrument_id] = short_weight
    return weights


def _apply_short_tilt(
    weights: dict[str, float],
    base_shorts: list[str],
    *,
    eligible_short_ids: set[str],
    tilt_multiplier: float,
) -> dict[str, float]:
    if not base_shorts:
        return weights
    scales = dict.fromkeys(base_shorts, 1.0)
    for instrument_id in scales:
        if instrument_id in eligible_short_ids:
            scales[instrument_id] *= tilt_multiplier
    total_scale = sum(scales.values())
    for instrument_id, scale in scales.items():
        weights[instrument_id] = -0.5 * (scale / total_scale)
    return weights


def build_conditioned_weight_plan(
    composite_scores: dict[str, float],
    active_ids: list[str],
    *,
    eligible_short_ids: set[str],
    mode: ConditionalMode,
    long_bucket_frac: float,
    short_bucket_frac: float,
    tilt_multiplier: float,
) -> ConditionalWeightPlan:
    longs, base_shorts = select_bucket_members(
        composite_scores,
        long_bucket_frac=long_bucket_frac,
        short_bucket_frac=short_bucket_frac,
    )
    weights = dict.fromkeys(composite_scores, 0.0)
    if longs:
        long_weight = 0.5 / len(longs)
        for instrument_id in longs:
            weights[instrument_id] = long_weight

    if mode == "baseline":
        return ConditionalWeightPlan(
            weights=_apply_short_weights(weights, base_shorts),
            longs=longs,
            shorts=base_shorts,
        )

    ordered_all = [
        instrument_id
        for instrument_id, _score in sorted(
            composite_scores.items(),
            key=lambda item: (item[1], item[0]),
        )
    ]
    target_short_count = max(1, round(len(active_ids) * short_bucket_frac))

    if mode == "hard_filter":
        selected_shorts = [
            instrument_id
            for instrument_id in ordered_all
            if instrument_id in active_ids and instrument_id in eligible_short_ids
        ][:target_short_count]
        return ConditionalWeightPlan(
            weights=_apply_short_weights(weights, selected_shorts),
            longs=longs,
            shorts=selected_shorts,
        )

    if mode == "post_rank_filter":
        selected_shorts = [
            instrument_id for instrument_id in base_shorts if instrument_id in eligible_short_ids
        ]
        return ConditionalWeightPlan(
            weights=_apply_short_weights(weights, selected_shorts),
            longs=longs,
            shorts=selected_shorts,
        )

    if mode == "short_tilt":
        return ConditionalWeightPlan(
            weights=_apply_short_tilt(
                weights,
                base_shorts,
                eligible_short_ids=eligible_short_ids,
                tilt_multiplier=tilt_multiplier,
            ),
            longs=longs,
            shorts=base_shorts,
        )

    raise ValueError(f"Unsupported conditional mode: {mode}")


class CryptoXSecTrendConditionalConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``CryptoXSecTrendConditionalStrategy`` instances.
    """

    universe_instrument_ids: tuple[InstrumentId, ...]
    eligible_short_instrument_ids: tuple[InstrumentId, ...]
    bar_types: tuple[Any, ...]
    leg_notional_usd: PositiveFloat
    rebalance_cadence: RebalanceCadence = "weekly"
    long_bucket_frac: PositiveFloat = 0.2
    short_bucket_frac: PositiveFloat = 0.2
    fast_window: int = 7
    slow_window: int = 21
    sma_window: int = 21
    vol_window: int = 20
    volume_window: int = 5
    min_history_bars: int = 25
    volatility_managed: bool = True
    conditional_mode: ConditionalMode = "post_rank_filter"
    tilt_multiplier: PositiveFloat = 2.0
    min_active_shorts: int = 3
    request_bars: bool = True
    history_lookback_days: int = 90
    subscribe_quote_ticks: bool = True
    use_passive_limits: bool = True
    allow_market_fallback: bool = False
    order_time_in_force: TimeInForce = TimeInForce.GTD
    order_expire_seconds: PositiveFloat = 900.0
    rebalance_execution_window_secs: PositiveFloat = 3600.0
    min_order_notional_usd: PositiveFloat = 5.0
    manual_approval_rebalances: int = 2
    approval_artifact_dir: str | None = None
    close_positions_on_stop: bool = False


class CryptoXSecTrendConditionalStrategy(Strategy):
    """
    Live-oriented conditional CTREND basket with a frozen short-eligibility band.
    """

    def __init__(self, config: CryptoXSecTrendConditionalConfig) -> None:
        self._validate_config(config)
        super().__init__(config)

        self._bar_types = self._build_bar_type_index(config.bar_types)
        self._eligible_short_ids = set(config.eligible_short_instrument_ids)
        self._instruments: dict[InstrumentId, Instrument] = {}
        self._latest_bars: dict[InstrumentId, Bar] = {}
        self._latest_quotes: dict[InstrumentId, QuoteTick] = {}
        self._close_history: dict[InstrumentId, deque[float]] = {}
        self._volume_history: dict[InstrumentId, deque[float]] = {}
        self._target_quantities: dict[InstrumentId, Decimal] = {}
        self._pending_approval_targets: dict[InstrumentId, Decimal] = {}
        self._pending_approval_preview_path: Path | None = None
        self._last_rebalance_ts: int | None = None
        self._rebalance_count = 0
        self._execution_deadline_ns: int | None = None
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

    def _validate_config(self, config: CryptoXSecTrendConditionalConfig) -> None:
        PyCondition.is_true(
            bool(config.universe_instrument_ids),
            "universe_instrument_ids must not be empty",
        )
        PyCondition.is_true(
            bool(config.eligible_short_instrument_ids),
            "eligible_short_instrument_ids must not be empty",
        )
        PyCondition.is_true(
            set(config.eligible_short_instrument_ids).issubset(set(config.universe_instrument_ids)),
            "eligible_short_instrument_ids must be a subset of universe_instrument_ids",
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
        PyCondition.is_true(config.min_history_bars > 0, "min_history_bars must be positive")
        PyCondition.is_true(config.min_active_shorts > 0, "min_active_shorts must be positive")
        PyCondition.is_true(config.history_lookback_days > 0, "history_lookback_days must be positive")

    def _build_bar_type_index(self, bar_types: tuple[Any, ...]) -> dict[InstrumentId, Any]:
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

            if self.config.request_bars:
                self.request_bars(
                    self._bar_types[instrument_id],
                    start=self._clock.utc_now() - pd.Timedelta(days=self.config.history_lookback_days),
                )
            self.subscribe_bars(self._bar_types[instrument_id])
            if self.config.subscribe_quote_ticks:
                self.subscribe_quote_ticks(instrument_id)

        self.log.info(
            "Started CryptoXSecTrendConditionalStrategy "
            f"with {len(self.config.universe_instrument_ids)} instruments, "
            f"eligible_shorts={len(self.config.eligible_short_instrument_ids)}, "
            f"cadence={self.config.rebalance_cadence}, "
            f"conditional_mode={self.config.conditional_mode}, "
            f"min_active_shorts={self.config.min_active_shorts}",
        )

    def on_historical_data(self, data: Any) -> None:
        if isinstance(data, Bar) and data.bar_type.instrument_id in self._close_history:
            self._ingest_bar(data)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        if tick.instrument_id in self._instruments:
            self._latest_quotes[tick.instrument_id] = tick
            self._try_execute_pending_approval()
            self._work_pending_target(tick.instrument_id)

    def on_bar(self, bar: Bar) -> None:
        instrument_id = bar.bar_type.instrument_id
        if instrument_id not in self._instruments:
            return

        self._latest_bars[instrument_id] = bar
        self._ingest_bar(bar)

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
            if self.config.subscribe_quote_ticks:
                self.unsubscribe_quote_ticks(instrument_id)
            if self.config.close_positions_on_stop:
                self.close_all_positions(instrument_id)

    def on_event(self, event: Event) -> None:
        instrument_id = getattr(event, "instrument_id", None)
        if not isinstance(instrument_id, InstrumentId):
            return
        if not isinstance(event, OrderFilled | OrderCanceled | OrderRejected):
            return
        self._work_pending_target(instrument_id)

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

    def _ingest_bar(self, bar: Bar) -> None:
        instrument_id = bar.bar_type.instrument_id
        self._close_history[instrument_id].append(float(bar.close.as_decimal()))
        self._volume_history[instrument_id].append(float(bar.volume.as_decimal()))

    def _has_signal_ready_snapshot(self, ts_event: int) -> bool:
        return len(self._active_signal_ids(ts_event)) == len(self.config.universe_instrument_ids)

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
        plan = build_conditioned_weight_plan(
            {
                instrument_id: payload["composite_score"]
                for instrument_id, payload in score_payload.items()
            },
            [str(instrument_id) for instrument_id in active_ids],
            eligible_short_ids={str(instrument_id) for instrument_id in self._eligible_short_ids},
            mode=self.config.conditional_mode,
            long_bucket_frac=self.config.long_bucket_frac,
            short_bucket_frac=self.config.short_bucket_frac,
            tilt_multiplier=float(self.config.tilt_multiplier),
        )

        targets: dict[InstrumentId, Decimal] = {
            instrument_id: Decimal(0)
            for instrument_id in self.config.universe_instrument_ids
            if Decimal(self.portfolio.net_position(instrument_id)) != 0
        }
        if conditioned_plan_tradeable(plan, self.config.min_active_shorts):
            gross_notional = Decimal(str(self.config.leg_notional_usd)) * 2
            for instrument_id in active_ids:
                weight = Decimal(str(plan.weights.get(str(instrument_id), 0.0)))
                if weight == 0:
                    continue
                targets[instrument_id] = self._signed_target_quantity(
                    instrument_id,
                    gross_notional * abs(weight),
                    is_long=weight > 0,
                )
        else:
            self.log.warning(
                "Conditional short intersection below threshold "
                f"({len(plan.shorts)} < {self.config.min_active_shorts}); flattening tracked targets",
            )

        if self._approval_required(ts_event):
            return self._stage_or_execute_approval(
                ts_event=ts_event,
                plan=plan,
                targets=targets,
            )

        self._activate_rebalance_targets(ts_event, targets)
        self._rebalance_count += 1
        self.log.info(
            "Rebalanced crypto conditional trend basket "
            f"#{self._rebalance_count}: longs={len(plan.longs)} shorts={len(plan.shorts)} "
            f"submitted_orders={len(targets)}",
        )

    def _approval_required(self, ts_event: int) -> bool:
        return (
            self.config.approval_artifact_dir is not None
            and self._rebalance_count < self.config.manual_approval_rebalances
            and (self._last_rebalance_ts is None or ts_event > self._last_rebalance_ts)
        )

    def _stage_or_execute_approval(
        self,
        *,
        ts_event: int,
        plan: ConditionalWeightPlan,
        targets: dict[InstrumentId, Decimal],
    ) -> None:
        preview_dir = Path(self.config.approval_artifact_dir or "")
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"rebalance_{self._rebalance_count + 1}_{ts_event}.json"
        approved_path = preview_path.with_suffix(".approved")
        if approved_path.exists():
            self._activate_rebalance_targets(ts_event, targets)
            self._pending_approval_targets = {}
            self._pending_approval_preview_path = None
            self._rebalance_count += 1
            self.log.info(
                f"Approval token detected for rebalance #{self._rebalance_count}; orders released",
            )
            return

        preview_payload = {
            "generated_at_ts_event": ts_event,
            "rebalance_number": self._rebalance_count + 1,
            "conditional_mode": self.config.conditional_mode,
            "eligible_short_count": len(self.config.eligible_short_instrument_ids),
            "longs": plan.longs,
            "shorts": plan.shorts,
            "targets": {str(instrument_id): str(quantity) for instrument_id, quantity in targets.items()},
        }
        preview_path.write_text(json.dumps(preview_payload, indent=2, sort_keys=True), encoding="utf-8")
        self._pending_approval_targets = dict(targets)
        self._pending_approval_preview_path = preview_path
        self.log.warning(
            "Manual approval required before live rebalance submission; "
            f"review {preview_path} and create {approved_path.name} to release orders",
        )

    def _try_execute_pending_approval(self) -> None:
        if not self._pending_approval_targets or self._pending_approval_preview_path is None:
            return
        approved_path = self._pending_approval_preview_path.with_suffix(".approved")
        if not approved_path.exists():
            return
        ts_event = max(bar.ts_event for bar in self._latest_bars.values())
        self._activate_rebalance_targets(ts_event, self._pending_approval_targets)
        self._rebalance_count += 1
        self.log.info(
            f"Approval token detected for rebalance #{self._rebalance_count}; orders released",
        )
        self._pending_approval_targets = {}
        self._pending_approval_preview_path = None

    def _activate_rebalance_targets(
        self,
        ts_event: int,
        targets: dict[InstrumentId, Decimal],
    ) -> None:
        self._cancel_all_open_orders()
        self._target_quantities = dict(targets)
        self._execution_deadline_ns = self.clock.timestamp_ns() + int(
            self.config.rebalance_execution_window_secs * 1_000_000_000,
        )

        submitted = 0
        for instrument_id in list(self._target_quantities):
            submitted += int(self._work_pending_target(instrument_id))

        self._last_rebalance_ts = ts_event
        self.log.info(f"Activated rebalance targets: submitted_orders={submitted}")

    def _signed_target_quantity(
        self,
        instrument_id: InstrumentId,
        target_notional_usd: Decimal,
        *,
        is_long: bool,
    ) -> Decimal:
        instrument = self._instruments[instrument_id]
        bar = self._latest_bars[instrument_id]
        price = Decimal(bar.close.as_decimal())
        multiplier = Decimal(instrument.multiplier.as_decimal())
        quantity = target_notional_usd / (price * multiplier)
        return quantity if is_long else -quantity

    def _cancel_all_open_orders(self) -> None:
        for instrument_id in self.config.universe_instrument_ids:
            self.cancel_all_orders(instrument_id)

    def _has_resting_orders(self, instrument_id: InstrumentId) -> bool:
        return bool(
            self.cache.orders_open(instrument_id=instrument_id, strategy_id=self.id)
            or self.cache.orders_inflight(instrument_id=instrument_id, strategy_id=self.id)
        )

    def _work_pending_target(self, instrument_id: InstrumentId) -> bool:
        if instrument_id not in self._target_quantities:
            return False
        if self._execution_deadline_ns is not None and self.clock.timestamp_ns() > self._execution_deadline_ns:
            self.log.warning(
                f"Execution window expired for {instrument_id}; leaving target unresolved until next rebalance",
            )
            self._target_quantities.pop(instrument_id, None)
            return False
        if self._has_resting_orders(instrument_id):
            return False

        target_quantity = self._target_quantities[instrument_id]
        current_quantity = Decimal(self.portfolio.net_position(instrument_id))
        delta_quantity = target_quantity - current_quantity
        if delta_quantity == 0:
            self._target_quantities.pop(instrument_id, None)
            return False

        submitted = self._submit_delta_order(instrument_id, target_quantity)
        if not submitted and not self.config.allow_market_fallback:
            return False
        return submitted

    def _submit_delta_order(self, instrument_id: InstrumentId, target_quantity: Decimal) -> bool:
        current_quantity = Decimal(self.portfolio.net_position(instrument_id))
        delta_quantity = target_quantity - current_quantity
        if delta_quantity == 0:
            return False

        instrument = self._instruments[instrument_id]
        bar = self._latest_bars.get(instrument_id)
        if bar is None:
            self.log.warning(f"Skipping order for {instrument_id}: no latest bar")
            return False

        price = Decimal(bar.close.as_decimal())
        multiplier = Decimal(instrument.multiplier.as_decimal())
        delta_notional = abs(delta_quantity) * price * multiplier
        if delta_notional < Decimal(str(self.config.min_order_notional_usd)):
            return False

        if self.config.use_passive_limits:
            quote = self._latest_quotes.get(instrument_id)
            if quote is not None:
                limit_order = self._build_passive_limit_order(
                    instrument_id=instrument_id,
                    delta_quantity=delta_quantity,
                    quote=quote,
                )
                self.submit_order(limit_order)
                return True
            if not self.config.allow_market_fallback:
                self.log.warning(f"Skipping order for {instrument_id}: no quote available for passive limit")
                return False

        order_side = OrderSide.BUY if delta_quantity > 0 else OrderSide.SELL
        quantity = instrument.make_qty(abs(delta_quantity), round_down=True)
        if quantity.as_decimal() <= 0:
            return False

        market_order: MarketOrder = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(market_order)
        return True

    def _build_passive_limit_order(
        self,
        *,
        instrument_id: InstrumentId,
        delta_quantity: Decimal,
        quote: QuoteTick,
    ) -> LimitOrder:
        instrument = self._instruments[instrument_id]
        order_side = OrderSide.BUY if delta_quantity > 0 else OrderSide.SELL
        raw_price = (
            quote.bid_price.as_decimal()
            if order_side == OrderSide.BUY
            else quote.ask_price.as_decimal()
        )
        price = instrument.make_price(raw_price)
        quantity = instrument.make_qty(abs(delta_quantity), round_down=True)
        return self.order_factory.limit(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            price=price,
            time_in_force=self.config.order_time_in_force,
            expire_time=self.clock.utc_now() + pd.Timedelta(seconds=self.config.order_expire_seconds),
            post_only=True,
        )


CryptoXSecTrendConditional = CryptoXSecTrendConditionalStrategy
