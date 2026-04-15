from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from statistics import pstdev


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class TrendSignalSpec:
    fast_window: int = 2
    slow_window: int = 4
    sma_window: int = 4
    vol_window: int = 4
    volume_window: int = 2
    min_history_bars: int = 5

    def validate(self) -> None:
        windows = (
            self.fast_window,
            self.slow_window,
            self.sma_window,
            self.vol_window,
            self.volume_window,
            self.min_history_bars,
        )
        if any(value <= 0 for value in windows):
            raise ValueError("TrendSignalSpec windows must all be positive")
        if self.min_history_bars < max(
            self.fast_window + 1,
            self.slow_window + 1,
            self.sma_window,
            self.vol_window + 1,
            self.volume_window * 2,
        ):
            raise ValueError("min_history_bars must cover every configured lookback")


def _window_return(prices: Sequence[float], window: int) -> float:
    previous = float(prices[-window - 1])
    current = float(prices[-1])
    if previous <= 0:
        return 0.0
    return current / previous - 1.0


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(fmean(values))


def _realized_vol(prices: Sequence[float], window: int) -> float:
    if len(prices) < window + 1:
        return 0.0

    returns = []
    start = len(prices) - window
    for index in range(start, len(prices)):
        previous = float(prices[index - 1])
        current = float(prices[index])
        if previous <= 0:
            continue
        returns.append(current / previous - 1.0)

    if len(returns) < 2:
        return 0.0
    return float(pstdev(returns))


def compute_trend_components(
    prices: Sequence[float],
    volumes: Sequence[float] | None = None,
    spec: TrendSignalSpec | None = None,
) -> dict[str, float]:
    spec = spec or TrendSignalSpec()
    spec.validate()
    if len(prices) < spec.min_history_bars:
        raise ValueError("Insufficient price history for trend signal computation")

    volume_values = list(volumes or [1.0] * len(prices))
    if len(volume_values) < spec.min_history_bars:
        volume_values = [1.0] * len(prices)

    fast_momentum = _window_return(prices, spec.fast_window)
    slow_momentum = _window_return(prices, spec.slow_window)
    moving_average = _average([float(value) for value in prices[-spec.sma_window :]])
    latest_price = float(prices[-1])
    ma_gap = 0.0 if moving_average <= 0 else latest_price / moving_average - 1.0
    realized_vol = _realized_vol(prices, spec.vol_window)

    recent_volume = _average([float(value) for value in volume_values[-spec.volume_window :]])
    prior_volume = _average(
        [float(value) for value in volume_values[-(spec.volume_window * 2) : -spec.volume_window]],
    )
    if prior_volume <= 0:
        volume_confirmation = 0.0
    else:
        volume_confirmation = recent_volume / prior_volume - 1.0

    return {
        "fast_momentum": fast_momentum,
        "slow_momentum": slow_momentum,
        "ma_gap": ma_gap,
        "volume_confirmation": volume_confirmation,
        "realized_vol": realized_vol,
    }


def centered_rank(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        key = next(iter(values))
        return {key: 0.0}

    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    denominator = len(ordered) - 1
    ranked: dict[str, float] = {}
    for index, (key, _) in enumerate(ordered):
        ranked[key] = index / denominator - 0.5
    return ranked


def build_cross_sectional_scores(
    components_by_instrument: Mapping[str, Mapping[str, float]],
    *,
    volatility_managed: bool,
) -> dict[str, dict[str, float]]:
    if not components_by_instrument:
        return {}

    component_names = ("fast_momentum", "slow_momentum", "ma_gap", "volume_confirmation")
    ranked_components: dict[str, dict[str, float]] = {}

    for component_name in component_names:
        values: dict[str, float] = {}
        for instrument_id, components in components_by_instrument.items():
            value = float(components[component_name])
            if volatility_managed and component_name != "volume_confirmation":
                realized_vol = max(float(components["realized_vol"]), EPSILON)
                value /= realized_vol
            values[instrument_id] = value
        ranked_components[component_name] = centered_rank(values)

    composite_scores: dict[str, dict[str, float]] = {}
    for instrument_id in components_by_instrument:
        component_scores = {
            component_name: ranked_components[component_name][instrument_id]
            for component_name in component_names
        }
        composite_scores[instrument_id] = {
            **component_scores,
            "composite_score": _average(list(component_scores.values())),
        }

    return composite_scores


def select_bucket_members(
    composite_scores: Mapping[str, float],
    *,
    long_bucket_frac: float,
    short_bucket_frac: float,
) -> tuple[list[str], list[str]]:
    if not composite_scores:
        return [], []

    if not (0 < long_bucket_frac < 1):
        raise ValueError("long_bucket_frac must be within (0, 1)")
    if not (0 < short_bucket_frac < 1):
        raise ValueError("short_bucket_frac must be within (0, 1)")

    ordered = sorted(composite_scores.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    long_count = max(1, round(count * long_bucket_frac))
    short_count = max(1, round(count * short_bucket_frac))
    if long_count + short_count > count:
        long_count = max(1, count // 2)
        short_count = max(1, count - long_count)

    shorts = [instrument_id for instrument_id, _ in ordered[:short_count]]
    longs = [instrument_id for instrument_id, _ in ordered[-long_count:]]
    return longs, shorts


def build_target_weights(
    composite_scores: Mapping[str, float],
    *,
    long_bucket_frac: float,
    short_bucket_frac: float,
) -> dict[str, float]:
    longs, shorts = select_bucket_members(
        composite_scores,
        long_bucket_frac=long_bucket_frac,
        short_bucket_frac=short_bucket_frac,
    )
    weights = dict.fromkeys(composite_scores, 0.0)
    if longs:
        long_weight = 0.5 / len(longs)
        for instrument_id in longs:
            weights[instrument_id] = long_weight
    if shorts:
        short_weight = -0.5 / len(shorts)
        for instrument_id in shorts:
            weights[instrument_id] = short_weight
    return weights
