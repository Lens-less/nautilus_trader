# mypy: disable-error-code=no-redef

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from .common import resolve_path
    from .common import write_json
    from .schemas import CostModel
    from .schemas import PreparedCatalogArtifact
    from .schemas import StrategyImportConfig
    from .schemas import UniverseSnapshot
    from .schemas import load_json
    from .schemas import load_prepared_catalog
    from .schemas import load_snapshot
except ImportError:  # pragma: no cover - script execution fallback
    from common import resolve_path
    from common import write_json
    from schemas import CostModel
    from schemas import PreparedCatalogArtifact
    from schemas import StrategyImportConfig
    from schemas import UniverseSnapshot
    from schemas import load_json
    from schemas import load_prepared_catalog
    from schemas import load_snapshot


CTREND_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class CTrendSignalConfig:
    fast_window: int = 2
    slow_window: int = 4
    sma_window: int = 4
    vol_window: int = 4
    volume_window: int = 2
    min_history_bars: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CTrendSignalConfig:
        return cls(
            fast_window=int(payload.get("fast_window", 2)),
            slow_window=int(payload.get("slow_window", 4)),
            sma_window=int(payload.get("sma_window", 4)),
            vol_window=int(payload.get("vol_window", 4)),
            volume_window=int(payload.get("volume_window", 2)),
            min_history_bars=int(payload.get("min_history_bars", 5)),
        )

    def validate(self) -> None:
        fields = (
            self.fast_window,
            self.slow_window,
            self.sma_window,
            self.vol_window,
            self.volume_window,
            self.min_history_bars,
        )
        if any(value <= 0 for value in fields):
            raise ValueError("CTREND signal windows must be positive")
        if self.min_history_bars < max(
            self.fast_window + 1,
            self.slow_window + 1,
            self.sma_window,
            self.vol_window + 1,
            self.volume_window * 2,
        ):
            raise ValueError("min_history_bars must cover all signal lookbacks")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CTrendPortfolioConfig:
    leg_notional_usd: float
    long_bucket_frac: float = 0.2
    short_bucket_frac: float = 0.2
    rebalance_cadence: str = "weekly"
    min_order_notional_usd: float = 25.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CTrendPortfolioConfig:
        return cls(
            leg_notional_usd=float(payload["leg_notional_usd"]),
            long_bucket_frac=float(payload.get("long_bucket_frac", 0.2)),
            short_bucket_frac=float(payload.get("short_bucket_frac", 0.2)),
            rebalance_cadence=str(payload.get("rebalance_cadence", "weekly")),
            min_order_notional_usd=float(payload.get("min_order_notional_usd", 25.0)),
        )

    def validate(self) -> None:
        if self.leg_notional_usd <= 0:
            raise ValueError("leg_notional_usd must be positive")
        if not 0 < self.long_bucket_frac < 1:
            raise ValueError("long_bucket_frac must be within (0, 1)")
        if not 0 < self.short_bucket_frac < 1:
            raise ValueError("short_bucket_frac must be within (0, 1)")
        if self.rebalance_cadence not in {"weekly", "biweekly"}:
            raise ValueError("rebalance_cadence must be weekly or biweekly")
        if self.min_order_notional_usd <= 0:
            raise ValueError("min_order_notional_usd must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CTrendConfig:
    research_name: str
    snapshot_path: str
    prepared_catalog_path: str
    output_dir: str
    report_output_path: str
    baseline_long_instrument_ids: list[str]
    strategy: StrategyImportConfig
    signal: CTrendSignalConfig
    portfolio: CTrendPortfolioConfig
    cost_models: dict[str, CostModel]
    schema_version: str = CTREND_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CTrendConfig:
        return cls(
            research_name=str(payload["research_name"]),
            snapshot_path=str(payload["snapshot_path"]),
            prepared_catalog_path=str(payload["prepared_catalog_path"]),
            output_dir=str(payload["output_dir"]),
            report_output_path=str(payload["report_output_path"]),
            baseline_long_instrument_ids=list(payload["baseline_long_instrument_ids"]),
            strategy=StrategyImportConfig.from_dict(payload["strategy"]),
            signal=CTrendSignalConfig.from_dict(payload["signal"]),
            portfolio=CTrendPortfolioConfig.from_dict(payload["portfolio"]),
            cost_models={
                name: CostModel.from_dict(model)
                for name, model in payload["cost_models"].items()
            },
            schema_version=str(payload.get("schema_version", CTREND_SCHEMA_VERSION)),
        )

    def validate(self) -> None:
        if not self.research_name:
            raise ValueError("research_name must not be empty")
        if len(self.baseline_long_instrument_ids) < 2:
            raise ValueError("baseline_long_instrument_ids must contain at least two instruments")
        if {"base", "stress"} - set(self.cost_models):
            raise ValueError("cost_models must contain base and stress")
        for model in self.cost_models.values():
            model.validate()
        self.strategy.validate()
        self.signal.validate()
        self.portfolio.validate()

    def resolved_paths(self, config_path: Path) -> dict[str, Path]:
        base_dir = config_path.parent
        return {
            "snapshot_path": resolve_path(self.snapshot_path, base_dir),
            "prepared_catalog_path": resolve_path(self.prepared_catalog_path, base_dir),
            "output_dir": resolve_path(self.output_dir, base_dir),
            "report_output_path": resolve_path(self.report_output_path, base_dir),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_name": self.research_name,
            "snapshot_path": self.snapshot_path,
            "prepared_catalog_path": self.prepared_catalog_path,
            "output_dir": self.output_dir,
            "report_output_path": self.report_output_path,
            "baseline_long_instrument_ids": self.baseline_long_instrument_ids,
            "strategy": self.strategy.to_dict(),
            "signal": self.signal.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "cost_models": {name: model.to_dict() for name, model in self.cost_models.items()},
        }


def load_ctrend_config(path: Path) -> CTrendConfig:
    config = CTrendConfig.from_dict(load_json(path))
    config.validate()
    return config


def load_ctrend_snapshot(path: Path) -> UniverseSnapshot:
    return load_snapshot(path)


def load_ctrend_prepared_catalog(path: Path) -> PreparedCatalogArtifact:
    return load_prepared_catalog(path)


def save_artifact(path: Path, payload: Any) -> None:
    write_json(path, payload)
