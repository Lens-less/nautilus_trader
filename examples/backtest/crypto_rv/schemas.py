from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


try:
    from .common import format_utc_timestamp
    from .common import load_json
    from .common import normalize_symbol
    from .common import parse_float
    from .common import parse_optional_float
    from .common import parse_optional_int
    from .common import parse_utc_timestamp
    from .common import resolve_path
    from .common import stable_payload_hash
    from .common import within_ratio
    from .common import write_json
except ImportError:  # pragma: no cover - script execution fallback
    from common import format_utc_timestamp
    from common import load_json
    from common import normalize_symbol
    from common import parse_float
    from common import parse_optional_float
    from common import parse_optional_int
    from common import parse_utc_timestamp
    from common import resolve_path
    from common import stable_payload_hash
    from common import within_ratio
    from common import write_json


SCHEMA_VERSION = "1.0"
ARTIFACT_SCHEMA_VERSION = "1.0"


def classify_funding_coverage(
    coverage_ratio: float,
    partial_threshold: float = 0.80,
    warning_threshold: float = 0.95,
) -> str:
    if coverage_ratio < partial_threshold:
        return "partial evidence only"
    if coverage_ratio < warning_threshold:
        return "credible hypothesis test with funding warning"
    return "credible hypothesis test"


def renormalize_long_weights(
    weights: dict[str, float],
    *,
    excluded_symbol: str,
) -> dict[str, float]:
    filtered = {
        normalize_symbol(symbol): float(weight)
        for symbol, weight in weights.items()
        if normalize_symbol(symbol) != normalize_symbol(excluded_symbol)
    }
    gross_weight = sum(filtered.values())
    if gross_weight <= 0:
        raise ValueError("Cannot renormalize an empty long basket")
    return {symbol: weight / gross_weight for symbol, weight in filtered.items()}


@dataclass(slots=True)
class LongLegMember:
    asset: str
    raw_symbol: str
    instrument_id: str
    weight: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LongLegMember:
        return cls(
            asset=normalize_symbol(str(payload["asset"])),
            raw_symbol=normalize_symbol(str(payload["raw_symbol"])),
            instrument_id=str(payload["instrument_id"]).strip(),
            weight=float(payload["weight"]),
        )

    def validate(self) -> None:
        if not self.instrument_id:
            raise ValueError(f"{self.asset} is missing instrument_id")
        if self.weight <= 0:
            raise ValueError(f"{self.asset} weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThresholdConfig:
    min_market_cap_usd: float
    max_market_cap_usd: float
    max_volume_24h_usd: float
    pre_window_history_days: int = 14
    min_price_coverage_ratio: float = 1.0
    funding_partial_evidence_threshold: float = 0.80
    funding_warning_threshold: float = 0.95

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ThresholdConfig:
        return cls(
            min_market_cap_usd=float(payload["min_market_cap_usd"]),
            max_market_cap_usd=float(payload["max_market_cap_usd"]),
            max_volume_24h_usd=float(payload["max_volume_24h_usd"]),
            pre_window_history_days=int(payload.get("pre_window_history_days", 14)),
            min_price_coverage_ratio=float(payload.get("min_price_coverage_ratio", 1.0)),
            funding_partial_evidence_threshold=float(
                payload.get("funding_partial_evidence_threshold", 0.80),
            ),
            funding_warning_threshold=float(payload.get("funding_warning_threshold", 0.95)),
        )

    def validate(self) -> None:
        if self.min_market_cap_usd <= 0:
            raise ValueError("min_market_cap_usd must be positive")
        if self.max_market_cap_usd < self.min_market_cap_usd:
            raise ValueError("max_market_cap_usd must be >= min_market_cap_usd")
        if self.max_volume_24h_usd <= 0:
            raise ValueError("max_volume_24h_usd must be positive")
        if self.pre_window_history_days < 14:
            raise ValueError("pre_window_history_days must be at least 14")
        if not within_ratio(self.min_price_coverage_ratio):
            raise ValueError("min_price_coverage_ratio must be within [0, 1]")
        if not within_ratio(self.funding_partial_evidence_threshold):
            raise ValueError("funding_partial_evidence_threshold must be within [0, 1]")
        if not within_ratio(self.funding_warning_threshold):
            raise ValueError("funding_warning_threshold must be within [0, 1]")
        if self.funding_warning_threshold < self.funding_partial_evidence_threshold:
            raise ValueError("funding_warning_threshold must be >= partial threshold")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CostModel:
    fee_bps: float
    slippage_bps: float
    short_borrow_bps_annual: float = 0.0
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CostModel:
        return cls(
            fee_bps=float(payload["fee_bps"]),
            slippage_bps=float(payload["slippage_bps"]),
            short_borrow_bps_annual=float(payload.get("short_borrow_bps_annual", 0.0)),
            notes=str(payload.get("notes", "")),
        )

    def validate(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0 or self.short_borrow_bps_annual < 0:
            raise ValueError("Cost model values must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StrategyImportConfig:
    strategy_path: str
    config_path: str
    allow_placeholder_runner: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategyImportConfig:
        return cls(
            strategy_path=str(payload["strategy_path"]).strip(),
            config_path=str(payload["config_path"]).strip(),
            allow_placeholder_runner=bool(payload.get("allow_placeholder_runner", True)),
        )

    def validate(self) -> None:
        if ":" not in self.strategy_path:
            raise ValueError("strategy_path must be in module:object format")
        if ":" not in self.config_path:
            raise ValueError("config_path must be in module:object format")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SensitivityConfig:
    enable_hype_out: bool = True
    short_counts: list[int] = field(default_factory=lambda: [20, 30, 50])
    rebalance_cadences: list[str] = field(default_factory=lambda: ["weekly", "biweekly"])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensitivityConfig:
        return cls(
            enable_hype_out=bool(payload.get("enable_hype_out", True)),
            short_counts=[int(value) for value in payload.get("short_counts", [20, 30, 50])],
            rebalance_cadences=list(payload.get("rebalance_cadences", ["weekly", "biweekly"])),
        )

    def validate(self) -> None:
        if not self.short_counts:
            raise ValueError("At least one short-count sensitivity is required")
        if not self.rebalance_cadences:
            raise ValueError("At least one rebalance cadence is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchConfig:
    research_name: str
    formation_ts: str
    test_window_start: str
    test_window_end: str
    universe_input_path: str
    history_manifest_path: str
    snapshot_output_path: str
    prepared_catalog_output_path: str
    run_output_dir: str
    report_output_path: str
    long_leg: list[LongLegMember]
    thresholds: ThresholdConfig
    cost_models: dict[str, CostModel]
    strategy: StrategyImportConfig
    sensitivity: SensitivityConfig
    primary_short_count: int = 30
    min_short_count: int = 20
    max_short_count: int = 50
    rebalance_cadence: str = "weekly"
    neutrality_tolerance_pct: float = 1.0
    venue: str = "BINANCE"
    base_currency: str = "USDT"
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResearchConfig:
        return cls(
            research_name=str(payload["research_name"]),
            formation_ts=format_utc_timestamp(str(payload["formation_ts"])),
            test_window_start=format_utc_timestamp(str(payload["test_window_start"])),
            test_window_end=format_utc_timestamp(str(payload["test_window_end"])),
            universe_input_path=str(payload["universe_input_path"]),
            history_manifest_path=str(payload["history_manifest_path"]),
            snapshot_output_path=str(payload["snapshot_output_path"]),
            prepared_catalog_output_path=str(payload["prepared_catalog_output_path"]),
            run_output_dir=str(payload["run_output_dir"]),
            report_output_path=str(payload["report_output_path"]),
            long_leg=[LongLegMember.from_dict(item) for item in payload["long_leg"]],
            thresholds=ThresholdConfig.from_dict(payload["thresholds"]),
            cost_models={
                name: CostModel.from_dict(model)
                for name, model in payload["cost_models"].items()
            },
            strategy=StrategyImportConfig.from_dict(payload["strategy"]),
            sensitivity=SensitivityConfig.from_dict(payload.get("sensitivity", {})),
            primary_short_count=int(payload.get("primary_short_count", 30)),
            min_short_count=int(payload.get("min_short_count", 20)),
            max_short_count=int(payload.get("max_short_count", 50)),
            rebalance_cadence=str(payload.get("rebalance_cadence", "weekly")),
            neutrality_tolerance_pct=float(payload.get("neutrality_tolerance_pct", 1.0)),
            venue=str(payload.get("venue", "BINANCE")),
            base_currency=str(payload.get("base_currency", "USDT")),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def validate(self) -> None:
        self._validate_window_and_limits()
        self._validate_long_leg()
        self.thresholds.validate()
        self.strategy.validate()
        self.sensitivity.validate()
        self._validate_cost_models()

    def _validate_window_and_limits(self) -> None:
        parse_utc_timestamp(self.formation_ts)
        start_ts = parse_utc_timestamp(self.test_window_start)
        end_ts = parse_utc_timestamp(self.test_window_end)
        formation_ts = parse_utc_timestamp(self.formation_ts)

        if formation_ts >= start_ts:
            raise ValueError("formation_ts must be before test_window_start")
        if start_ts >= end_ts:
            raise ValueError("test_window_start must be before test_window_end")
        if self.primary_short_count < self.min_short_count:
            raise ValueError("primary_short_count must be >= min_short_count")
        if self.primary_short_count > self.max_short_count:
            raise ValueError("primary_short_count must be <= max_short_count")
        if self.rebalance_cadence not in {"weekly", "biweekly"}:
            raise ValueError("rebalance_cadence must be weekly or biweekly")
        if self.neutrality_tolerance_pct <= 0:
            raise ValueError("neutrality_tolerance_pct must be positive")

    def _validate_long_leg(self) -> None:
        if {member.asset for member in self.long_leg} != {"BTC", "ETH", "SOL", "BNB", "HYPE"}:
            raise ValueError("Primary long leg must contain BTC, ETH, SOL, BNB, HYPE")

        weight_sum = sum(member.weight for member in self.long_leg)
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("Primary long-leg weights must sum to 1.0")

        for member in self.long_leg:
            member.validate()

    def _validate_cost_models(self) -> None:
        if {"base", "stress"} - set(self.cost_models):
            raise ValueError("cost_models must contain both base and stress")
        for model in self.cost_models.values():
            model.validate()

    def resolved_paths(self, config_path: Path) -> dict[str, Path]:
        base_dir = config_path.parent
        return {
            "universe_input_path": resolve_path(self.universe_input_path, base_dir),
            "history_manifest_path": resolve_path(self.history_manifest_path, base_dir),
            "snapshot_output_path": resolve_path(self.snapshot_output_path, base_dir),
            "prepared_catalog_output_path": resolve_path(
                self.prepared_catalog_output_path,
                base_dir,
            ),
            "run_output_dir": resolve_path(self.run_output_dir, base_dir),
            "report_output_path": resolve_path(self.report_output_path, base_dir),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_name": self.research_name,
            "formation_ts": self.formation_ts,
            "test_window_start": self.test_window_start,
            "test_window_end": self.test_window_end,
            "universe_input_path": self.universe_input_path,
            "history_manifest_path": self.history_manifest_path,
            "snapshot_output_path": self.snapshot_output_path,
            "prepared_catalog_output_path": self.prepared_catalog_output_path,
            "run_output_dir": self.run_output_dir,
            "report_output_path": self.report_output_path,
            "long_leg": [item.to_dict() for item in self.long_leg],
            "thresholds": self.thresholds.to_dict(),
            "cost_models": {name: model.to_dict() for name, model in self.cost_models.items()},
            "strategy": self.strategy.to_dict(),
            "sensitivity": self.sensitivity.to_dict(),
            "primary_short_count": self.primary_short_count,
            "min_short_count": self.min_short_count,
            "max_short_count": self.max_short_count,
            "rebalance_cadence": self.rebalance_cadence,
            "neutrality_tolerance_pct": self.neutrality_tolerance_pct,
            "venue": self.venue,
            "base_currency": self.base_currency,
        }


@dataclass(slots=True)
class UniverseCandidate:
    raw_symbol: str
    instrument_id: str
    market_cap_usd: float
    volume_24h_usd: float
    listed_ts: str
    delisted_ts: str | None
    first_price_ts: str
    price_history_start_ts: str
    price_history_end_ts: str
    price_coverage_ratio: float
    funding_coverage_ratio: float
    market_cap_source: str = ""
    volume_24h_source: str = ""
    metadata_source: str = ""
    screen_rank: int | None = None
    frozen_rank: int | None = None
    eligible: bool = False
    selected: bool = False
    selection_rank: int | None = None
    exclusion_reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_screening_row(cls, row: dict[str, Any]) -> UniverseCandidate:
        listed_key = "listed_ts" if "listed_ts" in row else "listing_ts"
        return cls(
            raw_symbol=normalize_symbol(str(row["raw_symbol"] if "raw_symbol" in row else row["symbol"])),
            instrument_id=str(row["instrument_id"]).strip(),
            market_cap_usd=parse_float(row.get("market_cap_usd"), "market_cap_usd"),
            volume_24h_usd=parse_float(row.get("volume_24h_usd"), "volume_24h_usd"),
            listed_ts=format_utc_timestamp(str(row[listed_key])),
            delisted_ts=(
                format_utc_timestamp(str(row["delisted_ts"]))
                if row.get("delisted_ts") not in (None, "")
                else None
            ),
            first_price_ts=format_utc_timestamp(str(row["first_price_ts"])),
            price_history_start_ts=format_utc_timestamp(str(row["price_history_start_ts"])),
            price_history_end_ts=format_utc_timestamp(str(row["price_history_end_ts"])),
            price_coverage_ratio=(
                parse_optional_float(row.get("price_coverage_ratio")) or 1.0
            ),
            funding_coverage_ratio=(
                parse_optional_float(row.get("funding_coverage_ratio")) or 0.0
            ),
            market_cap_source=str(row.get("market_cap_source", "")),
            volume_24h_source=str(row.get("volume_24h_source", "")),
            metadata_source=str(row.get("metadata_source", "")),
            screen_rank=parse_optional_int(row.get("screen_rank")),
        )

    def validate(self) -> None:
        if not self.instrument_id:
            raise ValueError(f"{self.raw_symbol} is missing instrument_id")
        for value, name in (
            (self.price_coverage_ratio, "price_coverage_ratio"),
            (self.funding_coverage_ratio, "funding_coverage_ratio"),
        ):
            if not within_ratio(value):
                raise ValueError(f"{self.raw_symbol} has invalid {name}: {value}")

        parse_utc_timestamp(self.listed_ts)
        parse_utc_timestamp(self.first_price_ts)
        parse_utc_timestamp(self.price_history_start_ts)
        parse_utc_timestamp(self.price_history_end_ts)
        if self.delisted_ts is not None:
            parse_utc_timestamp(self.delisted_ts)
        if self.frozen_rank is not None and self.frozen_rank <= 0:
            raise ValueError(f"{self.raw_symbol} frozen_rank must be positive")
        if self.selection_rank is not None and self.selection_rank <= 0:
            raise ValueError(f"{self.raw_symbol} selection_rank must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UniverseSnapshot:
    schema_version: str
    research_name: str
    formation_ts: str
    screening_ts: str
    test_window_start: str
    test_window_end: str
    thresholds: dict[str, Any]
    sources: dict[str, str]
    long_leg_assets: list[str]
    selected_short_count: int
    selected_short_symbols: list[str]
    selected_short_instrument_ids: list[str]
    selected_short_funding_coverage_ratio: float
    funding_coverage_classification: str
    funding_coverage_basis: str
    candidates: list[UniverseCandidate]

    def _validate_window(self) -> None:
        formation_ts = parse_utc_timestamp(self.formation_ts)
        screening_ts = parse_utc_timestamp(self.screening_ts)
        window_start = parse_utc_timestamp(self.test_window_start)
        window_end = parse_utc_timestamp(self.test_window_end)

        if screening_ts > formation_ts:
            raise ValueError("screening_ts must be <= formation_ts for PIT universe freezing")
        if formation_ts >= window_start:
            raise ValueError("formation_ts must be before test_window_start")
        if window_start >= window_end:
            raise ValueError("test_window_start must be before test_window_end")

    def _validate_candidate_ranks(
        self,
        eligible: list[UniverseCandidate],
    ) -> None:
        if any(candidate.frozen_rank is None for candidate in eligible):
            raise ValueError("Every eligible candidate must persist a frozen_rank")
        expected_frozen_ranks = list(range(1, len(eligible) + 1))
        actual_frozen_ranks = sorted(candidate.frozen_rank for candidate in eligible if candidate.frozen_rank is not None)
        if actual_frozen_ranks != expected_frozen_ranks:
            raise ValueError("Eligible candidate frozen_rank values must be contiguous from 1")

    def _validate_selected(
        self,
        selected: list[UniverseCandidate],
    ) -> None:
        if len(selected) != self.selected_short_count:
            raise ValueError("selected_short_count does not match selected candidate records")
        if [candidate.raw_symbol for candidate in selected] != self.selected_short_symbols:
            raise ValueError("selected_short_symbols do not match selected candidate records")
        if [candidate.instrument_id for candidate in selected] != self.selected_short_instrument_ids:
            raise ValueError("selected_short_instrument_ids do not match selected candidate records")

    def validate(self, config: ResearchConfig | None = None) -> None:
        self._validate_window()
        for candidate in self.candidates:
            candidate.validate()

        selected = [candidate for candidate in self.candidates if candidate.selected]
        eligible = [candidate for candidate in self.candidates if candidate.eligible]
        self._validate_selected(selected)
        self._validate_candidate_ranks(eligible)
        if config is not None and self.selected_short_count < config.min_short_count:
            raise ValueError("Snapshot selected fewer shorts than config.min_short_count")

    def eligible_candidates(self) -> list[UniverseCandidate]:
        return [candidate for candidate in self.candidates if candidate.eligible]

    def selected_candidates(self) -> list[UniverseCandidate]:
        return [candidate for candidate in self.candidates if candidate.selected]

    def ranked_eligible_candidates(self) -> list[UniverseCandidate]:
        return sorted(
            self.eligible_candidates(),
            key=frozen_rank_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_name": self.research_name,
            "formation_ts": self.formation_ts,
            "screening_ts": self.screening_ts,
            "test_window_start": self.test_window_start,
            "test_window_end": self.test_window_end,
            "thresholds": self.thresholds,
            "sources": self.sources,
            "long_leg_assets": self.long_leg_assets,
            "selected_short_count": self.selected_short_count,
            "selected_short_symbols": self.selected_short_symbols,
            "selected_short_instrument_ids": self.selected_short_instrument_ids,
            "selected_short_funding_coverage_ratio": self.selected_short_funding_coverage_ratio,
            "funding_coverage_classification": self.funding_coverage_classification,
            "funding_coverage_basis": self.funding_coverage_basis,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class HistoryManifestEntry:
    raw_symbol: str
    instrument_id: str
    price_path: str
    funding_path: str | None
    start_ts: str
    end_ts: str
    price_coverage_ratio: float
    funding_coverage_ratio: float
    data_cls: str = "nautilus_trader.model.data:Bar"
    bar_spec: str = "60-MINUTE-LAST"
    catalog_path: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> HistoryManifestEntry:
        return cls(
            raw_symbol=normalize_symbol(str(row["raw_symbol"] if "raw_symbol" in row else row["symbol"])),
            instrument_id=str(row["instrument_id"]).strip(),
            price_path=str(row["price_path"]).strip(),
            funding_path=(
                str(row["funding_path"]).strip()
                if row.get("funding_path") not in (None, "")
                else None
            ),
            start_ts=format_utc_timestamp(str(row["start_ts"])),
            end_ts=format_utc_timestamp(str(row["end_ts"])),
            price_coverage_ratio=parse_float(row.get("price_coverage_ratio", 1.0), "price_coverage_ratio"),
            funding_coverage_ratio=parse_float(
                row.get("funding_coverage_ratio", 0.0),
                "funding_coverage_ratio",
            ),
            data_cls=str(row.get("data_cls", "nautilus_trader.model.data:Bar")).strip(),
            bar_spec=str(row.get("bar_spec", "60-MINUTE-LAST")).strip(),
            catalog_path=(
                str(row["catalog_path"]).strip() if row.get("catalog_path") not in (None, "") else None
            ),
        )

    def validate(self) -> None:
        if not self.instrument_id:
            raise ValueError(f"{self.raw_symbol} is missing instrument_id")
        if ":" not in self.data_cls:
            raise ValueError(f"{self.raw_symbol} data_cls must be importable")
        if not within_ratio(self.price_coverage_ratio):
            raise ValueError(f"{self.raw_symbol} has invalid price_coverage_ratio")
        if not within_ratio(self.funding_coverage_ratio):
            raise ValueError(f"{self.raw_symbol} has invalid funding_coverage_ratio")
        parse_utc_timestamp(self.start_ts)
        parse_utc_timestamp(self.end_ts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedCatalogArtifact:
    schema_version: str
    research_name: str
    snapshot_path: str
    history_manifest_path: str
    artifact_root: str
    catalog_path: str | None
    catalog_ready: bool
    mode: str
    selected_short_funding_coverage_ratio: float
    classification: str
    notes: list[str]
    validated_symbols: list[str]
    staged_files: list[dict[str, str]]
    manifest_entries: list[HistoryManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_name": self.research_name,
            "snapshot_path": self.snapshot_path,
            "history_manifest_path": self.history_manifest_path,
            "artifact_root": self.artifact_root,
            "catalog_path": self.catalog_path,
            "catalog_ready": self.catalog_ready,
            "mode": self.mode,
            "selected_short_funding_coverage_ratio": self.selected_short_funding_coverage_ratio,
            "classification": self.classification,
            "notes": self.notes,
            "validated_symbols": self.validated_symbols,
            "staged_files": self.staged_files,
            "manifest_entries": [entry.to_dict() for entry in self.manifest_entries],
        }


@dataclass(slots=True)
class FeaturePanelManifest:
    schema_version: str
    variant_name: str
    placeholder: bool
    feature_schema_version: str
    feature_columns: list[str]
    date_range: dict[str, str]
    instrument_count: int
    row_count: int
    artifact_path: str
    feature_panel_path: str
    signal_artifact_path: str
    signal_panel_path: str

    def validate(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if self.instrument_count <= 0:
            raise ValueError("instrument_count must be positive")
        if self.row_count <= 0:
            raise ValueError("row_count must be positive")
        if not self.artifact_path:
            raise ValueError("artifact_path must not be empty")
        if not self.feature_panel_path:
            raise ValueError("feature_panel_path must not be empty")
        if not self.signal_artifact_path:
            raise ValueError("signal_artifact_path must not be empty")
        if not self.signal_panel_path:
            raise ValueError("signal_panel_path must not be empty")
        if {"start", "end"} - set(self.date_range):
            raise ValueError("date_range must contain start and end")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LaneArtifactManifest:
    schema_version: str
    research_name: str
    variant_name: str
    mode: str
    status: str
    placeholder: bool
    snapshot_id: str
    config_hash: str
    snapshot_path: str
    prepared_catalog_path: str
    run_request_path: str
    result_path: str
    feature_panel_manifest_path: str
    feature_panel_path: str
    signal_panel_path: str
    signal_metrics_path: str
    portfolio_metrics_path: str
    portfolio_timeseries_path: str
    summary_report_path: str

    def validate(self) -> None:
        required = (
            self.snapshot_id,
            self.config_hash,
            self.snapshot_path,
            self.prepared_catalog_path,
            self.run_request_path,
            self.result_path,
            self.feature_panel_manifest_path,
            self.feature_panel_path,
            self.signal_panel_path,
            self.signal_metrics_path,
            self.portfolio_metrics_path,
            self.portfolio_timeseries_path,
            self.summary_report_path,
        )
        if any(not value for value in required):
            raise ValueError("LaneArtifactManifest contains empty required paths or identifiers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_research_config(path: Path) -> ResearchConfig:
    config = ResearchConfig.from_dict(load_json(path))
    config.validate()
    return config


def load_snapshot(path: Path) -> UniverseSnapshot:
    payload = load_json(path)
    snapshot = UniverseSnapshot(
        schema_version=str(payload["schema_version"]),
        research_name=str(payload["research_name"]),
        formation_ts=format_utc_timestamp(payload["formation_ts"]),
        screening_ts=format_utc_timestamp(payload["screening_ts"]),
        test_window_start=format_utc_timestamp(payload["test_window_start"]),
        test_window_end=format_utc_timestamp(payload["test_window_end"]),
        thresholds=dict(payload["thresholds"]),
        sources=dict(payload["sources"]),
        long_leg_assets=list(payload["long_leg_assets"]),
        selected_short_count=int(payload["selected_short_count"]),
        selected_short_symbols=list(payload["selected_short_symbols"]),
        selected_short_instrument_ids=list(payload["selected_short_instrument_ids"]),
        selected_short_funding_coverage_ratio=float(
            payload["selected_short_funding_coverage_ratio"],
        ),
        funding_coverage_classification=str(payload["funding_coverage_classification"]),
        funding_coverage_basis=str(payload["funding_coverage_basis"]),
        candidates=[UniverseCandidate(**candidate) for candidate in payload["candidates"]],
    )
    snapshot.validate()
    return snapshot


def load_history_manifest(path: Path) -> list[HistoryManifestEntry]:
    try:
        from .common import load_records
    except ImportError:  # pragma: no cover - script execution fallback
        from common import load_records

    entries = [HistoryManifestEntry.from_dict(row) for row in load_records(path)]
    for entry in entries:
        entry.validate()
    return entries


def load_prepared_catalog(path: Path) -> PreparedCatalogArtifact:
    payload = load_json(path)
    return PreparedCatalogArtifact(
        schema_version=str(payload["schema_version"]),
        research_name=str(payload["research_name"]),
        snapshot_path=str(payload["snapshot_path"]),
        history_manifest_path=str(payload["history_manifest_path"]),
        artifact_root=str(payload["artifact_root"]),
        catalog_path=payload.get("catalog_path"),
        catalog_ready=bool(payload["catalog_ready"]),
        mode=str(payload["mode"]),
        selected_short_funding_coverage_ratio=float(
            payload["selected_short_funding_coverage_ratio"],
        ),
        classification=str(payload["classification"]),
        notes=list(payload["notes"]),
        validated_symbols=list(payload["validated_symbols"]),
        staged_files=list(payload["staged_files"]),
        manifest_entries=[HistoryManifestEntry(**entry) for entry in payload["manifest_entries"]],
    )


def load_feature_panel_manifest(path: Path) -> FeaturePanelManifest:
    payload = load_json(path)
    manifest = FeaturePanelManifest(
        schema_version=str(payload["schema_version"]),
        variant_name=str(payload["variant_name"]),
        placeholder=bool(payload["placeholder"]),
        feature_schema_version=str(payload["feature_schema_version"]),
        feature_columns=list(payload["feature_columns"]),
        date_range=dict(payload["date_range"]),
        instrument_count=int(payload["instrument_count"]),
        row_count=int(payload["row_count"]),
        artifact_path=str(payload.get("artifact_path") or payload["feature_panel_path"]),
        feature_panel_path=str(payload.get("feature_panel_path") or payload["artifact_path"]),
        signal_artifact_path=str(payload.get("signal_artifact_path") or payload["signal_panel_path"]),
        signal_panel_path=str(payload.get("signal_panel_path") or payload["signal_artifact_path"]),
    )
    manifest.validate()
    return manifest


def load_lane_artifact_manifest(path: Path) -> LaneArtifactManifest:
    payload = load_json(path)
    manifest = LaneArtifactManifest(
        schema_version=str(payload["schema_version"]),
        research_name=str(payload["research_name"]),
        variant_name=str(payload["variant_name"]),
        mode=str(payload["mode"]),
        status=str(payload["status"]),
        placeholder=bool(payload["placeholder"]),
        snapshot_id=str(payload["snapshot_id"]),
        config_hash=str(payload["config_hash"]),
        snapshot_path=str(payload["snapshot_path"]),
        prepared_catalog_path=str(payload["prepared_catalog_path"]),
        run_request_path=str(payload["run_request_path"]),
        result_path=str(payload["result_path"]),
        feature_panel_manifest_path=str(payload["feature_panel_manifest_path"]),
        feature_panel_path=str(payload["feature_panel_path"]),
        signal_panel_path=str(payload["signal_panel_path"]),
        signal_metrics_path=str(payload["signal_metrics_path"]),
        portfolio_metrics_path=str(payload["portfolio_metrics_path"]),
        portfolio_timeseries_path=str(payload["portfolio_timeseries_path"]),
        summary_report_path=str(payload["summary_report_path"]),
    )
    manifest.validate()
    return manifest


def frozen_rank_key(candidate: UniverseCandidate) -> tuple[int, int, float, float, int, str]:
    return (
        0 if candidate.frozen_rank is not None else 1,
        candidate.frozen_rank if candidate.frozen_rank is not None else 10_000_000,
        candidate.volume_24h_usd,
        candidate.market_cap_usd,
        candidate.screen_rank if candidate.screen_rank is not None else 10_000_000,
        candidate.raw_symbol,
    )


def save_artifact(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def snapshot_identity(snapshot: UniverseSnapshot) -> str:
    return stable_payload_hash(snapshot.to_dict())[:12]
