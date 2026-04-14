#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


try:
    from .common import format_utc_timestamp
    from .common import load_records
    from .common import parse_utc_timestamp
    from .common import pre_window_cutoff
    from .schemas import SCHEMA_VERSION
    from .schemas import UniverseCandidate
    from .schemas import UniverseSnapshot
    from .schemas import classify_funding_coverage
    from .schemas import load_research_config
    from .schemas import load_snapshot
    from .schemas import renormalize_long_weights
    from .schemas import save_artifact
except ImportError:  # pragma: no cover - script execution fallback
    from common import format_utc_timestamp
    from common import load_records
    from common import parse_utc_timestamp
    from common import pre_window_cutoff
    from schemas import SCHEMA_VERSION
    from schemas import UniverseCandidate
    from schemas import UniverseSnapshot
    from schemas import classify_funding_coverage
    from schemas import load_research_config
    from schemas import load_snapshot
    from schemas import renormalize_long_weights
    from schemas import save_artifact

__all__ = [
    "UniverseSnapshot",
    "classify_funding",
    "evaluate_candidate",
    "load_snapshot",
    "renormalize_long_weights",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a point-in-time crypto RV short-universe snapshot.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the research config JSON.",
    )
    parser.add_argument(
        "--input",
        help="Optional override for the external screening CSV/JSON input.",
    )
    parser.add_argument(
        "--output",
        help="Optional override for the output snapshot JSON path.",
    )
    parser.add_argument(
        "--screening-ts",
        help="Override the screening timestamp recorded in the snapshot.",
    )
    parser.add_argument(
        "--selection-size",
        type=int,
        help="Override the primary short count for the emitted snapshot.",
    )
    return parser


def rank_key(candidate: UniverseCandidate) -> tuple[float, float, int, str]:
    explicit_rank = candidate.screen_rank if candidate.screen_rank is not None else 10_000_000
    return (
        candidate.volume_24h_usd,
        candidate.market_cap_usd,
        explicit_rank,
        candidate.raw_symbol,
    )


def classify_funding(snapshot_coverage: float, warning_threshold: float, partial_threshold: float) -> str:
    return classify_funding_coverage(
        snapshot_coverage,
        partial_threshold=partial_threshold,
        warning_threshold=warning_threshold,
    )


def evaluate_candidate(
    candidate: UniverseCandidate,
    *,
    formation_ts: str,
    window_start: str,
    window_end: str,
    min_market_cap_usd: float,
    max_market_cap_usd: float,
    max_volume_24h_usd: float,
    min_price_coverage_ratio: float,
    pre_window_history_cutoff: str,
) -> UniverseCandidate:
    candidate.validate()

    listed_ts = parse_utc_timestamp(candidate.listed_ts)
    formation = parse_utc_timestamp(formation_ts)
    history_cutoff = parse_utc_timestamp(pre_window_history_cutoff)
    window_end_ts = parse_utc_timestamp(window_end)

    reasons: list[str] = []
    if candidate.market_cap_usd < min_market_cap_usd or candidate.market_cap_usd > max_market_cap_usd:
        reasons.append("market_cap_out_of_range")
    if candidate.volume_24h_usd >= max_volume_24h_usd:
        reasons.append("volume_above_threshold")
    if listed_ts > formation:
        reasons.append("listed_after_formation")
    if candidate.delisted_ts and parse_utc_timestamp(candidate.delisted_ts) < window_end_ts:
        reasons.append("delisted_before_window_end")
    if parse_utc_timestamp(candidate.first_price_ts) > history_cutoff:
        reasons.append("first_price_after_pre_window_cutoff")
    if parse_utc_timestamp(candidate.price_history_start_ts) > history_cutoff:
        reasons.append("insufficient_pre_window_history")
    if parse_utc_timestamp(candidate.price_history_end_ts) < window_end_ts:
        reasons.append("history_ends_before_window")
    if candidate.price_coverage_ratio < min_price_coverage_ratio:
        reasons.append("price_coverage_below_floor")

    candidate.eligible = not reasons
    candidate.exclusion_reasons = reasons
    return candidate


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_research_config(config_path)
    paths = config.resolved_paths(config_path)

    input_path = Path(args.input).resolve() if args.input else paths["universe_input_path"]
    output_path = Path(args.output).resolve() if args.output else paths["snapshot_output_path"]
    screening_ts = format_utc_timestamp(args.screening_ts or config.formation_ts)
    selection_size = args.selection_size or config.primary_short_count

    raw_rows = load_records(input_path)
    candidates = [UniverseCandidate.from_screening_row(row) for row in raw_rows]

    pre_window_history_cutoff = pre_window_cutoff(
        config.test_window_start,
        config.thresholds.pre_window_history_days,
    )
    evaluated = [
        evaluate_candidate(
            candidate,
            formation_ts=config.formation_ts,
            window_start=config.test_window_start,
            window_end=config.test_window_end,
            min_market_cap_usd=config.thresholds.min_market_cap_usd,
            max_market_cap_usd=config.thresholds.max_market_cap_usd,
            max_volume_24h_usd=config.thresholds.max_volume_24h_usd,
            min_price_coverage_ratio=config.thresholds.min_price_coverage_ratio,
            pre_window_history_cutoff=pre_window_history_cutoff,
        )
        for candidate in candidates
    ]

    eligible = sorted((candidate for candidate in evaluated if candidate.eligible), key=rank_key)
    if len(eligible) < config.min_short_count:
        raise ValueError(
            f"Only {len(eligible)} eligible shorts found, below required minimum "
            f"{config.min_short_count}",
        )

    for index, candidate in enumerate(eligible, start=1):
        candidate.frozen_rank = index

    selected = eligible[: min(selection_size, len(eligible))]
    for index, candidate in enumerate(selected, start=1):
        candidate.selected = True
        candidate.selection_rank = index

    selected_symbols = [candidate.raw_symbol for candidate in selected]
    selected_instrument_ids = [candidate.instrument_id for candidate in selected]
    funding_ratio = sum(candidate.funding_coverage_ratio for candidate in selected) / len(selected)
    classification = classify_funding(
        funding_ratio,
        config.thresholds.funding_warning_threshold,
        config.thresholds.funding_partial_evidence_threshold,
    )

    snapshot = UniverseSnapshot(
        schema_version=SCHEMA_VERSION,
        research_name=config.research_name,
        formation_ts=config.formation_ts,
        screening_ts=screening_ts,
        test_window_start=config.test_window_start,
        test_window_end=config.test_window_end,
        thresholds={
            **config.thresholds.to_dict(),
            "selection_size": selection_size,
        },
        sources={
            "market_cap": "external_screening_input",
            "volume_24h": "external_screening_input",
            "instrument_metadata": "external_screening_input",
        },
        long_leg_assets=[member.asset for member in config.long_leg],
        selected_short_count=len(selected),
        selected_short_symbols=selected_symbols,
        selected_short_instrument_ids=selected_instrument_ids,
        selected_short_funding_coverage_ratio=funding_ratio,
        funding_coverage_classification=classification,
        funding_coverage_basis="equal-weight notional-days assumption across selected shorts",
        candidates=evaluated,
    )
    snapshot.validate(config)
    save_artifact(output_path, snapshot.to_dict())

    print(
        f"Wrote snapshot to {output_path} with {len(selected)} selected shorts "
        f"from {len(eligible)} eligible candidates ({len(evaluated)} screened).",
    )
    print(
        "Selected short funding coverage ratio "
        f"{snapshot.selected_short_funding_coverage_ratio:.2%} "
        f"-> {snapshot.funding_coverage_classification}",
    )


if __name__ == "__main__":
    main()
