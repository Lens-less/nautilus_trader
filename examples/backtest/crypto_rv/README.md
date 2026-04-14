# Crypto RV Research Scaffolding

This directory contains the top-level research scripts and config schema for the approved
`Binance USDT perpetuals` relative-value plan:

- long basket fixed to `BTC`, `ETH`, `SOL`, `BNB`, `HYPE`
- short basket frozen point-in-time from externally screened low-liquidity alts
- USD notional neutrality
- mandatory `base-cost` and `stress-cost`
- mandatory `HYPE-out`, `20/30/50`, and `weekly/biweekly` sensitivity branches

The scope here is intentionally the *top-level research workflow*. The package strategy class now
exists under `nautilus_trader/examples/strategies/crypto_rv_basket.py`, while this directory owns
the point-in-time universe snapshot, prepared-catalog artifact, run planning, and report
scaffolding. `run_backtest.py` still defaults to a bounded `plan-only` mode until a verified
Nautilus catalog is available.

## Files

- `prepare_universe.py`: loads external screening inputs and emits a point-in-time snapshot JSON
- `prepare_catalog.py`: validates and stages normalized historical inputs into a local research
  artifact bundle, and marks whether a real Nautilus catalog is already available
- `run_backtest.py`: builds `ImportableStrategyConfig` + `BacktestRunConfig` requests for the
  baseline and required sensitivities; optionally executes if a real strategy import and catalog
  exist
- `report.py`: compiles a research summary from generated run outputs
- `schemas.py`: config / snapshot / prepared-catalog dataclasses and validation
- `configs/`: example config, sample screening input, and sample history manifest

## Quick Start

Run the sample dry-run workflow from the repo root:

```bash
cd /Users/lens/Desktop/ns-st/nautilus_trader
uv run python examples/backtest/crypto_rv/prepare_universe.py \
  --config examples/backtest/crypto_rv/configs/research.example.json

uv run python examples/backtest/crypto_rv/prepare_catalog.py \
  --config examples/backtest/crypto_rv/configs/research.example.json

uv run python examples/backtest/crypto_rv/run_backtest.py \
  --config examples/backtest/crypto_rv/configs/research.example.json

uv run python examples/backtest/crypto_rv/report.py \
  --config examples/backtest/crypto_rv/configs/research.example.json
```

## External Input Contracts

### Screener input

`prepare_universe.py` accepts CSV or JSON records with at least:

- `raw_symbol`
- `instrument_id`
- `market_cap_usd`
- `volume_24h_usd`
- `listed_ts`
- `delisted_ts` (optional)
- `first_price_ts`
- `price_history_start_ts`
- `price_history_end_ts`
- `price_coverage_ratio`
- `funding_coverage_ratio`
- `screen_rank` (optional)

The script persists the *full candidate set*, not just the selected names, and records exclusions
and reasons.

### History manifest

`prepare_catalog.py` accepts CSV or JSON rows with at least:

- `raw_symbol`
- `instrument_id`
- `price_path`
- `funding_path` (optional)
- `start_ts`
- `end_ts`
- `price_coverage_ratio`
- `funding_coverage_ratio`
- `data_cls`
- `bar_spec`
- `catalog_path` (optional)

When `catalog_path` is missing or unverified, the script still stages local inputs and writes a
research artifact, but `run_backtest.py` will stay in `plan-only` mode.

## Current Integration Gaps

- A verified Nautilus `ParquetDataCatalog` still has to be connected before `--execute` can run a
  real backtest.
- `report.py` currently compiles scaffolding / placeholders until realized result files exist.

## Worktree Workflow

Generated research artifacts under `output/` are local working evidence and are ignored by default
for the seed/worktree flow. The seed baseline should commit the source tree, configs, strategy
module, and tests without dragging large local catalogs or run outputs into git history.

The approved worktree layout is managed by [`scripts/crypto_xsec_worktrees.py`](../../../scripts/crypto_xsec_worktrees.py):

```bash
cd /Users/lens/Desktop/ns-st/nautilus_trader

# Review the approved lane map.
uv run python scripts/crypto_xsec_worktrees.py plan

# Ensure the rolling integration branch exists after creating the seed baseline.
uv run python scripts/crypto_xsec_worktrees.py init-branches --seed-start-point HEAD

# Create the first kernel worktree.
uv run python scripts/crypto_xsec_worktrees.py add 00-kernel

# Check which branches/worktrees currently exist.
uv run python scripts/crypto_xsec_worktrees.py status
```

The intended execution order remains:

1. `00-kernel`
2. `01-liq-reversal` pilot
3. `02-carry-basis` and `03-vol-trend`
4. `04-conditional` and optional `05-overlays`
5. `06-combo`
