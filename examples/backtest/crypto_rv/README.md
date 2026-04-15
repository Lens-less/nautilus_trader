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
- `run_ctrend_backtest.py`: runs the CTREND cross-sectional trend lane and emits machine-readable
  artifacts plus an offline baseline comparison
- `report.py`: compiles a research summary from generated run outputs
- `report_ctrend.py`: compiles the CTREND lane summary and replacement verdict
- `schemas.py`: config / snapshot / prepared-catalog dataclasses and validation
- `ctrend_schemas.py`: CTREND-specific config schema so the old RV contract stays untouched
- `signals/vol_managed_trend.py`: pure-Python CTREND v1 signal logic shared by the lane runner and
  the importable strategy container
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

# Reuse the same snapshot/catalog artifacts for the CTREND lane.
uv run python examples/backtest/crypto_rv/run_ctrend_backtest.py \
  --config examples/backtest/crypto_rv/configs/ctrend.example.json

uv run python examples/backtest/crypto_rv/report_ctrend.py \
  --config examples/backtest/crypto_rv/configs/ctrend.example.json
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
- `run_ctrend_backtest.py` currently computes research-grade offline portfolio metrics from the
  staged price paths so the lane can be compared before full engine execution is wired.

## RV Baseline vs CTREND Lane

- `run_backtest.py` + `CryptoRVBasketStrategy` remain the preserved fixed-basket RV baseline.
- `run_ctrend_backtest.py` + `CryptoXSecTrendStrategy` are the new CTREND mainline candidate.
- The CTREND replacement rule is only positive when net return is higher than baseline and max
  drawdown is not worse.

## Shared Kernel Contract

The kernel lane now emits a stable machine-readable contract per variant under `output/runs/<variant>/`,
even in `plan-only` mode. These artifacts are intentionally placeholder-valued until a verified catalog
and lane-specific signal logic are connected, but the file layout and schemas are fixed for downstream
lanes:

- `lane_manifest.json`
- `feature_panel_manifest.json`
- `feature_panel.parquet`
- `signal_panel.parquet`
- `signal_metrics.json`
- `portfolio_metrics.json`
- `portfolio_timeseries.parquet`
- `summary_report.md`

This lets later lanes such as `01-liq-reversal`, `04-conditional`, and `06-combo` depend on a stable
artifact contract without having to guess output locations or column names.

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

The helper auto-detects the shared `worktrees/` root both from the seed repo and from an active
lane worktree. You only need `--worktrees-root` if you intentionally keep lane worktrees
somewhere else.

The intended execution order remains:

1. `00-kernel`
2. `01-liq-reversal` pilot
3. `02-carry-basis` and `03-vol-trend`
4. `04-conditional` and optional `05-overlays`
5. `06-combo`
