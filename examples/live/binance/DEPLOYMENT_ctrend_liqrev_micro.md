# CTREND Binance Micro-Live Deployment Runbook

This runbook captures the deployment path that actually worked for the accepted
micro-live lane:

- venue: Binance USD-M futures
- strategy: `vol-managed-biweekly-base | post_rank_filter | rank_1_20`
- live optimization: elastic fallback `rank_1_20 -> rank_1_30 -> rank_1_50` when fewer than `4` CTREND shorts remain
- defaults: `75U` per leg, `min_active_shorts=4`, first two rebalances approval-gated

## Canonical Entry Point

Use the runtime manager instead of ad-hoc shell commands:

```bash
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py check
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py start
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py process-status
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py stop
```

Create `examples/live/binance/runtime/live.env` from
`examples/live/binance/live.env.example` before running `check` or `start`.

## Why This Flow Exists

The original "just build and run" path failed in ways that are easy to repeat:

1. Small Linux boxes can fail during `cargo build` / `ld.bfd` linking.
   The working fix was to `uv sync --no-install-project` and then overlay the
   official NautilusTrader wheel into the repo before running live code.
2. Research configs that point at machine-specific absolute paths break as soon
   as the repo moves to another worktree or server.
3. Exchange-side prepare steps must be idempotent.
   Binance can return "already set" responses such as `-4059`.
4. Credential format mistakes should fail at `status`, before `prepare` or the
   live runner starts.
5. Passive-only orders should not quietly give up for the rest of the biweekly
   window. If the execution window expires, the strategy now schedules a fresh
   rebalance retry from the latest synchronized snapshot.

## Required Inputs

- `examples/live/binance/runtime/live.env`
- repo-local research artifacts reachable through
  `examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json`
- a funded Binance account dedicated to this lane

The runtime manager checks:

- credentials exist
- `_FILE` secret references resolve if used
- research config exists
- `selected_universe.json`, `catalog_50`, and `real_backtest_metrics.json` exist
- no stale runner PID is already active

## Runtime Files

The lane uses these files under `examples/live/binance/runtime/`:

- `live.env`: operator-managed secrets and runtime parameters
- `bootstrap-live.log`: authoritative bootstrap + runner log
- `bootstrap-nohup.out`: detached launcher output
- `live-runner.pid`: detached runtime manager / runner PID
- `ctrend_liqrev_micro_status.json`: latest `status` payload
- `approvals/`: first-two-rebalances approval artifacts

## Start Procedure

1. Run `check`.
2. Review the rendered config summary.
3. Start the runner with `start`.
4. Confirm `process-status` shows an active PID.
5. Tail `bootstrap-live.log` until you see:
   - successful `status`
   - successful `prepare`
   - `TradingNode: STARTING`
   - `DataClient-BINANCE: Connected`
   - `ExecClient-BINANCE: Binance API key authenticated`

## Approval Gate

The first two rebalances are still manual-approval events.
When a rebalance preview is emitted, review the JSON under
`examples/live/binance/runtime/approvals/` and create the sibling `.approved`
file to release it.

## Operational Checks

Before trusting the lane, confirm:

- `dedicated_account_ready` is `true`
- `foreign_open_positions` is empty
- `kill_switch_triggered` is `false`
- the account-level thresholds in `live.env` match the funded account size

Do not reuse the default `200 / 20 / 6 / 50` kill-switch numbers if the real
account balance is materially larger.

## Failure Signatures

### `API-key format invalid`

The key/secret pair is not accepted by Binance.
Fix `live.env` and rerun `check` or `start`.

### `No need to change position side`

This is an idempotent prepare response from Binance, not a fatal condition.
The ops helper treats it as "already in one-way mode".

### `FileNotFoundError` for `selected_universe.json` or `catalog_50`

The research config still points at missing artifacts.
Keep those paths repo-relative and ensure the real-public Q1 outputs are synced.

### `ld.bfd` / `cargo build` killed by OOM

Do not go back to full source builds on small servers.
Use the wheel-overlay runtime path.

## References

- NautilusTrader Binance integration:
  <https://nautilustrader.io/docs/latest/integrations/binance/>
- uv CLI reference:
  <https://docs.astral.sh/uv/reference/cli/>
