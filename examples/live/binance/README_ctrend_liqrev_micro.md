# CTREND Conditional Micro-Live

This lane is the production-shaped runner for the accepted research winner:

- venue: Binance USD-M futures
- strategy: `vol-managed-biweekly-base | post_rank_filter | rank_1_20`
- live optimization: when `rank_1_20` leaves fewer than `4` active shorts, widen to `rank_1_30`, then `rank_1_50`
- capital policy: `200U` account, `150U gross`, `50U` idle
- operations policy: first two rebalances human-supervised, frozen `rank_1_20` for six weeks

## Commands

Primary runtime entry point:

```bash
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py check
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py start
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py process-status
uv run python examples/live/binance/binance_futures_ctrend_liqrev_runtime.py stop
```

Seed `examples/live/binance/runtime/live.env` from
`examples/live/binance/live.env.example` and review
`examples/live/binance/DEPLOYMENT_ctrend_liqrev_micro.md` before using `start`.

Direct exchange-side operations are still available when needed.

Prepare the account and symbol-level settings:

```bash
export BINANCE_DEMO_API_KEY="..."
export BINANCE_DEMO_API_SECRET="..."

uv run python examples/live/binance/binance_futures_ctrend_liqrev_ops.py \
  --binance-environment demo \
  prepare
```

Fetch account status and kill-switch inputs:

```bash
uv run python examples/live/binance/binance_futures_ctrend_liqrev_ops.py \
  --binance-environment demo \
  status
```

Run the micro-live node:

```bash
uv run python examples/live/binance/binance_futures_ctrend_liqrev_micro.py \
  --binance-environment demo
```

Trigger a kill switch manually:

```bash
uv run python examples/live/binance/binance_futures_ctrend_liqrev_ops.py \
  --binance-environment demo \
  killswitch --force
```

## Default Risk Settings

- `starting_equity_usdt = 200`
- `max_total_drawdown_usdt = 20`
- `max_daily_loss_usdt = 6`
- `min_available_balance_usdt = 50`
- `min_active_shorts = 4`
- `initial_leverage = 2`

## Notes

- The strategy requests historical daily bars on start so it does not need to wait 25 days to warm up.
- The strategy uses passive limit orders by default and will not automatically fall back to market orders.
- If a post-only execution window fully expires, the runner resets the rebalance timestamp and re-queues a fresh rebalance instead of silently waiting until the next biweekly cadence.
- The ops helper writes JSON status reports under `examples/live/binance/runtime/`.
- The runtime manager writes bootstrap and detached-process logs under `examples/live/binance/runtime/`.
- Funding remains outside signal generation. The live lane keeps that research boundary intact.
- The runner refuses to start if unrelated Binance futures positions are already open in the account.
- The first two rebalances are approval-gated. Review the preview JSON under `examples/live/binance/runtime/approvals/` and create the sibling `.approved` file to release the rebalance.
- The runtime manager uses the official Nautilus wheel overlay by default to avoid source-build failures on small Linux hosts.
