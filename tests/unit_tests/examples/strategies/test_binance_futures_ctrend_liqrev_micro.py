from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from examples.live.binance import binance_futures_ctrend_liqrev_micro as micro_module


def test_monitor_kill_switch_persists_latest_status_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "ctrend_liqrev_micro_status.json"
    args = SimpleNamespace(
        research_config="examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json",
        binance_environment="live",
        short_band="rank_1_20",
        initial_leverage=2,
        starting_equity_usdt=200.0,
        max_total_drawdown_usdt=20.0,
        max_daily_loss_usdt=6.0,
        min_available_balance_usdt=50.0,
        kill_switch_check_interval_secs=0,
    )
    writes: list[tuple[Path, dict[str, object]]] = []

    monkeypatch.setattr(
        micro_module,
        "build_ops_namespace",
        lambda *call_args, **call_kwargs: SimpleNamespace(
            output_json=str(report_path),
            command=call_kwargs["command"],
        ),
    )
    monkeypatch.setattr(
        micro_module.asyncio,
        "run",
        lambda coroutine_result: {
            "generated_at": "2026-04-21T00:00:00Z",
            "dedicated_account_ready": True,
            "kill_switch_triggered": False,
        },
    )
    monkeypatch.setattr(micro_module, "command_status", lambda _args: object())
    monkeypatch.setattr(micro_module, "command_killswitch", lambda _args: object())
    monkeypatch.setattr(
        micro_module,
        "write_json_report",
        lambda path, payload: writes.append((Path(path), payload)),
    )

    class OneShotStopEvent:
        def __init__(self) -> None:
            self._calls = 0

        def wait(self, _seconds: int) -> bool:
            self._calls += 1
            return self._calls > 1

    micro_module.monitor_kill_switch(args, OneShotStopEvent())

    assert writes == [
        (
            report_path,
            {
                "generated_at": "2026-04-21T00:00:00Z",
                "dedicated_account_ready": True,
                "kill_switch_triggered": False,
            },
        ),
    ]
