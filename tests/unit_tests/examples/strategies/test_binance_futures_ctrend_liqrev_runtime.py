from pathlib import Path

from examples.live.binance.binance_futures_ctrend_liqrev_runtime import RuntimeConfig
from examples.live.binance.binance_futures_ctrend_liqrev_runtime import RuntimePaths
from examples.live.binance.binance_futures_ctrend_liqrev_runtime import build_effective_env
from examples.live.binance.binance_futures_ctrend_liqrev_runtime import build_micro_command
from examples.live.binance.binance_futures_ctrend_liqrev_runtime import build_paths
from examples.live.binance.binance_futures_ctrend_liqrev_runtime import load_runtime_config
from examples.live.binance.binance_futures_ctrend_liqrev_runtime import parse_env_file


def test_parse_env_file_handles_export_and_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / "live.env"
    env_file.write_text(
        """# comment
export BINANCE_API_KEY="abc123"
CTREND_SHORT_BAND=rank_1_20
CTREND_LOG_LEVEL='DEBUG'
""",
        encoding="utf-8",
    )

    payload = parse_env_file(env_file)

    assert payload == {
        "BINANCE_API_KEY": "abc123",
        "CTREND_SHORT_BAND": "rank_1_20",
        "CTREND_LOG_LEVEL": "DEBUG",
    }


def test_build_effective_env_resolves_file_backed_secrets(tmp_path: Path) -> None:
    key_file = tmp_path / "binance_api_key.txt"
    secret_file = tmp_path / "binance_api_secret.txt"
    key_file.write_text("key-from-file\n", encoding="utf-8")
    secret_file.write_text("secret-from-file\n", encoding="utf-8")

    env_file = tmp_path / "live.env"
    env_file.write_text(
        "\n".join(
            [
                f"BINANCE_API_KEY_FILE={key_file.name}",
                f"BINANCE_API_SECRET_FILE={secret_file.name}",
            ],
        ),
        encoding="utf-8",
    )

    env = build_effective_env(build_paths(str(env_file)))

    assert env["BINANCE_API_KEY"] == "key-from-file"
    assert env["BINANCE_API_SECRET"] == "secret-from-file"
    assert env["CTREND_BINANCE_API_KEY_SOURCE"] == str(key_file.resolve())
    assert env["CTREND_BINANCE_API_SECRET_SOURCE"] == str(secret_file.resolve())


def test_load_runtime_config_resolves_repo_relative_research_inputs(tmp_path: Path) -> None:
    env_file = tmp_path / "live.env"
    env_file.write_text(
        """BINANCE_API_KEY="inline-key"
BINANCE_API_SECRET="inline-secret"
""",
        encoding="utf-8",
    )

    paths = build_paths(str(env_file))
    env = build_effective_env(paths)
    config = load_runtime_config(paths, env)

    assert config.research_config.exists()
    assert config.selected_universe_path.exists()
    assert config.catalog_path.exists()
    assert config.baseline_metrics_path.exists()
    assert config.short_fallback_bands == ("rank_1_30", "rank_1_50")
    assert config.min_active_shorts == 4


def test_build_micro_command_uses_runtime_settings(tmp_path: Path) -> None:
    base_dir = tmp_path / "project"
    paths = RuntimePaths(
        root_dir=base_dir,
        runtime_dir=base_dir / "examples/live/binance/runtime",
        env_file=base_dir / "examples/live/binance/runtime/live.env",
        log_path=base_dir / "examples/live/binance/runtime/bootstrap-live.log",
        nohup_path=base_dir / "examples/live/binance/runtime/bootstrap-nohup.out",
        pid_path=base_dir / "examples/live/binance/runtime/live-runner.pid",
        status_json_path=base_dir / "examples/live/binance/runtime/ctrend_liqrev_micro_status.json",
        approval_dir=base_dir / "examples/live/binance/runtime/approvals",
        wheel_cache_dir=tmp_path / "nautilus-wheel",
        wheel_target_dir=tmp_path / "nautilus-wheel/site-packages",
    )
    config = RuntimeConfig(
        research_config=base_dir / "examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json",
        research_root=base_dir / "examples/backtest/crypto_rv/output/real_public_2026q1",
        selected_universe_path=base_dir
        / "examples/backtest/crypto_rv/output/real_public_2026q1/selected_universe.json",
        catalog_path=base_dir
        / "examples/backtest/crypto_rv/output/real_public_2026q1/followups/catalog/catalog_50",
        baseline_metrics_path=base_dir
        / "examples/backtest/crypto_rv/output/real_public_2026q1/real_backtest_metrics.json",
        binance_environment="live",
        trader_id="CTREND-LIVE-001",
        leg_notional_usd=75.0,
        short_band="rank_1_20",
        short_fallback_bands=("rank_1_30", "rank_1_50"),
        min_active_shorts=4,
        history_lookback_days=90,
        order_expire_seconds=900,
        log_level="INFO",
        initial_leverage=2,
        starting_equity_usdt=200.0,
        max_total_drawdown_usdt=20.0,
        max_daily_loss_usdt=6.0,
        min_available_balance_usdt=50.0,
        kill_switch_check_interval_secs=60,
        wheel_spec="nautilus_trader==1.226.0a20260414",
        credential_sources={
            "BINANCE_API_KEY": "inline",
            "BINANCE_API_SECRET": "inline",
        },
    )

    command = build_micro_command(paths, config)

    assert command[0] == str(paths.project_python)
    assert "--binance-environment" in command
    assert "live" in command
    assert "--leg-notional-usd" in command
    assert "75.0" in command
    assert "--short-fallback-bands" in command
    assert "rank_1_30,rank_1_50" in command
    assert "--kill-switch-check-interval-secs" in command
