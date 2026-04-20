#!/usr/bin/env python3
# mypy: disable-error-code=no-redef
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


INDEX_URL = "https://packages.nautechsystems.io/simple"
ENVIRONMENT_CHOICES = ("demo", "testnet", "live")
SHORT_BAND_CHOICES = ("rank_1_20", "rank_1_30", "rank_1_50")
TRUE_VALUES = {"1", "true", "yes", "on"}


def load_research_config(path: Path) -> Any:
    try:
        from examples.backtest.crypto_rv.run_ctrend_real_public import (
            load_config as _load_research_config,
        )
    except ImportError:  # pragma: no cover - script execution fallback
        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from examples.backtest.crypto_rv.run_ctrend_real_public import (
            load_config as _load_research_config,
        )

    return _load_research_config(path)


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    root_dir: Path
    runtime_dir: Path
    env_file: Path
    log_path: Path
    nohup_path: Path
    pid_path: Path
    status_json_path: Path
    approval_dir: Path
    wheel_cache_dir: Path
    wheel_target_dir: Path

    @property
    def project_python(self) -> Path:
        return self.root_dir / ".venv" / "bin" / "python"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    research_config: Path
    research_root: Path
    selected_universe_path: Path
    catalog_path: Path
    baseline_metrics_path: Path
    binance_environment: str
    trader_id: str
    leg_notional_usd: float
    short_band: str
    short_fallback_bands: tuple[str, ...]
    min_active_shorts: int
    history_lookback_days: int
    order_expire_seconds: int
    log_level: str
    initial_leverage: int
    starting_equity_usdt: float
    max_total_drawdown_usdt: float
    max_daily_loss_usdt: float
    min_available_balance_usdt: float
    kill_switch_check_interval_secs: int
    wheel_spec: str
    credential_sources: dict[str, str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def build_paths(env_file: str | None) -> RuntimePaths:
    script_dir = Path(__file__).resolve().parent
    runtime_dir = script_dir / "runtime"
    env_path = Path(env_file).expanduser().resolve() if env_file else (runtime_dir / "live.env")
    wheel_cache_dir = Path(
        os.environ.get("CTREND_WHEEL_CACHE_DIR", "/tmp/nautilus-wheel"),  # noqa: S108
    ).expanduser()
    return RuntimePaths(
        root_dir=script_dir.parents[2],
        runtime_dir=runtime_dir,
        env_file=env_path,
        log_path=runtime_dir / "bootstrap-live.log",
        nohup_path=runtime_dir / "bootstrap-nohup.out",
        pid_path=runtime_dir / "live-runner.pid",
        status_json_path=runtime_dir / "ctrend_liqrev_micro_status.json",
        approval_dir=runtime_dir / "approvals",
        wheel_cache_dir=wheel_cache_dir,
        wheel_target_dir=wheel_cache_dir / "site-packages",
    )


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _resolve_optional_path(value: str, *, base_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def resolve_secret_value(raw_env: dict[str, str], key: str, *, base_dir: Path) -> tuple[str, str]:
    direct_value = raw_env.get(key)
    file_value = raw_env.get(f"{key}_FILE")

    if direct_value and file_value:
        raise ValueError(f"{key} and {key}_FILE cannot both be set")

    if file_value:
        file_path = _resolve_optional_path(file_value, base_dir=base_dir)
        if not file_path.exists():
            raise ValueError(f"{key}_FILE path does not exist: {file_path}")
        return file_path.read_text(encoding="utf-8").strip(), str(file_path)

    if direct_value:
        return direct_value, "inline"

    raise ValueError(f"{key} not found in runtime environment")


def build_effective_env(paths: RuntimePaths) -> dict[str, str]:
    raw_env = parse_env_file(paths.env_file)
    merged = os.environ.copy()
    merged.update(raw_env)
    env_base_dir = paths.env_file.parent
    existing_pythonpath = merged.get("PYTHONPATH")
    merged["PYTHONPATH"] = (
        f"{paths.root_dir}:{existing_pythonpath}"
        if existing_pythonpath
        else str(paths.root_dir)
    )

    api_key, api_key_source = resolve_secret_value(merged, "BINANCE_API_KEY", base_dir=env_base_dir)
    api_secret, api_secret_source = resolve_secret_value(
        merged,
        "BINANCE_API_SECRET",
        base_dir=env_base_dir,
    )
    merged["BINANCE_API_KEY"] = api_key
    merged["BINANCE_API_SECRET"] = api_secret
    merged["CTREND_BINANCE_API_KEY_SOURCE"] = api_key_source
    merged["CTREND_BINANCE_API_SECRET_SOURCE"] = api_secret_source
    return merged


def env_str(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, default).strip()


def env_int(env: dict[str, str], key: str, default: int) -> int:
    return int(env.get(key, str(default)).strip())


def env_float(env: dict[str, str], key: str, default: float) -> float:
    return float(env.get(key, str(default)).strip())


def env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def mask_value(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def load_runtime_config(paths: RuntimePaths, env: dict[str, str]) -> RuntimeConfig:
    research_config = _resolve_optional_path(
        env_str(
            env,
            "CTREND_RESEARCH_CONFIG",
            "examples/backtest/crypto_rv/configs/ctrend.real_public_q1.json",
        ),
        base_dir=paths.root_dir,
    )
    real_public = load_research_config(research_config)

    return RuntimeConfig(
        research_config=research_config,
        research_root=real_public.research_root,
        selected_universe_path=real_public.selected_universe_path,
        catalog_path=real_public.catalog_path,
        baseline_metrics_path=real_public.baseline_metrics_path,
        binance_environment=env_str(env, "CTREND_BINANCE_ENVIRONMENT", "live"),
        trader_id=env_str(env, "CTREND_TRADER_ID", "CTREND-LIVE-001"),
        leg_notional_usd=env_float(env, "CTREND_LEG_NOTIONAL_USD", 75.0),
        short_band=env_str(env, "CTREND_SHORT_BAND", "rank_1_20"),
        short_fallback_bands=tuple(
            band.strip()
            for band in env_str(env, "CTREND_SHORT_FALLBACK_BANDS", "rank_1_30,rank_1_50").split(",")
            if band.strip()
        ),
        min_active_shorts=env_int(env, "CTREND_MIN_ACTIVE_SHORTS", 4),
        history_lookback_days=env_int(env, "CTREND_HISTORY_LOOKBACK_DAYS", 90),
        order_expire_seconds=env_int(env, "CTREND_ORDER_EXPIRE_SECONDS", 900),
        log_level=env_str(env, "CTREND_LOG_LEVEL", "INFO"),
        initial_leverage=env_int(env, "CTREND_INITIAL_LEVERAGE", 2),
        starting_equity_usdt=env_float(env, "CTREND_STARTING_EQUITY_USDT", 200.0),
        max_total_drawdown_usdt=env_float(env, "CTREND_MAX_TOTAL_DRAWDOWN_USDT", 20.0),
        max_daily_loss_usdt=env_float(env, "CTREND_MAX_DAILY_LOSS_USDT", 6.0),
        min_available_balance_usdt=env_float(env, "CTREND_MIN_AVAILABLE_BALANCE_USDT", 50.0),
        kill_switch_check_interval_secs=env_int(env, "CTREND_KILL_SWITCH_CHECK_INTERVAL_SECS", 60),
        wheel_spec=env_str(env, "CTREND_WHEEL_SPEC", "nautilus_trader==1.226.0a20260414"),
        credential_sources={
            "BINANCE_API_KEY": env["CTREND_BINANCE_API_KEY_SOURCE"],
            "BINANCE_API_SECRET": env["CTREND_BINANCE_API_SECRET_SOURCE"],
        },
    )


def validate_runtime_config(  # noqa: C901 - explicit deployment gate checks are clearer inline here
    paths: RuntimePaths,
    config: RuntimeConfig,
    *,
    allowed_pids: set[int] | None = None,
) -> list[str]:
    errors: list[str] = []
    if config.binance_environment not in ENVIRONMENT_CHOICES:
        errors.append(
            f"CTREND_BINANCE_ENVIRONMENT must be one of {ENVIRONMENT_CHOICES}, got {config.binance_environment!r}",
        )
    if config.short_band not in SHORT_BAND_CHOICES:
        errors.append(f"CTREND_SHORT_BAND must be one of {SHORT_BAND_CHOICES}, got {config.short_band!r}")
    invalid_fallback_bands = [
        band_name for band_name in config.short_fallback_bands if band_name not in SHORT_BAND_CHOICES
    ]
    if invalid_fallback_bands:
        errors.append(
            "CTREND_SHORT_FALLBACK_BANDS contains unsupported values: "
            f"{invalid_fallback_bands!r}",
        )
    if config.leg_notional_usd <= 0:
        errors.append("CTREND_LEG_NOTIONAL_USD must be positive")
    if config.min_active_shorts <= 0:
        errors.append("CTREND_MIN_ACTIVE_SHORTS must be positive")
    if config.initial_leverage <= 0:
        errors.append("CTREND_INITIAL_LEVERAGE must be positive")
    if config.history_lookback_days <= 0:
        errors.append("CTREND_HISTORY_LOOKBACK_DAYS must be positive")
    if config.order_expire_seconds <= 0:
        errors.append("CTREND_ORDER_EXPIRE_SECONDS must be positive")
    if config.starting_equity_usdt <= 0:
        errors.append("CTREND_STARTING_EQUITY_USDT must be positive")
    if config.max_total_drawdown_usdt <= 0:
        errors.append("CTREND_MAX_TOTAL_DRAWDOWN_USDT must be positive")
    if config.max_daily_loss_usdt <= 0:
        errors.append("CTREND_MAX_DAILY_LOSS_USDT must be positive")
    if config.min_available_balance_usdt <= 0:
        errors.append("CTREND_MIN_AVAILABLE_BALANCE_USDT must be positive")
    if config.kill_switch_check_interval_secs <= 0:
        errors.append("CTREND_KILL_SWITCH_CHECK_INTERVAL_SECS must be positive")

    for path in (
        config.research_config,
        config.selected_universe_path,
        config.catalog_path,
        config.baseline_metrics_path,
    ):
        if not path.exists():
            errors.append(f"Required path does not exist: {path}")

    if paths.pid_path.exists():
        pid = read_pid(paths.pid_path)
        if pid and is_process_alive(pid) and pid not in (allowed_pids or set()):
            errors.append(f"Runner already active with pid {pid}; stop it before starting a new one")

    return errors


def write_log_line(log_path: Path, message: str) -> None:
    timestamped = f"[{utc_now_iso()}] {message}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")
    print(timestamped, flush=True)


def describe_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, label: str) -> None:
    write_log_line(log_path, label)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(  # noqa: S603 - command list is constructed from validated repo/runtime inputs
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def install_official_wheel(paths: RuntimePaths, config: RuntimeConfig, env: dict[str, str]) -> None:
    if paths.wheel_target_dir.exists():
        shutil.rmtree(paths.wheel_target_dir)
    paths.wheel_cache_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(paths.project_python),
            "--target",
            str(paths.wheel_target_dir),
            "--no-deps",
            "--no-build",
            "--index-url",
            INDEX_URL,
            config.wheel_spec,
        ],
        cwd=paths.wheel_cache_dir,
        env=env,
        log_path=paths.log_path,
        label=f"overlay official nautilus wheel shared objects ({config.wheel_spec})",
    )

    overlay_count = 0
    for src in paths.wheel_target_dir.rglob("*.so"):
        rel = src.relative_to(paths.wheel_target_dir)
        dst = paths.root_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        overlay_count += 1

    write_log_line(paths.log_path, f"overlayed {overlay_count} shared objects from {paths.wheel_target_dir}")


def import_smoke(paths: RuntimePaths, env: dict[str, str]) -> None:
    run_command(
        [
            str(paths.project_python),
            "-c",
            "from nautilus_trader.model.data import BarType; print('BarType', BarType)",
        ],
        cwd=paths.root_dir,
        env=env,
        log_path=paths.log_path,
        label="import smoke",
    )


def build_common_args(paths: RuntimePaths, config: RuntimeConfig) -> list[str]:
    return [
        "--research-config",
        str(config.research_config),
        "--binance-environment",
        config.binance_environment,
        "--short-band",
        config.short_band,
        "--starting-equity-usdt",
        str(config.starting_equity_usdt),
        "--max-total-drawdown-usdt",
        str(config.max_total_drawdown_usdt),
        "--max-daily-loss-usdt",
        str(config.max_daily_loss_usdt),
        "--min-available-balance-usdt",
        str(config.min_available_balance_usdt),
        "--output-json",
        str(paths.status_json_path),
    ]


def build_ops_command(paths: RuntimePaths, config: RuntimeConfig, command: str) -> list[str]:
    argv = [
        str(paths.project_python),
        "examples/live/binance/binance_futures_ctrend_liqrev_ops.py",
        *build_common_args(paths, config),
    ]
    if command == "prepare":
        argv.extend(["--leverage", str(config.initial_leverage)])
    argv.append(command)
    return argv


def build_micro_command(paths: RuntimePaths, config: RuntimeConfig) -> list[str]:
    return [
        str(paths.project_python),
        "examples/live/binance/binance_futures_ctrend_liqrev_micro.py",
        "--research-config",
        str(config.research_config),
        "--binance-environment",
        config.binance_environment,
        "--trader-id",
        config.trader_id,
        "--leg-notional-usd",
        str(config.leg_notional_usd),
        "--short-band",
        config.short_band,
        "--short-fallback-bands",
        ",".join(config.short_fallback_bands),
        "--min-active-shorts",
        str(config.min_active_shorts),
        "--history-lookback-days",
        str(config.history_lookback_days),
        "--order-expire-seconds",
        str(config.order_expire_seconds),
        "--log-level",
        config.log_level,
        "--initial-leverage",
        str(config.initial_leverage),
        "--starting-equity-usdt",
        str(config.starting_equity_usdt),
        "--max-total-drawdown-usdt",
        str(config.max_total_drawdown_usdt),
        "--max-daily-loss-usdt",
        str(config.max_daily_loss_usdt),
        "--min-available-balance-usdt",
        str(config.min_available_balance_usdt),
        "--kill-switch-check-interval-secs",
        str(config.kill_switch_check_interval_secs),
    ]


def summarize_process_status(paths: RuntimePaths) -> dict[str, Any]:
    pid = read_pid(paths.pid_path)
    payload: dict[str, Any] = {
        "pid_file": str(paths.pid_path),
        "runner_pid": pid,
        "runner_alive": bool(pid and is_process_alive(pid)),
        "log_path": str(paths.log_path),
        "nohup_path": str(paths.nohup_path),
        "status_json_path": str(paths.status_json_path),
        "approval_dir": str(paths.approval_dir),
    }
    if paths.status_json_path.exists():
        try:
            status_payload = json.loads(paths.status_json_path.read_text(encoding="utf-8"))
            payload["latest_status"] = {
                "generated_at": status_payload.get("generated_at"),
                "environment": status_payload.get("environment"),
                "dedicated_account_ready": status_payload.get("dedicated_account_ready"),
                "kill_switch_triggered": status_payload.get("kill_switch_triggered"),
            }
        except json.JSONDecodeError:
            payload["latest_status"] = "unreadable"
    return payload


def log_account_threshold_warning(paths: RuntimePaths, config: RuntimeConfig) -> None:
    if not paths.status_json_path.exists():
        return
    try:
        payload = json.loads(paths.status_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    snapshot = payload.get("account_snapshot")
    if not isinstance(snapshot, dict):
        return

    margin_balance = snapshot.get("margin_balance")
    if not isinstance(margin_balance, (int, float)):
        return

    if margin_balance > config.starting_equity_usdt * 1.5:
        write_log_line(
            paths.log_path,
            "warning: account margin balance "
            f"{margin_balance:.2f} exceeds configured starting equity "
            f"{config.starting_equity_usdt:.2f}; review kill-switch thresholds in live.env",
        )


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return int(raw)


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_exit(pid: int, *, timeout_secs: float) -> bool:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return True
        time.sleep(0.2)
    return not is_process_alive(pid)


def command_check(paths: RuntimePaths) -> int:
    env = build_effective_env(paths)
    config = load_runtime_config(paths, env)
    errors = validate_runtime_config(paths, config)
    summary = {
        "env_file": str(paths.env_file),
        "credential_preview": {
            "BINANCE_API_KEY": mask_value(env["BINANCE_API_KEY"]),
            "BINANCE_API_SECRET": mask_value(env["BINANCE_API_SECRET"]),
        },
        "runtime": {
            **asdict(config),
            "credential_sources": config.credential_sources,
        },
        "paths": {
            "log_path": str(paths.log_path),
            "nohup_path": str(paths.nohup_path),
            "pid_path": str(paths.pid_path),
            "status_json_path": str(paths.status_json_path),
        },
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 1 if errors else 0


def command_bootstrap(paths: RuntimePaths) -> int:
    env = build_effective_env(paths)
    config = load_runtime_config(paths, env)
    errors = validate_runtime_config(paths, config, allowed_pids={os.getpid()})
    if errors:
        raise RuntimeError("\n".join(errors))

    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.approval_dir.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("", encoding="utf-8")

    write_log_line(paths.log_path, f"runtime config {json.dumps(asdict(config), sort_keys=True, default=str)}")
    run_command(
        ["uv", "sync", "--frozen", "--no-install-project"],
        cwd=paths.root_dir,
        env=env,
        log_path=paths.log_path,
        label="sync deps (no project install)",
    )
    install_official_wheel(paths, config, env)
    import_smoke(paths, env)
    run_command(
        build_ops_command(paths, config, "status"),
        cwd=paths.root_dir,
        env=env,
        log_path=paths.log_path,
        label=f"preflight status: {describe_command(build_ops_command(paths, config, 'status'))}",
    )
    log_account_threshold_warning(paths, config)
    run_command(
        build_ops_command(paths, config, "prepare"),
        cwd=paths.root_dir,
        env=env,
        log_path=paths.log_path,
        label=f"prepare account: {describe_command(build_ops_command(paths, config, 'prepare'))}",
    )
    write_log_line(paths.log_path, f"starting live runner: {describe_command(build_micro_command(paths, config))}")

    with paths.log_path.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.dup2(handle.fileno(), sys.stdout.fileno())
        os.dup2(handle.fileno(), sys.stderr.fileno())
        os.execvpe(  # noqa: S606 - exec handoff is intentional once bootstrap validation is complete
            str(paths.project_python),
            build_micro_command(paths, config),
            env,
        )
    return 0


def command_start(paths: RuntimePaths) -> int:
    env = build_effective_env(paths)
    config = load_runtime_config(paths, env)
    errors = validate_runtime_config(paths, config)
    if errors:
        raise RuntimeError("\n".join(errors))

    paths.nohup_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.nohup_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(  # noqa: S603 - detached start uses validated local script and env paths
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--env-file",
                str(paths.env_file),
                "bootstrap",
            ],
            cwd=paths.root_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    paths.pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "runner_pid": process.pid,
                "pid_file": str(paths.pid_path),
                "nohup_path": str(paths.nohup_path),
                "log_path": str(paths.log_path),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return 0


def command_stop(paths: RuntimePaths, *, force: bool, timeout_secs: float) -> int:
    pid = read_pid(paths.pid_path)
    if pid is None:
        print(json.dumps({"stopped": False, "reason": "pid file missing"}, indent=2, sort_keys=True))
        return 0

    if not is_process_alive(pid):
        paths.pid_path.unlink(missing_ok=True)
        print(json.dumps({"stopped": False, "reason": "stale pid file", "pid": pid}, indent=2, sort_keys=True))
        return 0

    os.kill(pid, signal.SIGTERM)
    exited = wait_for_exit(pid, timeout_secs=timeout_secs)
    if not exited and force:
        os.kill(pid, signal.SIGKILL)
        exited = wait_for_exit(pid, timeout_secs=2.0)

    if exited:
        paths.pid_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {"pid": pid, "stopped": exited, "force": force, "timeout_secs": timeout_secs},
            indent=2,
            sort_keys=True,
        ),
    )
    return 0 if exited else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage runtime bootstrap, env validation, and process lifecycle for the CTREND Binance micro-live lane.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to the runtime env file. Defaults to examples/live/binance/runtime/live.env.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate env values, runtime files, and research artifacts.")
    subparsers.add_parser("bootstrap", help="Run sync, wheel overlay, preflight, prepare, and exec the live runner in the foreground.")
    subparsers.add_parser("start", help="Spawn bootstrap as a detached background process and persist its PID.")
    status_parser = subparsers.add_parser("process-status", help="Report the local runner process state and latest ops status file.")
    status_parser.set_defaults(command="process-status")
    stop_parser = subparsers.add_parser("stop", help="Stop the detached runner process.")
    stop_parser.add_argument("--force", action="store_true", help="Escalate to SIGKILL if SIGTERM does not stop the process.")
    stop_parser.add_argument(
        "--timeout-secs",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGTERM before failing or escalating.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = build_paths(args.env_file)

    if args.command == "check":
        raise SystemExit(command_check(paths))
    if args.command == "bootstrap":
        raise SystemExit(command_bootstrap(paths))
    if args.command == "start":
        raise SystemExit(command_start(paths))
    if args.command == "stop":
        raise SystemExit(command_stop(paths, force=args.force, timeout_secs=args.timeout_secs))

    print(json.dumps(summarize_process_status(paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
