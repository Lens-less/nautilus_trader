from __future__ import annotations

import csv
import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_utc_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone information: {value!r}")

    return dt.astimezone(UTC)


def format_utc_timestamp(value: str | datetime) -> str:
    return parse_utc_timestamp(value).isoformat().replace("+00:00", "Z")


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]

    raise ValueError(f"Unsupported record payload in {path}")


def normalize_symbol(value: str) -> str:
    return value.strip().upper()


def parse_float(value: Any, field_name: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Missing required numeric field: {field_name}")
    return float(value)


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unable to parse boolean value: {value!r}")


def within_ratio(value: float) -> bool:
    return 0.0 <= value <= 1.0


def pre_window_cutoff(start_ts: str, days: int) -> str:
    return format_utc_timestamp(parse_utc_timestamp(start_ts) - timedelta(days=days))

