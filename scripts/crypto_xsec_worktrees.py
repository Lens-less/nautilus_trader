#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaneSpec:
    key: str
    branch: str
    dir_name: str
    stage: str
    description: str


LANES: tuple[LaneSpec, ...] = (
    LaneSpec(
        key="00-kernel",
        branch="research/crypto-xsec/00-kernel",
        dir_name="nt-xsec-00-kernel",
        stage="wave0",
        description="Shared kernel contract, schemas, and report/cost baselines.",
    ),
    LaneSpec(
        key="01-liq-reversal",
        branch="research/crypto-xsec/01-liq-reversal",
        dir_name="nt-xsec-01-liq-reversal",
        stage="pilot",
        description="Pilot lane for liquidity / overshoot / reversal.",
    ),
    LaneSpec(
        key="02-carry-basis",
        branch="research/crypto-xsec/02-carry-basis",
        dir_name="nt-xsec-02-carry-basis",
        stage="wave1",
        description="Carry / funding / basis lane.",
    ),
    LaneSpec(
        key="03-vol-trend",
        branch="research/crypto-xsec/03-vol-trend",
        dir_name="nt-xsec-03-vol-trend",
        stage="wave1",
        description="Vol-managed cross-sectional trend lane.",
    ),
    LaneSpec(
        key="04-conditional",
        branch="research/crypto-xsec/04-conditional",
        dir_name="nt-xsec-04-conditional",
        stage="wave2",
        description="Conditional interaction / double-sort lane.",
    ),
    LaneSpec(
        key="05-overlays",
        branch="research/crypto-xsec/05-overlays",
        dir_name="nt-xsec-05-overlays",
        stage="wave2",
        description="Optional order-flow / attention / sector overlay lane.",
    ),
    LaneSpec(
        key="06-combo",
        branch="research/crypto-xsec/06-combo",
        dir_name="nt-xsec-06-combo",
        stage="final",
        description="Final combo / allocator / integration lane.",
    ),
)


def run_git(repo_root: Path, *args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=capture,
    )


def resolve_repo_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(result.stdout.strip())


def lane_by_key(key: str) -> LaneSpec:
    for lane in LANES:
        if lane.key == key:
            return lane
    raise KeyError(f"Unknown lane: {key}")


def branch_exists(repo_root: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def parse_worktree_list(repo_root: Path) -> list[dict[str, str]]:
    result = run_git(repo_root, "worktree", "list", "--porcelain")
    blocks = result.stdout.strip().split("\n\n") if result.stdout.strip() else []
    items: list[dict[str, str]] = []
    for block in blocks:
        item: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            item[key] = value
        items.append(item)
    return items


def print_plan() -> None:
    payload = [asdict(lane) for lane in LANES]
    print(json.dumps(payload, indent=2))


def print_status(repo_root: Path, worktrees_root: Path) -> None:
    existing_worktrees = {Path(item["worktree"]).resolve(): item for item in parse_worktree_list(repo_root)}
    payload = []
    for lane in LANES:
        path = (worktrees_root / lane.dir_name).resolve()
        payload.append(
            {
                "key": lane.key,
                "branch": lane.branch,
                "stage": lane.stage,
                "path": str(path),
                "branch_exists": branch_exists(repo_root, lane.branch),
                "worktree_exists": path in existing_worktrees,
            },
        )
    print(json.dumps(payload, indent=2))


def ensure_branch(repo_root: Path, branch: str, start_point: str) -> None:
    if branch_exists(repo_root, branch):
        return
    run_git(repo_root, "branch", branch, start_point)


def add_lane_worktree(repo_root: Path, worktrees_root: Path, lane: LaneSpec, start_point: str) -> None:
    worktrees_root.mkdir(parents=True, exist_ok=True)
    target = (worktrees_root / lane.dir_name).resolve()
    if target.exists():
        raise SystemExit(f"Worktree path already exists: {target}")

    if branch_exists(repo_root, lane.branch):
        run_git(repo_root, "worktree", "add", str(target), lane.branch, capture=False)
    else:
        run_git(repo_root, "worktree", "add", "-b", lane.branch, str(target), start_point, capture=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the approved crypto cross-sectional git worktree layout.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path inside the nautilus_trader git repo. Defaults to the current directory.",
    )
    parser.add_argument(
        "--worktrees-root",
        default="../worktrees",
        help="Directory where lane worktrees should be created.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Print the approved lane layout as JSON.")
    subparsers.add_parser("status", help="Print branch/worktree presence for all lanes.")

    add_parser = subparsers.add_parser("add", help="Create one lane worktree.")
    add_parser.add_argument("lane", choices=[lane.key for lane in LANES], help="Lane key to create.")
    add_parser.add_argument(
        "--start-point",
        default="research/crypto-xsec-integration",
        help="Start point for the new lane branch when it does not already exist.",
    )

    init_parser = subparsers.add_parser(
        "init-branches",
        help="Ensure the integration branch exists from the given seed start point.",
    )
    init_parser.add_argument(
        "--seed-start-point",
        default="HEAD",
        help="Start point used only when creating research/crypto-xsec-integration for the first time.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(args.repo_root).resolve())
    worktrees_root = Path(args.worktrees_root).resolve()

    if args.command == "plan":
        print_plan()
        return 0

    if args.command == "status":
        print_status(repo_root, worktrees_root)
        return 0

    if args.command == "init-branches":
        ensure_branch(repo_root, "research/crypto-xsec-integration", args.seed_start_point)
        print("Ensured branch research/crypto-xsec-integration")
        return 0

    if args.command == "add":
        lane = lane_by_key(args.lane)
        add_lane_worktree(repo_root, worktrees_root, lane, args.start_point)
        print(f"Created {lane.key} at {(worktrees_root / lane.dir_name).resolve()}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
