#!/usr/bin/env python3
# mypy: disable-error-code=no-redef

from __future__ import annotations

import argparse
from pathlib import Path


try:
    from .common import load_json
    from .ctrend_schemas import load_ctrend_config
    from .ctrend_schemas import save_artifact
except ImportError:  # pragma: no cover - script execution fallback
    from common import load_json
    from ctrend_schemas import load_ctrend_config
    from ctrend_schemas import save_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a CTREND lane summary report.")
    parser.add_argument("--config", required=True, help="Path to the CTREND research config JSON.")
    parser.add_argument("--output", help="Optional override for the report JSON path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_ctrend_config(config_path)
    paths = config.resolved_paths(config_path)
    output_dir = paths["output_dir"]
    output_path = Path(args.output).resolve() if args.output else paths["report_output_path"]

    index_payload = load_json(output_dir / "index.json")
    results = [
        load_json(Path(item["result_path"]))
        for item in index_payload["variants"]
    ]
    ranked = sorted(results, key=lambda item: (item["replace_mainline"], item["net_return"]), reverse=True)
    winner = ranked[0]

    report_payload = {
        "research_name": config.research_name,
        "winner": winner["variant_name"],
        "replace_mainline": winner["replace_mainline"],
        "variants": results,
        "decision_rule": {
            "net_return": "must be greater than baseline",
            "max_drawdown": "must be no worse than baseline",
        },
    }
    save_artifact(output_path, report_payload)

    markdown_lines = [
        f"# {config.research_name}",
        "",
        "## Winner",
        "",
        f"- Variant: `{winner['variant_name']}`",
        f"- Replace mainline: `{winner['replace_mainline']}`",
        f"- Net return: `{winner['net_return']:.4%}`",
        f"- Max drawdown: `{winner['max_drawdown']:.4%}`",
        f"- Baseline net return: `{winner['baseline_net_return']:.4%}`",
        f"- Baseline max drawdown: `{winner['baseline_max_drawdown']:.4%}`",
        "",
        "## Variants",
        "",
    ]
    for item in results:
        markdown_lines.extend(
            [
                f"### {item['variant_name']}",
                "",
                f"- Net return: `{item['net_return']:.4%}`",
                f"- Max drawdown: `{item['max_drawdown']:.4%}`",
                f"- Replace mainline: `{item['replace_mainline']}`",
                "",
            ],
        )

    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"Wrote CTREND report JSON to {output_path}")
    print(f"Wrote CTREND report Markdown to {markdown_path}")


if __name__ == "__main__":
    main()
