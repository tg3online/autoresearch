#!/usr/bin/env python3
"""Aggregate comparable passed trial manifests into baseline statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ROOT / ".autoresearch" / "runs"


def load_comparable(runs_dir: Path, time_budget: int, eval_tokens: int) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*/manifest.json")):
        data = json.loads(path.read_text())
        limits = data.get("limits", {})
        if (
            data.get("status") == "passed"
            and limits.get("time_budget_seconds") == time_budget
            and limits.get("eval_tokens") == eval_tokens
            and "val_bpb" in data.get("metrics", {})
        ):
            data["_manifest_path"] = str(path.relative_to(ROOT))
            manifests.append(data)
    return manifests


def summarize(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    if not manifests:
        raise ValueError("no comparable passed manifests")
    source_keys = {
        (
            item["source"]["files"]["train.py"],
            item["source"]["files"]["prepare.py"],
            item["source"]["files"]["uv.lock"],
        )
        for item in manifests
    }
    if len(source_keys) != 1:
        raise ValueError("manifests use different train/evaluation/dependency sources")

    values = [float(item["metrics"]["val_bpb"]) for item in manifests]
    peaks = [float(item["metrics"]["peak_vram_mb"]) for item in manifests]
    return {
        "schema_version": 1,
        "status": "established" if len(values) >= 3 else "provisional",
        "sample_count": len(values),
        "seeds": [item["seed"] for item in manifests],
        "val_bpb": {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "population_stdev": statistics.pstdev(values),
        },
        "peak_vram_mb": {
            "median": statistics.median(peaks),
            "max": max(peaks),
        },
        "source_hashes": {
            "train.py": manifests[0]["source"]["files"]["train.py"],
            "prepare.py": manifests[0]["source"]["files"]["prepare.py"],
            "uv.lock": manifests[0]["source"]["files"]["uv.lock"],
        },
        "manifests": [item["_manifest_path"] for item in manifests],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--time-budget-seconds", type=int, default=300)
    parser.add_argument("--eval-tokens", type=int, default=2**18)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = summarize(load_comparable(args.runs_dir, args.time_budget_seconds, args.eval_tokens))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"baseline report error: {exc}")
        return 1
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
