#!/usr/bin/env python3
"""Resource-gated runner for bounded autoresearch trials on Apple Silicon."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".autoresearch"
RUNS_DIR = STATE_DIR / "runs"
LOCK_PATH = STATE_DIR / "trial.lock"
PROTECTED_FILES = ("prepare.py", "pyproject.toml", "uv.lock")
METRIC_NAMES = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in METRIC_NAMES:
        match = re.search(
            rf"(?m)^{re.escape(name)}:\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
            text,
        )
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def available_memory_mb() -> float | None:
    """Best-effort available-memory estimate from macOS vm_stat."""
    try:
        output = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    page_match = re.search(r"page size of (\d+) bytes", output)
    if not page_match:
        return None
    page_size = int(page_match.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.", output, re.MULTILINE)
        if match:
            pages += int(match.group(1))
    return pages * page_size / (1024 * 1024)


def cache_size_bytes() -> int:
    cache = Path.home() / ".cache" / "autoresearch"
    if not cache.exists():
        return 0
    return sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=15).strip()


def protected_files_clean() -> tuple[bool, list[str]]:
    output = git_output("status", "--porcelain", "--", *PROTECTED_FILES)
    changed = [line for line in output.splitlines() if line]
    return not changed, changed


def source_manifest() -> dict[str, Any]:
    files = ("train.py", "prepare.py", "pyproject.toml", "uv.lock", "trial_gate.py")
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "files": {name: sha256_file(ROOT / name) for name in files},
        "cache_bytes": cache_size_bytes(),
    }


def acquire_lock() -> Any:
    STATE_DIR.mkdir(exist_ok=True)
    handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("another autoresearch trial is already running") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_trial(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    free_mb = available_memory_mb()
    disk_free_mb = shutil.disk_usage(ROOT).free / (1024 * 1024)
    if free_mb is not None and free_mb < args.min_available_memory_mb:
        raise RuntimeError(
            f"preflight failed: only {free_mb:.0f} MB available; "
            f"requires {args.min_available_memory_mb} MB"
        )
    if disk_free_mb < args.min_free_disk_mb:
        raise RuntimeError(
            f"preflight failed: only {disk_free_mb:.0f} MB disk free; "
            f"requires {args.min_free_disk_mb} MB"
        )
    clean, changed = protected_files_clean()
    if not clean:
        raise RuntimeError(f"protected files are modified: {changed}")

    started = dt.datetime.now(dt.timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"-seed{args.seed}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    manifest_path = run_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "description": args.description,
        "status": "running",
        "started_at": started.isoformat(),
        "seed": args.seed,
        "limits": {
            "timeout_seconds": args.timeout_seconds,
            "time_budget_seconds": args.time_budget_seconds,
            "eval_tokens": args.eval_tokens,
            "max_peak_memory_mb": args.max_peak_memory_mb,
            "min_available_memory_mb": args.min_available_memory_mb,
            "min_free_disk_mb": args.min_free_disk_mb,
        },
        "preflight": {
            "available_memory_mb": free_mb,
            "disk_free_mb": disk_free_mb,
        },
        "source": source_manifest(),
        "command": [os.path.relpath(sys.executable, ROOT), "train.py"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    env = os.environ.copy()
    env.update(
        {
            "AUTORESEARCH_SEED": str(args.seed),
            "AUTORESEARCH_TIME_BUDGET": str(args.time_budget_seconds),
            "AUTORESEARCH_EVAL_TOKENS": str(args.eval_tokens),
        }
    )

    start_monotonic = time.monotonic()
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            [sys.executable, "train.py"],
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        interrupted = False
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = 124
        except KeyboardInterrupt:
            interrupted = True
            terminate_process_group(process)
            return_code = 130
        finally:
            if process.poll() is None:
                terminate_process_group(process)

    elapsed = time.monotonic() - start_monotonic
    output = log_path.read_text(errors="replace")
    metrics = parse_metrics(output)
    peak_mb = metrics.get("peak_vram_mb")

    failures: list[str] = []
    if interrupted:
        failures.append("interrupted")
    if timed_out:
        failures.append("timeout")
    if return_code != 0:
        failures.append(f"exit_code_{return_code}")
    if "val_bpb" not in metrics:
        failures.append("missing_val_bpb")
    if peak_mb is None:
        failures.append("missing_peak_memory")
    elif peak_mb > args.max_peak_memory_mb:
        failures.append("peak_memory_limit")

    manifest.update(
        {
            "status": "passed" if not failures else "rejected",
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "return_code": return_code,
            "failures": failures,
            "metrics": metrics,
            "log_sha256": sha256_file(log_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return (0 if not failures else 1), manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-budget-seconds", type=int, default=300)
    parser.add_argument("--eval-tokens", type=int, default=2**18)
    parser.add_argument("--timeout-seconds", type=int, default=480)
    parser.add_argument("--max-peak-memory-mb", type=float, default=12 * 1024)
    parser.add_argument("--min-available-memory-mb", type=float, default=1024)
    parser.add_argument("--min-free-disk-mb", type=float, default=2048)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.time_budget_seconds <= 0 or args.timeout_seconds <= 0 or args.eval_tokens <= 0:
        print("budgets and eval tokens must be positive", file=sys.stderr)
        return 2
    lock_handle = None
    try:
        lock_handle = acquire_lock()
        code, manifest = run_trial(args)
        print(json.dumps(manifest, indent=2))
        return code
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        print(f"trial gate error: {exc}", file=sys.stderr)
        return 2
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
