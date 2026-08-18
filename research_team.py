#!/usr/bin/env python3
"""Two-model autoresearch coordinator.

A local Qwen server proposes exactly one allowlisted `train.py` hyperparameter
change; Claude reviews it with tools disabled and returns approve/reject only.
Nothing is written unless `--apply` is passed, and no trial runs unless
`--run-trial` is passed as well. The coordinator never merges, promotes,
installs dependencies, or edits protected harness files.

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".autoresearch"
LOCK_PATH = STATE_DIR / "coordinator.lock"
TRIAL_LOCK_PATH = STATE_DIR / "trial.lock"
REPORTS_DIR = STATE_DIR / "coordinator"
DEFAULT_WORKTREE_ROOT = ROOT.parent / f"{ROOT.name}-trials"
BASELINE_PATH = ROOT / "evidence" / "m4-mini-provisional-baseline.json"

TARGET_FILE = "train.py"
PROTECTED_FILES = ("prepare.py", "pyproject.toml", "uv.lock", "trial_gate.py")
DEFAULT_QWEN_URL = "http://127.0.0.1:8088/v1/chat/completions"
DEFAULT_QWEN_MODEL = "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
DEFAULT_CLAUDE_NATIVE = Path(
    "/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/node_modules/"
    "@anthropic-ai/claude-code-darwin-arm64/claude"
)
DEFAULT_CLAUDE_BIN = str(DEFAULT_CLAUDE_NATIVE) if DEFAULT_CLAUDE_NATIVE.exists() else "/opt/homebrew/bin/claude"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_RESPONSE_BYTES = 256 * 1024

# Mirrors of harness constants, refreshed from source when the files parse.
SEQ_LEN = 512  # prepare.MAX_SEQ_LEN
VOCAB_SIZE = 4096  # prepare.VOCAB_SIZE
EVAL_BATCH_SIZE = 16  # train.FINAL_EVAL_BATCH_SIZE

# trial_gate.py rejects above 12288 MB; gate proposals well under that.
MAX_ESTIMATED_PEAK_MB = 8192.0
MAX_GRAD_ACCUM_STEPS = 64

PROPOSAL_KEYS = ("parameter", "value", "rationale")


class CoordinatorError(RuntimeError):
    """Operational failure: bad environment, git state, or model transport."""


class ProposalError(ValueError):
    """A model returned something outside the allowlist, schema, or bounds."""


@dataclass(frozen=True)
class Spec:
    kind: type
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None
    choices: tuple[Any, ...] | None = None
    note: str = ""

    def describe(self, name: str, current: Any) -> str:
        if self.choices is not None:
            domain = "{" + ",".join(str(choice) for choice in self.choices) + "}"
        elif self.pattern is not None:
            domain = f"str matching {self.pattern}"
        else:
            domain = f"{self.kind.__name__} [{self.minimum},{self.maximum}]"
        return f"{name}: {domain} now={current!r}"


# The closed allowlist. Nothing outside these fourteen names may ever be edited.
PARAM_SPECS: dict[str, Spec] = {
    "ASPECT_RATIO": Spec(int, minimum=16, maximum=256),
    "HEAD_DIM": Spec(int, choices=(32, 64, 128)),
    "WINDOW_PATTERN": Spec(str, pattern=r"^[SL]{1,8}$"),
    "TOTAL_BATCH_SIZE": Spec(int, minimum=2048, maximum=65536),
    "EMBEDDING_LR": Spec(float, minimum=1e-4, maximum=3.0),
    "UNEMBEDDING_LR": Spec(float, minimum=1e-5, maximum=0.1),
    "MATRIX_LR": Spec(float, minimum=1e-4, maximum=0.5),
    "SCALAR_LR": Spec(float, minimum=1e-3, maximum=2.0),
    "WEIGHT_DECAY": Spec(float, minimum=0.0, maximum=1.0),
    "WARMUP_RATIO": Spec(float, minimum=0.0, maximum=0.5),
    "WARMDOWN_RATIO": Spec(float, minimum=0.05, maximum=1.0),
    "FINAL_LR_FRAC": Spec(float, minimum=0.0, maximum=1.0),
    "DEPTH": Spec(int, minimum=2, maximum=12),
    "DEVICE_BATCH_SIZE": Spec(int, minimum=1, maximum=16),
}
ALLOWLIST = tuple(PARAM_SPECS)

CLAUDE_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "rationale"],
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "reject"]},
        "rationale": {"type": "string", "maxLength": 400},
        "adjusted_parameter": {"type": "string", "enum": list(ALLOWLIST)},
        "adjusted_value": {"type": ["number", "string"]},
    },
}

QWEN_SYSTEM_PROMPT = (
    "You propose one bounded hyperparameter experiment. Reply with a single JSON "
    'object and nothing else: {"parameter": <name>, "value": <number or string>, '
    '"rationale": <one sentence>}. No other keys, no prose, no code fences. '
    "Integer parameters need integer values; float parameters need numbers; "
    "WINDOW_PATTERN needs a string of S and L characters."
)

CLAUDE_SYSTEM_PROMPT = (
    "You are a strict experiment reviewer. You have no tools and must not attempt "
    "to read files, run commands, or edit anything. Judge only the proposal text "
    "you are given and answer with the required JSON object. Reject anything that "
    "risks harness invariants, memory limits, or an uninterpretable result."
)


# ---------------------------------------------------------------------------
# Source parsing and deterministic patching
# ---------------------------------------------------------------------------


def _literal(node: ast.AST) -> Any:
    """Evaluate the small literal grammar used for harness constants."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _literal(node.operand)
        if not isinstance(operand, (int, float)) or isinstance(operand, bool):
            raise ProposalError("unsupported unary constant expression")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.Mult, ast.Add)):
        left, right = _literal(node.left), _literal(node.right)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ProposalError("unsupported binary constant expression")
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left + right
    if isinstance(node, ast.Tuple):
        return tuple(_literal(element) for element in node.elts)
    raise ProposalError("unsupported constant expression")


def _assignments(source: str) -> dict[str, ast.Assign]:
    """Module-level single-name assignments, keyed by name. Duplicates are dropped."""
    module = ast.parse(source)
    found: dict[str, ast.Assign] = {}
    duplicated: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in found:
            duplicated.add(target.id)
        found[target.id] = node
    for name in duplicated:
        found.pop(name, None)
    return found


def module_constants(source: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Read module-level constants without importing the module."""
    assignments = _assignments(source)
    values: dict[str, Any] = {}
    for name in names:
        node = assignments.get(name)
        if node is None:
            continue
        try:
            values[name] = _literal(node.value)
        except ProposalError:
            continue
    return values


def render_value(value: Any) -> str:
    """Render a validated value back into deterministic Python source."""
    if isinstance(value, bool):
        raise ProposalError("booleans are not valid hyperparameter values")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if not any(char in text for char in ".eE"):
            text += ".0"
        return text
    if isinstance(value, str):
        return json.dumps(value)
    raise ProposalError(f"cannot render value of type {type(value).__name__}")


def replace_constant(source: str, name: str, value: Any) -> str:
    """Replace exactly one module-level constant assignment, preserving comments."""
    if name not in PARAM_SPECS:
        raise ProposalError(f"{name} is not an allowlisted parameter")
    assignments = _assignments(source)
    node = assignments.get(name)
    if node is None:
        raise ProposalError(f"no unique module-level assignment for {name}")
    if node.lineno != node.end_lineno:
        raise ProposalError(f"{name} assignment spans multiple lines; refusing to patch")

    lines = source.splitlines(keepends=True)
    index = node.lineno - 1
    original = lines[index]
    newline = original[len(original.rstrip("\r\n")) :]
    comment = ""
    match = re.search(r"(\s+#.*)$", original.rstrip("\r\n"))
    if match:
        comment = match.group(1)
    lines[index] = f"{name} = {render_value(value)}{comment}{newline}"
    patched = "".join(lines)

    changed = [
        position
        for position, (before, after) in enumerate(zip(source.splitlines(), patched.splitlines()))
        if before != after
    ]
    if changed != [index]:
        raise ProposalError(f"patch touched lines {changed}, expected only line {node.lineno}")
    return patched


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_value(name: str, value: Any) -> Any:
    """Type- and bounds-check one allowlisted parameter value. Exact types only."""
    if name not in PARAM_SPECS:
        raise ProposalError(f"{name!r} is not in the allowlist: {', '.join(ALLOWLIST)}")
    spec = PARAM_SPECS[name]
    if isinstance(value, bool):
        raise ProposalError(f"{name}: booleans are not accepted")
    if spec.kind is int and not isinstance(value, int):
        raise ProposalError(f"{name}: expected int, got {type(value).__name__} ({value!r})")
    if spec.kind is float and not isinstance(value, (int, float)):
        raise ProposalError(f"{name}: expected number, got {type(value).__name__} ({value!r})")
    if spec.kind is str and not isinstance(value, str):
        raise ProposalError(f"{name}: expected str, got {type(value).__name__} ({value!r})")

    if spec.kind is float:
        value = float(value)
        if not math.isfinite(value):
            raise ProposalError(f"{name}: value must be finite")
    if spec.choices is not None and value not in spec.choices:
        raise ProposalError(f"{name}: {value!r} not in {list(spec.choices)}")
    if spec.pattern is not None and not re.fullmatch(spec.pattern, value):
        raise ProposalError(f"{name}: {value!r} does not match {spec.pattern}")
    if spec.minimum is not None and value < spec.minimum:
        raise ProposalError(f"{name}: {value!r} below minimum {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise ProposalError(f"{name}: {value!r} above maximum {spec.maximum}")
    return value


def model_dim_for(depth: int, aspect_ratio: int, head_dim: int) -> int:
    """Mirror of train.py's model_dim derivation."""
    return ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim


def parameter_count(depth: int, model_dim: int, vocab_size: int = VOCAB_SIZE) -> int:
    """Mirror of the train.py parameter layout (checked against the 7.3M baseline)."""
    value_embed_layers = (depth + 1) // 2
    embeddings = vocab_size * model_dim * (2 + value_embed_layers)
    blocks = depth * 12 * model_dim * model_dim
    return embeddings + blocks


def estimate_peak_memory_mb(constants: dict[str, Any]) -> float:
    """Coarse peak-memory estimate, calibrated against the 659.8 MB baseline run.

    Deliberately approximate: it exists to reject obviously oversized proposals
    before a trial burns wall-clock, not to replace trial_gate.py's hard ceiling.
    """
    depth = constants["DEPTH"]
    model_dim = model_dim_for(depth, constants["ASPECT_RATIO"], constants["HEAD_DIM"])
    device_batch = constants["DEVICE_BATCH_SIZE"]
    params = parameter_count(depth, model_dim)
    weights_mb = params * 10 / 1e6  # bf16 weights + fp32 Adam m/v + step temporaries
    logits_mb = max(device_batch, EVAL_BATCH_SIZE) * SEQ_LEN * VOCAB_SIZE * 4 * 4 / 1e6
    hidden_mb = device_batch * SEQ_LEN * model_dim * depth * 2 * 6 / 1e6
    return weights_mb + logits_mb + hidden_mb


def validate_change(current: dict[str, Any], name: str, value: Any) -> dict[str, Any]:
    """Validate one change against every relationship train.py relies on."""
    value = validate_value(name, value)
    missing = [key for key in ALLOWLIST if key not in current]
    if missing:
        raise CoordinatorError(f"could not read current values for: {', '.join(missing)}")
    if current[name] == value and type(current[name]) is type(value):
        raise ProposalError(f"{name} is already {value!r}; a no-op is not an experiment")

    effective = dict(current)
    effective[name] = value

    tokens_per_fwdbwd = effective["DEVICE_BATCH_SIZE"] * SEQ_LEN
    if effective["TOTAL_BATCH_SIZE"] % tokens_per_fwdbwd:
        raise ProposalError(
            f"TOTAL_BATCH_SIZE {effective['TOTAL_BATCH_SIZE']} is not divisible by "
            f"DEVICE_BATCH_SIZE*{SEQ_LEN} = {tokens_per_fwdbwd}"
        )
    grad_accum = effective["TOTAL_BATCH_SIZE"] // tokens_per_fwdbwd
    if grad_accum > MAX_GRAD_ACCUM_STEPS:
        raise ProposalError(
            f"gradient accumulation would be {grad_accum} steps, above {MAX_GRAD_ACCUM_STEPS}"
        )

    if effective["WARMUP_RATIO"] + effective["WARMDOWN_RATIO"] > 1.0:
        raise ProposalError("WARMUP_RATIO + WARMDOWN_RATIO must not exceed 1.0")

    head_dim = effective["HEAD_DIM"]
    if head_dim % 2:
        raise ProposalError("HEAD_DIM must be even for traditional RoPE")
    model_dim = model_dim_for(effective["DEPTH"], effective["ASPECT_RATIO"], head_dim)
    if model_dim % head_dim or model_dim // head_dim < 1:
        raise ProposalError(f"derived model_dim {model_dim} is not a positive multiple of {head_dim}")
    if model_dim < 32:
        raise ProposalError(f"derived model_dim {model_dim} is below the 32-channel value-embed gate")

    estimated_mb = estimate_peak_memory_mb(effective)
    if estimated_mb > MAX_ESTIMATED_PEAK_MB:
        raise ProposalError(
            f"estimated peak memory {estimated_mb:.0f} MB exceeds the "
            f"{MAX_ESTIMATED_PEAK_MB:.0f} MB coordinator ceiling"
        )

    effective["_derived"] = {
        "model_dim": model_dim,
        "n_head": model_dim // head_dim,
        "grad_accum_steps": grad_accum,
        "params_m": round(parameter_count(effective["DEPTH"], model_dim) / 1e6, 2),
        "estimated_peak_mb": round(estimated_mb, 1),
    }
    return effective


def validate_proposal(payload: Any, current: dict[str, Any]) -> dict[str, Any]:
    """Enforce the closed proposal schema, then the value and relationship rules."""
    if not isinstance(payload, dict):
        raise ProposalError(f"proposal must be a JSON object, got {type(payload).__name__}")
    keys = set(payload)
    unknown = keys - set(PROPOSAL_KEYS)
    if unknown:
        raise ProposalError(f"unknown proposal keys: {', '.join(sorted(unknown))}")
    missing = set(PROPOSAL_KEYS) - keys
    if missing:
        raise ProposalError(f"missing proposal keys: {', '.join(sorted(missing))}")
    if not isinstance(payload["parameter"], str):
        raise ProposalError("proposal parameter must be a string")
    if not isinstance(payload["rationale"], str):
        raise ProposalError("proposal rationale must be a string")

    name = payload["parameter"]
    effective = validate_change(current, name, payload["value"])
    return {
        "parameter": name,
        "value": effective[name],
        "current": current[name],
        "rationale": payload["rationale"].strip()[:400],
        "derived": effective["_derived"],
    }


# ---------------------------------------------------------------------------
# JSON extraction shared by both model transports
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> Any:
    """Pull the first balanced JSON object out of model output."""
    if not isinstance(text, str) or not text.strip():
        raise ProposalError("model returned empty content")
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(stripped[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ProposalError(f"model returned malformed JSON: {exc}") from exc
    raise ProposalError("model returned no JSON object")


# ---------------------------------------------------------------------------
# Stage 1: local Qwen proposal
# ---------------------------------------------------------------------------


def require_loopback(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or (parsed.hostname or "") not in LOOPBACK_HOSTS:
        raise CoordinatorError(f"qwen url must be http on loopback, got {url!r}")


def _post_json(
    url: str, payload: dict[str, Any], timeout: float, api_key: str | None = None
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback only
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise CoordinatorError("qwen response exceeded the response size limit")
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise CoordinatorError(f"qwen returned non-JSON transport payload: {exc}") from exc


def build_qwen_prompt(current: dict[str, Any], goal: str, baseline: dict[str, Any] | None) -> str:
    bounds = "\n".join(PARAM_SPECS[name].describe(name, current.get(name)) for name in ALLOWLIST)
    lines = [
        f"Goal: {goal}",
        f"Harness: MLX pretraining on a 16 GB M4 mini, seq_len={SEQ_LEN}, vocab={VOCAB_SIZE}, "
        "fixed time budget, metric val_bpb (minimize).",
    ]
    if baseline:
        lines.append(
            f"Baseline val_bpb median={baseline.get('median')} spread={baseline.get('spread')} "
            f"over {baseline.get('samples')} run(s)."
        )
    lines += [
        "Change exactly ONE of these constants. Hard rules: "
        f"TOTAL_BATCH_SIZE must stay divisible by DEVICE_BATCH_SIZE*{SEQ_LEN}; "
        "WARMUP_RATIO+WARMDOWN_RATIO<=1.0; the value must differ from the current one.",
        "",
        bounds,
        "",
        'Answer with only: {"parameter": ..., "value": ..., "rationale": ...}',
    ]
    return "\n".join(lines)


def call_qwen(
    url: str,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
    api_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Ask the local server for a proposal. Returns (content, raw transport payload).

    `api_key` comes from an environment variable, never a command-line argument,
    so it does not land in the process table or the JSON report.
    """
    require_loopback(url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    try:
        data = _post_json(url, payload, timeout, api_key)
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 404, 422):
            raise CoordinatorError(f"qwen http error {exc.code}: {exc.reason}") from exc
        # Older OpenAI-compatible servers reject response_format; retry once without it.
        payload.pop("response_format")
        try:
            data = _post_json(url, payload, timeout, api_key)
        except urllib.error.HTTPError as retry_exc:
            raise CoordinatorError(f"qwen http error {retry_exc.code}: {retry_exc.reason}") from retry_exc
        except (urllib.error.URLError, OSError) as retry_exc:
            raise CoordinatorError(f"qwen unreachable at {url}: {retry_exc}") from retry_exc
    except (urllib.error.URLError, OSError) as exc:
        raise CoordinatorError(f"qwen unreachable at {url}: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CoordinatorError(f"qwen response missing choices[0].message.content: {exc}") from exc
    if not isinstance(content, str):
        raise CoordinatorError("qwen returned a non-string message content")
    return content, data


# ---------------------------------------------------------------------------
# Stage 2: Claude review (no tools, no shell, no repo access)
# ---------------------------------------------------------------------------


def build_claude_command(binary: str, model: str, schema: dict[str, Any]) -> list[str]:
    """Review invocation with the entire tool surface removed."""
    return [
        binary,
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--tools",
        "",  # documented: empty string disables every built-in tool
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--no-session-persistence",
        "--system-prompt",
        CLAUDE_SYSTEM_PROMPT,
    ]


def build_claude_prompt(proposal: dict[str, Any], current: dict[str, Any], goal: str) -> str:
    name = proposal["parameter"]
    spec = PARAM_SPECS[name]
    derived = proposal["derived"]
    return "\n".join(
        [
            f"Goal: {goal}",
            f"Proposal: set {name} from {current[name]!r} to {proposal['value']!r}.",
            f"Proposer rationale: {proposal['rationale']}",
            f"Bounds: {spec.describe(name, current[name])}",
            f"Derived after change: model_dim={derived['model_dim']} n_head={derived['n_head']} "
            f"grad_accum={derived['grad_accum_steps']} params={derived['params_m']}M "
            f"estimated_peak={derived['estimated_peak_mb']}MB (hard ceiling 12288MB).",
            "The coordinator already checked the allowlist, types, bounds, divisibility, and "
            "memory estimate. Judge only whether this is a sound, interpretable single-variable "
            "experiment on a fixed time budget.",
            "Approve or reject. You may optionally return adjusted_parameter/adjusted_value to "
            "substitute a different single allowlisted change; it will be re-validated and "
            "rejected if it fails any check.",
        ]
    )


def _extract_claude_payload(stdout: str) -> Any:
    """Unwrap the CLI result envelope and return the structured review object."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return extract_json_object(stdout)
    if isinstance(envelope, list):
        results = [item for item in envelope if isinstance(item, dict) and item.get("type") == "result"]
        if not results:
            raise CoordinatorError("claude stream contained no result message")
        envelope = results[-1]
    if not isinstance(envelope, dict):
        raise CoordinatorError("claude returned an unexpected envelope type")
    if envelope.get("is_error"):
        raise CoordinatorError(f"claude reported an error: {str(envelope.get('result'))[:200]}")
    if "decision" in envelope:
        return envelope
    for key in ("structured_output", "structuredOutput", "structured_result", "result", "response"):
        if key not in envelope:
            continue
        value = envelope[key]
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return extract_json_object(value)
    raise CoordinatorError("claude envelope contained no structured review")


def validate_review(payload: Any, proposal: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Enforce the closed review schema and re-validate any adjusted value."""
    if not isinstance(payload, dict):
        raise ProposalError("review must be a JSON object")
    allowed = set(CLAUDE_REVIEW_SCHEMA["properties"])
    unknown = set(payload) - allowed
    if unknown:
        raise ProposalError(f"unknown review keys: {', '.join(sorted(unknown))}")
    decision = payload.get("decision")
    if decision not in ("approve", "reject"):
        raise ProposalError(f"review decision must be approve or reject, got {decision!r}")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ProposalError("review rationale must be a non-empty string")

    review: dict[str, Any] = {
        "decision": decision,
        "rationale": rationale.strip()[:400],
        "adjusted": False,
        "parameter": proposal["parameter"],
        "value": proposal["value"],
    }
    if "adjusted_value" not in payload and "adjusted_parameter" not in payload:
        return review
    if decision == "reject":
        raise ProposalError("a rejecting review must not carry an adjusted value")
    if "adjusted_value" not in payload:
        raise ProposalError("adjusted_parameter given without adjusted_value")

    name = payload.get("adjusted_parameter", proposal["parameter"])
    if not isinstance(name, str):
        raise ProposalError("adjusted_parameter must be a string")
    value = payload["adjusted_value"]
    if PARAM_SPECS.get(name) and PARAM_SPECS[name].kind is int and isinstance(value, float):
        # JSON Schema "number" can widen an integer; accept only exact integers.
        if not value.is_integer():
            raise ProposalError(f"{name}: expected int, got {value!r}")
        value = int(value)
    effective = validate_change(current, name, value)
    review.update(
        {
            "adjusted": True,
            "parameter": name,
            "value": effective[name],
            "derived": effective["_derived"],
        }
    )
    return review


def call_claude(
    binary: str, model: str, prompt: str, timeout: float
) -> tuple[dict[str, Any], str]:
    """Run the review in a scratch cwd so no repo path is even visible to it."""
    if not Path(binary).exists():
        raise CoordinatorError(f"claude binary not found at {binary}")
    command = build_claude_command(binary, model, CLAUDE_REVIEW_SCHEMA)
    with tempfile.TemporaryDirectory(prefix="research-team-review-") as scratch:
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=scratch,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CoordinatorError(f"claude review timed out after {timeout}s") from exc
        except OSError as exc:
            raise CoordinatorError(f"could not run claude: {exc}") from exc
    if completed.returncode != 0:
        raise CoordinatorError(
            f"claude exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )
    return _extract_claude_payload(completed.stdout), completed.stdout


# ---------------------------------------------------------------------------
# Git worktree helpers (deterministic, pure where possible)
# ---------------------------------------------------------------------------


def run_id_for(timestamp: dt.datetime, parameter: str) -> str:
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{parameter.lower()}"


def branch_name_for(run_id: str) -> str:
    return f"autoresearch/{run_id}"


def worktree_path_for(worktree_root: Path, run_id: str) -> Path:
    return worktree_root / run_id


def git(cwd: Path, *args: str, timeout: float = 30.0) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CoordinatorError(f"git {' '.join(args)} failed: {exc}") from exc
    if completed.returncode != 0:
        raise CoordinatorError(f"git {' '.join(args)} failed: {completed.stderr.strip()[:300]}")
    return completed.stdout.strip()


def require_clean_worktree(root: Path) -> None:
    status = git(root, "status", "--porcelain")
    if status:
        raise CoordinatorError(
            "repository is not clean; commit or stash first:\n" + status
        )


def verify_only_target_changed(porcelain: str, target: str = TARGET_FILE) -> None:
    """Reject anything other than a single modification of the target file."""
    entries = [line for line in porcelain.splitlines() if line.strip()]
    if len(entries) != 1:
        raise CoordinatorError(f"expected exactly one changed file, got {entries!r}")
    status, _, path = entries[0][:2], entries[0][2], entries[0][3:]
    if status != " M":
        raise CoordinatorError(f"expected an unstaged modification, got status {status!r}")
    path = path.strip().strip('"')
    if path != target:
        raise CoordinatorError(f"expected only {target} to change, got {path!r}")
    if path in PROTECTED_FILES:
        raise CoordinatorError(f"refusing to touch protected file {path}")


def verify_single_line_diff(numstat: str, target: str = TARGET_FILE) -> None:
    entries = [line for line in numstat.splitlines() if line.strip()]
    if len(entries) != 1:
        raise CoordinatorError(f"expected one file in diff, got {entries!r}")
    added, removed, path = entries[0].split("\t", 2)
    if (added, removed, path) != ("1", "1", target):
        raise CoordinatorError(f"expected a one-line replacement in {target}, got {entries[0]!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def venv_python(root: Path = ROOT) -> str:
    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def acquire_lock(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise CoordinatorError(f"another process holds {path.name}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def release_lock(handle: Any) -> None:
    if handle is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def load_context(root: Path = ROOT) -> dict[str, Any]:
    """Read current constants from source; never import the training modules."""
    global SEQ_LEN, VOCAB_SIZE, EVAL_BATCH_SIZE
    train_source = (root / TARGET_FILE).read_text()
    current = module_constants(train_source, ALLOWLIST)
    missing = [name for name in ALLOWLIST if name not in current]
    if missing:
        raise CoordinatorError(f"{TARGET_FILE} is missing constants: {', '.join(missing)}")
    extras = module_constants(train_source, ("FINAL_EVAL_BATCH_SIZE",))
    if isinstance(extras.get("FINAL_EVAL_BATCH_SIZE"), int):
        EVAL_BATCH_SIZE = extras["FINAL_EVAL_BATCH_SIZE"]
    prepare_path = root / "prepare.py"
    if prepare_path.exists():
        harness = module_constants(prepare_path.read_text(), ("MAX_SEQ_LEN", "VOCAB_SIZE"))
        if isinstance(harness.get("MAX_SEQ_LEN"), int):
            SEQ_LEN = harness["MAX_SEQ_LEN"]
        if isinstance(harness.get("VOCAB_SIZE"), int):
            VOCAB_SIZE = harness["VOCAB_SIZE"]
    return {"current": current, "train_source": train_source}


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    val = data.get("val_bpb")
    if not isinstance(val, dict):
        return None
    return {
        "median": val.get("median"),
        "spread": round(float(val.get("max", 0)) - float(val.get("min", 0)), 6),
        "samples": data.get("sample_count"),
        "status": data.get("status"),
    }


def run_review_stage(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    """Both model stages. Touches no files and creates no worktrees."""
    current = context["current"]
    baseline = load_baseline()
    qwen_prompt = build_qwen_prompt(current, args.goal, baseline)
    content, _ = call_qwen(
        args.qwen_url,
        args.qwen_model,
        qwen_prompt,
        args.qwen_timeout,
        args.qwen_max_tokens,
        os.environ.get(args.qwen_api_key_env) or None,
    )
    proposal = validate_proposal(extract_json_object(content), current)

    claude_prompt = build_claude_prompt(proposal, current, args.goal)
    payload, _ = call_claude(args.claude_bin, args.claude_model, claude_prompt, args.claude_timeout)
    review = validate_review(payload, proposal, current)

    return {
        "baseline": baseline,
        "proposal": proposal,
        "review": review,
        "change": {
            "parameter": review["parameter"],
            "from": current[review["parameter"]],
            "to": review["value"],
        },
    }


def apply_change(args: argparse.Namespace, outcome: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Create an isolated worktree, patch one constant, and verify the result."""
    require_clean_worktree(ROOT)
    base_commit = git(ROOT, "rev-parse", "HEAD")
    branch = branch_name_for(run_id)
    worktree = worktree_path_for(Path(args.worktree_root).resolve(), run_id)
    if worktree.exists():
        raise CoordinatorError(f"worktree path already exists: {worktree}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    git(ROOT, "worktree", "add", "-b", branch, str(worktree), base_commit, timeout=120.0)

    target = worktree / TARGET_FILE
    patched = replace_constant(target.read_text(), outcome["change"]["parameter"], outcome["change"]["to"])
    target.write_text(patched)

    verify_only_target_changed(git(worktree, "status", "--porcelain"))
    verify_single_line_diff(git(worktree, "diff", "--numstat"))

    interpreter = venv_python()
    try:
        compiled = subprocess.run(
            [interpreter, "-m", "py_compile", TARGET_FILE],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=120.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CoordinatorError(f"py_compile failed to run: {exc}") from exc
    if compiled.returncode != 0:
        raise CoordinatorError(f"py_compile rejected the patch: {compiled.stderr.strip()[:300]}")

    return {
        "branch": branch,
        "path": str(worktree),
        "base_commit": base_commit,
        "interpreter": interpreter,
        "train_py_sha256": sha256_file(target),
        "py_compile": "ok",
        "changed_files": [TARGET_FILE],
    }


MAX_EVIDENCE_BYTES = 8 * 1024 * 1024


def copy_evidence(worktree_run_dir: Path, destination: Path) -> dict[str, str]:
    """Copy the worktree's trial evidence into the local report directory.

    The worktree is disposable, so the manifest and log are pulled back here
    before anything else can remove it. Copies are read-only; hashes are taken
    from the copy so the archived bytes are what got verified.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in ("manifest.json", "run.log"):
        source = worktree_run_dir / name
        if not source.is_file() or source.is_symlink():
            continue
        if source.stat().st_size > MAX_EVIDENCE_BYTES:
            continue
        target = destination / name
        shutil.copyfile(source, target)
        target.chmod(0o444)
        copied[name] = sha256_file(target)
    return copied


def run_trial(
    args: argparse.Namespace, worktree: Path, description: str, run_id: str
) -> dict[str, Any]:
    """Run the worktree's own trial_gate.py under the parent venv, bounded."""
    command = [
        venv_python(),
        "trial_gate.py",
        "--description",
        description,
        "--seed",
        str(args.seed),
        "--time-budget-seconds",
        str(args.time_budget_seconds),
        "--eval-tokens",
        str(args.eval_tokens),
        "--timeout-seconds",
        str(args.trial_timeout_seconds),
    ]
    # Hold the parent trial lock too: one training job at a time on a 16 GB box.
    parent_lock = acquire_lock(TRIAL_LOCK_PATH)
    try:
        completed = subprocess.run(
            command,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=args.trial_timeout_seconds + 120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CoordinatorError("trial exceeded the coordinator timeout") from exc
    except OSError as exc:
        raise CoordinatorError(f"could not launch trial_gate.py: {exc}") from exc
    finally:
        release_lock(parent_lock)

    manifests = sorted((worktree / ".autoresearch" / "runs").glob("*/manifest.json"))
    if not manifests:
        raise CoordinatorError(
            f"trial produced no manifest (exit {completed.returncode}): "
            f"{completed.stderr.strip()[:300]}"
        )
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text())
    evidence_dir = REPORTS_DIR / run_id
    evidence = copy_evidence(manifest_path.parent, evidence_dir)
    return {
        "command": command,
        "return_code": completed.returncode,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "evidence_dir": str(evidence_dir),
        "evidence_sha256": evidence,
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "failures": manifest.get("failures", []),
        "metrics": manifest.get("metrics", {}),
        "log_sha256": manifest.get("log_sha256"),
        "source_files": manifest.get("source", {}).get("files", {}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: no prefix may stand in for an escalation flag, and
    # `--qwen-api-key X` must fail loudly instead of silently becoming the
    # *name* of the variable to read.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--apply", action="store_true", help="create a worktree and patch train.py (default: dry run)"
    )
    parser.add_argument(
        "--run-trial", action="store_true", help="with --apply, run the bounded worktree trial"
    )
    parser.add_argument(
        "--goal",
        default="reduce val_bpb at a fixed time budget without raising peak memory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-url", default=DEFAULT_QWEN_URL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--qwen-api-key-env",
        default="LOCAL_QWEN_API_KEY",
        help="environment variable holding a bearer token, if the local server requires one",
    )
    parser.add_argument("--qwen-timeout", type=float, default=120.0)
    parser.add_argument("--qwen-max-tokens", type=int, default=300)
    parser.add_argument("--claude-bin", default=DEFAULT_CLAUDE_BIN)
    parser.add_argument("--claude-model", default="sonnet")
    parser.add_argument("--claude-timeout", type=float, default=180.0)
    parser.add_argument("--worktree-root", default=str(DEFAULT_WORKTREE_ROOT))
    parser.add_argument("--time-budget-seconds", type=int, default=300)
    parser.add_argument("--eval-tokens", type=int, default=2**18)
    parser.add_argument("--trial-timeout-seconds", type=int, default=480)
    parser.add_argument("--json", action="store_true", help="print only the JSON report")
    return parser


def execute(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.run_trial and not args.apply:
        raise CoordinatorError("--run-trial requires --apply")
    for name in ("qwen_timeout", "claude_timeout"):
        if getattr(args, name) <= 0:
            raise CoordinatorError(f"--{name.replace('_', '-')} must be positive")
    if args.time_budget_seconds <= 0 or args.eval_tokens <= 0 or args.trial_timeout_seconds <= 0:
        raise CoordinatorError("trial budgets must be positive")

    started = dt.datetime.now(dt.timezone.utc)
    context = load_context()
    outcome = run_review_stage(args, context)
    run_id = run_id_for(started, outcome["change"]["parameter"])

    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": started.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "goal": args.goal,
        "models": {"proposer": args.qwen_model, "reviewer": args.claude_model},
        "baseline": outcome["baseline"],
        "proposal": outcome["proposal"],
        "review": outcome["review"],
        "change": outcome["change"],
        "promotion": "manual review required; this coordinator never merges or promotes",
    }

    if outcome["review"]["decision"] != "approve":
        report["status"] = "rejected"
        return 1, report
    if not args.apply:
        report["status"] = "approved-dry-run"
        report["next_step"] = "re-run with --apply to create the worktree and patch"
        return 0, report

    report["worktree"] = apply_change(args, outcome, run_id)
    report["status"] = "applied"
    if not args.run_trial:
        report["next_step"] = "re-run with --apply --run-trial to measure, or inspect the worktree"
        write_report(report)
        return 0, report

    description = (
        f"{outcome['change']['parameter']} {outcome['change']['from']!r}"
        f"->{outcome['change']['to']!r}"
    )
    report["trial"] = run_trial(args, Path(report["worktree"]["path"]), description, run_id)
    report["status"] = "trial-" + str(report["trial"].get("status", "unknown"))
    report["next_step"] = "human review; compare against baseline before any promotion"
    write_report(report)
    return (0 if report["trial"].get("status") == "passed" else 1), report


def write_report(report: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    report["report_path"] = str(path)
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock_handle = None
    try:
        lock_handle = acquire_lock(LOCK_PATH)
        code, report = execute(args)
        print(json.dumps(report, indent=2))
        if not args.json:
            print(
                f"\n{report['status']}: {report['change']['parameter']} "
                f"{report['change']['from']!r} -> {report['change']['to']!r}",
                file=sys.stderr,
            )
        return code
    except (CoordinatorError, ProposalError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    finally:
        release_lock(lock_handle)


if __name__ == "__main__":
    raise SystemExit(main())
