"""Unit tests for the autoresearch coordinator.

Every model call, git command, and subprocess is mocked. Nothing here starts a
server, spends tokens, creates a worktree, or runs training.
"""

import argparse
import contextlib
import copy
import datetime as dt
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error


@contextlib.contextmanager
def quiet_argparse():
    """Swallow argparse's usage message when a parse failure is the assertion."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield

import research_team as rt


CURRENT = {
    "ASPECT_RATIO": 64,
    "HEAD_DIM": 128,
    "WINDOW_PATTERN": "SSSL",
    "TOTAL_BATCH_SIZE": 8192,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.2,
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
    "DEPTH": 4,
    "DEVICE_BATCH_SIZE": 4,
}


def envelope(structured):
    return json.dumps({"is_error": False, "subtype": "success", "structured_output": structured})


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(["mock"], returncode, stdout, stderr)


def namespace(**overrides):
    args = argparse.Namespace(
        apply=False,
        run_trial=False,
        goal="reduce val_bpb",
        seed=42,
        qwen_url=rt.DEFAULT_QWEN_URL,
        qwen_model=rt.DEFAULT_QWEN_MODEL,
        qwen_api_key_env="QWEN_API_KEY_ABSENT_IN_TESTS",
        qwen_timeout=5.0,
        qwen_max_tokens=300,
        claude_bin="/opt/homebrew/bin/claude",
        claude_model="sonnet",
        claude_timeout=5.0,
        worktree_root="/tmp/does-not-exist",
        time_budget_seconds=300,
        eval_tokens=1024,
        trial_timeout_seconds=480,
        json=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class AllowlistTests(unittest.TestCase):
    def test_allowlist_matches_real_train_py_constants(self):
        source = (rt.ROOT / "train.py").read_text()
        found = rt.module_constants(source, rt.ALLOWLIST)
        self.assertEqual(sorted(found), sorted(rt.ALLOWLIST))

    def test_current_train_py_values_satisfy_every_bound(self):
        source = (rt.ROOT / "train.py").read_text()
        for name, value in rt.module_constants(source, rt.ALLOWLIST).items():
            with self.subTest(name=name):
                rt.validate_value(name, value)

    def test_protected_files_are_not_reachable(self):
        self.assertNotIn(rt.TARGET_FILE, rt.PROTECTED_FILES)
        for name in ("prepare.py", "pyproject.toml", "uv.lock", "trial_gate.py"):
            self.assertIn(name, rt.PROTECTED_FILES)


class ValueValidationTests(unittest.TestCase):
    def test_rejects_name_outside_allowlist(self):
        for name in ("os", "__builtins__", "MAX_SEQ_LEN", "VOCAB_SIZE", "DEPTH "):
            with self.subTest(name=name), self.assertRaises(rt.ProposalError):
                rt.validate_value(name, 4)

    def test_rejects_out_of_bounds(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("DEPTH", 13)
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("DEPTH", 1)
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("WEIGHT_DECAY", -0.1)

    def test_rejects_bool_and_non_finite_and_wrong_type(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("DEPTH", True)
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("MATRIX_LR", float("nan"))
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("MATRIX_LR", float("inf"))
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("DEPTH", 6.5)
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("DEPTH", "6")

    def test_enum_and_pattern(self):
        self.assertEqual(rt.validate_value("HEAD_DIM", 64), 64)
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("HEAD_DIM", 96)
        self.assertEqual(rt.validate_value("WINDOW_PATTERN", "SSLL"), "SSLL")
        for bad in ("", "SSSX", "S" * 9, "sssl"):
            with self.subTest(bad=bad), self.assertRaises(rt.ProposalError):
                rt.validate_value("WINDOW_PATTERN", bad)

    def test_int_is_widened_to_float_for_float_specs_only(self):
        self.assertIsInstance(rt.validate_value("WEIGHT_DECAY", 1), float)
        with self.assertRaises(rt.ProposalError):
            rt.validate_value("DEPTH", 4.0000001)


class RelationshipTests(unittest.TestCase):
    def test_accepts_a_sound_change(self):
        effective = rt.validate_change(CURRENT, "DEPTH", 6)
        self.assertEqual(effective["DEPTH"], 6)
        self.assertEqual(effective["_derived"]["model_dim"], 384)
        self.assertEqual(effective["_derived"]["n_head"], 3)
        self.assertEqual(effective["_derived"]["grad_accum_steps"], 4)

    def test_rejects_no_op(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_change(CURRENT, "DEPTH", 4)

    def test_rejects_indivisible_batch_relationship(self):
        with self.assertRaises(rt.ProposalError) as caught:
            rt.validate_change(CURRENT, "DEVICE_BATCH_SIZE", 3)
        self.assertIn("divisible", str(caught.exception))

    def test_rejects_excessive_grad_accum(self):
        # 65536 tokens / (1 * 512) = 128 micro-batches, twice the ceiling.
        with self.assertRaises(rt.ProposalError) as caught:
            rt.validate_change(dict(CURRENT, DEVICE_BATCH_SIZE=1), "TOTAL_BATCH_SIZE", 65536)
        self.assertIn("accumulation", str(caught.exception))

    def test_rejects_warmup_plus_warmdown_over_one(self):
        current = dict(CURRENT, WARMUP_RATIO=0.6)
        with self.assertRaises(rt.ProposalError) as caught:
            rt.validate_change(current, "WARMDOWN_RATIO", 0.9)
        self.assertIn("WARMUP_RATIO", str(caught.exception))

    def test_rejects_oversized_memory_estimate(self):
        with self.assertRaises(rt.ProposalError) as caught:
            rt.validate_change(dict(CURRENT, ASPECT_RATIO=256), "DEPTH", 12)
        self.assertIn("peak memory", str(caught.exception))

    def test_memory_estimate_is_calibrated_to_the_baseline(self):
        # The recorded baseline peak is 659.8 MB; stay within an order of magnitude.
        estimate = rt.estimate_peak_memory_mb(CURRENT)
        self.assertGreater(estimate, 200.0)
        self.assertLess(estimate, 4000.0)

    def test_parameter_count_matches_the_7_3m_baseline(self):
        model_dim = rt.model_dim_for(4, 64, 128)
        self.assertEqual(model_dim, 256)
        self.assertAlmostEqual(rt.parameter_count(4, model_dim) / 1e6, 7.34, places=2)

    def test_missing_current_values_is_an_operational_error(self):
        partial = {key: value for key, value in CURRENT.items() if key != "DEPTH"}
        with self.assertRaises(rt.CoordinatorError):
            rt.validate_change(partial, "MATRIX_LR", 0.05)


class ProposalSchemaTests(unittest.TestCase):
    def test_accepts_exact_schema(self):
        proposal = rt.validate_proposal(
            {"parameter": "MATRIX_LR", "value": 0.05, "rationale": "slightly faster matrices"},
            CURRENT,
        )
        self.assertEqual(proposal["parameter"], "MATRIX_LR")
        self.assertEqual(proposal["value"], 0.05)
        self.assertEqual(proposal["current"], 0.04)

    def test_rejects_extra_or_missing_keys(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_proposal(
                {"parameter": "DEPTH", "value": 6, "rationale": "x", "file": "prepare.py"}, CURRENT
            )
        with self.assertRaises(rt.ProposalError):
            rt.validate_proposal({"parameter": "DEPTH", "value": 6}, CURRENT)

    def test_rejects_non_object(self):
        for payload in ([], "DEPTH=6", 7, None):
            with self.subTest(payload=payload), self.assertRaises(rt.ProposalError):
                rt.validate_proposal(payload, CURRENT)

    def test_rationale_is_truncated(self):
        proposal = rt.validate_proposal(
            {"parameter": "DEPTH", "value": 6, "rationale": "y" * 900}, CURRENT
        )
        self.assertEqual(len(proposal["rationale"]), 400)


class ReviewValidationTests(unittest.TestCase):
    def setUp(self):
        self.proposal = rt.validate_proposal(
            {"parameter": "DEPTH", "value": 6, "rationale": "more depth"}, CURRENT
        )

    def test_approve_and_reject(self):
        approved = rt.validate_review(
            {"decision": "approve", "rationale": "sound"}, self.proposal, CURRENT
        )
        self.assertEqual(approved["decision"], "approve")
        self.assertFalse(approved["adjusted"])
        rejected = rt.validate_review(
            {"decision": "reject", "rationale": "too risky"}, self.proposal, CURRENT
        )
        self.assertEqual(rejected["decision"], "reject")

    def test_rejects_unknown_keys_and_bad_decision(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_review(
                {"decision": "approve", "rationale": "ok", "run_command": "rm -rf /"},
                self.proposal,
                CURRENT,
            )
        for decision in ("maybe", "APPROVE", None, True):
            with self.subTest(decision=decision), self.assertRaises(rt.ProposalError):
                rt.validate_review({"decision": decision, "rationale": "x"}, self.proposal, CURRENT)

    def test_rejects_empty_rationale(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_review({"decision": "approve", "rationale": "   "}, self.proposal, CURRENT)

    def test_adjustment_is_revalidated(self):
        review = rt.validate_review(
            {
                "decision": "approve",
                "rationale": "smaller step",
                "adjusted_parameter": "DEPTH",
                "adjusted_value": 5,
            },
            self.proposal,
            CURRENT,
        )
        self.assertTrue(review["adjusted"])
        self.assertEqual(review["value"], 5)

    def test_adjustment_cannot_escape_the_allowlist_or_bounds(self):
        for value in (99, 0, "6"):
            with self.subTest(value=value), self.assertRaises(rt.ProposalError):
                rt.validate_review(
                    {
                        "decision": "approve",
                        "rationale": "x",
                        "adjusted_parameter": "DEPTH",
                        "adjusted_value": value,
                    },
                    self.proposal,
                    CURRENT,
                )
        with self.assertRaises(rt.ProposalError):
            rt.validate_review(
                {
                    "decision": "approve",
                    "rationale": "x",
                    "adjusted_parameter": "MAX_SEQ_LEN",
                    "adjusted_value": 1024,
                },
                self.proposal,
                CURRENT,
            )

    def test_integral_float_is_narrowed_but_fractional_is_rejected(self):
        review = rt.validate_review(
            {
                "decision": "approve",
                "rationale": "x",
                "adjusted_parameter": "DEPTH",
                "adjusted_value": 6.0,
            },
            self.proposal,
            CURRENT,
        )
        self.assertIsInstance(review["value"], int)
        with self.assertRaises(rt.ProposalError):
            rt.validate_review(
                {
                    "decision": "approve",
                    "rationale": "x",
                    "adjusted_parameter": "DEPTH",
                    "adjusted_value": 6.5,
                },
                self.proposal,
                CURRENT,
            )

    def test_rejecting_review_cannot_carry_an_adjustment(self):
        with self.assertRaises(rt.ProposalError):
            rt.validate_review(
                {
                    "decision": "reject",
                    "rationale": "no",
                    "adjusted_parameter": "DEPTH",
                    "adjusted_value": 5,
                },
                self.proposal,
                CURRENT,
            )


class JsonExtractionTests(unittest.TestCase):
    def test_plain_fenced_and_embedded(self):
        want = {"parameter": "DEPTH", "value": 6, "rationale": "r"}
        for text in (
            json.dumps(want),
            "```json\n" + json.dumps(want) + "\n```",
            "Sure!\n" + json.dumps(want) + "\nHope that helps.",
        ):
            with self.subTest(text=text[:20]):
                self.assertEqual(rt.extract_json_object(text), want)

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        self.assertEqual(
            rt.extract_json_object('note {"rationale": "a } brace", "value": 6}'),
            {"rationale": "a } brace", "value": 6},
        )

    def test_empty_and_objectless_output(self):
        for text in ("", "   ", "no json here"):
            with self.subTest(text=text), self.assertRaises(rt.ProposalError):
                rt.extract_json_object(text)


class PatchingTests(unittest.TestCase):
    SOURCE = (
        "import os\n"
        "DEPTH = 4\n"
        "MATRIX_LR = 0.04  # tuned\n"
        'WINDOW_PATTERN = "SSSL"\n'
        "TOTAL_BATCH_SIZE = 2**13\n"
        "def f():\n"
        "    DEPTH = 99\n"
        "    return DEPTH\n"
    )

    def test_changes_exactly_one_line_and_keeps_comment(self):
        patched = rt.replace_constant(self.SOURCE, "MATRIX_LR", 0.05)
        self.assertIn("MATRIX_LR = 0.05  # tuned\n", patched)
        before, after = self.SOURCE.splitlines(), patched.splitlines()
        self.assertEqual(len(before), len(after))
        self.assertEqual([i for i, (a, b) in enumerate(zip(before, after)) if a != b], [2])

    def test_does_not_touch_local_variables(self):
        patched = rt.replace_constant(self.SOURCE, "DEPTH", 6)
        self.assertIn("DEPTH = 6\n", patched)
        self.assertIn("    DEPTH = 99\n", patched)

    def test_expression_valued_constant_is_replaced_with_a_literal(self):
        patched = rt.replace_constant(self.SOURCE, "TOTAL_BATCH_SIZE", 16384)
        self.assertIn("TOTAL_BATCH_SIZE = 16384\n", patched)

    def test_string_values_are_quoted(self):
        self.assertIn(
            'WINDOW_PATTERN = "SSLL"\n', rt.replace_constant(self.SOURCE, "WINDOW_PATTERN", "SSLL")
        )

    def test_refuses_non_allowlisted_and_absent_names(self):
        with self.assertRaises(rt.ProposalError):
            rt.replace_constant(self.SOURCE, "os", 1)
        with self.assertRaises(rt.ProposalError):
            rt.replace_constant("X = 1\n", "DEPTH", 6)

    def test_duplicate_assignment_is_refused(self):
        with self.assertRaises(rt.ProposalError):
            rt.replace_constant("DEPTH = 4\nDEPTH = 5\n", "DEPTH", 6)

    def test_patched_real_train_py_still_compiles(self):
        source = (rt.ROOT / "train.py").read_text()
        patched = rt.replace_constant(source, "DEPTH", 6)
        self.assertNotEqual(patched, source)
        compile(patched, "train.py", "exec")

    def test_render_value_round_trips(self):
        self.assertEqual(rt.render_value(6), "6")
        self.assertEqual(rt.render_value(1.0), "1.0")
        self.assertEqual(rt.render_value("SL"), '"SL"')
        with self.assertRaises(rt.ProposalError):
            rt.render_value(True)
        with self.assertRaises(rt.ProposalError):
            rt.render_value(None)


class GitGuardTests(unittest.TestCase):
    def test_accepts_a_single_train_py_modification(self):
        rt.verify_only_target_changed(" M train.py\n")
        rt.verify_single_line_diff("1\t1\ttrain.py\n")

    def test_rejects_extra_or_wrong_files(self):
        for porcelain in (
            "",
            " M train.py\n M prepare.py\n",
            " M prepare.py\n",
            "?? scratch.py\n",
            "M  train.py\n",
            " D train.py\n",
        ):
            with self.subTest(porcelain=porcelain), self.assertRaises(rt.CoordinatorError):
                rt.verify_only_target_changed(porcelain)

    def test_rejects_multi_line_or_wrong_file_diffs(self):
        for numstat in ("2\t2\ttrain.py\n", "1\t1\tprepare.py\n", "1\t1\ttrain.py\n1\t1\tx.py\n", ""):
            with self.subTest(numstat=numstat), self.assertRaises(rt.CoordinatorError):
                rt.verify_single_line_diff(numstat)

    def test_require_clean_worktree(self):
        with mock.patch.object(rt, "git", return_value=""):
            rt.require_clean_worktree(rt.ROOT)
        with mock.patch.object(rt, "git", return_value=" M train.py"):
            with self.assertRaises(rt.CoordinatorError):
                rt.require_clean_worktree(rt.ROOT)

    def test_naming_is_deterministic(self):
        stamp = dt.datetime(2026, 8, 17, 12, 30, 0, tzinfo=dt.timezone.utc)
        run_id = rt.run_id_for(stamp, "DEPTH")
        self.assertEqual(run_id, "20260817T123000Z-depth")
        self.assertEqual(rt.branch_name_for(run_id), "autoresearch/20260817T123000Z-depth")
        self.assertEqual(
            rt.worktree_path_for(Path("/tmp/trials"), run_id), Path("/tmp/trials") / run_id
        )


class TransportTests(unittest.TestCase):
    def test_requires_loopback(self):
        rt.require_loopback("http://127.0.0.1:8088/v1/chat/completions")
        for url in (
            "http://evil.example.com/v1/chat/completions",
            "https://127.0.0.1/v1",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url), self.assertRaises(rt.CoordinatorError):
                rt.require_loopback(url)

    def test_call_qwen_returns_content(self):
        payload = {"choices": [{"message": {"content": '{"parameter":"DEPTH"}'}}]}
        with mock.patch.object(rt, "_post_json", return_value=payload) as post:
            content, _ = rt.call_qwen(rt.DEFAULT_QWEN_URL, "qwen", "hi", 5.0, 300)
        self.assertEqual(content, '{"parameter":"DEPTH"}')
        self.assertEqual(post.call_args.args[1]["max_tokens"], 300)

    def test_call_qwen_retries_once_without_response_format(self):
        error = urllib.error.HTTPError(rt.DEFAULT_QWEN_URL, 400, "Bad Request", {}, None)
        good = {"choices": [{"message": {"content": "{}"}}]}
        # call_qwen mutates the payload between attempts, so snapshot each call.
        seen = []
        results = iter([error, good])

        def record(url, payload, timeout, api_key=None):
            seen.append(copy.deepcopy(payload))
            outcome = next(results)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with mock.patch.object(rt, "_post_json", side_effect=record):
            rt.call_qwen(rt.DEFAULT_QWEN_URL, "qwen", "hi", 5.0, 300)
        self.assertEqual(len(seen), 2)
        self.assertIn("response_format", seen[0])
        self.assertNotIn("response_format", seen[1])

    def test_bearer_header_is_sent_only_when_a_key_is_present(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["headers"] = dict(request.header_items())
            raise urllib.error.URLError("stop here")

        for key, expected in ((None, False), ("secret-token", True)):
            with self.subTest(key=key):
                with mock.patch.object(rt.urllib.request, "urlopen", fake_urlopen):
                    with self.assertRaises(urllib.error.URLError):
                        rt._post_json(rt.DEFAULT_QWEN_URL, {"a": 1}, 5.0, key)
                headers = {name.lower(): value for name, value in captured["headers"].items()}
                self.assertEqual("authorization" in headers, expected)
                if expected:
                    self.assertEqual(headers["authorization"], "Bearer secret-token")

    def test_api_key_is_read_from_the_environment_not_the_command_line(self):
        args = rt.build_parser().parse_args([])
        self.assertEqual(args.qwen_api_key_env, "LOCAL_QWEN_API_KEY")
        # Only the variable *name* is a flag; the secret itself has no flag.
        self.assertFalse(hasattr(args, "qwen_api_key"))
        with quiet_argparse(), self.assertRaises(SystemExit):
            rt.build_parser().parse_args(["--qwen-api-key", "secret"])

    def test_call_qwen_surfaces_unreachable_server(self):
        with mock.patch.object(rt, "_post_json", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(rt.CoordinatorError):
                rt.call_qwen(rt.DEFAULT_QWEN_URL, "qwen", "hi", 5.0, 300)

    def test_call_qwen_rejects_malformed_transport_payload(self):
        with mock.patch.object(rt, "_post_json", return_value={"error": "nope"}):
            with self.assertRaises(rt.CoordinatorError):
                rt.call_qwen(rt.DEFAULT_QWEN_URL, "qwen", "hi", 5.0, 300)


class ClaudeInvocationTests(unittest.TestCase):
    def test_command_disables_tools_and_pins_the_schema(self):
        command = rt.build_claude_command("/opt/homebrew/bin/claude", "sonnet", rt.CLAUDE_REVIEW_SCHEMA)
        self.assertEqual(command[0], "/opt/homebrew/bin/claude")
        self.assertEqual(command[1], "-p")
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", command)
        self.assertEqual(command[command.index("--mcp-config") + 1], '{"mcpServers":{}}')
        self.assertEqual(command[command.index("--output-format") + 1], "json")
        schema = json.loads(command[command.index("--json-schema") + 1])
        self.assertEqual(schema, rt.CLAUDE_REVIEW_SCHEMA)
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertNotIn("--allow-dangerously-skip-permissions", command)
        self.assertNotIn("--add-dir", command)

    def test_prompt_never_leaks_repo_paths(self):
        proposal = rt.validate_proposal(
            {"parameter": "DEPTH", "value": 6, "rationale": "r"}, CURRENT
        )
        prompt = rt.build_claude_prompt(proposal, CURRENT, "goal")
        self.assertNotIn(str(rt.ROOT), prompt)
        self.assertIn("DEPTH", prompt)

    def test_call_claude_passes_the_prompt_on_stdin_from_a_scratch_cwd(self):
        stdout = envelope({"decision": "approve", "rationale": "ok"})
        with mock.patch.object(Path, "exists", return_value=True), mock.patch.object(
            rt.subprocess, "run", return_value=completed(stdout)
        ) as run:
            payload, _ = rt.call_claude("/opt/homebrew/bin/claude", "sonnet", "prompt", 5.0)
        self.assertEqual(payload["decision"], "approve")
        self.assertEqual(run.call_args.kwargs["input"], "prompt")
        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)
        self.assertNotEqual(Path(run.call_args.kwargs["cwd"]), rt.ROOT)

    def test_call_claude_reports_missing_binary_timeout_and_failure(self):
        with mock.patch.object(Path, "exists", return_value=False):
            with self.assertRaises(rt.CoordinatorError):
                rt.call_claude("/nope/claude", "sonnet", "p", 5.0)
        with mock.patch.object(Path, "exists", return_value=True), mock.patch.object(
            rt.subprocess, "run", side_effect=subprocess.TimeoutExpired("claude", 5.0)
        ):
            with self.assertRaises(rt.CoordinatorError):
                rt.call_claude("/opt/homebrew/bin/claude", "sonnet", "p", 5.0)
        with mock.patch.object(Path, "exists", return_value=True), mock.patch.object(
            rt.subprocess, "run", return_value=completed("", 1, "boom")
        ):
            with self.assertRaises(rt.CoordinatorError):
                rt.call_claude("/opt/homebrew/bin/claude", "sonnet", "p", 5.0)

    def test_envelope_unwrapping(self):
        want = {"decision": "approve", "rationale": "ok"}
        self.assertEqual(rt._extract_claude_payload(envelope(want)), want)
        self.assertEqual(
            rt._extract_claude_payload(json.dumps({"is_error": False, "result": json.dumps(want)})),
            want,
        )
        self.assertEqual(rt._extract_claude_payload(json.dumps(want)), want)

    def test_error_envelope_is_surfaced(self):
        with self.assertRaises(rt.CoordinatorError):
            rt._extract_claude_payload(json.dumps({"is_error": True, "result": "rate limited"}))


class EvidenceTests(unittest.TestCase):
    def test_copies_manifest_and_log_and_hashes_the_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run"
            source.mkdir()
            (source / "manifest.json").write_text('{"status": "passed"}')
            (source / "run.log").write_text("val_bpb: 2.7\n")
            (source / "checkpoint.bin").write_bytes(b"\x00" * 16)
            destination = Path(directory) / "report"
            copied = rt.copy_evidence(source, destination)

        self.assertEqual(sorted(copied), ["manifest.json", "run.log"])
        self.assertEqual(
            copied["manifest.json"],
            rt.hashlib.sha256(b'{"status": "passed"}').hexdigest(),
        )
        self.assertFalse((destination / "checkpoint.bin").exists())

    def test_missing_evidence_is_reported_as_absent_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run"
            source.mkdir()
            self.assertEqual(rt.copy_evidence(source, Path(directory) / "report"), {})


class LockTests(unittest.TestCase):
    def test_second_acquisition_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coordinator.lock"
            handle = rt.acquire_lock(path)
            try:
                with self.assertRaises(rt.CoordinatorError):
                    rt.acquire_lock(path)
            finally:
                rt.release_lock(handle)
            rt.release_lock(rt.acquire_lock(path))


class ExecuteTests(unittest.TestCase):
    """End-to-end flow with both models mocked and every side effect stubbed."""

    def setUp(self):
        self.context = {"current": copy.deepcopy(CURRENT), "train_source": "DEPTH = 4\n"}
        patcher = mock.patch.object(rt, "load_context", return_value=self.context)
        patcher.start()
        self.addCleanup(patcher.stop)
        baseline = mock.patch.object(rt, "load_baseline", return_value=None)
        baseline.start()
        self.addCleanup(baseline.stop)

    def _models(self, proposal=None, review=None):
        proposal = proposal or {"parameter": "DEPTH", "value": 6, "rationale": "deeper"}
        review = review or {"decision": "approve", "rationale": "sound single-variable test"}
        qwen = mock.patch.object(rt, "call_qwen", return_value=(json.dumps(proposal), {}))
        claude = mock.patch.object(rt, "call_claude", return_value=(review, ""))
        return qwen, claude

    def test_dry_run_calls_both_models_and_mutates_nothing(self):
        qwen, claude = self._models()
        with qwen as qwen_mock, claude as claude_mock, mock.patch.object(
            rt, "apply_change"
        ) as apply_mock, mock.patch.object(rt, "run_trial") as trial_mock, mock.patch.object(
            rt, "write_report"
        ) as report_mock, mock.patch.object(rt, "git") as git_mock:
            code, report = rt.execute(namespace())

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "approved-dry-run")
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["change"], {"parameter": "DEPTH", "from": 4, "to": 6})
        qwen_mock.assert_called_once()
        claude_mock.assert_called_once()
        apply_mock.assert_not_called()
        trial_mock.assert_not_called()
        report_mock.assert_not_called()
        git_mock.assert_not_called()

    def test_rejected_review_stops_before_apply(self):
        qwen, claude = self._models(review={"decision": "reject", "rationale": "unsound"})
        with qwen, claude, mock.patch.object(rt, "apply_change") as apply_mock:
            code, report = rt.execute(namespace(apply=True))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "rejected")
        apply_mock.assert_not_called()

    def test_apply_without_run_trial_never_trains(self):
        qwen, claude = self._models()
        worktree = {"branch": "autoresearch/x", "path": "/tmp/x", "py_compile": "ok"}
        with qwen, claude, mock.patch.object(
            rt, "apply_change", return_value=worktree
        ) as apply_mock, mock.patch.object(rt, "run_trial") as trial_mock, mock.patch.object(
            rt, "write_report"
        ):
            code, report = rt.execute(namespace(apply=True))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "applied")
        apply_mock.assert_called_once()
        trial_mock.assert_not_called()

    def test_run_trial_requires_apply(self):
        qwen, claude = self._models()
        with qwen, claude, self.assertRaises(rt.CoordinatorError):
            rt.execute(namespace(run_trial=True))

    def test_full_trial_records_evidence_and_never_promotes(self):
        qwen, claude = self._models()
        worktree = {"branch": "autoresearch/x", "path": "/tmp/x", "py_compile": "ok"}
        trial = {"status": "passed", "metrics": {"val_bpb": 2.7}, "evidence_sha256": {"run.log": "a"}}
        with qwen, claude, mock.patch.object(
            rt, "apply_change", return_value=worktree
        ), mock.patch.object(rt, "run_trial", return_value=trial) as trial_mock, mock.patch.object(
            rt, "write_report"
        ):
            code, report = rt.execute(namespace(apply=True, run_trial=True))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "trial-passed")
        self.assertEqual(trial_mock.call_args.args[3], report["run_id"])
        self.assertIn("never merges", report["promotion"])
        self.assertNotIn("merge", report.get("next_step", "").lower())

    def test_failed_trial_is_a_nonzero_exit(self):
        qwen, claude = self._models()
        with self._models()[0], self._models()[1], mock.patch.object(
            rt, "apply_change", return_value={"path": "/tmp/x"}
        ), mock.patch.object(
            rt, "run_trial", return_value={"status": "rejected", "failures": ["peak_memory_limit"]}
        ), mock.patch.object(rt, "write_report"):
            code, report = rt.execute(namespace(apply=True, run_trial=True))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "trial-rejected")

    def test_bad_proposal_from_qwen_is_rejected_before_claude_is_called(self):
        qwen, claude = self._models(
            proposal={"parameter": "os.system", "value": 1, "rationale": "r"}
        )
        with qwen, claude as claude_mock, self.assertRaises(rt.ProposalError):
            rt.execute(namespace())
        claude_mock.assert_not_called()

    def test_nonpositive_budgets_are_refused(self):
        for override in (
            {"qwen_timeout": 0},
            {"claude_timeout": -1},
            {"apply": True, "run_trial": True, "time_budget_seconds": 0},
            {"apply": True, "run_trial": True, "eval_tokens": 0},
            {"apply": True, "run_trial": True, "trial_timeout_seconds": 0},
        ):
            with self.subTest(override=override), self.assertRaises(rt.CoordinatorError):
                rt.execute(namespace(**override))


class MainTests(unittest.TestCase):
    def test_defaults_are_dry_run(self):
        args = rt.build_parser().parse_args([])
        self.assertFalse(args.apply)
        self.assertFalse(args.run_trial)
        self.assertEqual(args.claude_bin, rt.DEFAULT_CLAUDE_BIN)

    def test_escalation_flags_cannot_be_abbreviated(self):
        for argv in (["--app"], ["--run"], ["--appl"]):
            with self.subTest(argv=argv), quiet_argparse(), self.assertRaises(SystemExit):
                rt.build_parser().parse_args(argv)

    def test_errors_exit_two_without_a_traceback(self):
        with mock.patch.object(rt, "acquire_lock", return_value=None), mock.patch.object(
            rt, "execute", side_effect=rt.CoordinatorError("qwen unreachable")
        ), mock.patch("builtins.print") as printer:
            self.assertEqual(rt.main([]), 2)
        printed = json.loads(printer.call_args_list[0].args[0])
        self.assertEqual(printed["status"], "error")


if __name__ == "__main__":
    unittest.main()
