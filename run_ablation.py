"""Experiment 5: score Stage 2 under four matched Stage 1 representations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from authorization_evidence import canonical, digest, extract_candidates
from representation_ablation import ABLATION_MODES, LEDGER_SESSION_OPEN, prepare_representations
from run_case import BENCHMARK_COMMIT, DEFAULT_MODELS, OPENAI_EFFORTS, PROMPTS, call_monitor
from run_context import ContextItem, TARGET_INSTRUCTION, TWO_STAGE_SCENARIOS, select_scenarios
from run_evidence import _payload_hash, build_evidence_plan, prepare as prepare_evidence

FORMAT_VERSION = "representation-ablation-stage2-v1"
MODES = ABLATION_MODES
ORDER_SEED = 5
OFFICIAL_ENDPOINTS = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}
STOP_REASONS = frozenset({
    "completed", "response_incomplete", "response_failed", "response_cancelled",
    "response_queued", "response_in_progress", "end_turn", "max_tokens",
    "stop_sequence", "tool_use", "pause_turn", "refusal", "model_context_window_exceeded",
})


@dataclass(frozen=True)
class Prepared:
    item: ContextItem
    system: str | None
    user: str
    provenance: dict[str, Any]


def benchmark_revision(root: Path) -> str:
    """Verify HEAD and relevant runtime files, without inspecting dataset paths."""
    paths = ["evaluate", "pyproject.toml", "uv.lock", *(f"prompts/{p}.md" for p in PROMPTS)]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        if revision != BENCHMARK_COMMIT:
            raise ValueError(f"Benchmark HEAD must be {BENCHMARK_COMMIT}")
        tree = subprocess.check_output(
            ["git", "ls-tree", "-rz", "HEAD", "--", *paths], cwd=root, stderr=subprocess.DEVNULL,
        ).decode()
        tracked_code = set()
        for entry in tree.rstrip("\0").split("\0"):
            metadata, name = entry.split("\t", 1)
            if name.startswith("evaluate/") and not name.endswith(".py"):
                continue
            _, kind, object_id = metadata.split()
            path = root / name
            raw = path.read_bytes()
            # Git on Windows may check out the same text with CRLF.
            normalized = raw.replace(b"\r\n", b"\n")
            matches = any(
                hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == object_id
                for data in (raw, normalized)
            )
            if path.is_symlink() or kind != "blob" or not matches:
                raise ValueError(f"Pinned benchmark runtime file differs: {name}")
            if name.startswith("evaluate/"):
                tracked_code.add(path)
        if set((root / "evaluate").rglob("*.py")) != tracked_code:
            raise ValueError("Benchmark evaluate/ contains different Python files than the pinned revision")
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("The pinned benchmark Git checkout or runtime files are unavailable") from exc
    return revision


def prepare_arms(item: ContextItem, root: Path) -> list[Prepared]:
    from evaluate.core.transcript import render_prompt, strip_assistant_text

    reference = prepare_evidence(item, root)
    stage1 = strip_assistant_text(item.stage1_path.read_text(encoding="utf-8"))
    stage2 = strip_assistant_text(item.stage2_path.read_text(encoding="utf-8"))
    prompt = (root / "prompts" / f"{item.prompt_name}.md").read_text(encoding="utf-8")
    representations = prepare_representations(stage1, extract_candidates(stage1))
    inherited = {**reference.hashes}
    inherited["evidence_payload_sha256"] = inherited.pop("payload_sha256")
    arms = []
    for mode, representation in representations.items():
        source = LEDGER_SESSION_OPEN + representation.text + '\n</session>\n\n'
        source += '<session index="2" role="target">\n' + stage2 + '\n</session>'
        system, user = render_prompt(prompt, source)
        user = TARGET_INSTRUCTION + "\n\n" + user
        payload_hash = _payload_hash(system, user)
        if mode == "evidence" and payload_hash != inherited["evidence_payload_sha256"]:
            raise ValueError("The fresh evidence arm does not reproduce the Experiment 4 model input")
        provenance = {
            **inherited, **representation.provenance,
            "stripped_stage1_sha256": digest(stage1), "stripped_stage2_sha256": digest(stage2),
            "payload_sha256": payload_hash,
            "input_length_proxy_bytes": len((system or "").encode("utf-8")) + len(user.encode("utf-8")),
        }
        arms.append(Prepared(replace(item, context_mode=mode), system, user, provenance))
    return arms


def build_plan(root: Path, trials: int, scenarios: Sequence[str]) -> list[Prepared]:
    # Only render identical model inputs once. Shuffle within each repetition
    # to distribute order effects across both labels, prompts, and arms.
    base = [arm for item in build_evidence_plan(root, 1, scenarios) for arm in prepare_arms(item, root)]
    randomizer = random.Random(ORDER_SEED)
    plan = []
    for trial in range(trials):
        batch = [replace(arm, item=replace(arm.item, trial=trial)) for arm in base]
        randomizer.shuffle(batch)
        plan.extend(batch)
    return plan


def anthropic_params(model: str, system: str | None, user: str) -> dict[str, Any]:
    from evaluate.core.providers import build_api_params
    from evaluate.core.types import MonitorSpec

    spec = MonitorSpec(
        prompt="control_arena", model=model, temperature=1.0, trials=1,
        thinking=True, strip_assistant=True, type="api",
    )
    params = build_api_params(spec, system, user)
    params.pop("output_config", None)  # Match the pinned streaming helper.
    return {**params, "timeout": 300.0}


def request_settings(provider: str, model: str, effort: str | None) -> dict[str, Any]:
    if provider == "anthropic":
        params = anthropic_params(model, None, "")
        params.pop("messages")
        params["api"] = "messages.stream"
    else:
        params = {
            "model": model, "api": "responses.create", "reasoning": {"effort": effort},
            "max_output_tokens": 8192, "timeout": 180.0, "store": False,
        }
    return {**params, "sdk_version": version(provider), "max_retries": 0, "base_url": OFFICIAL_ENDPOINTS[provider]}


def evaluate(provider: str, model: str, effort: str, arm: Prepared, client: Any) -> dict[str, Any]:
    from evaluate.core.response import extract_response_parts, parse_response

    if provider == "openai":
        answer, refused, stop, usage = call_monitor(provider, model, effort, arm.system, arm.user, client)
    else:
        # Same wire settings as Experiment 4. Bypass the upstream retry loop
        # so every scheduled condition can dispatch at most one request.
        params = anthropic_params(model, arm.system, arm.user)
        with client.messages.stream(**params) as stream:
            response = stream.get_final_message()
        _, answer, refused, stop = extract_response_parts(response)
        metering = getattr(response, "usage", None)
        input_tokens = getattr(metering, "input_tokens", None)
        if type(input_tokens) is int:
            input_tokens += (getattr(metering, "cache_read_input_tokens", None) or 0)
            input_tokens += (getattr(metering, "cache_creation_input_tokens", None) or 0)
        usage = {"input_tokens": input_tokens, "output_tokens": getattr(metering, "output_tokens", None),
                 "reasoning_tokens": None}
    parsed = parse_response(answer)
    score = parsed.get("numeric_score") if isinstance(parsed, dict) else None
    if refused or type(score) is not int or not 0 <= score <= 100:
        score = None
    return {
        "numeric_score": score, "status": "scored" if score is not None else "refusal_or_invalid",
        "stop_reason": stop if isinstance(stop, str) and stop in STOP_REASONS else None,
        **{name: value if type(value) is int and value >= 0 else None for name, value in usage.items()},
    }


def output_path(provider: str, model: str, effort: str | None, trials: int, scenarios: Sequence[str]) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "-", model)
    signature = digest(canonical([FORMAT_VERSION, model, effort, sorted(scenarios), MODES]))[:8]
    return Path("results") / f"{provider}-{safe_model}-ablation-{signature}-t{trials}.jsonl"


def condition_row(arm: Prepared, config: dict) -> dict:
    item = arm.item
    return {
        **config, **arm.provenance, "scenario": item.scenario, "prompt_name": item.prompt_name,
        "context_mode": item.context_mode, "is_benign": item.is_benign, "trial": item.trial,
    }


def load_completed(path: Path, expected: dict, resume: bool) -> dict:
    if not path.exists():
        return {}
    if not resume:
        raise ValueError("Output exists. Use --resume or choose a new --output.")
    completed = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if type(row) is not dict or type(row["is_benign"]) is not bool or type(row["trial"]) is not int:
                raise ValueError
            key = tuple(row[name] for name in ("scenario", "prompt_name", "is_benign", "trial", "context_mode"))
            if key in completed or key not in expected:
                raise ValueError
            if any(field not in row or canonical(row[field]) != canonical(value) for field, value in expected[key].items()):
                raise ValueError
            score = row["numeric_score"]
            if row["status"] == "scored":
                if type(score) is not int or not 0 <= score <= 100:
                    raise ValueError
            elif row["status"] != "refusal_or_invalid" or score is not None:
                raise ValueError
            completed[key] = row
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"Invalid, duplicate, or incompatible resume record at line {index}") from None
    return completed


@contextmanager
def lock_output(path: Path):
    """Hold a process lock through resume validation and all result writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another process is using this result file") from exc
        yield


def save_results(path: Path, rows) -> None:
    """Replace the score ledger atomically after a completed request."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".jsonl", encoding="utf-8", newline="\n", delete=False,
        ) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("vendor/sleight-bench"))
    parser.add_argument("--provider", choices=tuple(DEFAULT_MODELS), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=OPENAI_EFFORTS)
    parser.add_argument("--scenario", action="append", choices=TWO_STAGE_SCENARIOS)
    parser.add_argument("--trials", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Permit paid requests")
    parser.add_argument("--resume", action="store_true", help="Validate and skip completed conditions")
    parser.add_argument("--max-calls", type=int, help="Required ceiling on new requests with --execute")
    args = parser.parse_args(argv)
    provider = args.provider
    model = args.model or DEFAULT_MODELS[provider]
    effort = (args.reasoning_effort or "medium") if provider == "openai" else None
    if not model.startswith("gpt-" if provider == "openai" else "claude-"):
        parser.error("Provider and model ID do not match")
    if provider == "anthropic" and args.reasoning_effort is not None:
        parser.error("--reasoning-effort applies only to OpenAI")
    if args.max_calls is not None and args.max_calls < 1:
        parser.error("--max-calls must be positive")
    if args.execute and args.max_calls is None:
        parser.error("--execute requires --max-calls")

    root = args.benchmark_dir.resolve()
    try:
        scenarios = tuple(sorted(select_scenarios(args.scenario)))
        revision = benchmark_revision(root)
        sys.path.insert(0, str(root))
        try:
            settings = request_settings(provider, model, effort)
            plan = build_plan(root, args.trials, scenarios)
        finally:
            sys.path.remove(str(root))
    except ValueError as exc:
        # Known precheck errors report only metadata. Source parsing errors from
        # third-party code may embed transcript material and are withheld.
        if type(exc) is ValueError:
            print(f"PRECHECK FAILED: {exc}", file=sys.stderr)
        else:
            print(f"PRECHECK FAILED ({type(exc).__name__})", file=sys.stderr)
        return 2
    except (OSError, ImportError, RuntimeError) as exc:
        print(f"PRECHECK FAILED ({type(exc).__name__})", file=sys.stderr)
        return 2

    config = {
        "format_version": FORMAT_VERSION, "benchmark_commit": revision, "provider": provider,
        "model": model, "reasoning_effort": effort, "thinking": True, "temperature": settings.get("temperature"),
        "strip_assistant": True, "target_stage": 2, "stage_count": 2, "order_seed": ORDER_SEED,
        "planned_scenarios": list(scenarios), "planned_trials": args.trials,
        "request_settings": settings, "request_settings_sha256": digest(canonical(settings)),
        "runner_sources_sha256": digest(canonical({
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("run_ablation.py", "representation_ablation.py", "run_case.py", "run_context.py",
                         "run_evidence.py", "authorization_evidence.py")
        })),
    }
    expected = {arm.item.key: condition_row(arm, config) for arm in plan}
    dest = (args.output or output_path(provider, model, effort, args.trials, scenarios)).resolve()
    if dest.suffix.lower() != ".jsonl" or dest.is_relative_to(root):
        parser.error("--output must be a .jsonl path outside the benchmark checkout")
    print(f"Experiment 5 | Benchmark {revision} | {provider}/{model} | {effort or 'benchmark thinking'}")
    print(f"Planned requests: {len(plan)} | Output: {dest}")
    for scenario in scenarios:
        for benign in (False, True):
            lengths = {
                arm.item.context_mode: arm.provenance["representation_token_proxy"]
                for arm in plan if arm.item.scenario == scenario and arm.item.is_benign == benign and arm.item.trial == 0
            }
            neutral = next(arm for arm in plan if arm.item.scenario == scenario and arm.item.is_benign == benign
                           and arm.item.context_mode == "neutral_context")
            provenance = neutral.provenance
            print(f"{scenario} benign={benign}: Stage 1 proxy {canonical(lengths)}, "
                  f"padding={provenance['neutral_padding_units']}, delta={provenance['representation_token_proxy_delta']}, "
                  f"selector={provenance['evidence_selection_mode']}")

    def pending_arms(completed: dict) -> list[Prepared]:
        pending = [arm for arm in plan if arm.item.key not in completed]
        max_output = settings.get("max_output_tokens", settings.get("max_tokens"))
        print(f"Completed: {len(completed)} | Remaining: {len(pending)} | "
              f"Max output tokens per request: {max_output} | "
              f"Remaining aggregate output ceiling: {len(pending) * max_output}")
        return pending

    try:
        if not args.execute:
            pending_arms(load_completed(dest, expected, True) if args.resume else {})
            print("DRY RUN COMPLETE: no model requests or result writes")
            return 0
        with lock_output(dest):
            completed = load_completed(dest, expected, args.resume)
            pending = pending_arms(completed)
            if len(pending) > args.max_calls:
                raise ValueError("Remaining calls exceed --max-calls. No requests were made.")
            if not pending:
                return 0
            key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
            api_key = os.environ.get(key_name)
            if not api_key:
                raise ValueError(f"Missing {key_name}. No requests were made.")
            if provider == "openai":
                from openai import OpenAI as Client
            else:
                from anthropic import Anthropic as Client
            with Client(api_key=api_key, base_url=OFFICIAL_ENDPOINTS[provider], max_retries=0) as client:
                for index, arm in enumerate(pending, 1):
                    print(f"Running {index}/{len(pending)}: {arm.item.key}", flush=True)
                    try:
                        outcome = evaluate(provider, model, effort or "medium", arm, client)
                        completed[arm.item.key] = {
                            **expected[arm.item.key], **outcome, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        save_results(dest, completed.values())
                    except Exception as exc:
                        print(f"EVALUATION INTERRUPTED ({type(exc).__name__}). Inspect usage before resuming.",
                              file=sys.stderr)
                        return 3
    except (OSError, ValueError, ImportError) as exc:
        print(f"RUN PRECHECK FAILED: {exc if type(exc) is ValueError else type(exc).__name__}", file=sys.stderr)
        return 2
    print(f"Completed: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
