"""Build-time profiling support for USE_CMAKE_INSTRUMENTATION.

This script backs cmake/Instrumentation.cmake in two roles:

  collect --out-dir DIR INDEX
    Callback invoked by CMake's instrumentation hooks with an index file.
    Copies the snippet data and CMake's Google trace (trace.json, viewable
    at https://ui.perfetto.dev) into DIR/runs/<index>/, merges any pending
    time-step events, and writes summary.txt. The summary is also printed.

  time-step --name NAME --out-dir DIR -- CMD...
    Runs CMD, recording its wall time as an extra profile event. Used to
    time steps that are otherwise folded into another build command, such
    as POST_BUILD steps that Ninja appends to the link command.

Only the standard library is used; CMake invokes this with whatever Python
configured the build.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    Snippet = dict[str, Any]

# Roles describing a whole build phase rather than an individual command.
PHASE_ROLES = ("configure", "generate", "build", "cmakeBuild", "cmakeInstall")

# Test executions picked up from ctest runs; not build profiling data.
IGNORED_ROLES = ("test", "ctest")


def snippet_name(snippet: Snippet) -> str:
    role = snippet.get("role", "unknown")
    if role == "compile" and snippet.get("source"):
        return Path(snippet["source"]).name
    if snippet.get("outputs"):
        return Path(snippet["outputs"][0]).name
    return role


def fmt_seconds(ms: float) -> str:
    return f"{ms / 1000:.1f}s"


def build_summary(snippets: list[Snippet], steps: list[Snippet], hook: str) -> str:
    by_role: dict[str, list[Snippet]] = {}
    for s in snippets:
        by_role.setdefault(s.get("role", "unknown"), []).append(s)

    lines = [f"== build profile ({hook}) =="]
    for role in PHASE_ROLES:
        for s in by_role.get(role, []):
            lines.append(f"{role}: {fmt_seconds(s['duration'])}")

    installs = by_role.get("install", [])
    if installs:
        total = sum(s["duration"] for s in installs)
        lines.append(f"install: {fmt_seconds(total)} ({len(installs)} script(s))")

    for step in steps:
        result = "" if step.get("result") == 0 else f" (exit {step.get('result')})"
        lines.append(f"step {step['name']}: {fmt_seconds(step['duration'])}{result}")

    targets: dict[str, dict[str, float]] = {}
    for s in by_role.get("compile", []) + by_role.get("link", []):
        stats = targets.setdefault(
            s.get("target") or "<unknown>", {"files": 0, "compile": 0.0, "link": 0.0}
        )
        if s["role"] == "compile":
            stats["files"] += 1
            stats["compile"] += s["duration"]
        else:
            stats["link"] += s["duration"]
    if targets:
        lines.append("")
        lines.append(f"{'target':<40} {'files':>6} {'compile':>10} {'link':>8}")
        ranked = sorted(
            targets.items(), key=lambda kv: -(kv[1]["compile"] + kv[1]["link"])
        )
        for name, t in ranked:
            lines.append(
                f"{name:<40} {t['files']:>6} {fmt_seconds(t['compile']):>10}"
                f" {fmt_seconds(t['link']):>8}"
            )
        lines.append("(compile is CPU time summed over parallel jobs, link is wall"
                     " time and includes POST_BUILD commands)")

    sections = (("custom", "slowest custom commands"), ("compile", "slowest compiles"))
    for role, title in sections:
        slowest = sorted(by_role.get(role, []), key=lambda s: -s["duration"])[:10]
        if slowest:
            lines.append("")
            lines.append(f"{title}:")
            lines.extend(
                f"  {fmt_seconds(s['duration']):>8}  {snippet_name(s)}" for s in slowest
            )
    return "\n".join(lines)


def cmd_collect(args: argparse.Namespace) -> int:
    with open(args.index) as f:
        index = json.load(f)
    data_dir = Path(index["dataDir"])
    out_dir = Path(args.out_dir)

    snippets: list[Snippet] = []
    for rel in index["snippets"]:
        # An interrupted command can leave a truncated/empty snippet behind
        # that gets picked up by the next indexing; skip it rather than
        # failing the whole callback.
        try:
            with open(data_dir / rel) as f:
                snippet = json.load(f)
        except json.JSONDecodeError:
            print(f"build_instrumentation: skipping bad snippet {rel}", file=sys.stderr)
            continue
        if snippet.get("role") not in IGNORED_ROLES:
            snippets.append(snippet)

    steps_dir = out_dir / "steps"
    pending_steps = sorted(steps_dir.glob("*.json")) if steps_dir.is_dir() else []
    steps = [json.loads(p.read_text()) for p in pending_steps]

    if not snippets and not steps:
        return 0

    # Index file names have limited timestamp resolution and can repeat
    # within one cmake run; keep every invocation's data separate.
    base_dir = out_dir / "runs" / Path(args.index).stem
    run_dir, n = base_dir, 2
    while run_dir.exists():
        run_dir = base_dir.with_name(f"{base_dir.name}-{n}")
        n += 1
    raw_dir = run_dir / "data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.index, run_dir / Path(args.index).name)
    for rel in index["snippets"]:
        dest = raw_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_dir / rel, dest)
    if index.get("trace"):
        # Strip the source-root prefix so trace event names are relative
        # (paths under the build dir become build/...).
        trace = (data_dir / index["trace"]).read_text()
        trace = trace.replace(args.src_dir.rstrip("/") + "/", "")
        (run_dir / "trace.json").write_text(trace)
    for path in pending_steps:
        shutil.move(str(path), raw_dir / path.name)

    summary = build_summary(snippets, steps, index.get("hook", "manual"))
    (run_dir / "summary.txt").write_text(summary + "\n")
    print(f"{summary}\n\nfull data in {run_dir} (open trace.json in ui.perfetto.dev)")
    return 0


def cmd_time_step(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    start = time.time()
    result = subprocess.call(command)
    event = {
        "name": args.name,
        "timeStart": int(start * 1000),
        "duration": int((time.time() - start) * 1000),
        "result": result,
        "command": shlex.join(command),
    }
    steps_dir = Path(args.out_dir) / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    stamp = steps_dir / f"step-{event['timeStart']}-{os.getpid()}.json"
    stamp.write_text(json.dumps(event))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect", help="instrumentation index callback")
    collect.add_argument("--out-dir", required=True)
    collect.add_argument("--src-dir", required=True)
    collect.add_argument("index", help="index file path, appended by CMake")
    collect.set_defaults(func=cmd_collect)

    step = sub.add_parser("time-step", help="run and time a build step")
    step.add_argument("--name", required=True)
    step.add_argument("--out-dir", required=True)
    step.add_argument("command", nargs=argparse.REMAINDER)
    step.set_defaults(func=cmd_time_step)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
