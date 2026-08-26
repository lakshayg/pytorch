#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[assignment, misc]


class ProfileSummary:
    def __init__(self, yaml_path: Path):
        self.n_functions: int = 0
        self.n_blocks: int = 0
        self.n_samples: int = 0

        with open(yaml_path) as f:
            profile = yaml.load(f, Loader=Loader)
            functions = profile.get('functions', [])
            self.n_functions = len(functions)
            for func in functions:
                blocks = func.get('blocks', [])
                self.n_blocks += len(blocks)
                for block in blocks:
                    for succ in block.get('succ', []):
                        self.n_samples += succ.get('cnt', 0)

    def __repr__(self):
        return f"ProfileSummary({self.n_functions=:_}, {self.n_blocks=:_}, {self.n_samples=:_})"


class OptimizationSummary:
    _PROFILED_RE = re.compile(
        r"BOLT-INFO: (?P<n_funcs_with_profile>\d+) out of "
        r"(?P<n_funcs_in_binary>\d+) functions in the binary"
        r" .* have non-empty execution profile"
    )
    _UNOPTIMIZED_RE = re.compile(
        r"BOLT-INFO: (?P<n_funcs_unoptimizable>\d+) functions with profile could not be optimized"
    )
    _STALE_FUNCTIONS_RE = re.compile(
        r"BOLT-WARNING: (\d+) \([^)]*of all profiled\) functions have invalid"
    )
    _STALE_SAMPLES_RE = re.compile(
        r"BOLT-WARNING: (\d+) out of (\d+) samples in the binary .*"
        r"belong to functions with invalid"
    )
    _INFERRED_RE = re.compile(
        r"BOLT-INFO: inferred profile for (\d+) \([^)]*\) functions responsible for "
        r"[0-9.]+% samples \((\d+) out of (\d+)\)"
    )
    _MATCH_RE = re.compile(
        r"BOLT-INFO: inference found an? (.*?) for [0-9.]+% of basic blocks "
        r"\((\d+) out of (\d+) stale\) responsible for [0-9.]+% samples "
        r"\((\d+) out of (\d+) stale\)"
    )
    _IGNORED_RE = re.compile(r"BOLT-INFO: profile for (\d+) objects was ignored")

    def __init__(self, log_path: Path):
        self.n_funcs_with_profile: int = 0
        self.n_funcs_in_binary: int = 0

        log = log_path.read_text(encoding="utf-8")

        if match := self._PROFILED_RE.search(log):
            self.n_funcs_with_profile = int(match.group("n_funcs_with_profile"))
            self.n_funcs_in_binary = int(match.group("n_funcs_in_binary"))

    def __repr__(self):
        return "\n".join([
            f"{self.n_funcs_with_profile=:_}",
            f"{self.n_funcs_in_binary=:_}",
        ])


@dataclass(frozen=True)
class InferenceMatch:
    blocks: int
    total_blocks: int
    samples: int
    total_samples: int


@dataclass(frozen=True)
class BoltMetrics:
    profiled_functions: int
    binary_functions: int
    unoptimized_functions: int
    stale_functions: int
    stale_samples: int
    binary_samples: int
    inferred_functions: int
    inferred_samples: int
    inferred_sample_total: int
    ignored_functions: int
    matches: dict[str, InferenceMatch]


def yaml_profile_summary(path: Path) -> ProfileSummary:
    summary = ProfileSummary()
    with open(path) as f:
        profile = yaml.load(f, Loader=Loader)
        functions = profile.get('functions', [])
        summary.n_functions = len(functions)
        for func in functions:
            blocks = func.get('blocks', [])
            summary.n_blocks += len(blocks)
            for block in blocks:
                for succ in block.get('succ', []):
                    summary.n_samples += succ.get('cnt', 0)
    return summary


def _match(pattern: re.Pattern[str], text: str, description: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"BOLT log does not contain {description}")
    return match


def _optional_count(pattern: re.Pattern[str], text: str) -> int:
    match = pattern.search(text)
    return int(match.group(1)) if match is not None else 0


def parse_bolt_log(path: Path) -> BoltMetrics:
    text = path.read_text(encoding="utf-8")
    profiled = _match(_PROFILED_RE, text, "profiled function counts")
    stale_functions = _match(_STALE_FUNCTIONS_RE, text, "stale function counts")
    stale_samples = _match(_STALE_SAMPLES_RE, text, "stale sample counts")
    inferred = _match(_INFERRED_RE, text, "inferred profile counts")
    matches = {
        match.group(1): InferenceMatch(*(int(value) for value in match.groups()[1:]))
        for match in _MATCH_RE.finditer(text)
    }
    if not matches:
        raise ValueError("BOLT log does not contain stale inference match counts")

    return BoltMetrics(
        profiled_functions=int(profiled.group(1)),
        binary_functions=int(profiled.group(2)),
        unoptimized_functions=_optional_count(_UNOPTIMIZED_RE, text),
        stale_functions=int(stale_functions.group(1)),
        stale_samples=int(stale_samples.group(1)),
        binary_samples=int(stale_samples.group(2)),
        inferred_functions=int(inferred.group(1)),
        inferred_samples=int(inferred.group(2)),
        inferred_sample_total=int(inferred.group(3)),
        ignored_functions=_optional_count(_IGNORED_RE, text),
        matches=matches,
    )


def _percent(value: int, total: int) -> float:
    return 100 * value / total if total else 0


def _sum_matches(metrics: BoltMetrics, names: tuple[str, ...]) -> InferenceMatch:
    matches = [metrics.matches[name] for name in names if name in metrics.matches]
    if not matches:
        return InferenceMatch(0, 0, 0, 0)
    return InferenceMatch(
        blocks=sum(match.blocks for match in matches),
        total_blocks=matches[0].total_blocks,
        samples=sum(match.samples for match in matches),
        total_samples=matches[0].total_samples,
    )


def _format_match(name: str, match: InferenceMatch) -> str:
    return (
        f"    {name:<10} {match.blocks:,}/{match.total_blocks:,} blocks "
        f"({_percent(match.blocks, match.total_blocks):.2f}%), "
        f"{match.samples:,}/{match.total_samples:,} samples "
        f"({_percent(match.samples, match.total_samples):.2f}%)"
    )


def format_summary(library: str, symbols: int, metrics: BoltMetrics) -> str:
    exact = _sum_matches(metrics, ("exact match", "exact pseudo probe match"))
    call = _sum_matches(metrics, ("call match",))
    loose = _sum_matches(metrics, ("loose match", "loose pseudo probe match"))
    total_blocks = max(match.total_blocks for match in metrics.matches.values())
    total_samples = max(match.total_samples for match in metrics.matches.values())
    unmatched = InferenceMatch(
        max(total_blocks - exact.blocks - call.blocks - loose.blocks, 0),
        total_blocks,
        max(total_samples - exact.samples - call.samples - loose.samples, 0),
        total_samples,
    )

    return "\n".join(
        [
            f"BOLT profile summary: {library}",
            f"  YAML profile: {symbols:,} symbols",
            "  Functions:",
            f"    profiled    {metrics.profiled_functions:,}/{symbols:,} YAML symbols "
            f"({_percent(metrics.profiled_functions, symbols):.2f}%); "
            f"{metrics.profiled_functions:,}/{metrics.binary_functions:,} binary functions "
            f"({_percent(metrics.profiled_functions, metrics.binary_functions):.2f}%)",
            f"    stale       {metrics.stale_functions:,}/{symbols:,} YAML symbols "
            f"({_percent(metrics.stale_functions, symbols):.2f}%)",
            f"    inferred    {metrics.inferred_functions:,}/{symbols:,} YAML symbols "
            f"({_percent(metrics.inferred_functions, symbols):.2f}%)",
            f"    ignored     {metrics.ignored_functions:,}/{symbols:,} YAML symbols "
            f"({_percent(metrics.ignored_functions, symbols):.2f}%)",
            f"    unoptimized {metrics.unoptimized_functions:,}/{symbols:,} YAML symbols "
            f"({_percent(metrics.unoptimized_functions, symbols):.2f}%)",
            "  Samples:",
            f"    stale       {metrics.stale_samples:,}/{metrics.binary_samples:,} "
            f"({_percent(metrics.stale_samples, metrics.binary_samples):.2f}%)",
            f"    inferred    {metrics.inferred_samples:,}/{metrics.inferred_sample_total:,} "
            f"({_percent(metrics.inferred_samples, metrics.inferred_sample_total):.2f}%)",
            "  Stale inference quality:",
            _format_match("exact", exact),
            _format_match("call", call),
            _format_match("loose", loose),
            _format_match("unmatched", unmatched),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLVM BOLT profile quality")
    parser.add_argument("profile", type=Path, help="BOLT YAML profile")
    parser.add_argument("log", type=Path, help="llvm-bolt log for the profile")
    args = parser.parse_args()

    profile_summary = ProfileSummary(args.profile)
    print(f"{profile_summary=}")

    optimization_summary = OptimizationSummary(args.log)
    print(optimization_summary)

    relevant_funcs_in_profile = round(100 * (optimization_summary.n_funcs_with_profile / profile_summary.n_functions), 1)
    print(f"relevant_profile={relevant_funcs_in_profile}%")
    # try:
    #     # metrics = parse_bolt_log(args.log)
    #     # print(format_summary(args.profile.stem, symbols, metrics))
    # except (OSError, ValueError) as error:
    #     parser.error(str(error))


if __name__ == "__main__":
    main()
