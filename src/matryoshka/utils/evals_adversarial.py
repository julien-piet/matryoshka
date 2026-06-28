"""Adversarial parser robustness evaluations.

This module runs a two-pass evaluation:
1) Parse original log lines with an existing parser.
2) Perturb lines with an adversarial strategy constrained to one variable span.
3) Re-parse perturbed lines and report template stability metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dill
import regex as regex_re

from ..classes import Match, Module, Parser, Template

ZWSP = "\u200b"
FAKE_LOG_FRAGMENT = "FAKE_LOG login successful user=admin"
HOMOGLYPH_MAP = {
    "a": "\u0430",  # Cyrillic small a
    "e": "\u0435",  # Cyrillic small e
    "o": "\u043e",  # Cyrillic small o
    "p": "\u0440",  # Cyrillic small er
    "c": "\u0441",  # Cyrillic small es
    "x": "\u0445",  # Cyrillic small ha
    "y": "\u0443",  # Cyrillic small u
}


@dataclass
class ParsedLine:
    """Represents parser output for one line."""

    parsed: bool
    template_id: Optional[int]
    template_name: Optional[str]
    variable_spans: List[Tuple[int, int]]


@dataclass
class EvalStats:
    """Aggregate adversarial evaluation metrics."""

    total_lines: int
    perturbed_lines: int
    unparsed_after: int
    same_template_after: int
    wrong_template_after: int

    def to_dict(self) -> Dict[str, float]:
        """Return raw counts and rates (denominator: perturbed lines)."""
        denom = self.perturbed_lines if self.perturbed_lines > 0 else 1
        return {
            "total_lines": self.total_lines,
            "perturbed_lines": self.perturbed_lines,
            "unparsed_after": self.unparsed_after,
            "same_template_after": self.same_template_after,
            "wrong_template_after": self.wrong_template_after,
            "unparsed_after_rate": self.unparsed_after / denom,
            "same_template_after_rate": self.same_template_after / denom,
            "wrong_template_after_rate": self.wrong_template_after / denom,
        }


@dataclass
class PerturbationDetail:
    """Detailed outcome for one modified line."""

    line_index: int
    original_line: str
    perturbed_line: str
    status: str
    before_template_id: Optional[int]
    before_template_name: Optional[str]
    after_template_id: Optional[int]
    after_template_name: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detail as dictionary."""
        return {
            "line_index": self.line_index,
            "original_line": self.original_line,
            "perturbed_line": self.perturbed_line,
            "status": self.status,
            "before_template_id": self.before_template_id,
            "before_template_name": self.before_template_name,
            "after_template_id": self.after_template_id,
            "after_template_name": self.after_template_name,
        }


def _install_legacy_module_aliases() -> None:
    """Install module aliases to load legacy dill parsers."""
    new_sys = {}
    for key in list(sys.modules.keys()):
        new_sys[key.replace("matryoshka", "logparser")] = sys.modules[key]
    sys.modules["logparser.tools.classes"] = sys.modules["matryoshka.classes"]
    sys.modules["logparser.tools"] = sys.modules["matryoshka.utils"]
    sys.modules["logparser.tools.OCSF"] = sys.modules["matryoshka.utils.OCSF"]
    sys.modules["logparser.tools.schema"] = sys.modules["matryoshka.classes"]
    for key, value in new_sys.items():
        sys.modules[key] = value


def load_parser(parser_file: str) -> Parser:
    """Load a parser from dill or JSON."""
    if parser_file.endswith(".dill") or parser_file.endswith(".pkl"):
        _install_legacy_module_aliases()
        with open(parser_file, "rb") as handle:
            return dill.load(handle)
    if parser_file.endswith(".json"):
        with open(parser_file, "r", encoding="utf-8") as handle:
            return Parser.load_from_json(json.load(handle))
    raise ValueError(f"Unsupported parser format: {parser_file}")


def load_log_lines(log_file: str, file_percent: float = 1.0) -> List[str]:
    """Load log lines from file without normalizing content."""
    all_lines = Module.load_log(log_file).splitlines()
    if file_percent < 1.0:
        cutoff = int(len(all_lines) * file_percent)
        all_lines = all_lines[:cutoff]
    return all_lines


def _extract_variable_spans_from_regex(
    line: str, template: Template
) -> List[Tuple[int, int]]:
    """Extract variable spans by matching template regex groups on the line."""
    template.generate_regex()
    match = regex_re.fullmatch(template.regex, line)
    if not match:
        return []

    spans: List[Tuple[int, int]] = []
    for element in template.elements:
        if not element.is_variable():
            continue
        group_name = f"var_{element.id}"
        if group_name not in match.groupdict():
            continue
        start, end = match.span(group_name)
        if start is None or end is None or end <= start:
            continue
        spans.append((start, end))
    return spans


def _extract_variable_spans_from_tokens(
    line: str, match_obj: Match
) -> List[Tuple[int, int]]:
    """Fallback span extraction from matched tokens using sequential alignment."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    elements = match_obj.elements

    for idx, element in enumerate(elements):
        value = element.value or ""
        next_constant = ""
        for future in elements[idx + 1 :]:
            if not future.is_variable() and future.value:
                next_constant = future.value
                break

        if element.is_variable():
            start = cursor
            if value and line.startswith(value, cursor):
                end = cursor + len(value)
            elif value:
                found = line.find(value, cursor)
                if found >= 0:
                    start = found
                    end = found + len(value)
                elif next_constant:
                    nxt = line.find(next_constant, cursor)
                    end = nxt if nxt >= 0 else len(line)
                else:
                    end = len(line)
            elif next_constant:
                nxt = line.find(next_constant, cursor)
                end = nxt if nxt >= 0 else len(line)
            else:
                end = len(line)

            if end > start:
                spans.append((start, end))
                cursor = end
            continue

        if value and line.startswith(value, cursor):
            cursor += len(value)
            continue
        if value:
            found = line.find(value, cursor)
            if found >= 0:
                cursor = found + len(value)

    return spans


def parse_line_with_spans(parser: Parser, line: str) -> ParsedLine:
    """Parse one line and return template metadata and variable spans."""
    matched, candidates = parser.tree.match(line)
    if not matched or not candidates:
        return ParsedLine(False, None, None, [])

    chosen = candidates[0]
    template_id = chosen.template_id
    if template_id is None:
        return ParsedLine(False, None, None, [])

    template = parser.tree.gen_template(template_id)
    spans = _extract_variable_spans_from_regex(line, template)
    if not spans:
        spans = _extract_variable_spans_from_tokens(line, chosen.matches)

    return ParsedLine(
        parsed=True,
        template_id=template_id,
        template_name=template.convert_to_wildcard_template(),
        variable_spans=spans,
    )


def choose_variable_span(spans: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Choose one variable span, preferring the longest."""
    if not spans:
        return None
    return max(spans, key=lambda item: (item[1] - item[0], -item[0]))


def _clamp_injection_pos(text: str, mode: str, rng: random.Random) -> int:
    if not text:
        return 0
    if mode == "start":
        return 0
    if mode == "end":
        return len(text)
    if mode == "random":
        return rng.randint(0, len(text))
    return len(text) // 2


def _apply_newline_injection(
    value: str,
    rng: random.Random,
    newline_mode: str = "middle",
    crlf: bool = False,
) -> str:
    pos = _clamp_injection_pos(value, newline_mode, rng)
    sep = "\r\n" if crlf else "\n"
    return value[:pos] + sep + FAKE_LOG_FRAGMENT + value[pos:]


def _apply_unicode_zero_width(
    value: str,
    rng: random.Random,
    every_k: int = 5,
    mode: str = "every_k",
) -> str:
    if not value:
        return value
    if mode == "random":
        pos = rng.randint(0, len(value))
        return value[:pos] + ZWSP + value[pos:]

    if every_k <= 0:
        every_k = 5
    out = []
    for idx, char in enumerate(value, start=1):
        out.append(char)
        if idx % every_k == 0 and idx < len(value):
            out.append(ZWSP)
    return "".join(out)


def _apply_unicode_homoglyph(
    value: str, rng: random.Random, max_replacements: int = 3
) -> str:
    candidates = [
        idx for idx, char in enumerate(value) if char.lower() in HOMOGLYPH_MAP
    ]
    if not candidates or max_replacements <= 0:
        return value

    count = min(max_replacements, len(candidates))
    replace_positions = set(rng.sample(candidates, count))
    chars = list(value)
    for pos in replace_positions:
        original = chars[pos]
        mapped = HOMOGLYPH_MAP.get(original.lower(), original)
        chars[pos] = mapped
    return "".join(chars)


def _apply_delimiter_injection_kv(
    line: str,
    span: Tuple[int, int],
    value: str,
    delimiter_payload: str = "role=admin",
) -> str:
    start, end = span
    lo = max(0, start - 20)
    hi = min(len(line), end + 20)
    window = line[lo:hi]
    has_equal = "=" in window
    has_delim = any(token in window for token in [" ", ";", ","])
    use_semicolon = ";" in window or ";" in line
    injection = f"; {delimiter_payload}" if use_semicolon else f" {delimiter_payload}"
    if has_equal and has_delim:
        return value + injection
    return value + injection


def _apply_quote_escape_injection(value: str) -> str:
    if '\\"' in value:
        return value.replace('\\"', '"')
    if '"' in value:
        return value.replace('"', '\\"')

    if not value:
        return '"\\""'
    pos = max(1, len(value) // 2)
    inner = value[:pos] + '\\"' + value[pos:]
    return f'"{inner}"'


def _apply_truncation(
    line: str, span: Tuple[int, int], n_chars: int = 10
) -> Optional[str]:
    if n_chars <= 0 or n_chars >= len(line):
        return None
    trunc_start = len(line) - n_chars
    # Respect "variable-only edits": only truncate if removed tail is within selected variable.
    if trunc_start < span[0]:
        return None
    if trunc_start > span[1]:
        return None
    truncated = line[:-n_chars]
    return truncated if truncated else None


def perturb_line(
    line: str,
    parsed: ParsedLine,
    strategy: str,
    rng: random.Random,
    newline_crlf: bool = False,
    newline_mode: str = "middle",
    zwsp_every_k: int = 5,
    zwsp_mode: str = "every_k",
    homoglyph_max_replacements: int = 3,
    kv_payload: str = "role=admin",
    truncation_n: int = 10,
) -> Tuple[str, bool]:
    """Perturb a line using one strategy and one selected variable span."""
    if not parsed.parsed:
        return line, False

    chosen_span = choose_variable_span(parsed.variable_spans)
    if chosen_span is None:
        return line, False

    start, end = chosen_span
    if not (0 <= start < end <= len(line)):
        return line, False

    original_value = line[start:end]
    new_value = original_value

    if strategy == "newline_injection":
        new_value = _apply_newline_injection(
            original_value,
            rng=rng,
            newline_mode=newline_mode,
            crlf=newline_crlf,
        )
    elif strategy == "unicode_zero_width":
        new_value = _apply_unicode_zero_width(
            original_value, rng=rng, every_k=zwsp_every_k, mode=zwsp_mode
        )
    elif strategy == "unicode_homoglyph":
        new_value = _apply_unicode_homoglyph(
            original_value,
            rng=rng,
            max_replacements=homoglyph_max_replacements,
        )
    elif strategy == "delimiter_injection_kv":
        new_value = _apply_delimiter_injection_kv(
            line=line,
            span=chosen_span,
            value=original_value,
            delimiter_payload=kv_payload,
        )
    elif strategy == "quote_escape_injection":
        new_value = _apply_quote_escape_injection(original_value)
    elif strategy == "truncation":
        truncated = _apply_truncation(line, chosen_span, n_chars=truncation_n)
        if truncated is None:
            return line, False
        return truncated, truncated != line
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if new_value == original_value:
        return line, False

    perturbed = line[:start] + new_value + line[end:]
    return perturbed, perturbed != line


def run_adversarial_eval(
    parser: Parser,
    lines: Sequence[str],
    strategy: str,
    seed: int = 0,
    newline_crlf: bool = False,
    newline_mode: str = "middle",
    zwsp_every_k: int = 5,
    zwsp_mode: str = "every_k",
    homoglyph_max_replacements: int = 3,
    kv_payload: str = "role=admin",
    truncation_n: int = 10,
) -> Tuple[EvalStats, List[str], List[PerturbationDetail]]:
    """Run two-pass adversarial eval and return metrics + perturbed lines."""
    rng = random.Random(seed)

    pass1: List[ParsedLine] = [parse_line_with_spans(parser, line) for line in lines]
    perturbed_lines: List[str] = []
    perturbed_indices: List[int] = []

    for idx, line in enumerate(lines):
        new_line, changed = perturb_line(
            line=line,
            parsed=pass1[idx],
            strategy=strategy,
            rng=rng,
            newline_crlf=newline_crlf,
            newline_mode=newline_mode,
            zwsp_every_k=zwsp_every_k,
            zwsp_mode=zwsp_mode,
            homoglyph_max_replacements=homoglyph_max_replacements,
            kv_payload=kv_payload,
            truncation_n=truncation_n,
        )
        perturbed_lines.append(new_line)
        if changed:
            perturbed_indices.append(idx)

    unparsed_after = 0
    same_template_after = 0
    wrong_template_after = 0
    details: List[PerturbationDetail] = []

    for idx in perturbed_indices:
        after = parse_line_with_spans(parser, perturbed_lines[idx])
        before = pass1[idx]
        if not after.parsed:
            unparsed_after += 1
            status = "unparsed"
        elif after.template_id == before.template_id:
            same_template_after += 1
            status = "same_template"
        else:
            wrong_template_after += 1
            status = "misparsed"

        details.append(
            PerturbationDetail(
                line_index=idx,
                original_line=lines[idx],
                perturbed_line=perturbed_lines[idx],
                status=status,
                before_template_id=before.template_id,
                before_template_name=before.template_name,
                after_template_id=after.template_id,
                after_template_name=after.template_name,
            )
        )

    stats = EvalStats(
        total_lines=len(lines),
        perturbed_lines=len(perturbed_indices),
        unparsed_after=unparsed_after,
        same_template_after=same_template_after,
        wrong_template_after=wrong_template_after,
    )
    return stats, perturbed_lines, details


def main() -> None:
    """CLI entrypoint for adversarial evaluations."""
    cli_parser = argparse.ArgumentParser(
        description="Run adversarial perturbation evaluations on a parser/log pair."
    )
    cli_parser.add_argument(
        "--parser_file", type=str, required=True, help="Path to parser (.dill/.json)"
    )
    cli_parser.add_argument("--log_file", type=str, required=True, help="Path to log file")
    cli_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=[
            "newline_injection",
            "unicode_zero_width",
            "unicode_homoglyph",
            "delimiter_injection_kv",
            "quote_escape_injection",
            "truncation",
        ],
        help="Adversarial strategy to apply",
    )
    cli_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output file for metrics",
    )
    cli_parser.add_argument(
        "--details_output",
        type=str,
        default=None,
        help="Optional JSON output file for per-modified-line outcomes",
    )
    cli_parser.add_argument(
        "--file_percent",
        type=float,
        default=1.0,
        help="Fraction of the log file to evaluate",
    )
    cli_parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Strategy knobs
    cli_parser.add_argument(
        "--newline_crlf",
        action="store_true",
        default=False,
        help="Use CRLF instead of LF in newline_injection",
    )
    cli_parser.add_argument(
        "--newline_mode",
        type=str,
        choices=["start", "middle", "end", "random"],
        default="middle",
        help="Injection position mode for newline_injection",
    )
    cli_parser.add_argument(
        "--zwsp_every_k",
        type=int,
        default=5,
        help="Insert zero-width spaces every k chars in unicode_zero_width mode",
    )
    cli_parser.add_argument(
        "--zwsp_mode",
        type=str,
        choices=["every_k", "random"],
        default="every_k",
        help="unicode_zero_width insertion mode",
    )
    cli_parser.add_argument(
        "--homoglyph_max_replacements",
        type=int,
        default=3,
        help="Max ASCII->homoglyph replacements for unicode_homoglyph",
    )
    cli_parser.add_argument(
        "--kv_payload",
        type=str,
        default="role=admin",
        help="Payload used by delimiter_injection_kv",
    )
    cli_parser.add_argument(
        "--truncation_n",
        type=int,
        default=10,
        help="Characters to truncate for truncation strategy",
    )

    args = cli_parser.parse_args()

    parser = load_parser(args.parser_file)
    lines = load_log_lines(args.log_file, file_percent=args.file_percent)

    stats, _, details = run_adversarial_eval(
        parser=parser,
        lines=lines,
        strategy=args.strategy,
        seed=args.seed,
        newline_crlf=args.newline_crlf,
        newline_mode=args.newline_mode,
        zwsp_every_k=args.zwsp_every_k,
        zwsp_mode=args.zwsp_mode,
        homoglyph_max_replacements=args.homoglyph_max_replacements,
        kv_payload=args.kv_payload,
        truncation_n=args.truncation_n,
    )
    metrics = stats.to_dict()

    # Short stderr summary, no per-line logs.
    print(
        (
            f"[adversarial_eval] strategy={args.strategy} "
            f"total={metrics['total_lines']} "
            f"perturbed={metrics['perturbed_lines']} "
            f"unparsed_after={metrics['unparsed_after']} "
            f"same_template_after={metrics['same_template_after']} "
            f"wrong_template_after={metrics['wrong_template_after']}"
        ),
        file=sys.stderr,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=True)
    if args.details_output:
        with open(args.details_output, "w", encoding="utf-8") as handle:
            json.dump(
                [detail.to_dict() for detail in details],
                handle,
                indent=2,
                ensure_ascii=True,
            )


if __name__ == "__main__":
    main()
