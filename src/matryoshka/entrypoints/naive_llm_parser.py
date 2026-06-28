import argparse
import json
import os

from ..genai_api.api import Caller, backend_choices, get_backend
from ..naive_llm_parser.run import parse_log_file
from ..utils.logging import get_logger, setup_logger


def _read_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def main():
    parser = argparse.ArgumentParser(
        description="Naive LLM-powered, line-by-line log parser."
    )
    parser.add_argument("--log_file", required=True, help="Path to log file")
    parser.add_argument(
        "--output",
        type=str,
        default="output/naive_llm_parsed.json",
        help="Where to write the parsed output (JSON).",
    )
    parser.add_argument(
        "--backend",
        choices=backend_choices(),
        default=backend_choices()[0],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        default=8,
        help="Number of parallel worker threads for LLM calls",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name to query",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to request from the model",
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        default=1.0,
        help="Fraction of the file to parse (0-1]",
    )
    parser.add_argument(
        "--line_limit",
        type=int,
        default=-1,
        help="Optional hard limit on number of lines to parse",
    )
    parser.add_argument(
        "--include_empty_lines",
        action="store_true",
        help="Process empty/whitespace lines as well",
    )
    parser.add_argument(
        "--system_prompt_file",
        type=str,
        default=None,
        help="Path to a file containing a custom system prompt",
    )
    args = parser.parse_args()

    setup_logger()

    system_prompt = (
        _read_system_prompt(args.system_prompt_file)
        if args.system_prompt_file
        else None
    )

    caller = Caller(
        parallelism=args.thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )

    parsed = parse_log_file(
        log_file=args.log_file,
        caller=caller,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        file_percent=args.file_percent,
        line_limit=args.line_limit if args.line_limit >= 0 else None,
        include_empty=args.include_empty_lines,
        system_prompt=system_prompt,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"parsed": parsed}, handle, indent=2, ensure_ascii=True)

    get_logger().info(
        "Parsed %d lines from %s into %s",
        len(parsed),
        args.log_file,
        args.output,
    )


if __name__ == "__main__":
    main()
