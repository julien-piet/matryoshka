import argparse
import json
import os

import dill

from .semantics.mapping.normalize_attributes import NormAttributes
from .tools.api import BACKEND, Caller
from .tools.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Add types to variables.")
    parser.add_argument(
        "--parser", type=str, help="Path to the parser", default=None
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--few-shot-len",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND,
        default=BACKEND[0],
        help="Select the backend to use",
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output file", default=None
    )
    parser.add_argument(
        "--openai_thread_count",
        type=int,
        help="number of threads to use for openai",
        default=16,
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=0.1,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Backend model to use",
        default="gemini-1.5-flash",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description of the log file (pointer to file containing the description)",
        default=None,
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        default=False,
    )
    # Parse the arguments
    args = parser.parse_args()

    if not args.config_file and not args.parser:
        raise ValueError("Please provide a config file or a parser path")

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not args.parser:
            if not args.golden:
                args.parser = (
                    config["results_path"] + "_schemas/results/saved.dill"
                )
            else:
                args.parser = (
                    config["results_path"] + "_schemas_fixed/results/saved.dill"
                )
        if not args.output:
            if not args.golden:
                args.output = config["results_path"] + "_schemas_normalized"
            else:
                args.output = (
                    config["results_path"] + "_schemas_normalized_fixed"
                )
        if not args.description:
            args.description = config["description_path"]
    setup_logger()

    caller = Caller(
        args.openai_thread_count,
        backend=args.backend,
        distribute_parallel_requests=True,
    )
    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    os.makedirs(args.output, exist_ok=True)

    # Load descrpition:
    if args.description:
        with open(args.description, "r", encoding="utf-8") as f:
            log_desc = f.read().strip()
    else:
        log_desc = "No description provided."

    matcher = NormAttributes(
        caller,
        parser,
        log_desc,
        output_dir=args.output,
        model=args.model,
        few_shot_len=args.few_shot_len,
    )
    matcher(None)

    del caller
