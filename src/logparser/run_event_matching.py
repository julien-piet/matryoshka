import argparse
import json
import os
import pdb
import sys

import dill

from .semantics.mapping.identify_event import IdentifyEvent
from .tools.api import Caller
from .tools.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Add types to variables.")
    parser.add_argument("--parser", type=str, help="Path to the parser")
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--few-shot-len",
        type=int,
        default=3,
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
                if args.golden:
                    args.parser = (
                        config["results_path"] + "_types_fixed/parser.dill"
                    )
                else:
                    args.parser = config["results_path"] + "_types/parser.dill"

            if not args.output:
                if args.golden:
                    args.output = config["results_path"] + "_events_fixed"
                else:
                    args.output = config["results_path"] + "_events"
    setup_logger()

    caller = Caller(args.openai_thread_count)
    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    os.makedirs(args.output, exist_ok=True)

    matcher = IdentifyEvent(caller, parser, output_dir=args.output)
    matcher(None)

    del caller
