import argparse
import json
import os
import pdb
import sys

import dill

from .syntax.run import VariableParser
from .tools.api import Caller
from .tools.classes import Parser
from .tools.logging import setup_logger

# pdb.Pdb().set_trace()


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "--existing_templates", type=str, help="Path to the existing templates"
    )
    parser.add_argument(
        "--existing_parser", type=str, help="Path to the existing parser"
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output file", default="output/"
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
        default=1,
    )
    # Parse the arguments
    args = parser.parse_args()

    if not args.config_file and not args.log_file:
        raise ValueError("Please provide a config file or a log file")

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        args.log_file = config["data_path"]
        args.existing_templates = config["example_path"]
        args.output = config["results_path"] + "_parser"
    setup_logger()

    caller = Caller(args.openai_thread_count, distribute_parallel_requests=True)
    template_example = (
        args.existing_templates if args.existing_templates else None
    )

    os.makedirs(args.output, exist_ok=True)
    if not args.existing_parser:
        parser = VariableParser(
            caller=caller,
            init_templates=template_example,
            debug_folder=args.output,
        )
    else:
        with open(args.existing_parser, "rb") as f:
            parser = dill.load(f)
        parser.init_caller(caller)

    parser.parse(args.log_file, percentage=args.file_percent)

    with open(os.path.join(args.output, "parser.dill"), "wb") as f:
        dill.dump(
            Parser(
                parser.tree,
                parser.values,
                parser.entries_per_template,
                parser.naive_distance,
            ),
            f,
        )
