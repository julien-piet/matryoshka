import argparse
import json
import os
import re
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 10)

import dill

from ..genai_api.api import Caller, backend_choices, get_backend
from ..utils.logging import setup_logger
from ..utils.OCSF import OCSFSchemaClient
from ..utils.structured_log import TreeEditor


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument(
        "--baseline", type=str, help="Path to the target parser"
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output results file",
        default=None,
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=1.0,
    )
    parser.add_argument(
        "--backend",
        choices=backend_choices(),
        default=backend_choices()[0],
        help="Select the backend to use",
    )
    parser.add_argument(
        "--openai_thread_count",
        type=int,
        help="number of threads to use for openai",
        default=16,
    )
    # Parse the arguments
    args = parser.parse_args()

    if not args.config_file and not args.baseline:
        raise ValueError(
            "Please provide a config file or a path to a baseline parser"
        )

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not args.log_file:
            args.log_file = config["data_path"]
        if not args.baseline:
            args.baseline = config["baseline_path"]
        if not args.output:
            os.makedirs(config["results_path"], exist_ok=True)
            args.output = os.path.join(
                config["results_path"], f"results_{args.stage}.tsv"
            )
    setup_logger()

    # Load parsers
    print("...Loading parsers")

    with open(args.baseline, "rb") as f:
        baseline_parser = dill.load(f)

    baseline_parser.tree.reset_regex()

    with open(args.baseline, "wb") as f:
        dill.dump(baseline_parser, f)

    for val in baseline_parser.var_mapping.values():
        if "mappings" in val.__dict__:
            if not val.mappings:
                val.mapping = {}
            else:
                breakpoint()

    # Setup caller and OCSF client
    print("...Setting up caller and OCSF client")
    caller = Caller(
        args.openai_thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )
    ocsf_client = OCSFSchemaClient(caller=caller, saved_path=".cache/OCSF")

    # Load lines
    print("...Loading lines")
    with open(args.log_file, "r", encoding="utf-8") as f:
        lines = f.read()
    all_lines = re.split("\n", lines)
    if args.file_percent < 1:
        all_lines = all_lines[: int(len(all_lines) * args.file_percent)]
    all_lines = [re.sub(r"\s", " ", line).strip() for line in all_lines if line]

    # Parse the lines
    print("...Baseline parsing")
    baseline = TreeEditor(
        baseline_parser,
        all_lines,
        lines_per_template=-1,
        caller=caller,
        client=ocsf_client,
        output=args.output,
    )

    # Save the results
    baseline.save_parser_as_json(path=args.output)

    # Clean up
    del caller
