import argparse
import csv
import json
import os
import re

import dill
from tqdm import tqdm

from logparser.tools.OCSF import (
    OCSFMapping,
    OCSFSchemaClient,
    VariableSemantics,
)

from ..tools.api import BACKEND, Caller
from ..tools.classes import Parser, Value
from ..tools.embedding import NaiveDistance
from ..tools.logging import setup_logger
from ..tools.module import Module
from ..tools.structured_log import TreeEditor


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("target", type=str, help="Path to the target parser")
    parser.add_argument(
        "--baseline", type=str, help="Path to the baseline parser"
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output results file",
        default="results.tsv",
    )
    parser.add_argument(
        "--baseline_query_path",
        type=str,
        help="Path to the baseline query file",
    )
    parser.add_argument(
        "--target_query_path",
        type=str,
        help="Path to the target query file",
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=1.0,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="parser",
        choices=["parser", "schema", "mapping", "end_to_end"],
        help="Stage of the evaluation",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND,
        default=BACKEND[0],
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
            args.output = os.path.join(
                config["results_path"], f"results_{args.stage}.tsv"
            )
        if not args.baseline_query_path:
            args.baseline_query_path = config["query_path"]
    setup_logger()

    # Setup caller and OCSF client
    print("...Setting up caller and OCSF client")
    caller = Caller(
        args.openai_thread_count,
        backend=args.backend,
        distribute_parallel_requests=True,
    )
    ocsf_client = OCSFSchemaClient(caller=caller)

    # Load parsers
    print("...Loading parsers")
    with open(args.baseline, "rb") as f:
        baseline_parser = dill.load(f)
    with open(args.target, "rb") as f:
        target_parser = dill.load(f)

    # Load lines
    print("...Loading lines")
    with open(args.log_file, "r", encoding="utf-8") as f:
        lines = f.read()
    all_lines = re.split("\n", lines)
    if args.file_percent < 1:
        all_lines = all_lines[: int(len(all_lines) * args.file_percent)]
    all_lines = [re.sub(r"\s", " ", line).strip() for line in all_lines if line]

    # Parse the lines
    print("...Target parsing")
    target = TreeEditor(
        target_parser,
        all_lines,
        lines_per_template=-1,
        caller=caller,
        client=ocsf_client,
        query_file=args.target_query_path,
    )

    print("...Baseline parsing")
    baseline = TreeEditor(
        baseline_parser,
        all_lines,
        lines_per_template=-1,
        caller=caller,
        client=ocsf_client,
        query_file=args.baseline_query_path,
    )

    # Evaluate the parsers
    if args.stage == "parser":
        group_accuracy = baseline.group_accuracy(target)
        parser_accuracy = baseline.parser_accuracy(target)
        group_similarity = baseline.group_similarity(target)
        template_similarity = baseline.template_similarity(target)
        print(
            f"Parser Evaluation Results:\n"
            f"Group Accuracy: {group_accuracy}\n"
            f"Parser Accuracy: {parser_accuracy}\n"
            f"Group Similarity: {group_similarity}\n"
            f"Template Similarity: {template_similarity}"
        )
        with open(args.output, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "group_accuracy",
                    "parser_accuracy",
                    "group_similarity",
                    "template_similarity",
                ]
            )
            writer.writerow(
                [
                    group_accuracy,
                    parser_accuracy,
                    group_similarity,
                    template_similarity,
                ]
            )
    elif args.stage == "schema":
        schema_group_accuracy = baseline.schema_group_accuracy(target)
        schema_group_similarity = baseline.schema_group_similarity(target)
        print(
            f"Schema Evaluation Results:\n"
            f"Schema Group Accuracy: {schema_group_accuracy}\n"
            f"Schema Group Similarity: {schema_group_similarity}"
        )
        with open(args.output, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                ["schema_group_accuracy", "schema_group_similarity"]
            )
            writer.writerow([schema_group_accuracy, schema_group_similarity])
    elif args.stage == "mapping":
        mapping_score = baseline.mapping_score(target)
        print(
            f"Mapping Evaluation Results:\n" f"Mapping Score: {mapping_score}"
        )
        with open(args.output, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["mapping_score"])
            writer.writerow([mapping_score])
    elif args.stage == "end_to_end":
        end_to_end_score = baseline.end_to_end_score(target)
        print(
            f"End-to-End Evaluation Results:\n"
            f"End-to-End Score: {end_to_end_score}"
        )
        with open(args.output, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["end_to_end_score"])
            writer.writerow([end_to_end_score])

    # Clean up
    del caller
