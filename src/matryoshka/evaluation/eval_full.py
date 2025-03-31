import argparse
import csv
import json
import os
import re

import dill

from ..genai_api.api import Caller, backend_choices, get_backend
from ..utils.logging import setup_logger
from ..utils.OCSF import OCSFSchemaClient
from ..utils.structured_log import TreeEditor


def compare_results(baseline_lines, raw_results_path):
    precision, recall, count = 0, 0, 0
    target_lines_per_query = {}
    for key, lines in baseline_lines.items():
        target_file_name = os.path.join(
            raw_results_path, key.split("_")[0] + "_" + key.split("_")[2]
        )
        if not os.path.exists(target_file_name):
            continue
        count += 1

        with open(target_file_name, "r", encoding="utf-8") as f:
            raw_lines = [
                re.sub(r"\s+", " ", line.strip()).strip()
                for line in f.readlines()
                if line.strip()
            ]

        target_lines = [
            re.sub(r"\s+", " ", line.strip()).strip()
            for line in lines
            if line.strip()
        ]

        local_precision = len(set(target_lines) & set(raw_lines)) / len(
            target_lines
        )
        local_recall = len(set(target_lines) & set(raw_lines)) / len(raw_lines)

        print(
            f"Query: {key}. "
            f"Precision: {local_precision:.4f} "
            f"Recall: {local_recall:.4f} "
        )
        precision += local_precision
        recall += local_recall
        target_lines_per_query[key] = raw_lines

    if count:
        return precision / count, recall / count, target_lines_per_query
    else:
        return 0, 0, {}


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
        default=None,
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
        nargs="+",
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
        choices=["syntax", "schemas", "mapping", "end_to_end", "stats"],
        help="Stage of the evaluation",
    )
    parser.add_argument(
        "--backend",
        choices=backend_choices(),
        default=backend_choices()[0],
        help="Select the backend to use",
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        help="number of threads to use for openai",
        default=16,
    )
    parser.add_argument(
        "--raw_results_path",
        type=str,
        help="Path to a directory to containing raw results to eval against (for querying)",
        default=None,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path to the cache directory",
        default=".cache/",
    )
    parser.add_argument(
        "--save_to_file",
        type=str,
        help="Path to save baseline query lines to a file",
        default=None,
    )

    # Parse the arguments
    args = parser.parse_args()

    if not args.config_file and not args.baseline and args.stage != "stats":
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
        if not args.baseline_query_path:
            args.baseline_query_path = config["query_path"]
    setup_logger()

    # Setup caller and OCSF client
    print("Setting up caller and OCSF client")
    caller = Caller(
        args.thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )
    ocsf_client = OCSFSchemaClient(
        caller, saved_path=os.path.join(args.cache_dir, "OCSF")
    )

    # Special handling for stats
    if args.stage == "stats":
        print("Loading parsers")
        with open(args.target, "rb") as f:
            target_parser = dill.load(f)
        print("Target parsing")
        target = TreeEditor(
            target_parser,
            [],
            lines_per_template=-1,
            caller=caller,
            client=ocsf_client,
            query_file=args.target_query_path,
            run_parse=False,
        )

        stats = target.tree_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        return

    # Load parsers
    print("Loading parsers")
    with open(args.baseline, "rb") as f:
        baseline_parser = dill.load(f)
    with open(args.target, "rb") as f:
        target_parser = dill.load(f)

    # Load lines
    print("Loading lines")
    with open(args.log_file, "r", encoding="utf-8") as f:
        lines = f.read()
    all_lines = re.split("\n", lines)
    if args.file_percent < 1:
        all_lines = all_lines[: int(len(all_lines) * args.file_percent)]
    all_lines = [re.sub(r"\s", " ", line).strip() for line in all_lines if line]

    # Parse the lines
    print("Target parsing")
    target = TreeEditor(
        target_parser,
        all_lines,
        lines_per_template=-1,
        caller=caller,
        client=ocsf_client,
        query_file=args.target_query_path,
    )

    print("Baseline parsing")
    baseline = TreeEditor(
        baseline_parser,
        all_lines,
        lines_per_template=-1,
        caller=caller,
        client=ocsf_client,
        query_file=args.baseline_query_path,
    )

    # Evaluate the parsers
    if args.stage == "syntax":
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
        precision, recall, baseline_lines, target_lines = (
            baseline.end_to_end_score(target, save_to_file=args.save_to_file)
        )
        precision_ocsf, recall_ocsf, _, _ = baseline.end_to_end_score(
            target, ocsf=True
        )
        # Save baseline and target lines
        for key, lines in baseline_lines.items():
            with open(
                os.path.join(
                    os.path.dirname(args.output),
                    f"baseline_lines_query_{key}.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(lines))
        for key, lines in target_lines.items():
            with open(
                os.path.join(
                    os.path.dirname(args.output),
                    f"target_lines_query_{key}.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("\n".join(lines))
        if not args.raw_results_path:
            print(
                f"End-to-End Evaluation Results:\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"Precision (OCSF): {precision_ocsf:.4f}\n"
                f"Recall (OCSF): {recall_ocsf:.4f}"
            )
            with open(args.output, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["type", "precision", "recall"])
                writer.writerow(["custom", precision, recall])
                writer.writerow(["ocsf", precision_ocsf, recall_ocsf])
        else:
            precision_substring, recall_substring, substring_lines = (
                compare_results(baseline_lines, args.raw_results_path)
            )
            print(
                f"End-to-End Evaluation Results:\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"Precision (OCSF): {precision_ocsf:.4f}\n"
                f"Recall (OCSF): {recall_ocsf:.4f}\n"
                f"Precision (substring): {precision_substring:.4f}\n"
                f"Recall (substring): {recall_substring:.4f}"
            )
            for key, lines in substring_lines.items():
                with open(
                    os.path.join(
                        os.path.dirname(args.output),
                        f"substring_lines_query_{key}.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("\n".join(lines))
            with open(args.output, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["type", "precision", "recall"])
                writer.writerow(["custom", precision, recall])
                writer.writerow(["ocsf", precision_ocsf, recall_ocsf])
                writer.writerow(
                    ["substring", precision_substring, recall_substring]
                )

    # Clean up
    del caller
