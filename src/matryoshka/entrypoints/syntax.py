import argparse
import json
import os

import dill

from ..classes import Parser
from ..genai_api.api import Caller, backend_choices, get_backend
from ..syntax.run import VariableParser
from ..utils.logging import get_logger, setup_logger


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
        "--output", type=str, help="Path to the output file", default="output/"
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        help="number of threads to use for genai",
        default=16,
    )
    parser.add_argument(
        "--backend",
        choices=backend_choices(),
        default=backend_choices()[0],
        help="Select the backend to use",
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=1,
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        help="Buffer size for processing",
        default=2500,
    )
    parser.add_argument(
        "--force_overlap",
        action="store_true",
        help="Force overlapping template resolution without asking user",
        default=False,
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        help="Frequency of checkpoints for saving progress",
        default=50,
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        help="How many lines to keep per template in memory. ",
        default=10000,
    )
    parser.add_argument(
        "--use_fewshot",
        action="store_true",
        help="Use few-shot examples for parsing",
        default=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for generation",
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--validation_model",
        type=str,
        help="Model to use for validation",
        default="gemini-2.5-pro",
    )
    parser.add_argument(
        "--save_contents",
        action="store_true",
        help="Save the contents of the log in the parser",
        default=False,
    )
    parser.add_argument(
        "--parser",
        type=str,
        help="Path to an existing parser",
        default=None,
    )
    parser.add_argument(
        "--few_shot_length",
        type=int,
        help="Number of few-shot examples to use",
        default=5,
    )
    parser.add_argument(
        "--validation_frequency",
        type=int,
        help="How often to run validation during parsing",
        default=25,
    )
    parser.add_argument(
        "--total_validation_rounds",
        type=int,
        help="Total number of validation rounds to run during parsing. Validation is most useful early on, so we can stop it after a fixed number of iterations. -1 for infinite.",
        default=-1,
    )
    parser.add_argument(
        "--no_reparse_overlaps",
        action="store_true",
        help="Disable reparse of overlapping templates when a new template is added",
        default=False,
    )
    parser.add_argument(
        "--ablation_fewshot",
        action="store_true",
        help="Ablation: Disable few-shot examples",
        default=False,
    )
    parser.add_argument(
        "--ablation_description",
        action="store_true",
        help="Ablation: Disable description embedding",
        default=False,
    )
    parser.add_argument(
        "--ablation_self_correction",
        action="store_true",
        help="Ablation: Disable self-correction during parsing",
        default=False,
    )
    parser.add_argument(
        "--ablation_no_overlap_avoidance",
        action="store_true",
        help="Ablation: Disable overlap avoidance during parsing",
        default=False,
    )
    # Parse the arguments
    args = parser.parse_args()

    if not args.config_file and not args.log_file:
        raise ValueError("Please provide a config file or a log file")

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        args.log_file = config["data_path"]
        args.existing_templates = (
            config["example_path"] if "example_path" in config else None
        )
        args.output = os.path.join(config["results_path"], "syntax")
    setup_logger()

    caller = Caller(
        args.thread_count,
        distribute_parallel_requests=True,
        backend=get_backend(args.backend),
    )
    template_example = (
        args.existing_templates if args.existing_templates else None
    )

    # Load existing parser if provided
    if args.parser:
        with open(args.parser, "rb") as f:
            existing_parser = dill.load(f)
    else:
        existing_parser = None

    # Remove examples from existing tree
    if existing_parser and existing_parser.tree:
        existing_parser.tree.examples = [
            [] for _ in existing_parser.tree.examples
        ]

    os.makedirs(args.output, exist_ok=True)
    parser = VariableParser(
        caller=caller,
        init_templates=template_example,
        debug_folder=args.output,
        buffer_len=args.buffer_size,
        force_overlap=args.force_overlap,
        checkpoint_frequency=args.checkpoint_frequency,
        max_memory=args.max_memory,
        use_description_distance=not args.ablation_description,
        model=args.model,
        use_fewshot=args.use_fewshot,
        parser=existing_parser if args.parser else None,
        few_shot_length=args.few_shot_length,
        validation_size=args.validation_frequency,
        run_validation=args.validation_frequency > 0,
        validation_model=args.validation_model,
        skip_cluster=False,
        total_validation_rounds=args.total_validation_rounds,
        reparse_overlaps=not args.no_reparse_overlaps,
        ablation_fewshot=args.ablation_fewshot,
        ablation_self_correction=args.ablation_self_correction,
        ablation_no_overlap_avoidance=args.ablation_no_overlap_avoidance,
    )

    parser.parse(args.log_file, percentage=args.file_percent)

    with open(os.path.join(args.output, "parser.dill"), "wb") as f:
        if args.save_contents:
            dill.dump(
                Parser(
                    tree=parser.tree,
                    values=parser.values,
                    entries_per_template=parser.entries_per_template,
                    event_types=(
                        existing_parser.event_types if existing_parser else None
                    ),
                    var_mapping=(
                        existing_parser.var_mapping if existing_parser else None
                    ),
                    schema_mapping=(
                        existing_parser.schema_mapping
                        if existing_parser
                        else None
                    ),
                    schemas=(
                        existing_parser.schemas if existing_parser else None
                    ),
                    template_mapping=(
                        existing_parser.template_mapping
                        if existing_parser
                        else None
                    ),
                ),
                f,
            )
        else:
            dill.dump(
                Parser(
                    tree=parser.tree,
                    event_types=(
                        existing_parser.event_types if existing_parser else None
                    ),
                    var_mapping=(
                        existing_parser.var_mapping if existing_parser else None
                    ),
                    schema_mapping=(
                        existing_parser.schema_mapping
                        if existing_parser
                        else None
                    ),
                    schemas=(
                        existing_parser.schemas if existing_parser else None
                    ),
                    template_mapping=(
                        existing_parser.template_mapping
                        if existing_parser
                        else None
                    ),
                ),
                f,
            )

    get_logger().info(f"Parser saved to {args.output}/parser.dill")
    del caller
