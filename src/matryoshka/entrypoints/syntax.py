import argparse
import json
import os

import dill

from ..classes import Parser
from ..genai_api.api import Caller, backend_choices, get_backend
from ..syntax.run import VariableParser
from ..utils.logging import setup_logger


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
        "--no-description-embedding",
        action="store_true",
        help="Don't use description embedding",
        default=False,
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
        "--save_contents",
        action="store_true",
        help="Save the contents of the log in the parser",
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
        args.existing_templates = config["example_path"]
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

    os.makedirs(args.output, exist_ok=True)
    if not args.existing_parser:
        parser = VariableParser(
            caller=caller,
            init_templates=template_example,
            debug_folder=args.output,
            buffer_len=args.buffer_size,
            force_overlap=args.force_overlap,
            checkpoint_frequency=args.checkpoint_frequency,
            max_memory=args.max_memory,
            use_description_distance=not args.no_description_embedding,
            model=args.model,
            use_fewshot=args.use_fewshot,
        )
    else:
        with open(args.existing_parser, "rb") as f:
            parser = dill.load(f)
        parser.init_caller(caller)
        parser.apply_setting(
            force_overlap=args.force_overlap,
            checkpoint_frequency=args.checkpoint_frequency,
            max_memory=args.max_memory,
        )

    parser.parse(args.log_file, percentage=args.file_percent)

    with open(os.path.join(args.output, "parser.dill"), "wb") as f:
        if args.save_contents:
            dill.dump(
                Parser(
                    parser.tree,
                    parser.values,
                    parser.entries_per_template,
                    parser.naive_distance,
                ),
                f,
            )
        else:
            dill.dump(
                Parser(
                    parser.tree,
                ),
                f,
            )
