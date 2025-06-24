import argparse
import json
import os

import dill

from ..genai_api.api import Caller, backend_choices, get_backend
from ..schema.run import CreateAttributes
from ..utils.ingest import Ingest
from ..utils.logging import get_logger, setup_logger


def main():
    parser = argparse.ArgumentParser(description="Add types to variables.")
    parser.add_argument(
        "--parser", type=str, help="Path to the parser", default=None
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "--few-shot-len",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--backend",
        choices=backend_choices(),
        default=backend_choices()[0],
        help="Select the backend to use",
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output file", default=None
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        help="number of threads to use for openai",
        default=16,
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=1.0,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Backend model to use",
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description of the log file (pointer to file containing the description)",
        default=None,
    )
    parser.add_argument(
        "--erase",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-description-embedding",
        action="store_true",
        help="Don't use description embedding",
        default=False,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path to the cache directory for the backend",
        default=".cache/",
    )
    parser.add_argument(
        "--save_contents",
        action="store_true",
        help="Save the parsed lines with the parser",
        default=False,
    )

    # Parse the arguments
    args = parser.parse_args()

    if not args.config_file and not args.parser:
        raise ValueError("Please provide a config file or a parser path")

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not args.log_file:
            args.log_file = config["data_path"]
        if not args.parser:
            args.parser = os.path.join(
                config["results_path"], "syntax/parser.dill"
            )

        if not args.output:
            args.output = os.path.join(config["results_path"], "schemas")
        if not args.description:
            args.description = config["description_path"]
    setup_logger()

    caller = Caller(
        args.thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )
    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    if args.erase:
        parser.var_mapping = None
        parser.schema_mapping = None

    os.makedirs(args.output, exist_ok=True)

    # Load descrpition:
    if args.description:
        with open(args.description, "r", encoding="utf-8") as f:
            log_desc = f.read().strip()
    else:
        log_desc = "No description provided."

    # Reparse with the syntax parser
    get_logger().info("Parsing log file.")
    parser = Ingest(parser, caller, output=None).process(
        args.log_file, percentage=args.file_percent
    )

    get_logger().info("Creating attributes for variables.")
    CreateAttributes(
        caller,
        parser,
        log_desc,
        output_dir=args.output,
        model=args.model,
        few_shot_len=args.few_shot_len,
        use_description_distance=not args.no_description_embedding,
        cache_dir=args.cache_dir,
        save_contents=args.save_contents,
    )(args.log_file)

    del caller
