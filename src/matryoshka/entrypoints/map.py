import argparse
import json
import os

import dill

from ..genai_api.api import Caller, backend_choices, get_backend
from ..mapping import MapToAttributes, MapToEvents, Typer
from ..utils.ingest import Ingest
from ..utils.logging import get_logger, setup_logger
from ..utils.OCSF import OCSFSchemaClient


def main():
    parser = argparse.ArgumentParser(
        description="Map templates to the OCSF format."
    )
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
        default=3,
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
        help="number of threads to use for language model calls",
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
        "--cache_dir",
        type=str,
        help="Path to cache directory",
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
        if not args.parser:
            args.parser = os.path.join(
                config["results_path"], "schemas/parser.dill"
            )

        if not args.output:
            args.output = os.path.join(config["results_path"], "OCSF")

        if not args.description:
            args.description = config["description_path"]

        if not args.log_file:
            args.log_file = config["data_path"]
    setup_logger()

    # Setup caller
    caller = Caller(
        args.thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )
    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load descrpition:
    if args.description:
        with open(args.description, "r", encoding="utf-8") as f:
            log_desc = f.read().strip()
    else:
        log_desc = "No description provided."

    # Setup OCSF client
    get_logger().info("Setting up OCSF client.")
    client = OCSFSchemaClient(caller, os.path.join(args.cache_dir, "OCSF"))

    # Ingest log file
    get_logger().info("Parsing log file.")
    parser = Ingest(parser, caller, output=None).process(
        args.log_file, percentage=args.file_percent
    )

    # Type variables
    get_logger().info("Typing.")
    typer_output = os.path.join(args.output, "type")
    parser = Typer(
        caller,
        parser,
        output_dir=typer_output,
        model=args.model,
        few_shot_len=args.few_shot_len,
        cache_dir=args.cache_dir,
        ocsf_client=client,
    ).run()

    # Map to events
    get_logger().info("Mapping to OCSF events.")
    event_output = os.path.join(args.output, "events")
    parser = MapToEvents(
        caller,
        parser,
        output_dir=event_output,
        model=args.model,
        few_shot_len=args.few_shot_len,
        cache_dir=args.cache_dir,
        ocsf_client=client,
    )(args.log_file)

    # Map to variables
    get_logger().info("Mapping variables to OCSF attributes.")
    map_output = os.path.join(args.output, "mapping")
    MapToAttributes(
        caller,
        parser,
        log_desc,
        output_dir=map_output,
        model=args.model,
        few_shot_len=args.few_shot_len,
        cache_dir=args.cache_dir,
        ocsf_client=client,
        save_contents=args.save_contents,
    )(args.log_file)

    del caller
