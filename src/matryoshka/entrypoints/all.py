import argparse
import json
import os

import dill

from ..classes import Parser
from ..genai_api.api import Caller, backend_choices, get_backend
from ..mapping import MapToAttributes, MapToEvents, Typer
from ..schema.run import CreateAttributes
from ..syntax.run import VariableParser
from ..utils.ingest import Ingest
from ..utils.logging import get_logger, setup_logger
from ..utils.OCSF import OCSFSchemaClient


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
        "--cache_dir",
        type=str,
        help="Path to cache directory",
        default=".cache/",
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
        args.output = config["results_path"]
        if not args.description:
            args.description = config["description_path"]
    setup_logger()

    # Setup caller
    caller = Caller(
        args.thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )

    # Load few-shot examples if any
    template_example = (
        args.existing_templates if args.existing_templates else None
    )

    # Setup OCSF client
    client = OCSFSchemaClient(
        caller, saved_path=os.path.join(args.cache_dir, "OCSF")
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load descrpition:
    if args.description:
        with open(args.description, "r", encoding="utf-8") as f:
            log_desc = f.read().strip()
    else:
        log_desc = "No description provided."

    # Step 1: Syntax Generation
    get_logger().info("Generating syntax tree from log file.")
    output_path = os.path.join(args.output, "syntax")
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(os.path.join(output_path, "parser.dill")):
        parser = dill.load(open(os.path.join(output_path, "parser.dill"), "rb"))
    else:
        if args.existing_parser:
            with open(args.existing_parser, "rb") as f:
                parser = dill.load(f)
            parser.init_caller(caller)
            parser.apply_setting(
                force_overlap=args.force_overlap,
                checkpoint_frequency=args.checkpoint_frequency,
                max_memory=args.max_memory,
            )
        else:
            parser = VariableParser(
                caller=caller,
                init_templates=template_example,
                debug_folder=output_path,
                buffer_len=args.buffer_size,
                force_overlap=args.force_overlap,
                checkpoint_frequency=args.checkpoint_frequency,
                max_memory=args.max_memory,
                use_description_distance=not args.no_description_embedding,
                model=args.model,
                use_fewshot=args.use_fewshot,
            )

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

    # Step 2: Ingest and create schema
    get_logger().info("Schema creation.")
    output_dir = os.path.join(args.output, "schemas")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, "parser.dill")):
        parser = dill.load(open(os.path.join(output_dir, "parser.dill"), "rb"))
    else:
        parser = Ingest(parser, caller, output=None).process(
            args.log_file, percentage=args.file_percent
        )

        parser = CreateAttributes(
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

    # Step 3: Map to OCSF
    get_logger().info("OCSF Mapping.")
    parser = Ingest(parser, caller, output=None).process(
        args.log_file, percentage=args.file_percent
    )

    # Type variables
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
    parser = MapToAttributes(
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

    with open(os.path.join(args.output, "parser.dill"), "wb") as f:
        dill.dump(parser, f)

    del caller
