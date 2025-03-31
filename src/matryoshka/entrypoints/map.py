import argparse
import json
import os

import dill

from ..genai_api.api import Caller, backend_choices, get_backend
from ..mapping.OCSF import MapToAttributes, MapToEvents, Typer
from ..mapping.OCSF.map_strawman import (
    MapToAttributes as StrawmanMapToAttributes,
)
from ..mapping.UDM import MapToAttributes as UDMMapToAttributes
from ..schema.run import CreateAttributes
from ..utils.ingest import Ingest
from ..utils.logging import get_logger, setup_logger
from ..utils.OCSF import OCSFSchemaClient
from ..utils.UDM import UDMSchemaClient


def ocsf(args):
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
            args.description = (
                config["description_path"]
                if "description_path" in config
                else None
            )

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

    parser.tree.reset_regex()
    os.makedirs(args.output, exist_ok=True)

    # Load description
    if args.description:
        with open(args.description, "r", encoding="utf-8") as f:
            log_desc = f.read().strip()
    else:
        log_desc = "No description provided."

    if args.erase:
        parser.event_types = None
        for m in parser.var_mapping.values():
            m.mapping = None
            m.embedding = None
        for node in parser.tree.nodes:
            if node:
                node.type = None

        # Regen missing embeddings
        attribute_creation = CreateAttributes(
            caller,
            parser,
            log_desc,
            output_dir=args.output,
            model=args.model,
            cache_dir=args.cache_dir,
            save_contents=args.save_contents,
            validation_size=-1,
            ablation_fewshot=args.ablation_fewshot,
            ablation_self_correction=args.ablation_self_correction,
        )
        get_logger().info("Regenerating missing attribute embeddings.")
        attribute_creation.add_missing_embeddings(
            parser.var_mapping.keys(), use_tqdm=True
        )

    # Setup OCSF client
    get_logger().info("Setting up OCSF client.")
    client = OCSFSchemaClient(
        caller,
        saved_path=os.path.join(args.cache_dir, "OCSF"),
        load_path=(
            os.path.join(args.cache_dir, "OCSF")
            if not args.load_cache_dir
            else os.path.join(args.load_cache_dir, "OCSF")
        ),
        model=args.model,
    )

    if not args.only_validate:

        # Ingest log file
        get_logger().info("Parsing log file.")
        parser = Ingest(parser, caller, output=None).process(
            args.log_file, percentage=args.file_percent
        )

        gemini_caller = Caller(4, distribute_parallel_requests=True)

        # Type variables
        get_logger().info("Typing.")
        typer_output = os.path.join(args.output, "type")
        parser = Typer(
            gemini_caller,
            parser,
            output_dir=typer_output,
            model="gemini-2.5-pro",
            few_shot_len=args.few_shot_len,
            cache_dir=args.cache_dir,
            ocsf_client=client,
            ablation_fewshot=args.ablation_fewshot,
        ).run()

        # Map to events
        get_logger().info("Mapping to OCSF events.")
        event_output = os.path.join(args.output, "events")
        parser = MapToEvents(
            gemini_caller,
            parser,
            output_dir=event_output,
            model="gemini-2.5-pro",
            few_shot_len=args.few_shot_len,
            cache_dir=args.cache_dir,
            ocsf_client=client,
            ablation_fewshot=args.ablation_fewshot,
        )(args.log_file)

        # with open(
        #     os.path.join(args.output, "events", "parser.dill"),
        #     "r",
        #     encoding="utf-8",
        # ) as f:
        #     parser = dill.load(f)

    else:
        existing_file_path = os.path.join(
            args.output, "mapping", "parser_no_validation.dill"
        )
        if not os.path.exists(existing_file_path):
            existing_file_path = os.path.join(
                args.output, "mapping", "parser.dill"
            )
        if os.path.exists(existing_file_path):
            with open(existing_file_path, "rb") as f:
                parser = dill.load(f)
        else:
            raise ValueError(
                f"Could not find existing mapping at {existing_file_path}. Please run without --only_validate first."
            )

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
        max_ocsf_attributes=args.max_attributes,
        validation_model=args.validation_model,
        ablation_fewshot=args.ablation_fewshot,
    )(args.log_file, only_validate=args.only_validate)

    del caller


def udm(args):
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
            args.output = os.path.join(config["results_path"], "UDM")

        if not args.description:
            args.description = (
                config["description_path"]
                if "description_path" in config
                else None
            )

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

    parser.tree.reset_regex()
    os.makedirs(args.output, exist_ok=True)

    # Load description
    if args.description:
        with open(args.description, "r", encoding="utf-8") as f:
            log_desc = f.read().strip()
    else:
        log_desc = "No description provided."

    if args.erase:
        parser.event_types = None
        for m in parser.var_mapping.values():
            m.mapping = None
            m.embedding = None
        for node in parser.tree.nodes:
            if node:
                node.type = None

        # Regen missing embeddings
        attribute_creation = CreateAttributes(
            caller,
            parser,
            log_desc,
            output_dir=args.output,
            model=args.model,
            cache_dir=args.cache_dir,
            save_contents=args.save_contents,
            validation_size=-1,
        )
        get_logger().info("Regenerating missing attribute embeddings.")
        attribute_creation.add_missing_embeddings(
            parser.var_mapping.keys(), use_tqdm=True
        )

    # Setup UDM client
    get_logger().info("Setting up UDM client.")
    client = UDMSchemaClient(
        caller, saved_path=os.path.join(args.cache_dir, "UDM")
    )

    if not args.only_validate:
        # Ingest log file
        get_logger().info("Parsing log file.")
        parser = Ingest(parser, caller, output=None).process(
            args.log_file, percentage=args.file_percent
        )
    else:
        existing_file_path = os.path.join(
            args.output, "mapping", "parser_no_validation.dill"
        )
        if not os.path.exists(existing_file_path):
            existing_file_path = os.path.join(
                args.output, "mapping", "parser.dill"
            )
        if os.path.exists(existing_file_path):
            with open(existing_file_path, "rb") as f:
                parser = dill.load(f)
        else:
            raise ValueError(
                f"Could not find existing mapping at {existing_file_path}. Please run without --only_validate first."
            )

    # Map to variables
    get_logger().info("Mapping variables to UDM attributes.")
    map_output = os.path.join(args.output, "mapping")
    UDMMapToAttributes(
        caller,
        parser,
        log_desc,
        output_dir=map_output,
        model=args.model,
        few_shot_len=args.few_shot_len,
        cache_dir=args.cache_dir,
        udm_client=client,
        save_contents=args.save_contents,
        max_udm_attributes=args.max_attributes,
        validation_model=args.validation_model,
    )(args.log_file, only_validate=args.only_validate)

    del caller


def ocsf_strawman(args):
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
            args.description = (
                config["description_path"]
                if "description_path" in config
                else None
            )

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

    parser.tree.reset_regex()
    os.makedirs(args.output, exist_ok=True)

    # Ingest log file
    get_logger().info("Parsing log file.")
    parser = Ingest(parser, caller, output=None).process(
        args.log_file, percentage=args.file_percent
    )

    # Map to variables
    get_logger().info("Mapping variables to OCSF attributes using strawman.")
    map_output = os.path.join(args.output, "mapping")
    StrawmanMapToAttributes(
        caller,
        parser,
        output_dir=map_output,
        model=args.model,
        save_contents=args.save_contents,
    )(args.log_file)

    del caller


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
        "--validation_model",
        type=str,
        help="Model to use for validation",
        default="gemini-2.5-pro",
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
        "--load_cache_dir",
        type=str,
        help="Path to a cache directory to load from (if different from cache_dir, use for loading descriptions from other model)",
        default=None,
    )
    parser.add_argument(
        "--save_contents",
        action="store_true",
        help="Save the parsed lines with the parser",
        default=False,
    )
    parser.add_argument(
        "--only_validate",
        action="store_true",
        help="Only validate the mapping without performing mapping steps",
        default=False,
    )
    parser.add_argument(
        "--erase",
        action="store_true",
        help="Erase previous outputs before running",
        default=False,
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["OCSF", "UDM"],
        default="OCSF",
        help="Mapping variant to use: OCSF or UDM",
    )
    parser.add_argument(
        "--max_attributes",
        type=int,
        help="Maximum number of attributes to map to",
        default=1,
    )
    parser.add_argument(
        "--ablation_fewshot",
        action="store_true",
        help="Ablation: remove few-shot examples from prompts",
        default=False,
    )
    parser.add_argument(
        "--strawman",
        action="store_true",
        help="Use the strawman mapping approach",
        default=False,
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.strawman and args.variant != "OCSF":
        raise ValueError("Strawman mapping is only available for OCSF variant.")
    if args.strawman:
        ocsf_strawman(args)
    elif args.variant == "OCSF":
        ocsf(args)
    else:
        udm(args)
