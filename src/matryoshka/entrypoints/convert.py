import argparse
import json
import os

import dill

from ..converter.convert_value import OcsfDataTypeParser
from ..converter.fill_event import FillEvent
from ..genai_api.api import Caller, backend_choices, get_backend
from ..utils.ingest import Ingest
from ..utils.logging import get_logger, setup_logger
from ..utils.OCSF import OCSFSchemaClient


def main():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("parser", type=str, help="Path to the parser")
    parser.add_argument(
        "--output", type=str, help="Path to the output file", default="output/"
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
        "--model",
        type=str,
        help="Backend model to use",
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
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
                config["results_path"], "OCSF/map/parser.dill"
            )

        if not args.output:
            args.output = os.path.join(config["results_path"], "convert")

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
    client = OCSFSchemaClient(
        caller, saved_path=os.path.join(args.cache_dir, "OCSF")
    )

    # Ingest log file
    get_logger().info("Parsing log file.")
    parser = Ingest(parser, caller, output=None).process(
        args.log_file, percentage=args.file_percent
    )

    get_logger().info("Running Event Filler.")
    parser = FillEvent(
        caller,
        parser,
        output_dir=args.output,
        model=args.model,
        ocsf_client=client,
        cache_dir=args.cache_dir,
        save_contents=args.save_contents,
    )(None)

    get_logger().info("Creating OCSF conversions.")
    ocsf_data_type_parser = OcsfDataTypeParser(client, caller)
    var_parsers = []
    for node_id, values in parser.values:
        values = values.values
        node = parser.tree.nodes[node_id]
        if not node or not node.is_variable():
            continue
        if (
            node_id not in parser.var_mapping
            or not parser.var_mapping[node_id].mappings
        ):
            continue
        template_ids = parser.tree.templates_per_node[node_id]
        templates = [parser.tree.gen_template(t_id) for t_id in template_ids]
        for mapping in parser.var_mapping[node_id].mappings.values():
            for field in mapping.fields:
                var_parser = ocsf_data_type_parser.get_parser_for_variable_path(
                    field, node, templates, values
                )
                var_parsers.append(var_parser)
                with open(
                    os.path.join(args.output, "var_parser.dill"), "wb"
                ) as f:
                    dill.dump(var_parsers, f)

    del caller
