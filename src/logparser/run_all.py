import argparse
import os
import pdb
import sys

import dill

from .semantics.mapping.fill_event import FillEvent
from .semantics.mapping.identify_event import IdentifyEvent
from .semantics.mapping.map_variables import MapVariables
from .semantics.type_parsing.ocsf import OcsfDataTypeParser
from .semantics.typing import TemplateTyper
from .syntax.run import VariableParser
from .tools.api import Caller
from .tools.classes import Parser, ParserStep
from .tools.logging import setup_logger


def run_syntax(parser, caller, args):
    output_folder, few_shot_template_path, log_file = (
        args.output,
        args.existing_templates if args.existing_templates else None,
        args.log_file,
    )
    syntax_output_folder = os.path.join(output_folder, "syntax")
    os.makedirs(syntax_output_folder, exist_ok=True)

    # Check if a syntax parser already exists
    saved_states = [
        f
        for f in os.listdir(syntax_output_folder)
        if f.startswith("saved_states_")
    ]
    if saved_states:
        latest_state = max(saved_states, key=lambda x: int(x.split("_")[2]))
        with open(os.path.join(syntax_output_folder, latest_state), "rb") as f:
            syntax_parser = dill.load(f)
        syntax_parser.init_caller(caller)
    else:
        syntax_parser = VariableParser(
            caller=caller,
            init_templates=few_shot_template_path,
            debug_folder=syntax_output_folder,
            checkpoint_frequency=25,
        )

    syntax_parser.parse(log_file)

    parser = Parser(
        syntax_parser.tree,
        syntax_parser.values,
        syntax_parser.entries_per_template,
        syntax_parser.naive_distance,
    )
    parser.update_step(ParserStep.SYNTAX)
    save_parser(parser, output_folder)

    return parser


def run_typing(parser, caller, args):
    output_folder = args.output
    typing_output_folder = os.path.join(output_folder, "typing")
    os.makedirs(typing_output_folder, exist_ok=True)

    typer = TemplateTyper(
        caller, parser, output_dir=output_folder, few_shot_len=args.few_shot_len
    )
    typer.run()

    parser.tree = typer.tree
    parser.values = typer.values
    parser.entries_per_template = typer.entries_per_template
    parser.embedding = typer.embedding
    parser.update_step(ParserStep.TYPED)

    save_parser(parser, output_folder)
    return parser


def run_event_mapping(parser, caller, args):
    output_dir = args.output
    event_output_folder = os.path.join(output_dir, "events")
    os.makedirs(event_output_folder, exist_ok=True)

    matcher = IdentifyEvent(caller, parser, output_dir=output_dir)
    matcher(args.log_file)

    parser.tree = matcher.tree
    parser.values = matcher.values
    parser.entries_per_template = matcher.entries_per_template
    parser.event_types = matcher.event_types
    parser.embedding = matcher.embedding
    parser.update_step(ParserStep.EVENT_MAPPING)

    save_parser(parser, output_dir)
    return parser


def run_variable_mapping(parser, caller, args):
    output_dir = args.output
    event_output_folder = os.path.join(output_dir, "variables")
    os.makedirs(event_output_folder, exist_ok=True)

    matcher = MapVariables(
        caller, parser, output_dir=output_dir, model="gemini-1.5-flash"
    )
    matcher(args.log_file)

    parser.tree = matcher.tree
    parser.values = matcher.values
    parser.entries_per_template = matcher.entries_per_template
    parser.event_types = matcher.event_types
    parser.embedding = matcher.embedding
    parser.var_mapping = matcher.var_mapping
    parser.update_step(ParserStep.VARIABLE_MAPPING)
    save_parser(parser, output_dir)
    return parser


def run_event_filling(parser, caller, args):
    output_dir = args.output
    event_output_folder = os.path.join(output_dir, "filled_events")
    os.makedirs(event_output_folder, exist_ok=True)

    filler = FillEvent(
        caller, parser, output_dir=output_dir, model="gemini-1.5-flash"
    )
    filler(args.log_file)

    parser.tree = filler.tree
    parser.values = filler.values
    parser.entries_per_template = filler.entries_per_template
    parser.event_types = filler.event_types
    parser.embedding = filler.embedding
    parser.var_mapping = filler.var_mapping
    parser.template_mapping = filler.template_mapping
    parser.update_step(ParserStep.TEMPLATE_FILLING)
    save_parser(parser, output_dir)
    return parser


def save_parser(parser, output_folder):
    with open(os.path.join(output_folder, "parser.dill"), "wb") as f:
        dill.dump(parser, f)


def main():
    parser = argparse.ArgumentParser(
        description="Build parser to convert raw system logs to OCSF-formatted logs."
    )
    parser.add_argument("log_file", type=str, help="Path to the log file")
    parser.add_argument(
        "--existing_templates", type=str, help="Path to few-shot templates"
    )
    parser.add_argument(
        "--existing_parser",
        type=str,
        help="Path to an existing, possibly incomplete, parser",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output folder",
        default="output/",
    )
    parser.add_argument(
        "--openai_thread_count",
        type=int,
        help="number of threads to use for querying LLM backend",
        default=16,
    )
    # Parse the arguments
    args = parser.parse_args()
    setup_logger()
    os.makedirs(args.output, exist_ok=True)

    # Setup backend
    caller = Caller(args.openai_thread_count, distribute_parallel_requests=True)

    # Load or create the parser
    if args.parser:
        with open(args.parser, "rb") as f:
            parser = dill.load(f)
            completed_steps = parser.completed_steps

    else:
        parser = None
        completed_steps = ParserStep.INIT

    if completed_steps == ParserStep.INIT:
        parser = run_syntax(parser, caller, args)

    if not parser.has_completed(ParserStep.TYPED):
        parser = run_typing(parser, caller, args)

    if not parser.has_completed(ParserStep.EVENT_MAPPING):
        parser = run_event_mapping(parser, caller, args)

    if not parser.has_completed(ParserStep.VARIABLE_MAPPING):
        parser = run_variable_mapping(parser, caller, args)

    if not parser.has_completed(ParserStep.TEMPLATE_FILLING):
        parser = run_event_filling(parser, caller, args)

    del caller
