import argparse
import os
import pdb
import sys

import dill

from .semantics.mapping.fill_event import FillEvent
from .tools.api import BACKEND, Caller
from .tools.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("parser", type=str, help="Path to the parser")
    parser.add_argument(
        "--output", type=str, help="Path to the output file", default="output/"
    )
    parser.add_argument(
        "--backend_thread_count",
        type=int,
        help="number of threads to use for backend calls",
        default=1,
    )
    parser.add_argument(
        "--few-shot-len",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND,
        default=BACKEND[0],
        help="Select the backend to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Backend model to use",
        default="gemini-1.5-flash",
    )
    # Parse the arguments
    args = parser.parse_args()
    setup_logger()

    caller = Caller(
        args.backend_thread_count,
        backend=args.backend,
        distribute_parallel_requests=True,
    )
    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    os.makedirs(args.output, exist_ok=True)

    matcher = FillEvent(
        caller, parser, output_dir=args.output, model=args.model
    )
    matcher(None)

    del caller
