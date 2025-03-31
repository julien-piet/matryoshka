import argparse
import asyncio
import json
import os
import re
import sys
import threading

import dill

from ..classes.parser import Parser
from ..utils.logging import setup_logger
from ..utils.OCSF import OCSFSchemaClient
from ..utils.structured_log import TreeEditor


def main():
    cli_parser = argparse.ArgumentParser(description="Editable Parser backend.")
    cli_parser.add_argument("log_file", type=str)
    cli_parser.add_argument("parser_file", type=str)
    cli_parser.add_argument("output_file", type=str)
    cli_parser.add_argument(
        "--OCSF",
        action="store_true",
        help="Map to OCSF format when possible",
    )
    cli_parser.add_argument(
        "--save_parser_path",
        type=str,
        default="",
        help="Path to save the modified parser",
    )
    args = cli_parser.parse_args()

    setup_logger()

    # Load existing parser tree from dill:
    new_sys = {}
    for key in sys.modules.keys():
        new_sys[key.replace("matryoshka", "logparser")] = sys.modules[key]
    sys.modules["logparser.tools.classes"] = sys.modules["matryoshka.classes"]
    sys.modules["logparser.tools"] = sys.modules["matryoshka.utils"]
    sys.modules["logparser.tools.OCSF"] = sys.modules["matryoshka.utils.OCSF"]
    sys.modules["logparser.tools.schema"] = sys.modules["matryoshka.classes"]
    for key, value in new_sys.items():
        sys.modules[key] = value

    if args.parser_file.endswith(".dill"):
        with open(args.parser_file, "rb") as f:
            parser = dill.load(f)
    else:
        with open(args.parser_file, "r", encoding="utf-8") as f:
            json_parser = json.load(f)
        parser = Parser.load_from_json(json_parser)

    # Load lines
    with open(args.log_file, "r", encoding="utf-8") as f:
        lines = f.read()
    all_lines = re.split("\n", lines)
    all_lines = [
        re.sub(r"\s+", " ", line).strip() for line in all_lines if line
    ]

    editor = TreeEditor(
        parser,
        all_lines,
        output=args.output_file,
        caller=None,
        client=OCSFSchemaClient(None),
        lines_per_template=-1,
    )

    if args.save_parser_path:
        editor.save_parser_as_json(args.save_parser_path)

    parsed = editor.export(OCSF=args.OCSF)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
