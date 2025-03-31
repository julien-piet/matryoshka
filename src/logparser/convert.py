import argparse
import csv
import json
import os
import re

import dill
from tqdm import tqdm

from .tools.classes import Parser, Value
from .tools.embedding import NaiveDistance
from .tools.logging import setup_logger
from .tools.module import Module


class Ingest(Module):

    def __init__(
        self,
        tree,
        var_mapping=None,
        event_types=None,
        caller=None,
        variant=False,
        unit_regex=re.compile("\n"),
        output="output/",
        **kwargs,
    ) -> None:

        super().__init__(
            "Parse file to populate values, entries_per_template, and distance",
            caller=caller,
        )
        self.tree = tree

        self.unit_regex = unit_regex
        self.output = output
        self.lines = []
        self.json_lines = []
        self.variant = variant
        self.output = output
        if not (
            output.endswith(".json")
            or output.endswith(".csv")
            or output.endswith(".dill")
        ):
            if self.variant == "json":
                os.path.join(self.output, "parsed.json")
            elif self.variant == "template":
                os.path.join(self.output, "line_to_template.csv")
            else:
                os.path.join(self.output, "line_to_template_extended.json")

        self.var_mapping = var_mapping
        self.event_types = event_types

    def save(self):
        # Save the lines as a json file
        if self.variant == "json":
            with open(self.output, "w", encoding="utf-8") as f:
                json.dump(self.json_lines, f, indent=2)
        elif self.variant == "template":
            template_to_wildcard = {
                i: self.tree.gen_template(i)
                .convert_to_wildcard_template()
                .strip()
                for i in range(len(self.tree.templates))
            }
            with open(
                self.output,
                "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["line_number", "line_content", "templates"])
                for line_id, line_dict in enumerate(self.json_lines):
                    line = line_dict[0]["orig_line"]
                    template_list = [
                        l["template_id"]
                        for l in line_dict
                        if l["template_id"] is not None
                    ]
                    writer.writerow(
                        [
                            line_id,
                            line,
                            list(
                                set(
                                    [
                                        template_to_wildcard[i]
                                        for i in template_list
                                    ]
                                )
                            ),
                        ]
                    )
        else:
            template_to_wildcard = {
                i: self.tree.gen_template(i)
                .convert_to_wildcard_template()
                .strip()
                for i in range(len(self.tree.templates))
            }
            save_data = []
            for line_id, line_dict in enumerate(self.json_lines):
                template_ids = []
                templates = []
                matches = []
                full_line = line_dict[0]["orig_line"]
                suffix_line = line_dict[0]["orig_line"]
                for line in line_dict:
                    if line["template_id"] is not None:
                        template_ids.append(line["template_id"])
                        templates.append(
                            template_to_wildcard[line["template_id"]]
                        )
                        matches.append(
                            [val["value"] for val in line["variables"]]
                        )
                save_data.append(
                    (
                        line_id,
                        full_line,
                        suffix_line,
                        template_ids,
                        templates,
                        matches,
                    )
                )
            with open(
                self.output,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(save_data, f, indent=2)

    def match_to_json(self, candidate, line):
        struct = {
            "template_id": candidate.template_id,
            "orig_line": line,
            "variables": [],
            "OCSF_events": list(
                self.event_types.get(candidate.template_id, [])
            ),
        }
        for match in candidate.matches.elements:
            if match.is_variable():
                if (
                    match.id in self.var_mapping
                    and self.var_mapping[match.id].created_attribute
                ):
                    struct["variables"].append(
                        {
                            "id": match.id,
                            "value": match.value,
                            "type": self.tree.nodes[match.id].type,
                            "name": self.var_mapping[
                                match.id
                            ].created_attribute,
                            "description": self.var_mapping[
                                match.id
                            ].field_description,
                        }
                    )
                else:
                    struct["variables"].append(
                        {
                            "id": match.id,
                            "value": match.value,
                            "type": self.tree.nodes[match.id].type,
                            "name": None,
                            "description": None,
                        }
                    )
        return struct

    def parse(self, log_file, percentage=1, **kwargs) -> None:
        all_lines = [
            re.sub("\s", " ", line.strip())
            for line in self.load_and_split_log(log_file, self.unit_regex)
            if len(line)
        ]

        if percentage < 1:
            all_lines = all_lines[: int(len(all_lines) * percentage)]

        for line_id, line in tqdm(
            enumerate(all_lines), desc="Parsing log file", total=len(all_lines)
        ):
            match, candidates = self.tree.match(line)
            # If the current line matches a template, add it to the list of matches
            if match:
                json_line = []
                for candidate in candidates:
                    json_line.append(self.match_to_json(candidate, line))
                self.json_lines.append(json_line)
            else:
                self.json_lines.append(
                    [
                        {
                            "template_id": None,
                            "orig_line": line,
                            "variables": [],
                            "OCSF_events": [],
                        }
                    ]
                )

    def process(self, log_file, **kwargs):
        self.parse(log_file, **kwargs)
        self.save()


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    parser.add_argument("--parser", type=str, help="Path to the parser")
    parser.add_argument(
        "--output", type=str, help="Path to the output file", default=None
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=1.0,
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Format of the log file",
        default="template",
        choices=["json", "template", "extended_template"],
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
            if args.golden:
                args.parser = (
                    config["results_path"] + "_schemas_fixed/results/saved.dill"
                )
            else:
                args.parser = (
                    config["results_path"] + "_schemas/results/saved.dill"
                )

        if not args.output:
            if args.golden:
                args.output = config["results_path"] + "_schemas_fixed"
            else:
                args.output = config["results_path"] + "_schemas"
    setup_logger()

    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    converter = Ingest(
        parser.tree,
        output=args.output,
        var_mapping=parser.var_mapping,
        event_types=parser.event_types,
        variant=args.format,
    )
    converter.process(args.log_file, percentage=args.file_percent)
