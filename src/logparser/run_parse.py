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
        parser,
        caller=None,
        unit_regex=re.compile("\n"),
        output="output/",
        fix=False,
        **kwargs,
    ) -> None:

        super().__init__(
            "Parse file to populate values, entries_per_template, and distance",
            caller=caller,
        )
        self.tree = parser.tree
        self.var_mapping = parser.var_mapping
        self.schema_mapping = parser.schema_mapping
        self.event_types = parser.event_types
        self.fix = fix

        if fix:
            # Make sure that regexes are not greedy
            for node_id, node in enumerate(self.tree.nodes):
                if node and node.is_variable():
                    if node.regexp[-1] == "+" or node.regexp[-1] == "*":
                        node.regexp += "?"

            # Remove types
            for node_id, node in enumerate(self.tree.nodes):
                if node:
                    node.type = None

        self.tree.examples = [[] for _ in self.tree.examples]
        self.unit_regex = unit_regex
        self.output = output
        self.lines = []

        self.naive_distance = NaiveDistance()
        self.entries_per_template = {
            t_idx: [] for t_idx in range(len(self.tree.templates))
        }
        self.values = {
            v: Value(v)
            for v in range(len(self.tree.nodes))
            if v > 0 and self.tree.nodes[v] and self.tree.nodes[v].is_variable()
        }
        self.output = output
        self.unmatched = []

        # init distance
        for i in range(len(self.tree.templates)):
            if self.tree.templates[i]:
                self.naive_distance.add_template(i, self.tree.gen_template(i))
        self.tree.distances = [self.naive_distance]

    def save(self):
        # generate wildcard templates
        if self.output.endswith(".dill"):
            opt = self.output
        else:
            opt = os.path.join(self.output, "parser.dill")
        with open(opt, "wb") as f:
            dill.dump(
                Parser(
                    self.tree,
                    self.values,
                    self.entries_per_template,
                    self.naive_distance,
                    var_mapping=self.var_mapping,
                    schema_mapping=self.schema_mapping,
                    event_types=self.event_types,
                ),
                f,
            )

    def _ingest_values(self, matches):
        for elt in matches.elements:
            if elt.is_variable():
                if elt.id not in self.values:
                    self.values[elt.id] = Value(
                        elt.id,
                        values=[elt.value],
                    )
                else:
                    self.values[elt.id].append(elt.value)

        self.naive_distance.update(matches)

    def analyze(self):
        # List non matched lines
        print("The following lines did not match any template:")
        for line in self.unmatched:
            print(line)

        # List templates with no matches
        to_be_removed = []
        print("The following templates did not match any line:")
        for t_id, entries in self.entries_per_template.items():
            if len(entries) == 0 and self.tree.templates[t_id]:
                print(f"Template {t_id}: {self.tree.gen_template(t_id)}")
                if self.fix:
                    to_be_removed.append(t_id)

        for t_id in to_be_removed:
            self.tree.remove_template(t_id)
            del self.entries_per_template[t_id]

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
                for candidate in candidates:
                    t_id, matches = candidate.template_id, candidate.matches
                    self.entries_per_template[t_id].append(line)
                    self._ingest_values(matches)
                    self.tree.examples[t_id].append(line)
            else:
                self.unmatched.append(line)

    def process(self, log_file, **kwargs):
        self.parse(log_file, **kwargs)
        self.analyze()
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
        "--output", type=str, help="Path to the output file", default="output/"
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=1.0,
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="parser",
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
            args.parser = config["results_path"] + f"_{args.stage}/parser.dill"
        if not args.output:
            args.output = config["results_path"] + f"_{args.stage}/"
    setup_logger()

    with open(args.parser, "rb") as f:
        parser = dill.load(f)

    converter = Ingest(parser, output=args.output, fix=args.fix)
    converter.process(args.log_file, percentage=args.file_percent)
