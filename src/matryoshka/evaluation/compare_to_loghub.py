import argparse
import csv
import json
import os
import re

import dill
from tqdm import tqdm

from ..classes import Template


class LoghubTemplate:
    def __init__(self, event_id: int, template: str):
        self.event_id = event_id
        # normalize template to escape regex characters
        self.orig_template = template.strip()
        template = re.sub(r"\s+", "__SPACE__", template.strip())
        template = template.replace("<*>", "__WILDCARD__")
        template = re.escape(template)
        template = template.replace("__WILDCARD__", "(.*?)")
        template = template.replace("__SPACE__", "\\s+")
        self.template = re.compile(template)  # compile to regex

    def __repr__(self):
        return f"LoghubTemplate(event_id={self.event_id}, template={self.template})"

    def match(self, line: str):
        line = re.sub(r"\s+", " ", line.strip())
        return self.template.match(line)


def parse_loghub(
    lines: list,
    loghub_raw_lines_path: str,
    loghub_structured_lines_path: str,
    loghub_templates_path: str,
    correction_path: str = None,
):
    # Load csv data
    # Format: ["line"]
    with open(loghub_raw_lines_path, "r", encoding="utf-8") as f:
        loghub_raw_lines = [
            re.sub(r"\s+", " ", l.strip()) for l in f.readlines() if l.strip()
        ]
        line_to_id = {l: i for i, l in enumerate(loghub_raw_lines)}

    # Format: [["LineID", "Content", "EventID", "EventTemplate"]]
    with open(loghub_structured_lines_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        loghub_structured_lines = list(reader)
        loghub_structured_lines = {
            int(d[0]): d for d in loghub_structured_lines
        }
        for k in loghub_structured_lines:
            loghub_structured_lines[k][1] = re.sub(
                r"\s+", " ", loghub_structured_lines[k][1].strip()
            )

    # Format: [["EventID", "EventTemplate", "Occurences"]]
    with open(loghub_templates_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        loghub_templates = list(reader)

    # Format: [["EventID", "EventTemplate"]]
    if correction_path:
        with open(correction_path, "r", encoding="utf-8") as f:
            raw_content = [d.split(",", 1) for d in f.readlines() if d.strip()]
            raw_content = [d for d in raw_content if len(d) == 2]
            correction = {d[0]: [] for d in raw_content}
            for key, template in raw_content:
                correction[key].append(template)
    else:
        correction = {}

    # Generate templates
    templates = []
    for event_id, template, _ in loghub_templates:
        if event_id not in correction:
            templates.append(LoghubTemplate(len(templates), template))
        else:
            for t in correction[event_id]:
                templates.append(LoghubTemplate(len(templates), t))

    # Parse lines
    parsed = []
    for line_id, line in tqdm(
        enumerate(lines),
        total=len(lines),
        desc="Parsing with LogHub templates",
        unit="lines",
    ):
        matches, content = [], ""
        line = line.strip()
        if line in line_to_id:
            loghub_line_id = line_to_id[line]
            content = loghub_structured_lines[loghub_line_id + 1][1]
            for template in templates:
                mtch = template.match(content)
                if mtch:
                    groups = [g for g in mtch.groups()]
                    matches.append(
                        (template.event_id, template.orig_template, groups)
                    )
        parsed.append(
            (
                line_id,
                line,
                content,
                [m[0] for m in matches],
                [m[1] for m in matches],
                [m[2] for m in matches],
            )
        )

    return parsed, templates


def parse_tree(loghub_output, tree):
    bar = tqdm(
        total=len(loghub_output),
        desc="Parsing with tree",
        unit="lines",
    )
    parsed = []
    template_to_id = {}
    for line_id, line, content, _, _, _ in loghub_output:
        match, candidates = tree.match(line)
        if not match:
            parsed.append((line_id, line, content, [], [], []))
        else:
            matches = []
            for c in candidates:
                # Find the suffix of nodes that match the content
                prefix = re.sub(
                    r"\s+", "", line.strip()[: -len(content.strip())]
                )
                template_prefix = ""
                matched_prefix, suffix_start_idx = "", 0
                for elt_idx, elt in enumerate(c.matches.elements):
                    new_matched_prefix = matched_prefix + re.sub(
                        r"\s+", "", elt.value
                    )
                    suffix_start_idx = elt_idx + 1
                    if prefix in new_matched_prefix:
                        if prefix != new_matched_prefix:
                            buffer = matched_prefix[:]
                            for val_idx, val in enumerate(elt.value):
                                buffer += val.strip()
                                if prefix == buffer:
                                    matched_prefix = buffer
                                    template_prefix = (
                                        elt.value[val_idx + 1 :].strip()
                                        if not elt.is_variable()
                                        else "<*>"
                                    ) + " " * max(1, elt.trailing_whitespace)
                                    break
                            if buffer != matched_prefix:
                                breakpoint()
                                matched_prefix = new_matched_prefix
                        break
                    else:
                        matched_prefix = new_matched_prefix

                if (
                    suffix_start_idx == len(c.matches.elements)
                    and not template_prefix
                ):
                    breakpoint()
                    suffix_start_idx = 0

                if suffix_start_idx < len(c.matches.elements):
                    content_template = Template(
                        c.matches.elements[suffix_start_idx:]
                    )
                    content_matches = [
                        mtch.value
                        for mtch in content_template.elements
                        if mtch.is_variable()
                    ]
                    formatted_template = (
                        template_prefix
                        + re.sub(
                            r"\s+",
                            " ",
                            content_template.convert_to_wildcard_template(),
                        ).strip()
                    )
                else:
                    formatted_template = template_prefix.strip()
                    content_matches = (
                        [] if "<*>" != template_prefix else [content.strip()]
                    )

                if formatted_template not in template_to_id:
                    template_to_id[formatted_template] = len(template_to_id)

                matches.append(
                    (
                        template_to_id[formatted_template],
                        formatted_template,
                        content_matches,
                    )
                )
            parsed.append(
                (
                    line_id,
                    line,
                    content,
                    [m[0] for m in matches],
                    [m[1] for m in matches],
                    [m[2] for m in matches],
                )
            )
            bar.update(1)

    templates = [None for _ in range(max(template_to_id.values()) + 1)]
    for k, v in template_to_id.items():
        templates[v] = k
    return parsed, templates


def parse_tree_suffix(loghub_output, tree):
    bar = tqdm(
        total=len(loghub_output),
        desc="Parsing with tree",
        unit="lines",
    )
    parsed = []
    template_to_id = {}
    for line_id, line, content, _, _, _ in loghub_output:
        match, candidates = tree.match(content)
        bar.update(1)
        if not match:
            parsed.append((line_id, line, content, [], [], []))
        else:
            matches = []
            for c in candidates:
                matches.append(
                    (
                        c.template_id,
                        tree.gen_template(
                            c.template_id
                        ).convert_to_wildcard_template(),
                        [
                            mtch.value
                            for mtch in c.matches.elements
                            if mtch.is_variable()
                        ],
                    )
                )
            parsed.append(
                (
                    line_id,
                    line,
                    content,
                    [m[0] for m in matches],
                    [m[1] for m in matches],
                    [m[2] for m in matches],
                )
            )

    templates = [k for k in range(len(tree.templates))]
    return parsed, templates


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--file_percent",
        type=float,
        help="Percent of file to process",
        default=0.1,
    )
    parser.add_argument(
        "--results_baseline",
        type=str,
        help="Path to the results folder for the baseline",
    )
    parser.add_argument(
        "--results_parser",
        type=str,
        help="Path to the results folder for parser",
    )
    parser.add_argument(
        "--suffix_only",
        action="store_true",
        help="Only use the loghub suffix",
        default=False,
    )
    # Parse the arguments
    args = parser.parse_args()
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    log_file = config["data_path"]
    if args.suffix_only:
        log_file = log_file.replace("_suffix", "")
    base_loghub_folder = config["loghub_folder"]
    correction_file = (
        config["correction_file"] if "correction_file" in config else None
    )
    parser = config["results_path"] + "/syntax/parser.dill"
    output_baseline = (
        (config["results_path"] + "_parser_baseline/")
        if not args.results_baseline
        else args.results_baseline
    )
    output_parser = (
        (config["results_path"] + "_parser/")
        if not args.results_parser
        else args.results_parser
    )

    # Check if output file exists
    if os.path.exists(os.path.join(output_parser, "parsed_tree.csv")):
        return

    os.makedirs(output_baseline, exist_ok=True)
    os.makedirs(output_parser, exist_ok=True)

    # Find the full paths of the files ending in "_full.log", "_full.log_structured.csv", "_full.log_templates.csv" in base_loghub_folder
    raw_lines_loghub = None
    structured_lines_loghub = None
    templates_loghub = None
    for file in os.listdir(base_loghub_folder):
        if file.endswith("_full.log"):
            raw_lines_loghub = os.path.join(base_loghub_folder, file)
        elif file.endswith("_full.log_structured.csv"):
            structured_lines_loghub = os.path.join(base_loghub_folder, file)
        elif file.endswith("_full.log_templates.csv"):
            templates_loghub = os.path.join(base_loghub_folder, file)

    if not all([raw_lines_loghub, structured_lines_loghub, templates_loghub]):
        raise FileNotFoundError("Could not find all required LogHub files")

    with open(parser, "rb") as f:
        tree = dill.load(f).tree

    with open(log_file, "r", encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f.read().split("\n") if l.strip()]

    all_lines = [re.sub(r"\s+", " ", line) for line in raw_lines if line]

    if args.file_percent < 1:
        all_lines = all_lines[: int(len(all_lines) * args.file_percent)]

    loghub_output, templates = parse_loghub(
        all_lines,
        raw_lines_loghub,
        structured_lines_loghub,
        templates_loghub,
        correction_file,
    )

    if args.suffix_only:
        parsed, templates = parse_tree_suffix(loghub_output, tree)
    else:
        parsed, templates = parse_tree(loghub_output, tree)

    # Save the parsed data
    with open(
        os.path.join(output_parser, "parsed_tree.csv"), "w", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "LineID",
                "Content",
                "LogHubContent",
                "EventIDs",
                "EventTemplates",
                "EventMatches",
            ]
        )
        for line in parsed:
            writer.writerow(line)

    with open(
        os.path.join(output_baseline, "parsed_loghub.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "LineID",
                "Content",
                "LogHubContent",
                "EventIDs",
                "EventTemplates",
                "EventMatches",
            ]
        )
        for line in loghub_output:
            writer.writerow(line)

    with open(
        os.path.join(output_parser, "templates_tree.csv"), "w", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["EventID", "EventTemplate"])
        for i, template in enumerate(templates):
            writer.writerow([i, template])
    with open(
        os.path.join(output_baseline, "templates_loghub.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["EventID", "EventTemplate"])
        for i, template in enumerate(templates):
            writer.writerow([i, template])

    with open(os.path.join(output_baseline, "baseline.dill"), "wb") as f:
        dill.dump(loghub_output, f)
    with open(os.path.join(output_parser, "tree.dill"), "wb") as f:
        dill.dump(parsed, f)


if __name__ == "__main__":
    main()
