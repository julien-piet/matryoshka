import argparse
import csv
import json
import os
import re

import dill
from tqdm import tqdm

from ..tools.classes import Template


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
        line = re.sub("\s+", " ", line.strip())
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
                r"\s+", " ", loghub_structured_lines[k][1]
            )

    # Format: [["EventID", "EventTemplate", "Occurences"]]
    with open(loghub_templates_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        loghub_templates = [l[:2] for l in list(reader)]

    # Format: [["EventID", "EventTemplate"]]
    if correction_path:
        with open(correction_path, "r", encoding="utf-8") as f:
            raw_content = [d.split(",", 1) for d in f.readlines() if d.strip()]
            correction = {d[0]: [] for d in raw_content}
            for key, template in raw_content:
                correction[key].append(template)
    else:
        correction = {}

    # Generate templates
    templates = []
    for event_id, template in loghub_templates:
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
            content = loghub_structured_lines[loghub_line_id + 1][2]
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


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--results", type=str, help="Path to the results")
    parser.add_argument("--orig", type=str, help="Path to the original data")
    parser.add_argument("--output", type=str, help="Path to the output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find all raw files recursively
    raw_lines = {}
    for root, _, files in os.walk(args.orig):
        for file in files:
            if file.endswith("_full.log"):
                logname = file.split("_")[0]
                raw_lines[logname] = os.path.join(root, file)

    # Find the full paths of the files ending in "_full.log_structured.csv", "_full.log_templates.csv"
    templates = {}
    structured_lines = {}
    for file in os.listdir(args.results):
        if file.endswith("_full.log_structured.csv"):
            logname = file.split("_")[0]
            structured_lines[logname] = os.path.join(args.results, file)
        elif file.endswith("_full.log_templates.csv"):
            logname = file.split("_")[0]
            templates[logname] = os.path.join(args.results, file)

    common_keys = (
        set(raw_lines.keys())
        .intersection(set(structured_lines.keys()))
        .intersection(set(templates.keys()))
    )

    for logname in common_keys:
        with open(raw_lines[logname], "r", encoding="utf-8") as f:
            log_lines = [l.strip() for l in f.read().split("\n") if l.strip()]
        log_lines = [re.sub(r"\s+", " ", line) for line in log_lines if line]

        relwork_output, relwork_templates = parse_loghub(
            log_lines,
            raw_lines[logname],
            structured_lines[logname],
            templates[logname],
        )

        with open(
            os.path.join(args.output, logname + "_parsed.csv"),
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
            for line in relwork_output:
                writer.writerow(line)

        with open(os.path.join(args.output, logname + ".dill"), "wb") as f:
            dill.dump(relwork_output, f)


if __name__ == "__main__":
    main()
