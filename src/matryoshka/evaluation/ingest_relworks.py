import argparse
import csv
import os
import re

import dill
from tqdm import tqdm


class LoghubTemplate:
    def __init__(
        self,
        event_id: int,
        template: str,
        capture_groups: bool = False,
        merge_wildcards=True,
    ):
        self.event_id = event_id
        # normalize template to escape regex characters
        self.orig_template = template.strip()
        template = template.replace("<*>", "__WILDCARD__")
        new_template = re.sub(
            r"__WILDCARD__(\s+__WILDCARD__)+", "__WILDCARD__", template
        )  # merge multiple wildcards
        if new_template != template and merge_wildcards:
            print(
                f"Warning: Template {self.orig_template} contains multiple wildcards. Merging them."
            )
            print(new_template)
            template = new_template

        template = re.sub(r"\s+", "__SPACE__", template.strip())
        template = re.escape(template)
        if not capture_groups:
            template = template.replace("__WILDCARD__", "(.*?)")
        else:
            # Replace __WILDCARD__ with named capture groups, named as var_0, var_1, ...
            wildcard_count = 0
            while "__WILDCARD__" in template:
                template = template.replace(
                    "__WILDCARD__", f"(?P<var_{wildcard_count}>.*?)", 1
                )
                wildcard_count += 1

        template = template.replace("__SPACE__", "\\s+")
        self.template = re.compile(template)  # compile to regex

    def __repr__(self):
        return f"LoghubTemplate(event_id={self.event_id}, template={self.template})"

    def match(self, line: str):
        line = re.sub(r"\s+", " ", line.strip())
        return self.template.fullmatch(line)


def parse_loghub(
    lines: list,
    loghub_raw_lines_path: str,
    loghub_structured_lines_path: str,
    loghub_templates_path: str,
    correction_path: str = None,
    variant: str = "lilac",
):
    # Load csv data
    # Format: ["line"]
    with open(loghub_raw_lines_path, "r", encoding="utf-8") as f:
        loghub_raw_lines = [
            re.sub(r"\s+", " ", l.strip()) for l in f.readlines() if l.strip()
        ]
        line_to_id = {l: i for i, l in enumerate(loghub_raw_lines)}

    with open(loghub_structured_lines_path, "rb") as f:  # Open in binary mode
        # Read and clean the file content
        content = f.read().replace(b"\0", b"")

        # Create a file-like object from the cleaned content
        from io import StringIO

        cleaned_file = StringIO(content.decode("utf-8"))

        reader = csv.reader(cleaned_file)
        # Get ordering of columns
        header = next(reader)
        line_id_index = header.index("LineId") if "LineId" in header else -1
        content_index = header.index("Content") if "Content" in header else -1
        event_id_index = header.index("EventId") if "EventId" in header else -1
        event_template_index = (
            header.index("EventTemplate") if "EventTemplate" in header else -1
        )
        if any(
            idx == -1
            for idx in [
                line_id_index,
                content_index,
                event_id_index,
                event_template_index,
            ]
        ):
            raise ValueError(
                f"Structured lines CSV must contain 'LineId', 'Content', 'EventId', and 'EventTemplate' columns. Contains {header}"
            )
        loghub_structured_lines = list(reader)
        loghub_structured_lines = {
            int(d[line_id_index]): [
                d[content_index],
                d[event_id_index],
                d[event_template_index],
            ]
            for d in loghub_structured_lines
        }

        for k in loghub_structured_lines:
            loghub_structured_lines[k][0] = re.sub(
                r"\s+", " ", loghub_structured_lines[k][0]
            )
    # Format: [["EventID", "EventTemplate", "Occurences"]]
    if variant == "lilac":
        with open(loghub_templates_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            loghub_templates = [
                [l[0], re.sub(r"\s+", " ", l[1]).strip()]
                for l in list(reader)
                if len(l) > 1
            ]
    elif variant == "brain":
        with open(loghub_templates_path, "r", encoding="utf-8") as f:
            templates = [f for f in f.readlines() if f.strip()]
            templates = [
                re.sub(r"\s+\d+$", "", line).strip() for line in templates
            ]
            loghub_templates = [
                re.sub(r"\s+", " ", line).strip()
                for line in templates
                if line.strip()
            ]
            loghub_templates = [
                [f"E{l_idx}", l] for l_idx, l in enumerate(loghub_templates)
            ]
    else:
        raise ValueError("Variant must be either 'lilac' or 'brain'.")

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

    # Print stats
    print(len(lines), " lines to parse.")
    print(len(templates), " templates loaded.")
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
            content = loghub_structured_lines[loghub_line_id + 1][0]
            for template_id, template in enumerate(templates):
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


def load_lilac(path):
    # Find the full paths of the files ending in "_full.log_structured.csv", "_full.log_templates.csv"
    templates = {}
    structured_lines = {}
    for file in os.listdir(path):
        if file.endswith("log_structured.csv"):
            logname = file.split("_")[0]
            structured_lines[logname] = os.path.join(path, file)
        elif file.endswith("log_templates.csv"):
            logname = file.split("_")[0]
            templates[logname] = os.path.join(path, file)

    return structured_lines, templates


def load_brain(path):
    # Find the full paths of the files ending in "result.csv", "_template.csv"
    templates = {}
    structured_lines = {}
    for file in os.listdir(path):
        if file.endswith("result.csv"):
            logname = file.replace("result.csv", "")
            structured_lines[logname] = os.path.join(path, file)
        elif file.endswith("_template.csv"):
            logname = file.replace("_template.csv", "")
            templates[logname] = os.path.join(path, file)

    return structured_lines, templates


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--results", type=str, help="Path to the results")
    parser.add_argument("--orig", type=str, help="Path to the original data")
    parser.add_argument("--output", type=str, help="Path to the output")
    parser.add_argument(
        "--source",
        type=str,
        choices=["lilac", "brain"],
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find all raw files recursively
    raw_lines = {}
    for root, _, files in os.walk(args.orig):
        for file in files:
            if file.endswith("_full.log"):
                logname = file.split("_")[0]
                raw_lines[logname] = os.path.join(root, file)

    if args.source == "lilac":
        structured_lines, templates = load_lilac(args.results)
    elif args.source == "brain":
        structured_lines, templates = load_brain(args.results)
    else:
        raise ValueError("Source must be either 'lilac' or 'brain'.")

    common_keys = (
        set(raw_lines.keys())
        .intersection(set(structured_lines.keys()))
        .intersection(set(templates.keys()))
    )

    for logname in common_keys:
        print(logname)
        # if the output file already exists, skip
        output_file = os.path.join(args.output, logname + "_parsed.csv")
        if os.path.exists(output_file):
            print(
                f"Output file {output_file} already exists. Skipping {logname}."
            )
            continue

        with open(raw_lines[logname], "r", encoding="utf-8") as f:
            log_lines = [l.strip() for l in f.read().split("\n") if l.strip()]
        log_lines = [re.sub(r"\s+", " ", line) for line in log_lines if line]

        relwork_output, relwork_templates = parse_loghub(
            log_lines,
            raw_lines[logname],
            structured_lines[logname],
            templates[logname],
            variant=args.source,
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
