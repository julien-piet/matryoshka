import argparse
import csv
import os
import re
import sys
from io import StringIO

import dill
from tqdm import tqdm

sys.setrecursionlimit(10 * sys.getrecursionlimit())

from matryoshka.classes import (
    Element,
    ElementType,
    Parser,
    Template,
    TemplateTree,
)

from .ingest_relworks import LoghubTemplate, load_lilac


def convert_wildcard_string_to_template(w_string, original_line):
    # Convert a wildcard string (with *) to a template
    template = LoghubTemplate(
        0, w_string, capture_groups=True, merge_wildcards=False
    )
    match = template.match(original_line)
    if match is None:
        # Sometimes the templates have surrounding quotes, try removing them
        w_string = re.sub(r'^(["\'`\s]*)\s*', "", w_string)
        w_string = re.sub(r'\s*(["\'`\s]*)$', "", w_string)
        template = LoghubTemplate(
            0, w_string, capture_groups=True, merge_wildcards=False
        )
        match = template.match(original_line)
        if match is None:
            return None

    # Iterate through match groups to find the positions of the matches. Order them into intervals, and then create a constant element for each interval between match groups, and a variable element for each match group.
    elements = []
    last_end = 0

    # Get all match groups and their positions
    for i, group in enumerate(match.groups()):
        if group is not None:
            start, end = match.span(i + 1)  # span(0) is the entire match

            # Add constant element before this match group if there's text between
            if start > last_end:
                constant_text = original_line[last_end:start].strip()
                if constant_text:
                    elements.append(
                        {"is_variable": False, "value": constant_text}
                    )

            # Add variable element for this match group
            if group.strip():
                elements.append({"is_variable": True, "value": group.strip()})
            last_end = end

    # Add any remaining constant text after the last match group
    if last_end < len(original_line):
        constant_text = original_line[last_end:].strip()
        if constant_text:
            elements.append({"is_variable": False, "value": constant_text})

    try:
        template = Template.load_from_json(elements, original_line)
    except:
        breakpoint()

    return template


def convert_to_templates(structured_lines_filename):

    # Load the structured lines CSV
    with open(structured_lines_filename, "rb") as f:  # Open in binary mode
        # Read and clean the file content
        content = f.read().replace(b"\0", b"")

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
            ).strip()
            loghub_structured_lines[k][2] = re.sub(
                r"\s+", " ", loghub_structured_lines[k][2]
            ).strip()

    # Convert wildcard templates to Template objects
    tree = TemplateTree()
    templates = {}
    for content, event_id, event_template in loghub_structured_lines.values():
        if event_id not in templates:
            template = convert_wildcard_string_to_template(
                event_template, content
            )
            if not template:
                print(
                    f"Could not convert template '{event_template}' with content '{content}'"
                )
                continue
            templates[event_id] = tree.add_template(template, example=content)

    return tree


def main():
    # logunit = LogUnit(caller=caller)
    # sep = logunit(sys.argv[1])
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--results", type=str, help="Path to the results")
    parser.add_argument("--output", type=str, help="Path to the output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    structured_lines, templates = load_lilac(args.results)

    for logname, structure_lines_filename in structured_lines.items():
        print(f"Processing {logname}...")
        # if the output file already exists, skip
        output_file = os.path.join(args.output, logname + ".dill")
        if os.path.exists(output_file):
            print(
                f"Output file {output_file} already exists. Skipping {logname}."
            )
            continue

        tree = convert_to_templates(structure_lines_filename)

        parser = Parser(tree=tree)
        with open(output_file, "wb") as f:
            dill.dump(parser, f)


if __name__ == "__main__":
    main()
