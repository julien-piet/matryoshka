import argparse
import json
import re
from collections import defaultdict
from io import StringIO

import dill
import Levenshtein


def group_accuracy(parsed, reference, allow_subgroup=True):
    # Compute Group Accuracy
    pred_template_to_line = {}
    pred_line_to_templates = {}
    ref_template_to_line = {}
    ref_line_to_templates = {}

    def build_groups(template_to_line, line_to_template, l):
        for line in l:
            if isinstance(line, dict):
                line_id, template_ids = line["LineID"], line["EventIDs"]
            else:
                line_id, template_ids = line[0], line[3]
            for template_id in template_ids:
                if template_id not in template_to_line:
                    template_to_line[template_id] = set()
                template_to_line[template_id].add(line_id)
                if line_id not in line_to_template:
                    line_to_template[line_id] = set()
                line_to_template[line_id].add(template_id)

    build_groups(pred_template_to_line, pred_line_to_templates, parsed)
    build_groups(ref_template_to_line, ref_line_to_templates, reference)

    # Compute template inclusion map
    template_inclusion_map = {
        pt: {
            rt: (
                len(pred_group.intersection(ref_group))
                / len(pred_group.union(ref_group))
                if allow_subgroup
                else 1 if pred_group == ref_group else 0
            )
            for rt, ref_group in ref_template_to_line.items()
        }
        for pt, pred_group in pred_template_to_line.items()
    }

    # Compute line per line accuracy
    results = {}
    for line_id, ref_template_ids in ref_line_to_templates.items():
        if line_id not in pred_line_to_templates:
            results[line_id] = 0
            continue

        results[line_id] = max(
            template_inclusion_map[t1][t2]
            for t1 in pred_line_to_templates[line_id]
            for t2 in ref_template_ids
        )

    ga_line = sum(results.values()) / len(results)

    return ga_line


def parse_accuracy(parsed, reference, soft=True):
    pred_line_to_skel = {
        d[0]: [re.sub(r"\s+", " ", v).strip() for v in d[4]] for d in parsed
    }
    ref_line_to_skel = {
        d[0]: [re.sub(r"\s+", " ", v).strip() for v in d[4]] for d in reference
    }

    unique_ref_skels = {
        re.sub(r"\s+", " ", v).strip() for d in reference for v in d[4]
    }
    unique_pred_skels = {
        re.sub(r"\s+", " ", v).strip() for d in parsed for v in d[4]
    }
    similarity = {
        rs: {ps: 0 for ps in unique_pred_skels} for rs in unique_ref_skels
    }

    for rs in similarity:
        for ps in similarity[rs]:
            if soft:
                try:
                    similarity[rs][ps] = 1 - (
                        Levenshtein.distance(rs, ps) / max(len(rs), len(ps))
                    )
                except:
                    breakpoint()
            else:
                similarity[rs][ps] = 1 if rs == ps else 0

    results = {}
    for line_id, ref_skel_list in ref_line_to_skel.items():
        if line_id not in pred_line_to_skel:
            results[line_id] = 0
            continue
        pred_skel_list = pred_line_to_skel[line_id]

        overlaps = [
            similarity[ref_skel][pred_skel]
            for ref_skel in ref_skel_list
            for pred_skel in pred_skel_list
        ]
        results[line_id] = max(overlaps) if overlaps else 0

    pa_line = sum(results.values()) / len(results)

    return pa_line


def compare(*variants):
    id_to_templates = defaultdict(lambda: tuple([set() for _ in variants]))
    for variant_idx, variant in enumerate(variants):
        for line_id, _, _, _, templates, _ in variant:
            for template in enumerate(templates):
                id_to_templates[line_id][variant_idx].add(template)

    id_to_templates = {
        key: tuple([tuple(sorted(list(v))) for v in val])
        for key, val in id_to_templates.items()
    }
    inverse_mapping = {val: 0 for val in id_to_templates.values()}
    for val in id_to_templates.values():
        inverse_mapping[val] += 1

    inverse_mapping = {
        k: v / len(id_to_templates) for k, v in inverse_mapping.items()
    }

    sorted_inverse_mapping = sorted(
        inverse_mapping.items(), key=lambda x: x[1], reverse=True
    )
    # breakpoint()


def load_file(filename):
    if filename.endswith(".pkl") or filename.endswith(".dill"):
        with open(filename, "rb") as f:
            return dill.load(f)
    elif filename.endswith(".json"):
        with open(filename, "r") as f:
            return json.load(f)
    elif filename.endswith(".csv"):
        import csv

        with open(filename, "rb") as f:
            content = f.read().replace(b"\0", b"")
            cleaned_file = StringIO(content.decode("utf-8"))
            reader = csv.reader(cleaned_file)
            header = next(reader)
            return list(reader)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--allow_subgroup",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # Fields: Line_id, line, content, template_ids, templates, content_matches
    baseline = load_file(args.baseline)
    if isinstance(baseline[0], dict):
        baseline = [
            [
                d["LineID"],
                d["Content"],
                d["LogHubContent"],
                d["EventIDs"],
                d["EventTemplates"],
                d["EventMatches"],
            ]
            for d in baseline
        ]

    variants = []
    for input_file in args.predictions:
        variants.append(load_file(input_file))

    # results format: group accuracy (per line), group accuracy (per template), parse accuracy (per line), parse accuracy (per template)
    results = []
    for variant in variants:
        ga_line = group_accuracy(
            variant, baseline, allow_subgroup=args.allow_subgroup
        )
        pa_line = parse_accuracy(variant, baseline, soft=args.soft)
        results.append((ga_line, pa_line))

    for i, result in enumerate(results):
        print(f"Results for {args.predictions[i]}:")
        print(f"Group accuracy (per line): {result[0]}")
        print(f"Parse accuracy (per line): {result[1]}")

    compare(baseline, *variants)
