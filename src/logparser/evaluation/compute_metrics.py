import argparse
import json
import re
from collections import defaultdict

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
                pred_group.issubset(ref_group)
                if allow_subgroup
                else pred_group == ref_group
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

        results[line_id] = (
            1
            if any(
                template_inclusion_map[t1][t2]
                for t1 in pred_line_to_templates[line_id]
                for t2 in ref_template_ids
            )
            else 0
        )

    ga_line = sum(results.values()) / len(results)

    # Compute per_template group accuracy
    results = {}
    for template_id, ref_group in ref_template_to_line.items():
        if not allow_subgroup:
            # look for identical groups
            results[template_id] = int(
                any(
                    pred_group == ref_group
                    for pred_group in pred_template_to_line.values()
                )
            )
        else:
            # look for contained groups
            results[template_id] = int(
                any(
                    pred_group.issubset(ref_group)
                    for pred_group in pred_template_to_line.values()
                )
            )
    ga_template = sum(results.values()) / len(results)

    return ga_line, ga_template


def parse_accuracy(parsed, reference, soft=True):
    pred_line_to_skel = {
        d[0]: [re.sub(r"\s+", " ", v.replace("<*>", "")).strip() for v in d[4]]
        for d in parsed
    }
    ref_line_to_skel = {
        d[0]: [re.sub(r"\s+", " ", v.replace("<*>", "")).strip() for v in d[4]]
        for d in reference
    }

    unique_ref_skels = {
        re.sub(r"\s+", " ", v.replace("<*>", "")).strip()
        for d in reference
        for v in d[4]
    }
    unique_pred_skels = {
        re.sub(r"\s+", " ", v.replace("<*>", "")).strip()
        for d in parsed
        for v in d[4]
    }
    similarity = {
        rs: {ps: 0 for ps in unique_pred_skels} for rs in unique_ref_skels
    }

    for rs in similarity:
        for ps in similarity[rs]:
            if soft:
                similarity[rs][ps] = 1 - (
                    Levenshtein.distance(rs, ps) / max(len(rs), len(ps))
                )
            else:
                similarity[rs][ps] = 1 if rs == ps else 0

    # Compute template-wise parse accuracy
    per_template = []
    for rs, sims in similarity.items():
        per_template.append(max(sims.values()))

    pa_template = sum(per_template) / len(per_template)

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

    return pa_line, pa_template


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


def load_file(filename):
    if filename.endswith(".pkl") or filename.endswith(".dill"):
        with open(filename, "rb") as f:
            return dill.load(f)
    elif filename.endswith(".json"):
        with open(filename, "r") as f:
            return json.load(f)
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
    with open(args.baseline, "rb") as f:
        baseline = dill.load(f)

    variants = []
    for input_file in args.predictions:
        with open(input_file, "rb") as f:
            variants.append(dill.load(f))

    # results format: group accuracy (per line), group accuracy (per template), parse accuracy (per line), parse accuracy (per template)
    results = []
    for variant in variants:
        ga_line, ga_template = group_accuracy(
            variant, baseline, allow_subgroup=args.allow_subgroup
        )
        pa_line, pa_template = parse_accuracy(variant, baseline, soft=args.soft)
        results.append((ga_line, ga_template, pa_line, pa_template))

    for i, result in enumerate(results):
        print(f"Results for {args.predictions[i]}:")
        print(f"Group accuracy (per line): {result[0]}")
        print(f"Group accuracy (per template): {result[1]}")
        print(f"Parse accuracy (per line): {result[2]}")
        print(f"Parse accuracy (per template): {result[3]}")

    compare(baseline, *variants)
