import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Set, Tuple

from ..utils.structured_log import parse_datetime_to_timestamp, parse_float


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_condition(condition: List) -> Tuple:
    """
    Expand a query condition to the 8 expected fields used by the existing
    structured_log query engine:
    (attribute_name, attribute_value, exact, case_sensitive, comparison,
     negation, existence, static_match)
    """
    # Pad condition to length 8 with defaults from structured_log.match_attribute
    defaults = [None, None, True, True, None, False, False, False]
    expanded = list(condition) + defaults[len(condition) :]
    return tuple(expanded[:8])


def _match_value(
    value: str,
    target: str,
    *,
    exact: bool,
    case_sensitive: bool,
    comparison,
) -> bool:
    if value is None:
        return False
    if comparison:
        # Try datetime first, then numeric
        try:
            target_parsed = (
                float(target)
                if isinstance(target, (int, float))
                else parse_float(str(target))
            )
        except Exception:
            try:
                target_parsed = parse_datetime_to_timestamp(str(target))
            except Exception:
                return False

        try:
            value_parsed = (
                float(value)
                if isinstance(value, (int, float))
                else parse_float(str(value))
            )
        except Exception:
            try:
                value_parsed = parse_datetime_to_timestamp(str(value))
            except Exception:
                return False

        if comparison == ">":
            return value_parsed > target_parsed
        if comparison == "<":
            return value_parsed < target_parsed
        if comparison == "=":
            return value_parsed == target_parsed
        if comparison == ">=":
            return value_parsed >= target_parsed
        if comparison == "<=":
            return value_parsed <= target_parsed
        return False

    flags = 0 if case_sensitive else re.IGNORECASE
    regexp = re.compile(re.escape(target), flags)
    return bool(regexp.fullmatch(value) if exact else regexp.search(value))


def _extract_field_values(entry: Dict, name: str):
    """Return a list of values for a given attribute name, supporting dotted paths."""
    if name == "":
        return [
            entry.get("content", "")
            or entry.get("raw", "")
            or entry.get("message", "")
        ]

    def _collect_by_key(obj, key):
        hits = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    hits.append(v)
                hits.extend(_collect_by_key(v, key))
        elif isinstance(obj, list):
            for item in obj:
                hits.extend(_collect_by_key(item, key))
        return hits

    parts = name.split(".")
    current = entry
    for idx, part in enumerate(parts):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif idx == 0 and isinstance(entry.get("attributes"), dict):
            current = entry["attributes"].get(part)
        else:
            current = None
            break

    if current is None:
        collected = _collect_by_key(entry, name)
        return collected if collected else [None]

    return current if isinstance(current, list) else [current]


def _match_condition_on_entry(entry: Dict, condition: Tuple) -> bool:
    (
        attribute_name,
        attribute_value,
        exact,
        case_sensitive,
        comparison,
        negation,
        existence,
        static_match,
    ) = condition

    names = _ensure_list(attribute_name)
    values = _ensure_list(attribute_value)
    attrs = entry.get("attributes", {}) or {}
    content = (
        entry.get("content", "") or entry.get("raw", "") or entry.get("message", "")
    )

    matched = False
    saw_candidate = False

    if static_match:
        # Static match: search in raw content string
        for target in values or [""]:
            if _match_value(content, target, exact=exact, case_sensitive=case_sensitive, comparison=None):
                matched = True
                break
    elif existence:
        if not names:
            matched = bool(content.strip())
        else:
            matched = any(
                (name in attrs and attrs.get(name) not in (None, ""))
                for name in names
            )
    else:
        for name in names or [""]:
            candidates = _extract_field_values(entry, name)
            for candidate_value in candidates:
                if candidate_value is None:
                    continue
                saw_candidate = True
                for target in values or [""]:
                    if _match_value(
                        str(candidate_value),
                        str(target),
                        exact=exact,
                        case_sensitive=case_sensitive,
                        comparison=comparison,
                    ):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break

    if negation and not static_match and not existence and not saw_candidate:
        return False

    return not matched if negation else matched


def _run_query(entries: List[Dict], query: Dict) -> Set[int]:
    results: List[Set[int]] = []
    for condition in query["conditions"]:
        if isinstance(condition, dict):
            results.append(_run_query(entries, condition))
        else:
            normalized = _normalize_condition(condition)
            matches = {
                idx
                for idx, entry in enumerate(entries)
                if _match_condition_on_entry(entry, normalized)
            }
            results.append(matches)

    if not results:
        return set()

    if query["operator"] == "AND":
        return set.intersection(*results)
    if query["operator"] == "OR":
        return set.union(*results)
    raise ValueError(f"Unknown operator: {query['operator']}")


def _load_entries(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("entries") or data.get("parsed") or []


def _load_queries(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_queries(
    baseline_log_path: str,
    predicted_log_path: str,
    baseline_queries_path: str,
    predicted_queries_path: str,
) -> Dict:
    baseline_entries = _load_entries(baseline_log_path)
    predicted_entries = _load_entries(predicted_log_path)
    baseline_queries = _load_queries(baseline_queries_path)
    predicted_queries = _load_queries(predicted_queries_path)

    common = set(baseline_queries.keys()).intersection(
        set(predicted_queries.keys())
    )
    per_query = {}

    for qname in common:
        baseline_match = _run_query(baseline_entries, baseline_queries[qname])
        predicted_match = _run_query(
            predicted_entries, predicted_queries[qname]
        )
        intersect = baseline_match.intersection(predicted_match)
        if not baseline_match and not predicted_match:
            precision = recall = 1.0
        else:
            precision = len(intersect) / len(predicted_match) if predicted_match else 0.0
            recall = len(intersect) / len(baseline_match) if baseline_match else 0.0
        per_query[qname] = {
            "precision": precision,
            "recall": recall,
            "baseline_count": len(baseline_match),
            "predicted_count": len(predicted_match),
            "overlap_count": len(intersect),
        }

    # Compute macro averages excluding queries with zero expected baseline lines
    counted = [
        v
        for v in per_query.values()
        if v["baseline_count"] > 0
    ]
    if counted:
        macro_precision = sum(v["precision"] for v in counted) / len(counted)
        macro_recall = sum(v["recall"] for v in counted) / len(counted)
    else:
        macro_precision = macro_recall = 0.0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate parsed logs against baseline using the structured_log query language."
    )
    parser.add_argument(
        "--baseline_log",
        required=True,
        help="Path to baseline parsed log JSON (format like data/logs/parsed/*).",
    )
    parser.add_argument(
        "--predicted_log",
        required=True,
        help="Path to parsed log JSON produced by the naive parser.",
    )
    parser.add_argument(
        "--baseline_queries",
        required=True,
        help="Path to baseline queries JSON.",
    )
    parser.add_argument(
        "--predicted_queries",
        required=True,
        help="Path to predicted queries JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    args = parser.parse_args()

    results = evaluate_queries(
        baseline_log_path=args.baseline_log,
        predicted_log_path=args.predicted_log,
        baseline_queries_path=args.baseline_queries,
        predicted_queries_path=args.predicted_queries,
    )

    print(
        f"Macro Precision: {results['macro_precision']:.4f}, "
        f"Macro Recall: {results['macro_recall']:.4f}"
    )
    for qname, metrics in results["per_query"].items():
        print(
            f"{qname}: P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"(baseline={metrics['baseline_count']}, pred={metrics['predicted_count']}, "
            f"overlap={metrics['overlap_count']})"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
