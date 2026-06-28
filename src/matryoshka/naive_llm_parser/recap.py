import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _extract_line(entry: Dict[str, Any]) -> str:
    return (
        entry.get("raw")
        or entry.get("content")
        or entry.get("message")
        or str(entry)
    )


def _walk(
    obj: Any,
    path: Tuple[str, ...],
    entry: Dict[str, Any],
    counts: Dict[str, int],
    examples: Dict[str, Tuple[Any, str]],
):
    if obj is None:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, path + (k,), entry, counts, examples)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, path, entry, counts, examples)
    else:
        key = ".".join(path)
        counts[key] += 1
        if key not in examples:
            examples[key] = (obj, _extract_line(entry))


def recap_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("parsed") or data.get("entries") or []

    counts: Dict[str, int] = defaultdict(int)
    examples: Dict[str, Tuple[Any, str]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        _walk(entry, tuple(), entry, counts, examples)

    rows = [
        {
            "name": name,
            "count": counts[name],
            "example_value": examples.get(name, ("", ""))[0],
            "example_line": examples.get(name, ("", ""))[1],
        }
        for name in counts
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create a recap of field usage for a parsed log."
    )
    parser.add_argument("input", help="Path to parsed log JSON")
    parser.add_argument(
        "--output",
        help="Path to output recap JSON (default: <input>.recap.json)",
    )
    args = parser.parse_args()

    out = args.output or f"{args.input}.recap.json"
    recap_file(args.input, out)
    print(f"Wrote recap to {out}")


if __name__ == "__main__":
    main()
