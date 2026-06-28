import argparse
import os

from .eval_queries import evaluate_queries

LOG_TYPES = ["audit", "cron", "dhcp", "puppet", "sshd"]


def _resolve(path_no_ext: str):
    """
    Some naive outputs are saved without .json extension; if the given path
    doesn't exist, try with .json appended.
    """
    if os.path.exists(path_no_ext):
        return path_no_ext
    if os.path.exists(path_no_ext + ".json"):
        return path_no_ext + ".json"
    return path_no_ext


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate naive parsed logs against baseline parsed logs for multiple log types."
    )
    parser.add_argument(
        "--baseline_logs_dir",
        default="data/logs/parsed",
        help="Directory containing baseline parsed logs (e.g., audit.json).",
    )
    parser.add_argument(
        "--naive_logs_dir",
        default="experiments/naive_llm",
        help="Directory containing naive parsed logs.",
    )
    parser.add_argument(
        "--baseline_queries_dir",
        default="data/golden_parsers",
        help="Directory containing baseline query JSON files (e.g., audit_query.json).",
    )
    parser.add_argument(
        "--naive_queries_dir",
        default="experiments/naive_llm",
        help="Directory containing naive query JSON files (e.g., audit_queries.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to save aggregated results.",
    )
    args = parser.parse_args()

    overall = {}
    macro_precisions = []
    macro_recalls = []

    for log_type in LOG_TYPES:
        baseline_log = os.path.join(args.baseline_logs_dir, f"{log_type}.json")
        naive_log = _resolve(os.path.join(args.naive_logs_dir, log_type))
        baseline_queries = os.path.join(
            args.baseline_queries_dir, f"{log_type}_query.json"
        )
        naive_queries = os.path.join(
            args.naive_queries_dir, f"{log_type}_queries.json"
        )

        missing = [
            p
            for p in (baseline_log, naive_log, baseline_queries, naive_queries)
            if not os.path.exists(p)
        ]
        if missing:
            print(
                f"[skip] {log_type}: missing files: "
                + ", ".join(os.path.basename(m) for m in missing)
            )
            continue

        results = evaluate_queries(
            baseline_log_path=baseline_log,
            predicted_log_path=naive_log,
            baseline_queries_path=baseline_queries,
            predicted_queries_path=naive_queries,
        )
        overall[log_type] = results
        macro_precisions.append(results["macro_precision"])
        macro_recalls.append(results["macro_recall"])

        print(
            f"{log_type}: macro P={results['macro_precision']:.4f}, "
            f"R={results['macro_recall']:.4f}"
        )
        for qname, metrics in results["per_query"].items():
            print(
                f"  {qname}: P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f} "
                f"(baseline={metrics['baseline_count']}, "
                f"pred={metrics['predicted_count']}, "
                f"overlap={metrics['overlap_count']})"
            )

    if macro_precisions and macro_recalls:
        avg_p = sum(macro_precisions) / len(macro_precisions)
        avg_r = sum(macro_recalls) / len(macro_recalls)
        print(f"Overall macro average: P={avg_p:.4f}, R={avg_r:.4f}")
        overall["macro_precision_avg"] = avg_p
        overall["macro_recall_avg"] = avg_r

    if args.output:
        import json

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(overall, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
