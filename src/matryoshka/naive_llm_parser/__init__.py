"""
Naive LLM-based log parser utilities.
"""

# Avoid hard dependency on LLM backends when only the evaluators are used.
try:  # pragma: no cover - defensive import
    from .run import parse_log_file  # noqa: F401
except Exception:  # broad to keep package importable without genai deps
    parse_log_file = None
