import math
import os
from typing import Dict, Iterable, List, Optional

from ..genai_api.api import Caller
from ..genai_api.classes import LLMTask, ModelResponses
from ..utils.json import parse_json
from ..utils.logging import get_logger


DEFAULT_SYSTEM_PROMPT = (
    "You are a log parsing assistant. Given a single raw log line, extract a"
    " concise, structured JSON object that captures the event. Prefer keys"
    " like timestamp, severity, component, action, actor, target, and any"
    " other useful attributes you can infer. Always include the original line"
    ' under the key "raw". Return only JSON without Markdown.'
)


def _load_lines(
    log_file: str,
    file_percent: float = 1.0,
    line_limit: Optional[int] = None,
    include_empty: bool = False,
) -> List[str]:
    with open(log_file, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if not include_empty:
        lines = [line for line in lines if line.strip()]

    file_percent = max(0.0, min(file_percent, 1.0))
    if file_percent and file_percent < 1.0:
        keep = max(1, math.floor(len(lines) * file_percent))
        lines = lines[:keep]

    if line_limit is not None and line_limit >= 0:
        lines = lines[:line_limit]

    return lines


def _build_prompt(line: str) -> str:
    return (
        "Parse the following log line into a structured JSON object. "
        "If you cannot infer a value, set it to null. Avoid prose."
        f"\n\nLog line:\n{line}"
    )


def _build_tasks(
    lines: Iterable[str],
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: Optional[str],
) -> List[LLMTask]:
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    tasks: List[LLMTask] = []
    for line in lines:
        tasks.append(
            LLMTask(
                system_prompt=prompt,
                message=_build_prompt(line),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    return tasks


def _extract_candidate(response: ModelResponses) -> str:
    if not response or response.failed:
        return ""
    return response.candidates[0] if response.candidates else ""


def _parse_candidate(
    raw_candidate: str,
    line: str,
    caller: Caller,
    task: LLMTask,
    model: str,
) -> Dict:
    try:
        parsed = parse_json(
            raw_candidate,
            caller=caller,
            task=task,
            model=model,
        )
    except Exception as exc:
        get_logger().warning(
            "Falling back to raw response for line due to parse error: %s", exc
        )
        return {"raw": line, "error": str(exc), "raw_response": raw_candidate}

    if isinstance(parsed, dict) and "raw" not in parsed:
        parsed["raw"] = line
    return parsed if isinstance(parsed, dict) else {"raw": line, "parsed": parsed}


def parse_log_file(
    log_file: str,
    caller: Caller,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    file_percent: float = 1.0,
    line_limit: Optional[int] = None,
    include_empty: bool = False,
    system_prompt: Optional[str] = None,
    generation_kwargs: Optional[Dict] = None,
) -> List[Dict]:
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    lines = _load_lines(
        log_file,
        file_percent=file_percent,
        line_limit=line_limit,
        include_empty=include_empty,
    )
    if not lines:
        return []

    get_logger().info("Parsing %d lines with model %s", len(lines), model)

    tasks = _build_tasks(
        lines,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    responses = caller(
        tasks,
        use_tqdm=True,
        generation_kwargs=generation_kwargs
        or {"response_mime_type": "application/json"},
    )

    parsed: List[Dict] = []
    for line, task, response in zip(lines, tasks, responses):
        candidate = _extract_candidate(response)
        if not candidate:
            parsed.append(
                {"raw": line, "error": "Empty response from model"}
            )
            continue
        parsed.append(
            _parse_candidate(candidate, line, caller=caller, task=task, model=model)
        )

    return parsed
