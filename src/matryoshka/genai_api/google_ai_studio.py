# google_ai_studio.py
import asyncio
import os
import random
import threading
import traceback
from queue import Queue
from typing import List, Optional

from google import genai
from google.genai.types import (
    Content,
    EmbedContentConfig,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    ThinkingConfig,
)

from ..utils.logging import get_logger
from .classes import LLMTask, ModelResponses
from .globals import global_stop_event


class _ThreadCtx(threading.local):
    """
    Per-thread context to keep one asyncio loop and one AI Studio client alive.
    """

    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.client: Optional[genai.Client] = None

    def start(self) -> None:
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.client = _create_client()

    def run_sync(self, coro, timeout: Optional[float] = None):
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout)
        return self.loop.run_until_complete(coro)

    def shutdown(self) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()
        self.loop = None
        self.client = None


_ctx = _ThreadCtx()


def get_ctx() -> _ThreadCtx:
    _ctx.start()
    return _ctx


def _create_client() -> genai.Client:
    """
    Create an AI Studio client using the provided API key.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
        "AI_STUDIO_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY (or AI_STUDIO_API_KEY) with your AI Studio key."
        )
    return genai.Client(api_key=api_key)


def _safety_settings() -> List[SafetySetting]:
    # AI Studio rejects image/jailbreak categories; keep only text-supported ones.
    allowed_categories = {
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    }
    safe_categories = [
        c for c in HarmCategory if getattr(c, "name", str(c)) in allowed_categories
    ]
    return [
        SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_NONE)
        for c in safe_categories
    ]


def _convert_history(history, system=None) -> List[Content]:
    content: List[Content] = []
    if system:
        content.append(
            Content(parts=[Part.from_text(text=system)], role="user")
        )

    for h in history:
        if not h["content"].strip():
            continue
        role = "user" if h["role"] == "user" else "model"
        content.append(
            Content(parts=[Part.from_text(text=h["content"])], role=role)
        )
    return content


def google_ai_studio_chat_server(call_queue: Queue, leader: bool = False) -> None:
    """
    Worker loop that mirrors the Vertex-based gemini_chat_server but uses the
    AI Studio API key auth.
    """
    thread_id = threading.get_ident()
    ctx = get_ctx()
    failed_tasks = set()

    while True:
        if global_stop_event.is_set():
            get_logger().info(
                "Stopping Gemini AI Studio thread %d – stop event", thread_id
            )
            ctx.shutdown()
            return

        raw_task = call_queue.get(block=True)
        if raw_task is None:
            get_logger().info(
                "Stopping Gemini AI Studio thread %d – received None",
                thread_id,
            )
            ctx.shutdown()
            return

        compl_id, task, dest_queue, kwargs = raw_task

        try:
            if task.query_type == "embedding":
                rslt = _call_gemini_embed(task, ctx, **kwargs)
            else:
                rslt = _call_gemini(task, ctx, **kwargs)
        except Exception:
            get_logger().error(
                "Gemini AI Studio thread %d failed: %s",
                thread_id,
                traceback.format_exc(),
            )
            rslt = None

        if rslt:
            dest_queue.put((compl_id, task, rslt))
            continue

        coinflip = random.random()
        if not leader and coinflip < 0.25:
            call_queue.put(raw_task)
            get_logger().warning(
                "Throttling Gemini AI Studio threads – rate limit"
            )
            ctx.shutdown()
            return

        if (not leader) or (compl_id not in failed_tasks):
            failed_tasks.add(compl_id)
            get_logger().warning(
                "Retrying task %s in thread %d", compl_id, thread_id
            )
            call_queue.put(raw_task)
            continue

        get_logger().warning("Task %s failed twice -> giving up", compl_id)
        if task.query_type == "embedding":
            dest_queue.put((compl_id, task, []))
        else:
            dest_queue.put((compl_id, task, ModelResponses.default_failed()))


def _call_gemini_embed(
    task: LLMTask, ctx: _ThreadCtx, retry_count: int = 3, **_
) -> Optional[List[float]]:
    async def worker():
        model = task.model or "text-embedding-005"
        fn = ctx.client.models.embed_content
        for attempt in range(retry_count):
            try:
                res = fn(
                    model=model,
                    contents=(
                        [task.message]
                        if isinstance(task.message, str)
                        else list(task.message)
                    ),
                    config=EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                )
                return (
                    [e.values for e in res.embeddings]
                    if isinstance(task.message, list)
                    else res.embeddings[0].values
                )
            except Exception as e:
                txt = str(e).lower()
                if "model supports up to" in txt:
                    get_logger().warning(
                        "Truncating message for embedding task %s",
                        task.message,
                    )
                    task.message = task.message[: len(task.message) // 2]

                transient = any(
                    k in txt
                    for k in (
                        "rate",
                        "overloaded",
                        "timed out",
                        "internal error",
                        "resource_exhausted",
                    )
                )
                get_logger().warning(
                    "Embedding error attempt %d: %s",
                    attempt,
                    traceback.format_exc(),
                )
                if not transient:
                    return None
                await asyncio.sleep(2 ** (2 + attempt))
        return None

    return ctx.run_sync(worker())


def _call_gemini(
    task: LLMTask, ctx: _ThreadCtx, retry_count: int = 3, **_
) -> Optional[ModelResponses]:
    async def worker():
        gen_kwargs = {
            "temperature": float(task.temperature or 1.0),
            "top_p": float(task.top_p or 1.0),
            "stop_sequences": (
                task.stop
                if isinstance(task.stop, list)
                else list(task.stop or [])
            ),
            "candidate_count": 1,
            "safety_settings": _safety_settings(),
        }
        if task.system_prompt:
            gen_kwargs["system_instruction"] = task.system_prompt
        if task.max_tokens not in (None, float("inf")):
            gen_kwargs["max_output_tokens"] = int(task.max_tokens)
        if "2.5" in task.model and task.thinking_budget >= 0:
            if "2.5-flash" in task.model:
                gen_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_budget=task.thinking_budget
                )
            else:
                gen_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_budget=max(128, task.thinking_budget)
                )

        content = _convert_history(
            task.history or [], system=None
        )  # system already in kwargs
        if task.message:
            content.append(
                Content(
                    parts=[Part.from_text(text=str(task.message))], role="user"
                )
            )

        for attempt in range(retry_count):
            try:
                cfg = GenerateContentConfig(**gen_kwargs)
                res = await ctx.client.aio.models.generate_content(
                    model=task.model or "gemini-2.5-flash",
                    contents=content,
                    config=cfg,
                )
                resp = ModelResponses.load_from_gemini(res)
                if resp and resp.candidates and resp.candidates[0]:
                    return resp
                raise ValueError("Empty response")
            except Exception as e:
                txt = str(e).lower()
                if (
                    "empty response" in txt
                    and "2.5" in (task.model or "")
                    and task.thinking_budget >= 0
                ):
                    if "2.5-flash" in task.model:
                        gen_kwargs["thinking_config"] = ThinkingConfig(
                            thinking_budget=0
                        )
                    else:
                        gen_kwargs["thinking_config"] = ThinkingConfig(
                            thinking_budget=128
                        )
                    continue

                transient = any(
                    k in txt
                    for k in (
                        "rate",
                        "overloaded",
                        "timed out",
                        "internal error",
                        "temporarily unavailable",
                        "quota exceeded",
                    )
                )
                get_logger().warning(
                    "Generation error attempt %d: %s",
                    attempt,
                    traceback.format_exc(),
                )
                if not transient:
                    return None
                await asyncio.sleep(2 ** (3 + attempt))
        return None

    return ctx.run_sync(worker(), timeout=task.timeout or 300 + 30)
