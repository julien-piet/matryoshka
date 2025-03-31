# gemini.py
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

# ------------------------------------------------------------------ #
#  Per-thread context                                                 #
# ------------------------------------------------------------------ #


class _ThreadCtx(threading.local):
    """
    One instance of this object exists inside every OS thread that calls
    get_ctx().  It owns a *persistent* asyncio loop and a google-genai
    Client bound to that loop.
    """

    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.client: Optional[genai.Client] = None

    def start(self) -> None:
        if self.loop is not None:  # already started
            return

        # 1) create a private loop
        self.loop = asyncio.new_event_loop()

        # 2) run the loop in the *current* thread
        asyncio.set_event_loop(self.loop)

        # 3) instantiate one client bound to the loop
        self.client = _create_client()

    def run_sync(self, coro, timeout: Optional[float] = None):
        """
        Run *coro* inside the thread's private loop and return its result.
        """
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
    """
    Return the context that belongs to the current thread, starting it if
    necessary.
    """
    _ctx.start()
    return _ctx


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def _create_client() -> genai.Client:
    """
    Build one configured Client.  Do *not* create more than one per
    thread – AsyncClient pools sockets, which we want to reuse.
    """
    project = os.environ.get("GEMINI_PROJECT", "default-project")
    location = os.environ.get("GEMINI_LOCATION", "us-central1")

    return genai.Client(vertexai=True, project=project, location=location)


def _safety_settings() -> List[SafetySetting]:
    return [
        SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_NONE)
        for c in HarmCategory
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


# ------------------------------------------------------------------ #
#  Public synchronous wrapper                                         #
# ------------------------------------------------------------------ #


def gemini_chat_server(call_queue: Queue, leader: bool = False) -> None:
    """
    Each server thread        ---- owns one event-loop
                              ---- owns one genai.Client
    All calls are made asynchronously *inside that loop* and the worker
    thread waits for the result synchronously.
    """
    thread_id = threading.get_ident()
    ctx = get_ctx()  # <- creates loop & client
    failed_tasks = set()

    while True:
        if global_stop_event.is_set():
            get_logger().info(
                "Stopping Gemini thread %d – stop event", thread_id
            )
            ctx.shutdown()
            return

        raw_task = call_queue.get(block=True)
        if raw_task is None:  # poison pill
            get_logger().info(
                "Stopping Gemini thread %d – received None", thread_id
            )
            ctx.shutdown()
            return

        compl_id, task, dest_queue, kwargs = raw_task

        try:
            if task.query_type == "embedding":
                rslt = _call_gemini_embed(task, ctx, **kwargs)
            else:
                rslt = _call_gemini(task, ctx, **kwargs)
        except Exception:  # unexpected – log & retry
            get_logger().error(
                "Gemini thread %d failed: %s", thread_id, traceback.format_exc()
            )
            rslt = None

        if rslt:
            dest_queue.put((compl_id, task, rslt))
            continue

        # ----------------------------------------------
        # Fallback / retry strategy
        # ----------------------------------------------
        coinflip = random.random()
        if not leader and coinflip < 0.25:  # shrink worker pool
            call_queue.put(raw_task)
            get_logger().warning("Throttling Gemini threads – rate limit")
            ctx.shutdown()
            return

        if (not leader) or (compl_id not in failed_tasks):
            failed_tasks.add(compl_id)
            get_logger().warning(
                "Retrying task %s in thread %d", compl_id, thread_id
            )
            call_queue.put(raw_task)
            continue

        # Second failure – give up
        get_logger().warning("Task %s failed twice -> giving up", compl_id)
        if task.query_type == "embedding":
            dest_queue.put((compl_id, task, []))
        else:
            dest_queue.put((compl_id, task, ModelResponses.default_failed()))


# ------------------------------------------------------------------ #
#  Low-level wrappers                                                 #
# ------------------------------------------------------------------ #


#
# Embedding
#
def _call_gemini_embed(
    task: LLMTask, ctx: _ThreadCtx, retry_count: int = 3, **_
) -> Optional[List[float]]:
    """
    Synchronous wrapper around an *async* embedding call.
    """

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

                test = "model supports up to" in txt
                if test:
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


#
# Text generation
#
def _call_gemini(
    task: LLMTask, ctx: _ThreadCtx, retry_count: int = 3, **_
) -> Optional[ModelResponses]:
    """
    Synchronous wrapper around the async generate_content call.
    """

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

    # --- run coroutine inside the thread's loop
    return ctx.run_sync(worker(), timeout=task.timeout or 300 + 30)
