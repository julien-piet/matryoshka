import math
import random
import threading
import time
import traceback
from typing import List, Optional, Union

from openai import OpenAI

from ..utils.logging import get_logger
from .classes import ModelResponses
from .globals import global_stop_event

### Exported Functions


def openai_chat_server(call_queue, leader: bool = False) -> None:
    """
    Each server thread owns one OpenAI client.
    All calls are made synchronously.
    """
    thread_id = threading.get_ident()
    client = OpenAI()
    failed_tasks = set()

    while True:
        if global_stop_event.is_set():
            get_logger().info(
                "Stopping OpenAI thread %d – stop event", thread_id
            )
            return

        raw_task = call_queue.get(block=True)
        if raw_task is None:  # poison pill
            get_logger().info(
                "Stopping OpenAI thread %d – received None", thread_id
            )
            return

        compl_id, task, dest_queue, kwargs = raw_task

        try:
            if getattr(task, "query_type", None) == "embedding":
                rslt = call_openai_embedding(client, **task.__dict__, **kwargs)
            else:
                rslt = call_openai(client, **task.__dict__, **kwargs)
        except Exception:
            get_logger().error(
                "OpenAI thread %d failed: %s", thread_id, traceback.format_exc()
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
            get_logger().warning("Throttling OpenAI threads – rate limit")
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
        dest_queue.put((compl_id, task, ModelResponses.default_failed()))


def _convert_history(history, system=None):
    content = []
    if system:
        content.append({"role": "developer", "content": system})

    for h in history:
        if not h["content"].strip():
            continue
        role = "user" if h["role"] == "user" else "assistant"
        content.append({"role": role, "content": h["content"]})
    return content


def call_openai(
    client,
    message="",
    max_tokens=32,
    model="gpt-4o",
    temperature=1.0,
    top_p=1,
    presence_penalty=0,
    frequency_penalty=0,
    system_prompt=None,
    stop=None,
    timeout=None,
    n=1,
    history=None,
    retry_count=7,
    thinking_budget=0,
    **kwargs
):
    """
    Calls the OpenAI API to generate text based on the given parameters.

    Args:
        client (openai.api_client.Client): The OpenAI API client.
        message (str): The user's message prompt.
        max_tokens (int): The maximum number of tokens to generate.
        query_type (str): The type of completion to use. Defaults to "chat".
        model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4o".
        temperature (float, optional): Controls the "creativity" of the generated text. Higher values result in more diverse text. Defaults to 1.0.
        top_p (float, optional): Controls the "quality" of the generated text. Higher values result in higher quality text. Defaults to 1.
        presence_penalty (float, optional): Controls how much the model avoids repeating words or phrases from the prompt. Defaults to 0.
        frequency_penalty (float, optional): Controls how much the model avoids generating words or phrases that were already generated in previous responses. Defaults to 0.
        system_prompt (str, optional): A prompt to be included before the user's message prompt. Defaults to None.
        stop (str, optional): A stop sequence
        timeout (int, optional): The maximum time to wait for a response from the API, in seconds. Defaults to 10.
        n (int, optional): The number of responses to generate. Defaults to 1.
        history (List[dict], optional): A list of previous messages in the conversation. Defaults to None.
        retry_count (int, optional): The number of times to retry the request in case of failure. Defaults to 7.

    Returns:
        The generated responses from the OpenAI API.
    """

    if history is None:
        history = []

    def loop(f, params):
        for retry in range(retry_count):
            try:
                rtn = f(params)
                return ModelResponses.load_from_openai(rtn)
            except Exception as e:
                # Handle other exceptions
                get_logger().warning(
                    "Encountered a OpenAI error on retry #%d: %s",
                    retry,
                    traceback.format_exc(),
                )
                if "'instructions': string too long" in str(e):
                    params["input"] = [
                        {"role": "user", "content": params["instructions"]}
                    ] + params["input"]
                    del params["instructions"]
                    continue

                if "maximum context length" in str(
                    e
                ) or "exceeds the context window" in str(e):
                    return None
                if (
                    "rate" in str(e).lower()
                    or "overloaded" in str(e).lower()
                    or "timed out" in str(e).lower()
                    or "internal error" in str(e).lower().lower()
                    or "temporarily unavailable" in str(e).lower()
                    or "quota exceeded" in str(e).lower()
                ):
                    time.sleep(2 ** (3 + retry))
                    if "timed out" in str(e) and retry < 3:
                        params["timeout"] += 2 ** (3 + retry)
                continue
        return None

    request_params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": (
            max_tokens if max_tokens != math.inf and max_tokens > 0 else None
        ),
    }
    if timeout is not None:
        request_params["timeout"] = timeout

    if thinking_budget > 512 and model.startswith("o"):
        request_params["reasoning"] = {"effort": "high"}
        del request_params["temperature"]
        del request_params["top_p"]
    elif thinking_budget > 256 and model.startswith("o"):
        request_params["reasoning"] = {"effort": "medium"}
        del request_params["temperature"]
        del request_params["top_p"]
    elif thinking_budget > 0 or model.startswith("o"):
        request_params["reasoning"] = {"effort": "low"}
        del request_params["temperature"]
        del request_params["top_p"]
    elif not model.startswith("o"):
        request_params["reasoning"] = {"effort": "minimal"}
        del request_params["temperature"]
        del request_params["top_p"]

    messages = _convert_history(history or [])
    messages.append({"role": "user", "content": message})
    request_params["input"] = messages
    if system_prompt:
        request_params["instructions"] = system_prompt
    return loop(lambda x: client.responses.create(**x), request_params)


def call_openai_embedding(
    client,
    message: Union[str, List[str]],
    model: str = "text-embedding-3-large",
    timeout: int = 10,
    retry_count: int = 7,
    **kwargs: Optional[dict]
) -> Union[List[float], List[List[float]]]:
    """
    Calls the OpenAI API to generate embeddings for the given message.

    Args:
        client (openai.api_client.Client): The OpenAI API client.
        message (Union[str, List[str]]): The message or list of messages to embed.
        model (str): The name of the OpenAI model to use for embeddings. Defaults to "text-embedding-3-small".
        timeout (int): The maximum time to wait for a response from the API, in seconds. Defaults to 10.
        retry_count (int): The number of times to retry the request in case of failure. Defaults to 7.

    Returns:
        The generated embeddings from the OpenAI API.
    """

    available_models = [m.id for m in client.models.list()]
    if model not in available_models:
        model = "text-embedding-3-large"

    request_params = {
        "model": model,
        "input": message,
        "timeout": timeout,
    }

    def loop(f, params):
        for retry in range(retry_count):
            try:
                rtn = f(params)
                return rtn
            except Exception as e:
                get_logger().warning(
                    "Encountered a OpenAI error on retry #%d: %s",
                    retry,
                    traceback.format_exc(),
                )
                if "rate" in str(e).lower() or "overloaded" in str(e).lower():
                    time.sleep(2 ** (3 + retry))
                if "400" in str(e):
                    return None
                continue
        return None

    rtn = loop(lambda x: client.embeddings.create(**x), request_params)
    if not rtn and isinstance(message, str):
        return None
    elif not rtn:
        return [None for _ in message]
    elif rtn and isinstance(message, str):
        return rtn.data[0].embedding
    return [item.embedding if item else None for item in rtn.data]
