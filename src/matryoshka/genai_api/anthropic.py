import math
import os
import random
import threading
import time
import traceback
from typing import List, Optional, Union

from anthropic import Anthropic
from openai import OpenAI

from ..utils.logging import get_logger
from .classes import ModelResponses
from .globals import global_stop_event

### Exported Functions


def anthropic_chat_server(call_queue, leader: bool = False) -> None:
    """
    Each server thread owns one Anthropic client.
    For embeddings, it also creates an OpenAI client.
    All calls are made synchronously.
    """
    thread_id = threading.get_ident()
    anthropic_client = Anthropic()
    openai_client = OpenAI()  # For embeddings
    failed_tasks = set()

    while True:
        if global_stop_event.is_set():
            get_logger().info(
                "Stopping Anthropic thread %d – stop event", thread_id
            )
            return

        raw_task = call_queue.get(block=True)
        if raw_task is None:  # poison pill
            get_logger().info(
                "Stopping Anthropic thread %d – received None", thread_id
            )
            return

        compl_id, task, dest_queue, kwargs = raw_task

        try:
            if getattr(task, "query_type", None) == "embedding":
                # Use OpenAI for embeddings since Anthropic doesn't support them
                rslt = call_anthropic_embedding(
                    openai_client, **task.__dict__, **kwargs
                )
            else:
                rslt = call_anthropic(
                    anthropic_client, **task.__dict__, **kwargs
                )
        except Exception:
            get_logger().error(
                "Anthropic thread %d failed: %s",
                thread_id,
                traceback.format_exc(),
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
            get_logger().warning("Throttling Anthropic threads – rate limit")
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


def _convert_history(history):
    """
    Convert history to Anthropic's message format.
    Anthropic uses a different format than OpenAI for conversations.
    """
    messages = []

    for h in history:
        if not h["content"].strip():
            continue
        role = "user" if h["role"] == "user" else "assistant"
        messages.append({"role": role, "content": h["content"]})

    return messages


def call_anthropic(
    client,
    message="",
    max_tokens=math.inf,
    model="claude-sonnet-4-20250514",
    temperature=1.0,
    top_p=1,
    system_prompt=None,
    stop=None,
    timeout=None,
    n=1,
    history=None,
    retry_count=7,
    thinking_budget=2048,
    stream=True,
    **kwargs
):
    """
    Calls the Anthropic API to generate text based on the given parameters.

    Args:
        client (anthropic.Anthropic): The Anthropic API client.
        message (str): The user's message prompt.
        max_tokens (int): The maximum number of tokens to generate.
        model (str, optional): The name of the Anthropic model to use. Defaults to "claude-sonnet-4-20250514".
        temperature (float, optional): Controls the "creativity" of the generated text. Higher values result in more diverse text. Defaults to 1.0.
        top_p (float, optional): Controls the "quality" of the generated text. Higher values result in higher quality text. Defaults to 1.
        presence_penalty (float, optional): Not supported by Anthropic API. Included for compatibility.
        frequency_penalty (float, optional): Not supported by Anthropic API. Included for compatibility.
        system_prompt (str, optional): A system prompt to set the assistant's behavior. Defaults to None.
        stop (str, optional): A stop sequence
        timeout (int, optional): The maximum time to wait for a response from the API, in seconds. Defaults to None.
        n (int, optional): The number of responses to generate. Only n=1 is supported. Defaults to 1.
        history (List[dict], optional): A list of previous messages in the conversation. Defaults to None.
        retry_count (int, optional): The number of times to retry the request in case of failure. Defaults to 7.
        thinking_budget (int, optional): Not directly supported by Anthropic. Included for compatibility.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        The generated responses from the Anthropic API.
    """

    if history is None:
        history = []

    # Note: n>1 is not supported, always use n=1
    if n != 1:
        get_logger().warning(
            "Anthropic backend only supports n=1, ignoring n=%d", n
        )

    def loop(f, params, use_streaming=False):
        for retry in range(retry_count):
            try:
                if use_streaming:
                    # Handle streaming response
                    stream_response = f(params)
                    complete_text = ""
                    thinking_text = ""

                    for chunk in stream_response:
                        if chunk.type == "content_block_start":
                            # Handle start of content block
                            pass
                        elif chunk.type == "content_block_delta":
                            if hasattr(chunk.delta, "text"):
                                complete_text += chunk.delta.text
                        elif chunk.type == "thinking_start":
                            # Handle start of thinking block
                            pass
                        elif chunk.type == "thinking_delta":
                            if hasattr(chunk.delta, "text"):
                                thinking_text += chunk.delta.text
                        elif chunk.type == "message_start":
                            # Handle message start
                            pass
                        elif chunk.type == "message_delta":
                            # Handle message metadata updates
                            pass
                        elif chunk.type == "message_stop":
                            # Handle message completion
                            break

                    return ModelResponses(
                        [complete_text] if complete_text else []
                    )
                else:
                    # Handle non-streaming response
                    rtn = f(params)
                    return ModelResponses.load_from_anthropic(rtn)

            except Exception as e:
                # Handle other exceptions
                get_logger().warning(
                    "Encountered an Anthropic error on retry #%d: %s",
                    retry,
                    traceback.format_exc(),
                )

                error_str = str(e).lower()
                if "context_length_exceeded" in error_str:
                    return None
                if (
                    "rate" in error_str
                    or "overloaded" in error_str
                    or "timed out" in error_str
                    or "internal" in error_str
                    or "temporarily unavailable" in error_str
                    or "quota" in error_str
                ):
                    time.sleep(2 ** (3 + retry))
                    if "timed out" in error_str and retry < 3 and timeout:
                        params["timeout"] = params.get("timeout", 30) + 2 ** (
                            3 + retry
                        )
                continue
        return None

    # Convert history and prepare messages
    messages = _convert_history(history or [])
    messages.append({"role": "user", "content": message})

    # Prepare request parameters
    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    if max_tokens != math.inf and max_tokens > 0:
        request_params["max_tokens"] = max_tokens
    else:
        request_params["max_tokens"] = 64000

    # Add optional parameters
    if system_prompt:
        request_params["system"] = system_prompt

    if stop:
        if isinstance(stop, str):
            request_params["stop_sequences"] = [stop]
        else:
            request_params["stop_sequences"] = stop

    if thinking_budget > 0:
        thinking_model = (
            "opus-4" in model or "sonnet-4" in model or "3-7-sonnet" in model
        )
        if thinking_model:
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            del request_params["temperature"]
            del request_params["top_p"]

    return loop(
        lambda x: client.messages.create(**x),
        request_params,
        use_streaming=stream,
    )


def call_anthropic_embedding(
    openai_client,
    message: Union[str, List[str]],
    model: str = "text-embedding-3-large",
    timeout: int = 10,
    retry_count: int = 7,
    **kwargs: Optional[dict]
) -> Union[List[float], List[List[float]]]:
    """
    Since Anthropic doesn't have a native embedding API, this function
    uses OpenAI's embedding API as a fallback.

    Args:
        openai_client (openai.OpenAI): The OpenAI API client
        message (Union[str, List[str]]): The message or list of messages to embed
        model (str): The OpenAI embedding model to use. Defaults to "text-embedding-3-large"
        timeout (int): The maximum time to wait for a response from the API, in seconds. Defaults to 10.
        retry_count (int): The number of times to retry the request in case of failure. Defaults to 7.

    Returns:
        The generated embeddings from the OpenAI API.
    """

    # Check available models
    available_models = [m.id for m in openai_client.models.list()]
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
                    "Encountered an OpenAI embedding error on retry #%d: %s",
                    retry,
                    traceback.format_exc(),
                )
                if "rate" in str(e).lower() or "overloaded" in str(e).lower():
                    time.sleep(2 ** (3 + retry))
                continue
        return None

    rtn = loop(lambda x: openai_client.embeddings.create(**x), request_params)
    if isinstance(message, str):
        return rtn.data[0].embedding
    return [item.embedding for item in rtn.data]
