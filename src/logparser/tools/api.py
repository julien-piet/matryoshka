import math
import multiprocessing
import os
import random
import signal
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import google.generativeai as genai
from openai import OpenAI
from tqdm import tqdm

from .logging import get_logger

global_process_list = []


def openai_chat_server(call_queue, leader=False):
    """
    A function that listens to a call queue for incoming tasks, and processes them using OpenAI's API.

    Args:
    call_queue (Queue): A queue object that contains incoming tasks. These are made of the following elements:
        id: id for this task.
        message: a string representing the user's message prompt.
        max_tokens: an integer representing the maximum number of tokens to generate.
        kwargs: a dictionary containing optional keyword arguments to be passed to the call_openai function.
        dest_queue: a queue object where the result of the task will be put.

    Returns:
        None
    """
    client = OpenAI()

    while True:
        raw_task = call_queue.get(block=True)
        if raw_task is None:
            return

        compl_id, task, dest_queue = raw_task
        rslt = call_openai(client, **task.__dict__)
        if rslt == 0 and not leader:
            call_queue.put(raw_task)
            print("Reducing the number of OpenAI threads due to Rate Limit")
            return
        elif rslt == 0 and leader:
            call_queue.put(raw_task)
        else:
            dest_queue.put((compl_id, task, rslt))


def gemini_chat_server(call_queue, leader=False):
    """
    A function that listens to a call queue for incoming tasks, and processes them using Google's API.
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    while True:
        raw_task = call_queue.get(block=True)
        if raw_task is None:
            return

        compl_id, task, dest_queue, kwargs = raw_task
        try:
            if task.query_type == "embedding":
                rslt = call_gemini_embed(**task.__dict__, **kwargs)
            else:
                rslt = call_gemini(**task.__dict__, **kwargs)
        except Exception as e:
            dest_queue.put(
                (-1 if isinstance(compl_id, int) else (-1, -1), task, str(e))
            )
        coinflip = random.randint(0, 1)
        if rslt is None and not leader and coinflip < 0.25:
            call_queue.put(raw_task)
            get_logger().warning(
                "Reducing the number of Gemini threads due to Rate Limit"
            )
            return
        elif rslt is None:
            call_queue.put(raw_task)
        else:
            dest_queue.put((compl_id, task, rslt))


BACKEND = [gemini_chat_server, openai_chat_server]

### API Functions


def call_gemini_embed(
    run_config=False,
    message="",
    model="text-embedding-004",
    api_key=None,
    **kwargs,
):

    if not isinstance(message, list):
        run_batch = False
    else:
        run_batch = True

    if run_config:
        api_key = api_key if api_key else os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)

    task_type = "semantic_similarity"

    def loop():
        for i in range(3):
            try:
                result = genai.embed_content(
                    model=model, content=message, task_type=task_type
                )
                return result["embedding"]
            except Exception as e:
                if "unable to submit request" in str(e).lower():
                    raise e
                get_logger().warning(
                    "Encountered a GCP embedding error on retry #%d: %s", i, e
                )
                if "maximum context length" in str(e):
                    return None
                if (
                    "RATE_LIMIT_EXCEEDED" in str(e)
                    or "overloaded" in str(e)
                    or "timed out" in str(e)
                    or "internal error" in str(e).lower()
                ):
                    time.sleep(5 * (1 + i))
                continue
        return None

    return loop()


def call_gemini(
    run_config=False,
    message="",
    max_tokens=32,
    model="gemini-1.5-flash",
    temperature=1.0,
    top_p=1,
    presence_penalty=0,
    frequency_penalty=0,
    system_prompt=None,
    stop=None,
    n=1,
    history=None,
    api_key=None,
    generation_kwargs=None,
    **kwargs,
):

    generation_kwargs = generation_kwargs or {}

    if run_config:
        api_key = api_key if api_key else os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)

    if history is None:
        history = []
    if message:
        history += [{"role": "user", "content": message}]

    # Geminify history
    history = [
        {
            "role": "user" if c["role"] == "user" else "model",
            "parts": c["content"],
        }
        for c in history
        if c["content"].strip()
    ]

    model = genai.GenerativeModel(model, system_instruction=system_prompt)

    generation_kwargs["candidate_count"] = n
    generation_kwargs["max_output_tokens"] = max_tokens
    generation_kwargs["temperature"] = temperature
    generation_kwargs["top_p"] = top_p
    generation_kwargs["presence_penalty"] = presence_penalty
    generation_kwargs["frequency_penalty"] = frequency_penalty
    generation_kwargs["stop_sequences"] = (
        list(stop) if stop and not isinstance(stop, list) else stop
    )

    def loop():
        retry = 0
        while retry < 3:
            try:
                rtn = model.generate_content(
                    contents=history,
                    generation_config=genai.types.GenerationConfig(
                        **generation_kwargs
                    ),
                    safety_settings={
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    },
                )
                resp = ModelResponses.load_from_gemini(rtn)
                if not resp.candidates:
                    raise Exception("Invalid response from Gemini: %s" % rtn)
                return resp
            except Exception as e:
                if "unable to submit request" in str(e).lower():
                    raise e
                get_logger().warning(
                    "Encountered a GCP error on retry #%d: %s", retry, e
                )
                get_logger().warning("Parameters are: %s", generation_kwargs)
                if "maximum context length" in str(e):
                    return None
                if (
                    "RATE_LIMIT_EXCEEDED" in str(e)
                    or "overloaded" in str(e)
                    or "timed out" in str(e)
                    or "internal error" in str(e).lower()
                    or "temporarily unavailable" in str(e).lower()
                ):
                    time.sleep(30 * (1 + retry))
                retry += 1
                continue
        return None

    return loop()


def call_openai(
    client,
    message="",
    max_tokens=32,
    query_type="chat",
    model="gpt-3.5-turbo",
    temperature=1.0,
    top_p=1,
    presence_penalty=0,
    frequency_penalty=0,
    system_prompt=None,
    stop=None,
    timeout=None,
    n=1,
    history=None,
):
    """
    Calls the OpenAI API to generate text based on the given parameters.

    Args:
        client (openai.api_client.Client): The OpenAI API client.
        message (str): The user's message prompt.
        max_tokens (int): The maximum number of tokens to generate.
        query_type (str): The type of completion to use. Defaults to "chat".
        model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Controls the "creativity" of the generated text. Higher values result in more diverse text. Defaults to 1.0.
        top_p (float, optional): Controls the "quality" of the generated text. Higher values result in higher quality text. Defaults to 1.
        presence_penalty (float, optional): Controls how much the model avoids repeating words or phrases from the prompt. Defaults to 0.
        frequency_penalty (float, optional): Controls how much the model avoids generating words or phrases that were already generated in previous responses. Defaults to 0.
        system_prompt (str, optional): A prompt to be included before the user's message prompt. Defaults to None.
        stop (str, optional): A stop sequence
        timeout (int, optional): The maximum time to wait for a response from the API, in seconds. Defaults to 10.
        n (int, optional): The number of responses to generate. Defaults to 1.

    Returns:
        The generated responses from the OpenAI API.
    """

    if history is None:
        history = []

    def loop(f, params):
        retry = 0
        while retry < 7:
            try:
                rtn = f(params)
                return ModelResponses.load_from_openai(rtn)
            except Exception as e:
                if retry > 5:
                    print(f"Error {retry}: {e}\n{params}")
                if "maximum context length" in str(e):
                    print("Context length exceeded")
                    return None
                if (
                    "Rate limit" in str(e)
                    or "overloaded" in str(e)
                    or "timed out" in str(e)
                ):
                    if "timed out" in str(e) and retry < 2:
                        params["timeout"] += 30 * retry
                    elif retry < 1:
                        time.sleep(30 * (1 + retry))
                    else:
                        print(e)
                        return 0

                else:
                    time.sleep(3 * retry)
                retry += 1
                continue
        return None

    request_params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop,
    }

    if max_tokens != math.inf:
        request_params["max_tokens"] = max_tokens

    if timeout is not None:
        request_params["timeout"] = timeout

    if query_type == "chat":
        if system_prompt is not None and not len(history):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(message)},
            ]
        else:
            messages = [{"role": "user", "content": message}]
        request_params["messages"] = history + messages
        return loop(
            lambda x: client.chat.completions.create(**x), request_params
        )

    request_params["prompt"] = message
    return loop(lambda x: client.completions.create(**x), request_params)


def init_servers(number_of_processes=4, backend=gemini_chat_server):
    """
    Initializes multiple chat servers using multiprocessing.

    Args:
        number_of_processes (int): The number of server processes to start. Default is 4.

    Returns:
        tuple: A tuple containing a call queue and a global manager object.
    """
    global_manager = multiprocessing.Manager()
    call_queue = global_manager.Queue()

    for i in range(number_of_processes):
        p = multiprocessing.Process(
            target=backend,
            args=(call_queue, i == 0),
        )
        p.start()
        global_process_list.append(p)

    return call_queue, global_manager


def kill_servers():
    """
    Kill all processes
    """
    for p in global_process_list:
        p.terminate()
        p.join()


@dataclass
class ModelResponses:
    candidates: List[str] = field(default_factory=list)

    @classmethod
    def load_from_openai(cls, response):
        return cls(
            [
                {"content": c.message.content, "role": c.message.role}
                for c in response.choices
            ]
        )

    @classmethod
    def load_from_gemini(cls, response):
        obj = cls(
            [
                {
                    "content": "\n\n".join(
                        [p.text for p in c.content.parts if p.text]
                    ),
                    "role": c.content.role,
                }
                for c in response.candidates
                if c.finish_reason != "SAFETY"
            ]
        )
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


@dataclass
class OpenAITask:
    message: Union[str, List[str]] = ""
    max_tokens: float = math.inf
    query_type: str = "chat"
    model: str = "gpt-3.5-turbo"
    temperature: float = 1.0
    top_p: float = 1
    presence_penalty: float = 0
    frequency_penalty: float = 0
    system_prompt: Optional[str] = None
    stop: Optional[str] = None
    timeout: Optional[float] = None
    n: int = 1
    history: List[dict] = field(default_factory=list)

    def update_conversation(self, response, new_message):
        if not len(self.history) and self.system_prompt is not None:
            history = [{"role": "system", "content": self.system_prompt}]
        elif not len(self.history):
            history = []
        else:
            history = self.history

        history.append({"role": "user", "content": str(self.message)})
        history.append({"role": "assistant", "content": response})

        self.history = history
        self.message = new_message


class Caller:

    def __init__(
        self,
        parallelism=8,
        backend=gemini_chat_server,
        distribute_parallel_requests=False,
    ) -> None:
        self.queue, self.manager = init_servers(parallelism, backend=backend)
        self.results_queue = self.manager.Queue()
        self.distribute = distribute_parallel_requests

    def __del__(self):
        kill_servers()

    def _distribute_parallel_requests(self, tasks, use_tqdm=False, **kwargs):
        total_length = 0
        for t_id, task in enumerate(tasks):
            task_copy = OpenAITask(**task.__dict__)
            task_copy.n = 1
            for g_id in range(task.n):
                self.queue.put(
                    ((t_id, g_id), task_copy, self.results_queue, kwargs)
                )
                total_length += 1
        responses = [ModelResponses() for _ in tasks]
        for _ in tqdm(
            range(total_length),
            total=total_length,
            desc="Processing tasks",
            disable=not use_tqdm or total_length < 100,
        ):
            (t_id, _), task, resp = self.results_queue.get(block=True)
            if t_id == -1:
                breakpoint()
            responses[t_id].candidates.extend(resp.candidates)

        return responses

    def _dont_distribute_parallel_requests(
        self, tasks, use_tqdm=False, **kwargs
    ):
        for t_id, task in enumerate(tasks):
            self.queue.put((t_id, task, self.results_queue, kwargs))
        responses = [[] for _ in tasks]
        failed = set()
        for _ in tqdm(
            tasks,
            total=len(tasks),
            desc="Processing tasks",
            disable=not use_tqdm,
        ):
            idx, task, resp = self.results_queue.get(block=True)
            if t_id == -1:
                breakpoint()
            responses[idx] = resp

        for i in sorted(list(failed))[::-1]:
            del responses[i]

        return responses

    def __call__(
        self,
        tasks: Union[OpenAITask, List[OpenAITask]],
        use_tqdm=False,
        distribute_parallel_requests=None,
        **kwargs,
    ):

        if isinstance(tasks, OpenAITask):
            tasks = [tasks]
            return_single = True
        else:
            return_single = False

        if not tasks:
            return []

        distribute_parallel_requests = (
            distribute_parallel_requests
            if distribute_parallel_requests is not None
            else self.distribute
        )

        responses = (
            self._distribute_parallel_requests(tasks, use_tqdm, **kwargs)
            if distribute_parallel_requests
            else self._dont_distribute_parallel_requests(
                tasks, use_tqdm, **kwargs
            )
        )
        return responses if not return_single else responses[0]


def graceful_exit(_, __):
    """
    Kill all processes on SIGINT
    """
    kill_servers()
    exit()


signal.signal(signal.SIGINT, graceful_exit)
