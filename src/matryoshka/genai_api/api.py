import os
import signal
import threading
from queue import Queue
from typing import List, Union

from tqdm import tqdm

from ..utils.logging import get_logger
from .anthropic import anthropic_chat_server
from .classes import LLMTask, ModelResponses
from .globals import global_call_queue, global_process_list, global_stop_event
from .google import gemini_chat_server
from .openai import openai_chat_server


def backend_choices():
    return ["google", "openai", "anthropic"]


def get_backend(backend_name):
    """
    Returns the backend function based on the provided backend name.

    Args:
        backend_name (str): The name of the backend to retrieve.

    Returns:
        function: The backend function corresponding to the provided name.
    """
    if backend_name == "google":
        return gemini_chat_server
    elif backend_name == "openai":
        return openai_chat_server
    elif backend_name == "anthropic":
        return anthropic_chat_server
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def init_servers(number_of_processes=4, backend=gemini_chat_server):
    """
    Initializes multiple chat servers using multiprocessing.

    Args:
        number_of_processes (int): The number of server processes to start. Default is 4.

    Returns:
        tuple: A tuple containing a call queue and a global manager object.
    """
    global global_process_list
    global global_call_queue

    call_queue = Queue()

    for i in range(number_of_processes):
        p = threading.Thread(
            target=backend,
            args=(call_queue, i == 0),
        )
        p.environ = os.environ.copy()
        p.start()
        global_process_list.append(p)

    global_call_queue = call_queue
    return call_queue


def kill_servers():
    """
    Kill all processes
    """
    global global_process_list
    global global_call_queue
    global_stop_event.set()

    # Add a termination event to the queue
    if global_call_queue is not None:
        for _ in range(len(global_process_list)):
            global_call_queue.put(None)

    # Terminate all processes
    for p in global_process_list:
        p.join()

    global_process_list.clear()
    get_logger().info("All processes have been terminated.")


class Caller:

    def __init__(
        self,
        parallelism=8,
        backend=gemini_chat_server,
        distribute_parallel_requests=False,
    ) -> None:
        self.queue = init_servers(parallelism, backend=backend)
        self.results_queue = Queue()
        if backend == openai_chat_server:
            distribute_parallel_requests = True
        self.distribute = distribute_parallel_requests

    def __del__(self):
        kill_servers()

    def _distribute_parallel_requests(self, tasks, use_tqdm=False, **kwargs):
        total_length = 0
        for t_id, task in enumerate(tasks):
            task_copy = LLMTask(**task.__dict__)
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
            if t_id != -1:
                responses[t_id].candidates.extend(resp.candidates)
            else:
                get_logger().error(
                    f"Failed to process task: {task.message}. Error: {resp.candidates[0]}"
                )
                responses[t_id] = ModelResponses.default_failed()
                continue

        return responses

    def _dont_distribute_parallel_requests(
        self, tasks, use_tqdm=False, **kwargs
    ):
        for t_id, task in enumerate(tasks):
            self.queue.put((t_id, task, self.results_queue, kwargs))
        responses = [[] for _ in tasks]
        for _ in tqdm(
            tasks,
            total=len(tasks),
            desc="Processing tasks",
            disable=not use_tqdm,
        ):
            idx, task, resp = self.results_queue.get(block=True)
            if t_id != -1:
                responses[idx] = resp
            else:
                get_logger().error(
                    f"Failed to process task: {task.message}. Error: {resp.candidates[0]}"
                )
                responses[idx] = ModelResponses.default_failed()

        return responses

    def __call__(
        self,
        tasks: Union[LLMTask, List[LLMTask]],
        use_tqdm=False,
        distribute_parallel_requests=None,
        **kwargs,
    ):

        if isinstance(tasks, LLMTask):
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
    get_logger().info("Received SIGINT, shutting down servers...")
    kill_servers()
    exit()


signal.signal(signal.SIGINT, graceful_exit)
