import threading

global_process_list = []
global_stop_event = threading.Event()
global_call_queue = None
