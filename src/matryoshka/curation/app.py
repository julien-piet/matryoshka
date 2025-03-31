import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time

# ----------------- DateParser ----------------------------------------
from functools import wraps
from threading import Lock

import dill
from flask import Flask, jsonify, request, send_from_directory

from ..classes import Parser
from ..genai_api.api import Caller, backend_choices, get_backend
from ..syntax.run import VariableParser
from ..utils.logging import setup_logger
from ..utils.OCSF import (
    OCSFSchemaClient,
)
from ..utils.structured_log import TreeEditor

# ----------------- SaveManager ----------------------------------------


class SaveManager:
    def __init__(self, min_save_interval=5):
        self.min_save_interval = min_save_interval
        self.last_save_time = 0
        self.save_lock = Lock()
        self._loop = None
        self.save_queue = None
        self.is_saving = False

    def start(self, loop):
        """Initialize the save queue with the given event loop"""
        self._loop = loop
        self.save_queue = asyncio.Queue()

    async def process_save_queue(self, editor):
        """Process queued saves while respecting minimum intervals"""
        while True:
            try:
                # Wait for a save request
                await self.save_queue.get()

                # Check if we need to wait before saving
                time_since_last_save = time.time() - self.last_save_time
                if time_since_last_save < self.min_save_interval:
                    await asyncio.sleep(
                        self.min_save_interval - time_since_last_save
                    )

                # Perform the save operation with lock
                with self.save_lock:
                    if not self.is_saving:
                        self.is_saving = True
                        try:
                            editor.save()
                            self.last_save_time = time.time()
                        finally:
                            self.is_saving = False

                self.save_queue.task_done()
                print("Save operation completed.")

            except Exception as e:
                print(f"Error in save queue processing: {e}")
                await asyncio.sleep(1)

    def queue_save(self):
        """Queue a save operation"""
        if self._loop is None:
            print(
                "Warning: SaveManager not properly initialized with an event loop"
            )
            return

        if self.save_queue is not None and not self.save_queue.full():
            future = asyncio.run_coroutine_threadsafe(
                self.save_queue.put(True), self._loop
            )
            # Optionally handle future completion
            future.add_done_callback(
                lambda f: f.exception() if f.exception() else None
            )


def requires_save(f):
    """Decorator to queue a save after modifying operations"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if GLOBAL_EDITOR and hasattr(GLOBAL_EDITOR, "save_manager"):
            GLOBAL_EDITOR.save_manager.queue_save()
        return result

    return wrapper


# ----------------- Flask App: Expose the Editor as an API -------------
app = Flask(__name__)
GLOBAL_EDITOR = None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/api/get_node/<int:node_id>", methods=["GET"])
def api_get_node(node_id):
    return jsonify(GLOBAL_EDITOR.get_node(node_id))


@app.route("/api/set_node/<int:node_id>", methods=["POST"])
def api_set_node(node_id):
    """
    Manage node updates.
    Query param ?reparse=true/false controls whether to re-parse after changes.
    """
    reparse = request.args.get("reparse", "true").lower() == "true"
    data = request.json
    node_data = data["node"]
    force = data.get("force", False)
    new_parent_id = node_data.get("new_parent", None)
    missed = GLOBAL_EDITOR.set_node(
        node_id=node_id,
        node=node_data,
        force=force,
        new_parent=new_parent_id,
        reparse=reparse,
    )
    missed_lines = [GLOBAL_EDITOR.lines[idx[0]] for idx in missed]
    return jsonify(missed_lines)


@app.route("/api/add_node", methods=["POST"])
def api_add_node():
    """
    Add a node.
    Query param ?reparse=true/false controls whether to parse afterward.
    """
    reparse = request.args.get("reparse", "true").lower() == "true"
    data = request.json
    GLOBAL_EDITOR.add_node(
        data["new_node"],
        data["parent"],
        data.get("children", []),
        reparse=reparse,
    )
    return jsonify({"status": "ok"})


@app.route("/api/del_node/<int:node_id>", methods=["DELETE"])
def api_del_node(node_id):
    """
    Delete a single node by ID.
    """
    reparse = request.args.get("reparse", "true").lower() == "true"
    GLOBAL_EDITOR.del_node(node_id, reparse=reparse)
    return jsonify({"status": "ok"})


@app.route("/api/del_nodes", methods=["DELETE"])
def api_del_nodes():
    """
    Delete multiple nodes at once.
    """
    reparse = request.args.get("reparse", "true").lower() == "true"
    data = request.json
    node_ids = data.get("node_ids", [])
    GLOBAL_EDITOR.del_node(node_ids, reparse=reparse)
    return jsonify({"status": "ok"})


@app.route("/api/get_template/<int:template_id>", methods=["GET"])
def api_get_template(template_id):
    return jsonify(GLOBAL_EDITOR.get_template(template_id))


@app.route("/api/break", methods=["GET"])
def api_break():
    breakpoint()
    return jsonify({"status": "ok"})


@app.route("/api/add_template", methods=["POST"])
def api_add_template():
    data = request.json
    matched = GLOBAL_EDITOR.add_template(data)
    return jsonify(matched)


@app.route("/api/set_template/<int:template_id>", methods=["POST"])
def api_set_template(template_id):
    reparse = request.args.get("reparse", "true").lower() == "true"
    data = request.json
    matched = GLOBAL_EDITOR.set_template(template_id, data, reparse=reparse)
    return jsonify(matched)


@app.route("/api/del_template/<int:template_id>", methods=["DELETE"])
def api_del_template(template_id):
    reparse = request.args.get("reparse", "true").lower() == "true"
    GLOBAL_EDITOR.del_template(template_id, reparse=reparse)
    return jsonify({"status": "ok"})


@app.route("/api/get_tree", methods=["GET"])
def api_get_tree():
    """
    Return a simplistic JSON representation of the tree.
    We'll include trailing whitespace for hover info.
    """
    nodes_dict = {}
    for node_id, node in enumerate(GLOBAL_EDITOR.tree.nodes):
        if node:
            t_node = GLOBAL_EDITOR.tree.node_to_tree[node_id]
            nodes_dict[node_id] = {
                "id": node.id,
                "value": node.value,
                "is_variable": node.is_variable(),
                "regex": node.regexp,
                "trailing_whitespace": node.trailing_whitespace,
                "children": list(t_node.branches.keys()),
                "is_end_of_template": t_node.terminal,
                "parent": (t_node.parent.node if t_node.parent else 0),
            }
    # Add a synthetic "0" root pointer
    first_order_nodes = [
        nid for nid, node_item in nodes_dict.items() if node_item["parent"] == 0
    ]
    nodes_dict[0] = {
        "id": 0,
        "value": "",
        "is_variable": False,
        "regex": "",
        "trailing_whitespace": 0,
        "children": first_order_nodes,
        "is_end_of_template": False,
        "parent": None,
    }
    return jsonify(nodes_dict)


@app.route("/api/get_template_matches/<int:template_id>", methods=["GET"])
def api_get_template_matches(template_id):
    """
    Return up to 100 lines matched by a template.
    """
    line_ids = list(GLOBAL_EDITOR.line_id_per_template[template_id])
    return jsonify([GLOBAL_EDITOR.lines[lid] for lid in line_ids[:100]])


@app.route("/api/get_stats", methods=["GET"])
def api_get_stats():
    """
    Return stats: total lines, matched count, matched percentage.
    """
    total_lines = len(GLOBAL_EDITOR.lines)
    matched = set()
    for _, lids in GLOBAL_EDITOR.line_id_per_template.items():
        matched.update(lids)
    matched_count = len(matched)
    matched_percentage = (
        (matched_count / total_lines * 100) if total_lines else 0
    )
    return jsonify(
        {
            "total_lines": total_lines,
            "matched_count": matched_count,
            "matched_percentage": matched_percentage,
        }
    )


@app.route("/api/get_all_templates", methods=["GET"])
def api_get_all_templates():
    """
    Return each template + a subset of matched lines and all line_ids for overlap checks.
    """
    data = {"templates": {}}
    for t_id in range(len(GLOBAL_EDITOR.tree.templates)):
        nodes_json = GLOBAL_EDITOR.get_template(t_id)
        line_ids = list(GLOBAL_EDITOR.line_id_per_template[t_id])
        matched_lines = [GLOBAL_EDITOR.lines[lid] for lid in line_ids[:100]]
        data["templates"][t_id] = {
            "nodes": nodes_json,
            "matched_lines": matched_lines,
            "line_ids": line_ids,
        }
        data["templates"][t_id]["events"] = GLOBAL_EDITOR.get_template_events(
            t_id
        )
    return jsonify(data)


@app.route("/api/find_templates_ending_in/<int:node_id>", methods=["GET"])
def api_find_templates_ending_in(node_id):
    """
    Return all template IDs that end in node_id.
    """
    hits = []
    for t_id, templ in enumerate(GLOBAL_EDITOR.tree.templates):
        if templ and templ[-1] == node_id:
            hits.append(t_id)
    return jsonify(hits)


@app.route("/api/save", methods=["GET"])
def api_save():
    """
    Trigger an async save to disk.
    """
    if GLOBAL_EDITOR and hasattr(GLOBAL_EDITOR, "save_manager"):
        GLOBAL_EDITOR.save_manager.queue_save()
        return jsonify({"status": "save queued"})
    return (
        jsonify({"status": "error", "message": "Save manager not initialized"}),
        500,
    )


@app.route("/api/reparse", methods=["POST"])
def api_reparse():
    """
    Manually force a re-parse if user chooses not to auto reparse.
    """
    GLOBAL_EDITOR._validate_tree()
    GLOBAL_EDITOR._parse()
    GLOBAL_EDITOR._validate_tree()
    return jsonify({"status": "ok"})


@app.route("/api/set_node_semantics/<int:node_id>", methods=["POST"])
def api_set_node_semantics(node_id):
    """
    Set the semantics of a node.
    """
    data = request.json
    GLOBAL_EDITOR.set_node_semantics(node_id, data)
    return jsonify({"status": "ok"})


# Add an endpoint for batch semantics setting
@app.route("/api/batch_set_node_semantics", methods=["POST"])
def api_batch_set_node_semantics():
    data = request.json
    node_ids = data.get("node_ids", [])
    nodes_data = data.get("nodes", [])
    if len(node_ids) != len(nodes_data):
        return jsonify({"error": "Length mismatch"}), 400

    for nid, ndata in zip(node_ids, nodes_data):
        GLOBAL_EDITOR.set_node_semantics(nid, ndata)

    return jsonify({"status": "ok"})


@app.route("/api/get_all_nodes", methods=["GET"])
def api_get_all_nodes():
    all_nodes = {}
    for node_id, node in enumerate(GLOBAL_EDITOR.tree.nodes):
        if node:
            all_nodes[node_id] = GLOBAL_EDITOR.get_node(node_id)
    return jsonify(all_nodes)


@app.route("/api/get_template_events/<int:template_id>", methods=["GET"])
def api_get_template_events(template_id):
    return jsonify(GLOBAL_EDITOR.get_template_events(template_id))


@app.route("/api/set_template_events/<int:template_id>", methods=["POST"])
def api_set_template_events(template_id):
    data = request.json
    GLOBAL_EDITOR.set_template_events(int(template_id), data)
    return jsonify({"status": "ok"})


@app.route("/api/batch_set_template_events", methods=["POST"])
def api_batch_set_template_events():
    data = request.json
    for template_id, events in data.items():
        GLOBAL_EDITOR.set_template_events(int(template_id), events)
    return jsonify({"status": "ok"})


@app.route("/api/get_types", methods=["GET"])
def api_get_types():
    """
    Return the list of types.
    """
    return jsonify(GLOBAL_EDITOR.valid_types)


@app.route("/api/get_events", methods=["GET"])
def api_get_events():
    """
    Return the list of types.
    """
    return jsonify(GLOBAL_EDITOR.valid_events)


### Mapping API ###


@app.route("/api/set_attribute_mapping/", methods=["POST"])
def api_set_attribute_mapping():
    """
    Set the mappings of an attribute
    """
    data = request.json
    attribute_name = data.get("attribute_name", None)
    OCSF_fields = data.get("OCSF_fields", [])
    description = data.get("description", None)
    GLOBAL_EDITOR.set_attribute_mapping(
        attribute_name, OCSF_fields, description
    )
    return jsonify({"status": "ok"})


@app.route(
    "/api/get_attribute_mapping/<string:attribute_name>", methods=["GET"]
)
def api_get_attribute_mapping(attribute_name):
    """
    Get the existing mappings of an attribute
    """
    return jsonify(GLOBAL_EDITOR.get_attribute_mapping(attribute_name))


@app.route("/api/get_all_attribute_mappings", methods=["GET"])
def api_get_all_attribute_mappings():
    """
    Get all existing attribute mappings
    """
    return jsonify(GLOBAL_EDITOR.get_all_attribute_mappings())


@app.route("/api/get_mappings/<string:attribute_name>", methods=["GET"])
def api_get_possible_mappings(attribute_name):
    """
    Return the list of all possible mappings for an attribute
    """
    return jsonify(GLOBAL_EDITOR.get_all_possible_mappings(attribute_name))


@app.route(
    "/api/get_events_per_attribute/<string:attribute_name>", methods=["GET"]
)
def api_get_events_per_attribute(attribute_name):
    """
    Return the list of events associated with an attribute
    """
    return jsonify(GLOBAL_EDITOR.get_events_per_attribute(attribute_name))


@app.route("/api/get_all_field_names", methods=["GET"])
def api_get_all_field_names():
    """
    Return the list of all field names
    """
    return jsonify(GLOBAL_EDITOR.all_field_names)


@app.route("/api/run_query/", methods=["POST"])
def api_run_query():
    """
    Run a query on the log file
    """
    data = request.json
    query = data.get("query", None)
    matching_lines = GLOBAL_EDITOR.run_query(query)
    lines_and_ids = [
        (l_id, GLOBAL_EDITOR.lines[l_id]) for l_id in matching_lines
    ]
    return jsonify(lines_and_ids)


@app.route("/api/save_query/", methods=["POST"])
def api_save_query():
    """
    Save a query by name to a queries.json file in the 'self.output' directory.
    If the query name already exists, it is overwritten.
    """
    query_name = request.json.get("query_name", None)
    query = request.json.get("query", None)
    if not query_name or not isinstance(query, dict):
        return (
            jsonify({"status": "error", "message": "Invalid query data"}),
            400,
        )

    GLOBAL_EDITOR.save_query(query_name, query)
    return jsonify({"status": "ok"})


def main():
    global GLOBAL_EDITOR

    cli_parser = argparse.ArgumentParser(description="Editable Parser backend.")
    cli_parser.add_argument("--log_file", type=str)
    cli_parser.add_argument("--parser_file", type=str, default=None)
    cli_parser.add_argument("--size_reduction", type=int, default=-1)
    cli_parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
    cli_parser.add_argument("--output", type=str, default=None)
    cli_parser.add_argument("--file_percent", type=float, default=1.0)
    cli_parser.add_argument(
        "--listen_addr",
        type=str,
        default="127.0.0.1",
        help="Address to listen on",
    )
    cli_parser.add_argument(
        "--listen_port", type=int, default=5000, help="Port to listen on"
    )
    cli_parser.add_argument(
        "--thread_count",
        type=int,
        help="number of threads to use for LLM calls",
        default=16,
    )
    cli_parser.add_argument(
        "--backend",
        choices=backend_choices(),
        default=backend_choices()[0],
        help="Select the backend to use",
    )
    args = cli_parser.parse_args()

    if not args.config_file and not args.parser_file:
        raise ValueError("Please provide a config file or a parser path")

    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if args.parser_file is None:
            args.parser_file = config["results_path"] + "_parser/parser.dill"
        if args.output is None:
            args.output = config["results_path"] + "_parser_fixed/"
        args.log_file = config["data_path"]
    setup_logger()

    os.makedirs(args.output, exist_ok=True)

    # # Load existing parser tree from dill:
    # new_sys = {}
    # for key in sys.modules.keys():
    #     new_sys[key.replace("matryoshka", "logparser")] = sys.modules[key]
    # sys.modules["logparser.tools.classes"] = sys.modules["matryoshka.classes"]
    # sys.modules["logparser.tools"] = sys.modules["matryoshka.utils"]
    # sys.modules["logparser.tools.OCSF"] = sys.modules["matryoshka.utils.OCSF"]
    # sys.modules["logparser.tools.schema"] = sys.modules["matryoshka.classes"]
    # for key, value in new_sys.items():
    #     sys.modules[key] = value

    with open(args.parser_file, "rb") as f:
        parser = dill.load(f)

    # Remove unused parts of tree
    parser.tree.distances = []

    # Parse lines
    with open(args.log_file, "r", encoding="utf-8") as f:
        lines = f.read()
    all_lines = re.split("\n", lines)
    if args.file_percent < 1:
        all_lines = all_lines[: int(len(all_lines) * args.file_percent)]
    all_lines = [re.sub(r"\s", " ", line).strip() for line in all_lines if line]

    caller = Caller(
        args.thread_count,
        backend=get_backend(args.backend),
        distribute_parallel_requests=True,
    )
    GLOBAL_EDITOR = TreeEditor(
        parser,
        all_lines,
        output=args.output,
        caller=caller,
        client=OCSFSchemaClient(None),
        lines_per_template=args.size_reduction,
        save_manager=SaveManager(),
    )

    # Create and configure the event loop in the main thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize the save manager with the loop
    GLOBAL_EDITOR.save_manager.start(loop)

    # Start the save queue processor
    loop.create_task(
        GLOBAL_EDITOR.save_manager.process_save_queue(GLOBAL_EDITOR)
    )

    # Start the event loop in a separate thread
    def run_event_loop():
        loop.run_forever()

    event_loop_thread = threading.Thread(target=run_event_loop, daemon=True)
    event_loop_thread.start()

    # Run the Flask app
    try:
        app.run(
            host=args.listen_addr,
            port=args.listen_port,
            debug=True,
            use_reloader=False,
        )
    finally:
        # Clean up when the Flask app stops
        loop.call_soon_threadsafe(loop.stop)
        event_loop_thread.join(timeout=1.0)
        loop.close()


if __name__ == "__main__":
    main()
