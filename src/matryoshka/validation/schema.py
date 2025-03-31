import copy
import random
import re
import time
import traceback
from collections import defaultdict

from matryoshka.classes.semantics import VariableSemantics
from matryoshka.genai_api import api
from matryoshka.genai_api.classes import LLMTask, ModelResponses
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.schema.validate import (
    gen_prompt,
    graph_to_markdown,
)

from .validate import Validator


class SchemaAPI:

    def __init__(self, tree, var_mapping):
        self.tree = tree
        self.var_mapping = var_mapping
        self.tracker = []
        self.errors = []
        self.changes_per_node = {}

        self.previous_missing_keys = self.key_mismatch()

    def key_mismatch(self):
        # Check if there are broken key fields
        missing_keys = set()
        for template_id, template in enumerate(self.tree.templates):
            if not template:
                continue

            key_field_names = {}
            non_key_field_names = defaultdict(list)
            for node_id in template:
                if node_id not in self.var_mapping:
                    continue
                if not self.var_mapping[node_id].created_attribute:
                    continue
                if self.var_mapping[node_id].created_attribute.endswith("_KEY"):
                    key_field_names[
                        self.var_mapping[node_id].created_attribute
                    ] = node_id
                else:
                    non_key_field_names[
                        self.var_mapping[node_id].created_attribute
                    ].append(node_id)

            for key_field_name, key_node_id in key_field_names.items():
                field_name = key_field_name.replace("_KEY", "")
                if field_name not in non_key_field_names:
                    missing_keys.add((template_id, key_node_id))

        return missing_keys

    def node_exists(self, node_id):
        return node_id < len(self.tree.nodes) and (
            not node_id or self.tree.nodes[node_id]
        )

    def post_command(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            name = func.__name__.upper()
            self.tracker.append((name, f"Args: {args}, Kwargs: {kwargs}"))
            return result

        return wrapper

    def get_nodes_per_field(self, field_name, only_frozen=False):
        node_ids = []
        for k, v in self.var_mapping.items():
            if v.created_attribute == field_name:
                if only_frozen and not self.tree.nodes[k].fixed:
                    continue
                node_ids.append(k)
        return node_ids

    def log_error(self, error_msg):
        self.errors.append((error_msg, None, None))
        return False

    def save_changes(self, node_id, operation):
        if node_id not in self.changes_per_node:
            self.changes_per_node[node_id] = set()
        self.changes_per_node[node_id].add(operation)

    @post_command
    def set_syntax(self, node_id):
        if not isinstance(node_id, int):
            return self.log_error(
                f"Node ids must be numbers. You provided {node_id}"
            )

        if not self.node_exists(node_id):
            return self.log_error(f"Node #{node_id} does not exist")

        if self.tree.nodes[node_id].fixed:
            return self.log_error(
                f"Node #{node_id} is frozen and cannot be edited"
            )

        if node_id in self.var_mapping:
            self.var_mapping[node_id].created_attribute = "SYNTAX"
            self.var_mapping[node_id].field_description = "SYNTAX"
        else:
            self.var_mapping[node_id] = VariableSemantics(
                orig_node=node_id,
                created_attribute="SYNTAX",
                field_description="SYNTAX",
            )

        self.save_changes(node_id, "SET_SYNTAX")

    @post_command
    def set_key_field(self, node_id, value_node_id):
        if not isinstance(node_id, int):
            return self.log_error(
                f"Node ids must be numbers. You provided {node_id}"
            )

        if not self.node_exists(node_id):
            return self.log_error(f"Node #{node_id} does not exist")

        if self.tree.nodes[node_id].fixed:
            return self.log_error(
                f"Node #{node_id} is frozen and cannot be edited"
            )

        if not isinstance(value_node_id, int):
            return self.log_error(
                f"Value node ids must be numbers. You provided {value_node_id}"
            )

        if not self.node_exists(value_node_id):
            return self.log_error(f"Value node #{value_node_id} does not exist")

        if (
            value_node_id not in self.var_mapping
            or not self.var_mapping[value_node_id].created_attribute
        ):
            return self.log_error(
                f"The value node {value_node_id} does not have a field name"
            )

        value_name = self.var_mapping[value_node_id].created_attribute

        if value_name == "SYNTAX":
            return self.log_error(
                "You cannot create a key field for a value that is a SYNTAX constant"
            )
        elif value_name.endswith("_KEY"):
            return self.log_error(
                "You cannot create a key field for a value that is a key field"
            )

        if node_id not in self.var_mapping:
            self.var_mapping[node_id] = VariableSemantics(
                orig_node=node_id,
                created_attribute=value_name + "_KEY",
                field_description="KEY",
            )
        else:
            self.var_mapping[node_id].created_attribute = value_name + "_KEY"
            self.var_mapping[node_id].field_description = "KEY"

        self.save_changes(node_id, "SET_KEY_FIELD")

    @post_command
    def set_existing_field(self, node_id, field_name):
        if not isinstance(node_id, int):
            return self.log_error(
                f"Node ids must be numbers. You provided {node_id}"
            )

        if not self.node_exists(node_id):
            return self.log_error(f"Node #{node_id} does not exist")

        if self.tree.nodes[node_id].fixed:
            return self.log_error(
                f"Node #{node_id} is frozen and cannot be edited"
            )

        if not field_name:
            return self.log_error("Field name must be non-empty")

        if field_name == "SYNTAX":
            return self.log_error(
                f"Use SET_SYNTAX to set node {node_id} to SYNTAX"
            )

        if field_name.endswith("_KEY"):
            return self.log_error(
                f"Use SET_KEY_FIELD to mark node {node_id} as a key for another node"
            )

        existing_nodes = self.get_nodes_per_field(field_name)
        if not existing_nodes:
            return self.log_error(f"Field {field_name} does not exist")

        description = self.var_mapping[existing_nodes[0]].field_description

        if node_id not in self.var_mapping:
            self.var_mapping[node_id] = VariableSemantics(
                created_attribute=field_name,
                orig_node=node_id,
                field_description=description,
            )
        else:
            self.var_mapping[node_id].created_attribute = field_name
            self.var_mapping[node_id].field_description = description

        self.save_changes(node_id, "SET_EXISTING_FIELD")

    @post_command
    def set_new_field_name(self, node_id, field_name, description):
        if not isinstance(node_id, int):
            return self.log_error(
                f"Node ids must be numbers. You provided {node_id}"
            )

        if not self.node_exists(node_id):
            return self.log_error(f"Node #{node_id} does not exist")

        if self.tree.nodes[node_id].fixed:
            return self.log_error(
                f"Node #{node_id} is frozen and cannot be edited"
            )

        if not field_name:
            return self.log_error("Field name must be non-empty")

        if field_name == "SYNTAX":
            return self.log_error(
                f"Use SET_SYNTAX to set node {node_id} to SYNTAX"
            )

        if field_name.endswith("_KEY"):
            return self.log_error(
                f"Use SET_KEY_FIELD to mark node {node_id} as a key for another node."
            )

        if node_id in self.var_mapping:
            if (
                self.var_mapping[node_id].created_attribute == field_name
                and self.var_mapping[node_id].field_description == description
            ):
                return

        existing_nodes = self.get_nodes_per_field(field_name, only_frozen=True)
        if existing_nodes:
            templates = set()
            for node in existing_nodes:
                templates = templates.union(
                    set(self.tree.templates_per_node[node])
                )
            templates = list(templates)[:5]
            example_lines = []
            for template in templates:
                if not self.tree.templates[template]:
                    continue
                included_nodes = [
                    n
                    for n in existing_nodes
                    if n in self.tree.templates[template]
                ]
                if not included_nodes:
                    continue
                example_lines.append(
                    self.tree.gen_template(template).highlight(
                        included_nodes[0]
                    )
                )
            existing_description = self.var_mapping[
                existing_nodes[0]
            ].field_description
            error_msg = f"Field {field_name} already exists, is linked to nodes {existing_nodes}, and is described as {existing_description}"
            self.errors.append((error_msg, templates, example_lines))
            return

        if node_id not in self.var_mapping:
            self.var_mapping[node_id] = VariableSemantics(
                created_attribute=field_name,
                orig_node=node_id,
                field_description=description,
            )
        else:
            self.var_mapping[node_id].created_attribute = field_name
            self.var_mapping[node_id].field_description = description

        self.save_changes(node_id, "SET_NEW_FIELD_NAME")

    def change_description(self, field_name, description):
        node_ids = []
        for node_id in self.var_mapping:
            if self.var_mapping[node_id].created_attribute == field_name:
                node_ids.append(node_id)

        if not node_ids:
            return self.log_error(
                f"Field {field_name} does not exist, its description cannot be changed."
            )

        for node_id in node_ids:
            self.var_mapping[node_id].field_description = description


class SchemaValidator(Validator):
    """Validator for template generation"""

    def __init__(
        self,
        caller,
        parser=None,
        tree=None,
        var_mapping=None,
        model="gemini-2.5-pro",
        save_path="./saved_queries/",
        values=None,
        lines_per_template=5,
        entries_per_template=None,
    ) -> None:
        if not tree and not parser:
            raise ValueError("Either tree or parser must be provided.")
        if var_mapping is None and not parser:
            raise ValueError("Var mapping must be provided.")
        super().__init__(
            "syntax_validator",
            caller,
            parser,
            tree,
            model,
            save_path,
            values,
            entries_per_template,
            parser.var_mapping if parser else var_mapping,
        )
        self.lines_per_template = lines_per_template

    def _extract_answer(self, response, *, schema=None):
        # Identify the code block
        code_block = re.compile(r"```python(.*)```", re.DOTALL)
        match = code_block.search(response)
        if not match:
            raise ValueError("No code block found in response")
        return match.group(1).strip()

    def print_schema(self, template_id):
        resp = ""
        template = self.tree.gen_template(template_id)
        resp += f"#{template_id}: {template}"
        created_attributes = []
        for node_id in self.tree.templates[template_id]:
            if node_id in self.var_mapping:
                if self.var_mapping[node_id].created_attribute:
                    created_attributes.append(
                        self.var_mapping[node_id].created_attribute
                    )
                else:
                    created_attributes.append("VAR")
            else:
                created_attributes.append("CST")
        resp += "\n" + " | ".join(created_attributes)
        print(resp)
        return resp

    def _parse_answer(
        self, code_block, original_template_ids, lines, force=False
    ):
        editor = SchemaAPI(self.tree, copy.deepcopy(self.var_mapping))

        # Define the API functions
        api_function_map = {
            "SET_SYNTAX": editor.set_syntax,
            "SET_KEY_FIELD": editor.set_key_field,
            "SET_EXISTING_FIELD": editor.set_existing_field,
            "SET_NEW_FIELD_NAME": editor.set_new_field_name,
            "CHANGE_DESCRIPTION": editor.change_description,
        }

        # Run code on tree copy
        try:
            exec(code_block, api_function_map)
        except Exception as ex:
            error = traceback.format_exc()
            raise Exception(f"An error occurred during execution: {error}")

        # Check if nodes where changes multiple times
        for node_id, ops in editor.changes_per_node.items():
            if len(ops) > 1:
                editor.log_error(
                    f"Node #{node_id} was the target of multiple operations: {ops}. Please only change each node at most once to its final state."
                )

        if editor.errors:
            # Handle existing field errors
            existing_field_errors = [
                e for e in editor.errors if e[1] is not None
            ]
            lines = "\n".join(
                [
                    f"{e[0]}. It is highlighted using three stars (***) in the following lines: {e[2]}"
                    for e in existing_field_errors
                ]
            )
            error_string = f"""
{lines}

For each of the pre-existing names, consider its description.
* If the description matches the current field you are trying to name, you can keep this current field name and use SET_EXISTING_FIELD to set the field to an existing field name.
* If the description does not match and the original fields serve a different purpose than the new ones, create a new field name distinct from the pre-existing name for all relevant attributes.
* If the original fields serve the same purpose as the new fields, but the original description needs to be changed, you can use CHANGE_DESCRIPTION to do so.
"""
            existing_field_error_string = f"Some of the new field names you assigned already exist in the tree. {error_string}"

            # Handle other errors
            other_errors = [e for e in editor.errors if e[1] is None]
            other_error_string = "\n".join([e[0] for e in other_errors])

            if other_error_string and not existing_field_errors:
                raise ValueError(
                    f"The following errors occurred during execution:\n{other_error_string}"
                )
            elif existing_field_errors and not other_error_string:
                raise ValueError(f"{existing_field_error_string}")
            else:
                raise ValueError(
                    f"The following errors occurred during execution:\n{other_error_string}\n{existing_field_error_string}"
                )

        # Check if there are broken key fields
        missing_keys = editor.key_mismatch()
        new_missing_keys = missing_keys - editor.previous_missing_keys
        new_missing_keys = {
            (template_id, key_node_id)
            for template_id, key_node_id in new_missing_keys
            if template_id in original_template_ids
        }
        new_missing_keys_errors = []
        for key_node_id in new_missing_keys:
            key_field_name = editor.var_mapping[key_node_id].created_attribute
            field_name = editor.var_mapping[
                key_node_id
            ].created_attribute.replace("_KEY", "")
            new_missing_keys_errors.append(
                f"Node {key_node_id} is marked as a key with name {key_field_name}, but no other field in the same branch has name {field_name}."
            )
        if new_missing_keys_errors:
            new_missing_keys_errors = "\n".join(new_missing_keys_errors)
            raise ValueError(
                "The following fields are marked as keys, but after your changes there are no longer value fields with the corresponding name. Please fix these mistakes. Remember, keys must be set AFTER setting the field name of the main value.\n{new_missing_keys_errors}"
            )

        return editor.var_mapping

    def _explain_changes(self, self_old_mapping, new_mapping, tree_md):
        return_str = ""
        if not new_mapping:
            new_mapping = {}

        all_node_ids = set(self_old_mapping.keys()) | set(new_mapping.keys())
        for node_id in all_node_ids:
            old_semantics = self_old_mapping.get(node_id, None)
            new_semantics = new_mapping.get(node_id, None)
            if old_semantics and old_semantics.created_attribute == "SYNTAX":
                old_semantics = None
            if new_semantics and new_semantics.created_attribute == "SYNTAX":
                new_semantics = None

            if not old_semantics and not new_semantics:
                continue
            if not old_semantics and new_semantics:
                return_str += f"New attribute for node #{node_id}: {new_semantics.created_attribute} ({new_semantics.field_description})\n"
            elif new_semantics and old_semantics:
                if not new_semantics and old_semantics:
                    return_str += f"Removed attribute for node #{node_id}: {old_semantics.created_attribute}\n"
                if (
                    old_semantics
                    and new_semantics
                    and old_semantics.created_attribute
                    != new_semantics.created_attribute
                ):
                    return_str += f"Renamed attribute for node #{node_id} from {old_semantics.created_attribute} to {new_semantics.created_attribute}\n"

        return_str += f"####\n\nold tree\n####\n\n{tree_md}\n"

        return return_str

    def _apply_changes(self, var_mapping):
        self.var_mapping = var_mapping

    def run(self, template_ids, json_tree=False, **kwargs):
        kwargs = self._prepare_kwargs(**kwargs) or {}
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Filter to only keep templates with matches
        template_ids = [
            template_id
            for template_id in template_ids
            if self.entries_per_template.get(template_id, [])
        ]

        # Build field to name and description mappings
        field_to_name = {}
        field_to_description = {}
        for var_id, var_map in self.var_mapping.items():
            if not var_map:
                continue
            if var_map.created_attribute:
                field_to_name[var_id] = var_map.created_attribute
            if var_map.field_description:
                field_to_description[var_id] = var_map.field_description

        # Build prompt
        entries = []
        for template_id in template_ids:
            all_matches = self.entries_per_template.get(template_id, [])
            selected = random.sample(
                all_matches, min(self.lines_per_template, len(all_matches))
            )
            entries.extend(selected)

        tree = self.tree.create_networkx_graph(
            template_ids,
            field_names=field_to_name,
            field_descriptions=field_to_description,
            include_placeholders=False,
        )

        history, system = gen_prompt(entries, tree, json_tree=json_tree)
        task = api.LLMTask(
            system_prompt=system,
            history=history,
            thinking_budget=4096,
            timeout=1600,
            **kwargs,
        )

        get_logger().debug(
            "%s - Running schema validation.",
            time.strftime("%H:%M:%S", time.localtime()),
        )

        raw_response = self.caller(task)
        if raw_response.failed:
            raise ValueError("Generation failed")
        response = raw_response.candidates[0]

        new_var_mapping, response = self._self_correct(
            response, template_ids, task, entries
        )
        tree_md = graph_to_markdown(
            self.tree.create_networkx_graph(
                template_ids,
                field_names=field_to_name,
                field_descriptions=field_to_description,
                include_placeholders=False,
            ),
        )
        self._write_llm_call(
            task,
            response,
            self._explain_changes(self.var_mapping, new_var_mapping, tree_md),
        )

        if not new_var_mapping:
            raise ValueError("No tree returned from self-correct")
        return new_var_mapping
