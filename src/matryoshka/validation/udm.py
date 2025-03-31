import copy
import random
import re
import time
import traceback
from collections import defaultdict

from matryoshka.classes import Mapping
from matryoshka.genai_api.classes import LLMTask
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.mapping.UDM.validate import gen_prompt

from .validate import Validator


class MapAPI:
    def __init__(
        self,
        var_mapping,
        proposed_field_list,
        udm_client,
        allow_non_listed_fields=True,
    ):
        self.var_mapping = var_mapping
        self.proposed_field_list = proposed_field_list
        self.tracker = []
        self.errors = []
        self.new_mapping = copy.deepcopy(var_mapping)
        self.udm_client = udm_client
        self.allow_non_listed_fields = allow_non_listed_fields
        for value in self.new_mapping.values():
            value.mapping = None

        self.field_name_to_ids = defaultdict(list)
        for node_id, value in self.var_mapping.items():
            attr_name = value.created_attribute
            if (
                attr_name
                and attr_name != "SYNTAX"
                and not attr_name.endswith("_KEY")
            ):
                self.field_name_to_ids[attr_name].append(node_id)

    def field_exists(self, field_name):
        return field_name.lower() in self.field_name_to_ids

    def udm_field_exists(self, field_name):
        if self.allow_non_listed_fields:
            return field_name in self.udm_client.attributes
        else:
            return field_name in self.proposed_field_list

    def post_command(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            name = func.__name__.upper()
            self.tracker.append((name, f"Args: {args}, Kwargs: {kwargs}"))
            return result

        return wrapper

    @post_command
    def set_no_mapping(self, field_name):
        if not self.field_exists(field_name):
            self.errors.append(
                f"Field {field_name} does not exist in the original mapping."
            )
        for node_id in self.field_name_to_ids[field_name]:
            self.new_mapping[node_id].mapping = Mapping(
                field_list=[], demonstration="", candidates=[], type="UDM"
            )

    @post_command
    def map_field(self, field_name, udm_field):
        if not self.udm_field_exists(udm_field):
            self.errors.append(
                f"UDM field {udm_field} does not exist. Please only use valid UDM fields, which are leaves of the provided UDM tree."
            )
        if not self.field_exists(field_name):
            self.errors.append(
                f"Field {field_name} does not exist in the original mapping."
            )
        for node_id in self.field_name_to_ids[field_name]:
            if not self.new_mapping[node_id].mapping:
                self.new_mapping[node_id].mapping = Mapping(
                    field_list=[], demonstration="", candidates=[], type="UDM"
                )
            if udm_field not in self.new_mapping[node_id].mapping.field_list:
                self.new_mapping[node_id].mapping.field_list.append(udm_field)

    @post_command
    def map_fields(self, field_names, udm_field):
        for field_name in field_names:
            self.map_field(field_name, udm_field)


class MappingValidator(Validator):
    """Validator for template generation"""

    def __init__(
        self,
        caller,
        udm_client,
        parser=None,
        tree=None,
        var_mapping=None,
        model="gemini-2.5-pro",
        save_path="./saved_queries/",
        values=None,
        lines_per_template=5,
        entries_per_template=5,
        max_udm_attributes=1,
    ):
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
        self.udm_client = udm_client
        self.max_udm_attributes = max_udm_attributes

    def _extract_answer(self, response, *, schema=None):
        # Identify the code block
        code_block = re.compile(r"```python(.*?)```", re.DOTALL)
        match = code_block.search(response)
        if not match:
            raise ValueError("No code block found in response")
        return match.group(1).strip()

    def get_field_args(self, field_name, ids):
        # Get description
        description = self.var_mapping[ids[0]].field_description

        # Get values
        values = []
        for id in ids:
            if id in self.values:
                values.extend(list(self.values[id].value_counts.keys())[:5])
            else:
                values.append(self.tree.nodes[id].value)
        values = list(set(values))
        values = random.sample(values, min(len(values), 5))

        # Get templates
        templates = []
        for id in ids:
            templates.extend(list(self.tree.templates_per_node[id]))
        templates = list(set(templates))
        random.shuffle(templates)
        template_examples = []
        for template_id in templates:
            if len(template_examples) > 5:
                break
            if not self.entries_per_template.get(template_id, []):
                continue
            example_entry = random.sample(
                self.entries_per_template[template_id], 1
            )[0]
            target_field_indices = [
                n_id for n_id in self.tree.templates[template_id] if n_id in ids
            ]
            if not target_field_indices:
                continue
            highlighted_template = self.tree.gen_template(
                template_id
            ).highlight(
                target_field_indices[0],
                entry=example_entry,
                field_name=field_name.upper(),
                start_separator="<<<",
                end_separator=">>>",
            )
            template_examples.append(highlighted_template)

        # Get current fields
        udm_field_set = set()
        for node_id in ids:
            for udm_field in self.var_mapping[node_id].mapping.field_list:
                udm_field_set.add(udm_field)
        current_udm_fields = [
            (
                udm_field,
                self.udm_client.attributes[udm_field].global_description,
            )
            for udm_field in list(udm_field_set)
        ]

        # Get all proposed fields
        all_candidates = set()
        for node_id in ids:
            for candidate in self.var_mapping[node_id].mapping.candidates:
                all_candidates.add(candidate[0])
            for assigned_field in self.var_mapping[node_id].mapping.field_list:
                all_candidates.add(assigned_field)
        all_candidates = list(all_candidates)

        return (
            field_name,
            description,
            values,
            template_examples,
            current_udm_fields,
        ), all_candidates

    def _parse_answer(self, code_block, proposed_field_list, _, force=False):
        editor = MapAPI(self.var_mapping, proposed_field_list, self.udm_client)

        # Define the API functions
        api_function_map = {
            "SET_NO_MAPPING": editor.set_no_mapping,
            "MAP_FIELD": editor.map_field,
            "MAP_FIELDS": editor.map_fields,
        }

        # Run code on tree copy
        try:
            exec(code_block, api_function_map)
        except Exception:
            error = traceback.format_exc()
            raise Exception(f"An error occurred during execution: {error}")

        if editor.errors:
            error_string = "\n".join(editor.errors)
            raise ValueError(
                f"Errors occurred during execution:\n{error_string}"
            )

        if not editor.tracker and not force:
            raise ValueError(
                "No changes were made. Are you sure your code runs when it is passed to the `eval` function in python? If anything is nested in a function, make sure that function is being called."
            )

        return editor.new_mapping

    def _apply_changes(self, var_mapping):
        self.var_mapping = var_mapping

    def run(self, **kwargs):
        kwargs = self._prepare_kwargs(**kwargs) or {}
        if "model" not in kwargs:
            kwargs["model"] = self.model

        field_to_ids = defaultdict(list)
        for node_id, value in self.var_mapping.items():
            attr_name = value.created_attribute
            if (
                attr_name
                and attr_name != "SYNTAX"
                and not attr_name.endswith("_KEY")
            ):
                field_to_ids[attr_name].append(node_id)

        field_info = []
        candidate_fields = set()
        for field_name, ids in field_to_ids.items():
            prompt_info, local_candidates = self.get_field_args(field_name, ids)
            field_info.append(prompt_info)
            candidate_fields = candidate_fields.union(set(local_candidates))

        udm_tree = self.udm_client.create_networkx_graph(
            field_subset=candidate_fields, add_siblings=True
        )
        history, system = gen_prompt(
            field_info, udm_tree, max_field_count=self.max_udm_attributes
        )
        task = LLMTask(
            system_prompt=system,
            history=history,
            thinking_budget=4096,
            timeout=1600,
            **kwargs,
        )

        get_logger().debug(
            "%s - Running mapping validation.",
            time.strftime("%H:%M:%S", time.localtime()),
        )

        response = self.caller(task).candidates[0]
        new_var_mapping, response = self._self_correct(
            response, candidate_fields, task, None
        )
        self._write_llm_call(task, response)
        if not new_var_mapping:
            raise ValueError("No valid mapping returned from self-correct")
        return new_var_mapping
