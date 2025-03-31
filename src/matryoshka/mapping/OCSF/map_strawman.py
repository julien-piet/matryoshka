import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import dill
import torch
from tqdm import tqdm

from matryoshka.classes import Mapping, Module, Parser, VariableSemantics
from matryoshka.genai_api.api import LLMTask
from matryoshka.utils.json import parse_json
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.naming_heuristic.mapping import (
    gen_prompt as gen_confirm_prompt,
)

sys.setrecursionlimit(sys.getrecursionlimit() * 10)


class MapToAttributes(Module):

    def __init__(
        self,
        caller,
        parser,
        output_dir="output/",
        model="gemini-2.5-flash",
        save_contents=False,
    ):
        super().__init__("MapToAttributes", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.save_contents = save_contents

        self.event_types = parser.event_types if parser.event_types else {}
        self.schema_mapping = (
            parser.schema_mapping if parser.schema_mapping else {}
        )
        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.schema_mapping = parser.schema_mapping
        self.embedding = parser.embedding
        self.description_distance = None

        self.clusters = {}

        self.paths = {
            "prompts": os.path.join(self.output_dir, "prompts/"),
            "outputs": os.path.join(self.output_dir, "outputs/"),
            "results": os.path.join(self.output_dir, "results/"),
        }
        os.makedirs(self.paths["prompts"], exist_ok=True)
        os.makedirs(self.paths["outputs"], exist_ok=True)
        os.makedirs(self.paths["results"], exist_ok=True)

    def _save_dill(self, suffix=""):
        with open(
            os.path.join(self.output_dir, f"parser{suffix}.dill"),
            "wb",
        ) as f:
            if self.save_contents:
                dill.dump(
                    Parser(
                        tree=self.tree,
                        values=self.values,
                        entries_per_template=self.entries_per_template,
                        event_types=self.event_types,
                        embedding=self.embedding,
                        var_mapping=self.var_mapping,
                        schema_mapping=self.schema_mapping,
                    ),
                    f,
                )
            else:
                dill.dump(
                    Parser(
                        tree=self.tree,
                        event_types=self.event_types,
                        embedding=self.embedding,
                        var_mapping=self.var_mapping,
                        schema_mapping=self.schema_mapping,
                    ),
                    f,
                )

    def _write(self, query, response, element, save=True):
        # Placeholder for writing logic
        os.makedirs(
            os.path.join(self.paths["prompts"], "mapping"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.paths["outputs"], "mapping"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.paths["results"], "mapping"), exist_ok=True
        )
        with open(
            os.path.join(self.paths["prompts"], "mapping", str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(self.paths["outputs"], "mapping", str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response)

        if save:
            self._save_dill()

    def _get_prompt_parameters(self, element_ids):
        if isinstance(element_ids, tuple):
            element_ids = list(element_ids)
        if not isinstance(element_ids, list):
            element_ids = [element_ids]
            return_list = False
        else:
            return_list = True

        templates, indices, elements, matches = [], [], [], []
        for element_id in element_ids:
            if element_id in self.values:
                local_matches = list(
                    self.values[element_id].value_counts.keys()
                )
            else:
                local_matches = [self.tree.nodes[element_id].value]
            local_templates = [
                self.tree.gen_template(t)
                for t in self.tree.templates_per_node[element_id]
            ]
            local_element = self.tree.nodes[element_id]
            if not local_templates:
                continue

            elt_index = 0
            for elt_id, element in enumerate(local_templates[0].elements):
                if element.id == element_id:
                    elt_index = elt_id
                    break

            templates.append(local_templates)
            indices.append(elt_index)
            elements.append(local_element)
            matches.extend(list(set(local_matches + [local_element.value])))

        matches = list(set(matches))
        if return_list:
            return templates, indices, elements, matches
        else:
            return templates[0], indices[0], elements[0], matches

    def matching(
        self,
        element_id,
        **kwargs,
    ):
        element_ids = [element_id]

        # Get element info
        templates, indices, elements, matches = self._get_prompt_parameters(
            element_ids
        )

        # Map element to fields
        kwargs["temperature"] = 0.2
        kwargs["n"] = 1

        user, history, system = gen_confirm_prompt(
            (templates, indices, elements, matches),
            self.var_mapping[element_id].field_description,
        )
        task = LLMTask(
            system_prompt=system,
            model=self.model,
            max_tokens=12000,
            thinking_budget=4096,
            message=user,
            history=history,
            **kwargs,
        )

        candidate = self.caller(task).candidates[0]

        try:
            proposal = parse_json(candidate, self.caller, task=task)
        except json.JSONDecodeError:
            return
        try:
            proposal = list(proposal)
        except AttributeError:
            return

        field_list = list(sorted(proposal))
        self.var_mapping[element_id].mapping = Mapping(field_list=field_list)

        self._write(
            query=(
                task.system_prompt
                + "\n\n__________\n\n"
                + "\n+++\n".join(v["content"] for v in task.history or [])
                + "\n\n__________\n\n"
                + task.message
            ),
            response=candidate,
            element=self.tree.nodes[element_id],
            save=False,
        )

    def process(self, log_file=None, **kwargs):

        for element_id in self.var_mapping:
            self.var_mapping[element_id].mapping = None

        # Process each element
        for element_id in tqdm(
            self.var_mapping,
            total=len(self.var_mapping),
            desc="Generating Mappings",
            unit="element",
        ):

            self.matching(element_id)

            get_logger().info(
                "%s ==> %s",
                self.var_mapping[element_id].created_attribute,
                str(self.var_mapping[element_id].mapping),
            )

        self._save_dill()
