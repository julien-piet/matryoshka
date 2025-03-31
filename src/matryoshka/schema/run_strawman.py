import os
import random
import re
from collections import defaultdict
from copy import deepcopy

import dill
import tqdm

from ..classes import (
    Module,
    Parser,
    Schema,
    SchemaEntry,
    VariableSemantics,
)
from ..genai_api.api import LLMTask
from ..utils.json import parse_json
from ..utils.logging import get_logger
from ..utils.prompts.naming_heuristic.creation import (
    gen_prompt as gen_schema_creation_prompt,
)
from .cluster_variables import ClusterVariables


class CreateAttributes(Module):

    def __init__(
        self,
        caller,
        parser,
        output_dir="output/",
        model="gemini-2.5-flash",
        cache_dir=".cache/",
        save_contents=False,
    ):
        super().__init__("CreateAttributes", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.save_contents = save_contents

        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.schema_mapping = (
            parser.schema_mapping if parser.schema_mapping else {}
        )
        self.schemas = parser.schemas if parser.schemas else []
        self.embedding = parser.embedding
        self.description_distance = None

        self.clusters = parser.clusters if parser.clusters else {}
        self.cluster_inverse_mapping = list(
            set([tuple(v) for v in self.clusters.values()])
        )

        self.paths = {
            "prompts": os.path.join(self.output_dir, "prompts/"),
            "outputs": os.path.join(self.output_dir, "outputs/"),
            "results": os.path.join(self.output_dir, "results/"),
        }
        os.makedirs(self.paths["prompts"], exist_ok=True)
        os.makedirs(self.paths["outputs"], exist_ok=True)
        os.makedirs(self.paths["results"], exist_ok=True)

        self.cluster_variables = ClusterVariables(
            caller, parser, output_dir, model
        )

    def _save_dill(self):
        with open(
            os.path.join(self.output_dir, "parser.dill"),
            "wb",
        ) as f:
            if self.save_contents:
                dill.dump(
                    Parser(
                        tree=self.tree,
                        values=self.values,
                        entries_per_template=self.entries_per_template,
                        embedding=self.embedding,
                        var_mapping=self.var_mapping,
                        schema_mapping=self.schema_mapping,
                        schemas=self.schemas,
                        clusters=self.clusters,
                    ),
                    f,
                )
            else:
                dill.dump(
                    Parser(
                        tree=self.tree,
                        var_mapping=self.var_mapping,
                        schema_mapping=self.schema_mapping,
                        schemas=self.schemas,
                        clusters=self.clusters,
                    ),
                    f,
                )

    def _write(self, query, response, template_id, save=True):
        # Placeholder for writing logic
        os.makedirs(
            os.path.join(self.paths["prompts"], "creation"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.paths["outputs"], "creation"), exist_ok=True
        )
        with open(
            os.path.join(self.paths["prompts"], "creation", str(template_id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(self.paths["outputs"], "creation", str(template_id)),
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
            local_matches = list(self.values[element_id].value_counts.keys())
            local_templates = [
                self.tree.gen_template(t)
                for t in self.tree.templates_per_node[element_id]
            ]
            local_element = self.tree.nodes[element_id]
            if (not len(local_templates)) or (not local_element):
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

    def parse_answer(self, candidate, task, template_id, force=False):
        if "//" in candidate:
            candidate = re.sub("//.*$", "", candidate)
        try:
            content = parse_json(
                candidate, self.caller, task=deepcopy(task), model=self.model
            )
        except ValueError:
            raise ValueError("The JSON is invalid")

        schema = self.fix_schema(Schema.from_json(content, template_id))
        return schema

    def controlled_execution(self, tasks, template_id, depth=0, force=False):
        rerun, responses = [], [None for _ in tasks]
        candidates = self.caller(tasks)
        candidates = [
            c.candidates[0] if not c.failed else None for c in candidates
        ]

        for task_idx, candidate in enumerate(candidates):
            if not candidate:
                continue
            schema = None
            try:
                schema = self.parse_answer(
                    candidate, tasks[task_idx], template_id, force=force
                )
            except Exception as e:
                rerun.append((task_idx, candidate, str(e)))
            responses[task_idx] = schema

        if not depth:
            if all(r is None for r in responses) and not force:
                return self.controlled_execution(
                    tasks, template_id, depth=depth, force=True
                )
            if any(r is None for r in responses):
                get_logger().warning("Some responses were invalid")
            return responses, candidates

        new_tasks = []
        for task_idx, candidate, e in rerun:
            task = tasks[task_idx]
            task.update_conversation(
                candidate,
                f"Your response is invalid. {e}. Please fix it and return a valid response",
            )
            task.timeout = max(300, 420 - 60 * depth)
            new_tasks.append(task)

        if new_tasks:
            fixed_responses, fixed_candidates = self.controlled_execution(
                new_tasks, template_id, depth=depth - 1
            )
            for new_idx, (idx, _, _) in enumerate(rerun):
                if fixed_responses[new_idx] is not None:
                    responses[idx] = fixed_responses[new_idx]
                    candidates[idx] = fixed_candidates[new_idx]

        return responses, candidates

    def fix_schema(self, schema):
        fix_list = []
        new_list = []
        for field_name, field_value in schema.fields.items():
            field_value.orig_ids = [
                i
                for i in field_value.orig_ids
                if i in self.tree.templates[schema.orig_template]
            ]
            if not field_value.orig_ids:
                fix_list.append(field_name)

            existing_names = {
                self.var_mapping[i].created_attribute
                for i in field_value.orig_ids
                if i in self.var_mapping
                and self.var_mapping[i].created_attribute is not None
            }
            if len(existing_names):
                fix_list.append(field_name)
                for name in existing_names:
                    ids = [
                        i
                        for i in field_value.orig_ids
                        if i in self.var_mapping
                        and self.var_mapping[i].created_attribute == name
                    ]
                    new_description = self.var_mapping[ids[0]].field_description
                    new_list.append(
                        (
                            field_name,
                            {
                                "description": new_description,
                                "ids": ids,
                            },
                        )
                    )

        for field_name in fix_list:
            del schema.fields[field_name]

        for entry_name, entry_attrs in new_list:
            schema_entry = SchemaEntry.from_json(entry_attrs, name=entry_name)
            schema.fields[entry_name] = schema_entry

        return schema

    def schema_creation(
        self,
        template_id,
        **kwargs,
    ):
        lines = random.sample(
            self.entries_per_template[template_id],
            k=min(5, len(self.entries_per_template[template_id])),
        )
        template = self.tree.gen_template(template_id)
        unmapped_elements = [
            node_id
            for node_id in self.tree.templates[template_id]
            if self.tree.nodes[node_id].is_variable()
            and (
                node_id not in self.var_mapping
                or self.var_mapping[node_id].created_attribute is None
            )
        ]
        existing_names = {
            val.created_attribute for val in self.var_mapping.values()
        }

        # Map element to fields
        kwargs["temperature"] = 0.2
        kwargs["n"] = 1

        user, history, system = gen_schema_creation_prompt(
            (lines, template),
            field_mapping={
                k: self.var_mapping[k].created_attribute
                for k in self.var_mapping
                if self.var_mapping[k].created_attribute is not None
            },
            descriptions={
                k: self.var_mapping[k].field_description
                for k in self.var_mapping
                if self.var_mapping[k].created_attribute is not None
            },
        )
        tasks = [
            LLMTask(
                system_prompt=system,
                model=self.model,
                message=user,
                history=history[:],
                thinking_budget=2048,
                **kwargs,
            )
            for _ in range(5)
        ]

        responses, candidates = self.controlled_execution(
            tasks, template_id, depth=2
        )
        valid_candidates = defaultdict(list)

        for schema_idx, schema in enumerate(responses):
            if not schema:
                continue
            new_fields = tuple(
                sorted(
                    [
                        field_name
                        for field_name, field in schema.fields.items()
                        if any(i in unmapped_elements for i in field.orig_ids)
                    ]
                )
            )
            valid_candidates[new_fields].append((schema_idx, schema))

        if not valid_candidates:
            get_logger().error("No valid schemas found")
            return None

        schema_idx, schema = valid_candidates[
            max(
                valid_candidates.keys(),
                key=lambda x: (
                    len(valid_candidates[x]),
                    -len([f for f in x if existing_names]),
                ),
            )
        ][0]

        self.schema_mapping[template_id] = schema

        self._write(
            query=(
                tasks[schema_idx].system_prompt
                + "\n\n__________\n\n"
                + "\n+++\n".join(
                    v["content"] for v in tasks[schema_idx].history or []
                )
                + "\n\n__________\n\n"
                + tasks[schema_idx].message
            ),
            response="\n\n##########\n\n".join(candidates),
            template_id=template_id,
            save=False,
        )

        updated_fields = set()
        for field_name, field in schema.fields.items():
            for node_id in field.orig_ids:
                if node_id not in self.var_mapping:
                    self.var_mapping[node_id] = VariableSemantics(
                        orig_node=node_id,
                        field_description=field.description,
                        created_attribute=field_name,
                    )
                    updated_fields.add(node_id)
                elif (
                    node_id in self.var_mapping
                    and self.var_mapping[node_id].created_attribute
                    != field_name
                ):
                    self.var_mapping[node_id].created_attribute = field_name
                    self.var_mapping[node_id].field_description = (
                        field.description
                    )
                    updated_fields.add(node_id)

        return schema

    def process(self, **kwargs):

        ## Create schemas for every template
        template_list = [
            t for t, elts in enumerate(self.tree.templates) if elts
        ]
        template_order = sorted(
            template_list,
            key=lambda x: len(self.entries_per_template[x]),
            reverse=True,
        )
        for template in tqdm.tqdm(
            template_order, total=len(template_order), desc="Creating Schemas"
        ):
            if template not in self.schema_mapping:
                self.schema_creation(template)

        self._save_dill()
        return Parser(
            tree=self.tree,
            values=self.values,
            entries_per_template=self.entries_per_template,
            embedding=self.embedding,
            var_mapping=self.var_mapping,
            schema_mapping=self.schema_mapping,
            schemas=self.schemas,
            clusters=self.clusters,
        )
