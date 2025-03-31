import json
import os
import random
import re
import traceback
from collections import defaultdict
from copy import deepcopy

import dill
import torch

from matryoshka.validation.schema import SchemaValidator

from ..classes import (
    Module,
    Parser,
    Schema,
    SchemaEntry,
    VariableSemantics,
)
from ..genai_api.api import LLMTask
from ..utils.embedding import DescriptionDistance
from ..utils.json import parse_json
from ..utils.logging import get_logger
from ..utils.prompts.schema.creation import (
    gen_prompt as gen_schema_creation_prompt,
)
from ..utils.prompts.schema.description import CONSTANTS
from ..utils.prompts.schema.description import (
    gen_prompt as gen_schema_description_prompt,
)
from ..utils.prompts.schema.label_variables import (
    gen_prompt as gen_label_prompt,
)
from .cluster_variables import ClusterVariables


class CreateAttributes(Module):

    def __init__(
        self,
        caller,
        parser,
        log_description,
        few_shot_len=2,
        output_dir="output/",
        model="gemini-2.5-flash",
        validation_model="gemini-2.5-pro",
        use_description_distance=True,
        cache_dir=".cache/",
        save_contents=False,
        validation_size=25,
        ablation_fewshot=False,
        ablation_self_correction=False,
    ):
        super().__init__("CreateAttributes", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.few_shot_len = few_shot_len
        self.cache_dir = cache_dir
        self.save_contents = save_contents
        self.ablation_fewshot = ablation_fewshot
        self.ablation_self_correction = ablation_self_correction

        self.log_description = log_description

        self.use_description_distance = use_description_distance

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

        # Validation
        if validation_size > 0:
            self.validate_heuristic = SchemaValidator(
                caller,
                tree=self.tree,
                var_mapping=self.var_mapping,
                entries_per_template=self.entries_per_template,
                values=self.values,
                save_path=output_dir,
                model=validation_model,
            )
            self.validation_size = validation_size
        else:
            self.validate_heuristic = None
            self.validation_size = 0

        self.validation_size = int(validation_size)

        self.validated_templates = []
        for t_id, template in enumerate(self.tree.templates):
            if not template:
                continue
            if all(self.tree.nodes[e_id].fixed for e_id in template):
                self.validated_templates.append(t_id)

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

    def _write_desc(self, query, response, element, description):
        # Placeholder for writing logic
        os.makedirs(
            os.path.join(self.paths["prompts"], "description"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.paths["outputs"], "description"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.paths["results"], "description"),
            exist_ok=True,
        )
        with open(
            os.path.join(self.paths["prompts"], "description", str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(self.paths["outputs"], "description", str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response)

        with open(
            os.path.join(self.paths["results"], "description", str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{element.id}\n{element.value}\n{description}")

    def _write_schema_desc(self, t_id, query, response):
        # Placeholder for writing logic
        os.makedirs(
            os.path.join(self.paths["prompts"], "description_schema"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.paths["outputs"], "description_schema"),
            exist_ok=True,
        )
        with open(
            os.path.join(
                self.paths["prompts"], "description_schema", str(t_id)
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(
                self.paths["outputs"], "description_schema", str(t_id)
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response)

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

    def generate_descriptions(
        self,
        **kwargs,
    ):
        # Generate descriptions for all elements
        tasks = {}
        kwargs["temperature"] = 0.25
        kwargs["n"] = 4
        for t_id, template in enumerate(self.tree.templates):
            if not template:
                continue
            lines = random.sample(
                self.entries_per_template[t_id],
                k=min(5, len(self.entries_per_template[t_id])),
            )
            if not lines:
                continue
            template = self.tree.gen_template(t_id)
            user, history, system = gen_schema_description_prompt(
                (lines, template), self.log_description
            )
            if self.ablation_fewshot:
                history = []
            tasks[t_id] = LLMTask(
                system_prompt=system,
                history=history,
                model=self.model,
                message=user,
                thinking_budget=128,
                **kwargs,
            )

        task_ids = list(tasks.keys())
        tasks_array = [tasks[v] for v in task_ids]

        get_logger().info(
            "Generating descriptions for %d templates", len(tasks)
        )
        response = self.caller(
            tasks_array,
            use_tqdm=True,
        )

        descriptions = defaultdict(list)
        # Parse the outputs
        for resp_id, raw_llm_response in enumerate(response):
            task_id = task_ids[resp_id]
            task = tasks_array[resp_id]
            self._write_schema_desc(
                task_id,
                query=(
                    task.system_prompt
                    + "\n\n__________\n\n"
                    + "\n+++\n".join(v["content"] for v in task.history or [])
                    + "\n\n__________\n\n"
                    + task.message
                ),
                response="\n\n###########\n\n".join(
                    candidate for candidate in raw_llm_response.candidates
                ),
            )
            if raw_llm_response.failed:
                continue
            for candidate in raw_llm_response.candidates:
                try:
                    content = parse_json(
                        candidate,
                        self.caller,
                        task=deepcopy(task),
                        model=self.model,
                    )
                except ValueError:
                    continue

                try:
                    content = {int(k): v for k, v in content.items()}
                except ValueError:
                    continue

                # Sanity check
                candidate_ids = {str(i) for i in content}
                requested_ids = {
                    str(elt.id)
                    for elt in self.tree.gen_template(task_id).elements
                }
                if not candidate_ids.issubset(requested_ids):
                    get_logger().warning(
                        f"Description generation for template #{task_id} returned different ids than the requested template."
                    )
                    continue

                # Save descriptions
                for node_id, description in content.items():
                    if (
                        node_id
                        and node_id < len(self.tree.nodes)
                        and not self.tree.nodes[node_id]
                    ):
                        breakpoint()
                    if (
                        node_id
                        and node_id < len(self.tree.nodes)
                        and self.tree.nodes[node_id].is_variable()
                    ):
                        descriptions[node_id].append(description.strip())

        # Embed the descriptions
        node_order = sorted(descriptions.keys())
        orig_nodes = [
            node_id for node_id in node_order for _ in descriptions[node_id]
        ]
        if self.use_description_distance:
            all_descriptions = [
                desc for node_id in node_order for desc in descriptions[node_id]
            ]
        else:
            all_descriptions = [
                (
                    self.tree.nodes[node_id].value
                    if self.tree.nodes[node_id]
                    else "No Value"
                )
                for node_id in node_order
                for desc in descriptions[node_id]
            ]

        # Setup the LLMTasks for embedding
        chunk_size = 100
        tasks = []
        for i in range(0, len(all_descriptions), chunk_size):
            description_chunk = all_descriptions[i : i + chunk_size]
            tasks.append(
                LLMTask(
                    message=description_chunk,
                    query_type="embedding",
                    model="text-embedding-005",
                )
            )

        response = self.caller(
            tasks, distribute_parallel_requests=False, use_tqdm=False
        )
        all_embeddings = [torch.tensor(emb) for r in response for emb in r]
        embeddings = {node_id: [] for node_id in descriptions}
        for node_id, emb in zip(orig_nodes, all_embeddings):
            embeddings[node_id].append(emb)

        for node_id, local_descriptions in descriptions.items():
            # Get the embedding that is closest to the average of all embeddings
            local_embeddings = embeddings[node_id]
            average_embedding = torch.stack(local_embeddings).mean(dim=0)
            embedding_distances = torch.nn.CosineSimilarity()(
                average_embedding, torch.stack(local_embeddings)
            )
            closest_embedding = torch.argmax(embedding_distances).item()

            # Select this embedding and description
            description, embedding = (
                local_descriptions[closest_embedding],
                local_embeddings[closest_embedding],
            )

            # Save to the variable mapping
            self.var_mapping[node_id] = VariableSemantics(
                orig_node=node_id,
                field_description=description,
                embedding=embedding,
            )

        # Remove descriptions of uninformative constants
        for node_id, node in enumerate(self.tree.nodes):
            if node and not node.is_variable():
                if node_id in self.var_mapping:
                    if self.var_mapping[node_id].field_description in CONSTANTS:
                        del self.var_mapping[node_id]

        # Check for missing descriptions
        for node_id, node in enumerate(self.tree.nodes):
            if node and node.is_variable() and node_id not in self.var_mapping:
                get_logger().warning(
                    "Missing description for %s (%s)", node.value, node_id
                )
                try:
                    self.generate_description_element(node_id)
                except Exception as e:
                    traceback.print_exc()
                    breakpoint()

    def generate_description_element(self, element_id, **kwargs):

        if not self.tree.nodes[element_id].is_variable():
            return

        # Get element info
        templates, elt_index, element, matches = self._get_prompt_parameters(
            element_id
        )

        # Get list of few-shot examples
        few_shot_nodes = [
            val.element_id
            for val in self.values[element_id].get_closest(
                [
                    val
                    for id, val in self.values.items()
                    if id in self.var_mapping and id != element_id
                ],
                k=self.few_shot_len,
                tree=self.tree,
                embedding=self.embedding,
            )
        ]

        # Get relevant info for each few-shot example: original event, elt info, and solution
        few_shot_examples = []
        for node in few_shot_nodes:
            fs_templates, _, fs_element, fs_matches = (
                self._get_prompt_parameters(node)
            )

            few_shot_examples.append(
                (
                    fs_templates,
                    fs_element,
                    fs_matches,
                    self.var_mapping[node].field_description,
                )
            )

        # Generating 8 field descriptions
        tasks = []
        for _ in range(8):
            kwargs["temperature"], kwargs["n"] = 0.3, 1
            user, history, system = gen_label_prompt(
                (templates, elt_index, element, matches),
                few_shot_examples,
                self.log_description,
            )
            if self.ablation_fewshot:
                history = []
            tasks.append(
                LLMTask(
                    system_prompt=system,
                    history=history,
                    model=self.model,
                    message=user,
                    thinking_budget=128,
                    **kwargs,
                )
            )
        response = self.caller(tasks)
        if not isinstance(response, list):
            response = [response]

        descriptions = []
        for raw_llm_response in response:
            if raw_llm_response.failed:
                get_logger().warning(
                    "Description generation failed for %s", element_id
                )
                continue
            for candidate in raw_llm_response.candidates:
                try:
                    descriptions.append(
                        re.sub(
                            r"\s+",
                            " ",
                            json.loads(candidate.strip()),
                        )
                    )
                except json.JSONDecodeError:
                    descriptions.append(
                        re.sub(
                            '(^")|("$)',
                            "",
                            candidate.strip(),
                        )
                    )

        # Generate the embedding of each
        emb_tasks = [
            LLMTask(
                message=description,
                query_type="embedding",
                model="text-embedding-005",
            )
            for description in descriptions
        ]
        response = self.caller(
            emb_tasks, distribute_parallel_requests=False, use_tqdm=False
        )
        embeddings = [torch.tensor(emb) for emb in response]

        # Get the embedding that is closest to the average of all embeddings
        average_embedding = torch.stack(embeddings).mean(dim=0)
        embedding_distances = torch.nn.CosineSimilarity()(
            average_embedding, torch.stack(embeddings)
        )
        closest_embedding = torch.argmax(embedding_distances).item()

        # Select this embedding and description
        description, embedding = (
            descriptions[closest_embedding],
            embeddings[closest_embedding],
        )

        # Save to the variable mapping
        self.var_mapping[element_id] = VariableSemantics(
            orig_node=element_id,
            field_description=description,
            embedding=embedding,
        )

        self._write_desc(
            query=(
                tasks[0].system_prompt
                or ""
                + "\n\n__________\n\n"
                + "\n+++\n".join(v["content"] for v in tasks[0].history or [])
                + "\n\n__________\n\n"
                + tasks[0].message
                or ""
            ),
            response="\n\n###########\n\n".join(descriptions),
            element=element,
            description=description,
        )

    def get_fewshot_examples(self, template_id):
        """Use element proximity to find the closest templates"""
        missing_attributes = [
            id
            for id in self.tree.templates[template_id]
            if self.tree.nodes[id].is_variable()
            and self.var_mapping[id].created_attribute is None
        ]
        if not missing_attributes:
            return []
        closest_elements = {
            k: self.get_close_elements(self.clusters[k])
            for k in missing_attributes
        }
        template_distances = {k: {} for k in missing_attributes}
        for element, closest in closest_elements.items():
            if not closest:
                continue
            element_index = {
                node: idx / len(closest)
                for idx, nodes in enumerate(closest)
                for node in nodes
            }
            template_distances[element] = {
                t: min(element_index.get(node, 1) for node in elts)
                for t, elts in enumerate(self.tree.templates)
                if elts
            }
        template_distances = {
            t: sum(
                [template_distances[k].get(t, 1) for k in missing_attributes]
            )
            / len(missing_attributes)
            for t, elts in enumerate(self.tree.templates)
            if elts
        }
        ranked_templates = sorted(
            list(template_distances.keys()), key=template_distances.get
        )
        ranked_templates = [
            t for t in ranked_templates if t in self.schema_mapping
        ]
        get_logger().info(
            "Selected templates %s as few-shot for %s",
            ranked_templates[: self.few_shot_len],
            template_id,
        )
        return ranked_templates[: self.few_shot_len]

    def get_close_elements(self, element_ids):
        """Get few-shot examples for a given cluster of elements"""
        if isinstance(element_ids, tuple):
            element_ids = list(element_ids)
        if not isinstance(element_ids, list):
            element_ids = [element_ids]

        element_ids = [i for i in element_ids if i in self.var_mapping]
        if not element_ids:
            return []

        ## First criteria: find the closest few-shot examples based on the embeddings
        source_embedding = torch.stack(
            [self.var_mapping[i].embedding for i in element_ids]
        ).mean(dim=0)
        embedding_distances = self.description_distance.rank(
            source_embedding, k=-1
        )
        few_shot_nodes_embeddings = []
        for cluster, _ in embedding_distances:
            mapped_nodes = [c for c in cluster if c in self.var_mapping]
            if not mapped_nodes:
                continue
            if self.var_mapping[mapped_nodes[0]].created_attribute is not None:
                few_shot_nodes_embeddings.append(cluster)
        distance_per_cluster = {
            cluster: sim for cluster, sim in embedding_distances
        }

        ## Second criteria: variables that are close in the parsing tree and of the same type.
        # Priorize those that are mapped to the same event, and that are close in embedding.
        # The embedding is rounded to the nearest 0.2 so that if we have multiple candidates
        # that have close similarities, we can chose the one that has the same event.
        degrees_of_separation = {
            cluster: (
                max(
                    self.tree.degree_of_separation(
                        elt_id, other_elt_id, norm=True
                    )
                    for elt_id in element_ids
                    for other_elt_id in cluster
                ),
                int(5 * distance_per_cluster[cluster]) / 5,
            )
            for cluster in self.cluster_inverse_mapping
            if all(elt_id not in cluster for elt_id in element_ids)
            and any(self.var_mapping[c].created_attribute for c in cluster)
        }
        few_shot_nodes_tree = sorted(
            list(degrees_of_separation.keys()),
            key=lambda x: degrees_of_separation[x],
            reverse=True,
        )

        # Interleave the two lists, removing duplicates
        ordered_list_of_elements = []
        for i in range(
            2 * max(len(few_shot_nodes_embeddings), len(few_shot_nodes_tree))
        ):
            if i % 2 == 0 and few_shot_nodes_embeddings:
                ordered_list_of_elements.append(
                    few_shot_nodes_embeddings.pop(0)
                )
            elif i % 2 == 1 and few_shot_nodes_tree:
                ordered_list_of_elements.append(few_shot_nodes_tree.pop(0))

        seen = set()
        dedup = []
        for cluster in ordered_list_of_elements:
            if cluster not in seen:
                seen.add(cluster)
                dedup.append(cluster)

        return dedup

    def parse_answer(self, candidate, task, template_id, force=False):
        if "//" in candidate:
            candidate = re.sub("//.*$", "", candidate)
        try:
            content = parse_json(
                candidate, self.caller, task=deepcopy(task), model=self.model
            )
        except ValueError:
            raise ValueError("The JSON is invalid")

        schema = self.fix_schema(
            Schema.from_json(content, template_id), force=force
        )
        return schema

    def controlled_execution(self, tasks, template_id, depth=0, force=False):
        rerun, responses = [], [None for _ in tasks]
        candidates = self.caller(tasks)
        candidates = [
            c.candidates[0] if not c.failed else None for c in candidates
        ]

        if self.ablation_self_correction:
            depth = 0
            force = True

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

    def fix_schema(self, schema, force=False):
        # check that every key field has a value field]
        all_attributes = set(schema.fields.keys())
        key_attributes = [
            k.replace("_KEY", "") for k in all_attributes if k.endswith("_KEY")
        ]
        missing_keys = [
            k + "_KEY" for k in key_attributes if k not in all_attributes
        ]
        if missing_keys and not force:
            raise ValueError(
                "The following key fields do not have a corresponding value field:"
                " %s. These are mistakes. Key fields should either a. be associated"
                " with a value field, or b. be labeled as SYNTAX. In the first case,"
                " if you are confident this field corresponds to a value, rename the"
                " key to have the same name as the value field with _KEY appended. In"
                " the second case, label the field as SYNTAX. " % missing_keys
            )
        elif missing_keys and force:
            for k in missing_keys:
                if "SYNTAX" not in schema.fields:
                    schema.fields["SYNTAX"] = SchemaEntry(
                        name="SYNTAX",
                        orig_ids=[],
                        description="SYNTAX",
                    )

            transferable_ids, non_transferable_ids = [], []
            for orig_id in schema.fields[k].orig_ids:
                if orig_id not in self.var_mapping:
                    transferable_ids.append(orig_id)
                elif not self.var_mapping[orig_id].created_attribute:
                    transferable_ids.append(orig_id)
                elif self.var_mapping[orig_id].created_attribute == "SYNTAX":
                    transferable_ids.append(orig_id)
                elif not str(
                    self.var_mapping[orig_id].created_attribute
                ).endswith("_KEY"):
                    transferable_ids.append(orig_id)
                else:
                    non_transferable_ids.append(orig_id)

            if transferable_ids:
                schema.fields["SYNTAX"].orig_ids.extend(transferable_ids)
            if non_transferable_ids:
                schema.fields[k].orig_ids = non_transferable_ids
            else:
                del schema.fields[k]

        renamed_key_errors = []
        for k in key_attributes:
            node_ids = schema.fields[k].orig_ids
            for orig_id in node_ids:
                if orig_id not in self.var_mapping:
                    continue
                if not self.var_mapping[orig_id].created_attribute:
                    continue
                if self.var_mapping[orig_id].created_attribute == "SYNTAX":
                    continue
                if not str(
                    self.var_mapping[orig_id].created_attribute
                ).endswith("_KEY"):
                    continue
                renamed_key_errors.append(
                    (
                        orig_id,
                        k + "_KEY",
                        self.var_mapping[orig_id].created_attribute,
                    )
                )

        if renamed_key_errors:
            formatted_errors = "\n".join(
                f"Node {node_id} was previously marked as a key with name {old_key_field_name}, but"
                f" was changed to {key_field_name}."
                for node_id, key_field_name, old_key_field_name in renamed_key_errors
            )
            raise ValueError(
                "You changed the names of key fields. This is not allowed. You must keep the names of existing key fields the same. Please fix these mistakes:\n%s"
                % formatted_errors
            )

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
            and self.var_mapping[node_id].created_attribute is None
        ]
        existing_names = {
            val.created_attribute for val in self.var_mapping.values()
        }

        # Get few-shot examples
        few_shot_ids = self.get_fewshot_examples(template_id)
        few_shot_examples = []
        for fid in few_shot_ids:
            fs_lines = random.sample(
                self.entries_per_template[fid],
                k=min(5, len(self.entries_per_template[fid])),
            )
            fs_template = self.tree.gen_template(fid)
            fs_output = self.schema_mapping[fid]
            few_shot_examples.append((fs_lines, fs_template, fs_output))

        # Map element to fields
        kwargs["temperature"] = 0.2
        kwargs["n"] = 1

        user, history, system = gen_schema_creation_prompt(
            (lines, template),
            few_shot_examples,
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
        if self.ablation_fewshot:
            history = history[:2] if len(history) > 2 else history
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

        self.add_missing_embeddings(list(updated_fields))
        self.mark_values_as_fixed(template_id)

        return schema

    def mark_values_as_fixed(self, template_id):
        """Mark value fields associated with a fixed key as fixed"""
        fixed_key_fields = set()
        for node_id in self.tree.templates[template_id]:
            if node_id not in self.var_mapping:
                continue
            if not self.var_mapping[node_id].created_attribute:
                continue
            if not self.tree.nodes[node_id].fixed:
                continue
            attr_name = self.var_mapping[node_id].created_attribute
            if str(attr_name).endswith("_KEY"):
                fixed_key_fields.add(str(attr_name).replace("_KEY", ""))

        for node_id in self.tree.templates[template_id]:
            if node_id not in self.var_mapping:
                continue
            if not self.var_mapping[node_id].created_attribute:
                continue
            attr_name = self.var_mapping[node_id].created_attribute
            if attr_name in fixed_key_fields:
                get_logger().info(
                    "Marking %s as fixed because it is associated with a fixed key",
                    attr_name,
                )
                self.tree.nodes[node_id].fixed = True

    def add_missing_embeddings(self, node_ids, use_tqdm=False):
        filtered_node_ids = []
        for node_id in node_ids:
            if node_id not in self.var_mapping:
                continue
            mapping = self.var_mapping[node_id]
            if mapping.embedding is not None:
                continue
            if not mapping.field_description.strip():
                continue
            filtered_node_ids.append(node_id)

        chunk_size = 100
        tasks = []
        for chunk in range(0, len(filtered_node_ids), chunk_size):
            chunk_ids = filtered_node_ids[chunk : chunk + chunk_size]
            tasks.append(
                LLMTask(
                    message=[
                        self.var_mapping[node_id].field_description.strip()
                        for node_id in chunk_ids
                    ],
                    query_type="embedding",
                    model="text-embedding-005",
                )
            )

        response = self.caller(
            tasks, distribute_parallel_requests=False, use_tqdm=use_tqdm
        )
        embeddings = [torch.tensor(emb) for r in response for emb in r]
        for node_id, embedding in zip(filtered_node_ids, embeddings):
            self.var_mapping[node_id].embedding = embedding

    def validate(self, non_validated_templates):
        """Validates the syntax tree"""
        if not self.validate_heuristic:
            return False

        # Select most common validated templates
        if self.validation_size > 0:
            templates = self.validated_templates[: self.validation_size]
        else:
            templates = self.validated_templates

        # Select all new templates
        templates += non_validated_templates

        # Run validation
        new_var_mapping = None
        for _ in range(3):
            try:
                new_var_mapping = self.validate_heuristic.run(templates)
            except ValueError:
                get_logger().error("Could not validate template. Retrying...")
                continue

            if new_var_mapping is not None:
                break

        if not new_var_mapping:
            return False

        # Update var mapping
        all_node_ids = {
            node_id
            for template in templates
            for node_id in self.tree.templates[template]
        }

        for node_id in all_node_ids:
            new_semantics = new_var_mapping.get(node_id, None)
            if new_semantics:
                self.var_mapping[node_id] = new_semantics
            elif node_id in self.var_mapping:
                del self.var_mapping[node_id]

        # Create missing embeddings
        self.add_missing_embeddings(all_node_ids)

        # Update schemas
        for key in self.schema_mapping:
            self.schema_mapping[key] = Schema.from_var_mapping(
                key, self.tree.gen_template(key), self.var_mapping
            )

        # Set nodes as fixed
        for node_id in all_node_ids:
            if self.tree.nodes[node_id]:
                self.tree.nodes[node_id].fixed = True

        self.validated_templates.extend(non_validated_templates)

        return True

    def process(self, **kwargs):

        # Map descriptions to variables
        if not self.var_mapping:
            self.generate_descriptions()
            self._save_dill()

        if not self.clusters:
            # Generate clusters
            self.clusters = self.cluster_variables.run(self.var_mapping)
            self.cluster_inverse_mapping = list(
                set([tuple(v) for v in self.clusters.values()])
            )
            self._save_dill()

        # Initialize the description distance
        self.description_distance = DescriptionDistance(
            self.clusters,
            {k: v.embedding for k, v in self.var_mapping.items()},
        )

        ## Create schemas for every template
        template_list = [
            t for t, elts in enumerate(self.tree.templates) if elts
        ]
        template_order = sorted(
            template_list,
            key=lambda x: len(self.entries_per_template[x]),
            reverse=True,
        )
        for template in template_order:
            if template not in self.schema_mapping:
                self.schema_creation(template)

            if self.validation_size > 0 and self.validate_heuristic:
                non_validated_templates = [
                    t_id
                    for t_id in template_order
                    if t_id in self.schema_mapping
                    and t_id not in self.validated_templates
                ]
                if len(non_validated_templates) > self.validation_size:
                    self.validate(non_validated_templates)
                    self._save_dill()

        if self.validation_size > 0 and self.validate_heuristic:
            non_validated_templates = [
                t_id
                for t_id in template_order
                if t_id in self.schema_mapping
                and t_id not in self.validated_templates
            ]
            if non_validated_templates:
                self.validate(non_validated_templates)

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
