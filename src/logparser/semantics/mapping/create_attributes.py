import json
import os
import random
import re
from collections import defaultdict
from copy import deepcopy

import dill
import torch
from tqdm import tqdm

from ...tools.api import OpenAITask
from ...tools.classes import Parser, Value
from ...tools.embedding import DescriptionDistance
from ...tools.logging import get_logger
from ...tools.module import Module
from ...tools.OCSF import (
    OCSFSchemaClient,
    VariableSemantics,
)
from ...tools.prompts.label_variables import gen_prompt as gen_label_prompt
from ...tools.prompts.schema_creation import (
    gen_prompt as gen_schema_creation_prompt,
)
from ...tools.prompts.schema_description import (
    gen_prompt as gen_schema_description_prompt,
)
from ...tools.schema import Schema, SchemaEntry
from ...tools.utils import parse_json
from .cluster_variables import ClusterVariables


class CreateAttributes(Module):

    def __init__(
        self,
        caller,
        parser,
        log_description,
        few_shot_len=2,
        output_dir="output/",
        model="gemini-1.5-flash",
    ):
        super().__init__("CreateAttributes", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.few_shot_len = few_shot_len

        self.log_description = log_description

        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.schema_mapping = (
            parser.schema_mapping if parser.schema_mapping else {}
        )
        self.schemas = parser.schemas if parser.schemas else []
        self.embedding = parser.embedding
        self.description_distance = None
        self.client = OCSFSchemaClient(caller)

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
            caller, parser, self.client, output_dir, model
        )

    def _save_dill(self):
        with open(
            os.path.join(self.paths["results"], f"saved.dill"),
            "wb",
        ) as f:
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
            if not len(local_templates):
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
            tasks[t_id] = OpenAITask(
                system_prompt=system,
                max_tokens=12000,
                history=history,
                model="gemini-1.5-flash",
                message=user,
                **kwargs,
            )

        tasks_array = list(tasks.values())

        get_logger().info(
            "Generating descriptions for %d templates", len(tasks)
        )
        response = self.caller(
            tasks_array,
            use_tqdm=True,
        )

        descriptions = defaultdict(list)
        # Parse the outputs
        for task_id, raw_llm_response in enumerate(response):
            task = tasks_array[task_id]
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
                    candidate["content"]
                    for candidate in raw_llm_response.candidates
                ),
            )
            for candidate in raw_llm_response.candidates:
                try:
                    content = parse_json(
                        candidate["content"],
                        self.caller,
                        task=deepcopy(task),
                    )
                except ValueError:
                    continue

                content = {int(k): v for k, v in content.items()}
                for node_id, description in content.items():
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
        all_descriptions = [
            desc for node_id in node_order for desc in descriptions[node_id]
        ]
        task = OpenAITask(
            message=all_descriptions,
            query_type="embedding",
            model="models/text-embedding-004",
        )
        response = self.caller(
            task, distribute_parallel_requests=False, use_tqdm=False
        )
        all_embeddings = [torch.tensor(emb) for emb in response]
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

        # Check for missing descriptions
        for node_id, node in enumerate(self.tree.nodes):
            if node and node.is_variable() and node_id not in self.var_mapping:
                get_logger().warning(
                    f"Missing description for {node.value} ({node_id})"
                )
                self.generate_description_element(node_id)

    def generate_description_element(self, element_id, **kwargs):

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
            tasks.append(
                OpenAITask(
                    system_prompt=system,
                    max_tokens=256,
                    history=history,
                    model="gemini-1.5-flash",
                    message=user,
                    **kwargs,
                )
            )
        response = self.caller(tasks)
        if not isinstance(response, list):
            response = [response]

        descriptions = []
        for raw_llm_response in response:
            for candidate in raw_llm_response.candidates:
                try:
                    descriptions.append(
                        re.sub(
                            "\s+",
                            " ",
                            json.loads(candidate["content"].strip()),
                        )
                    )
                except json.JSONDecodeError:
                    descriptions.append(
                        re.sub(
                            '(^")|("$)',
                            "",
                            candidate["content"].strip(),
                        )
                    )

        # Generate the embedding of each
        task = OpenAITask(
            message=descriptions,
            query_type="embedding",
            model="models/text-embedding-004",
        )
        response = self.caller(
            task, distribute_parallel_requests=False, use_tqdm=False
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
                + "\n\n__________\n\n"
                + "\n+++\n".join(v["content"] for v in tasks[0].history or [])
                + "\n\n__________\n\n"
                + tasks[0].message
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

        ## First criteria: find the closest few-shot examples based on the embeddings
        source_embedding = torch.stack(
            [self.var_mapping[i].embedding for i in element_ids]
        ).mean(dim=0)
        embedding_distances = self.description_distance.rank(
            source_embedding, k=-1
        )
        few_shot_nodes_embeddings = [
            cluster
            for cluster, _ in embedding_distances
            if self.var_mapping[cluster[0]].created_attribute is not None
        ]
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

    def fix_schema(self, schema):
        fix_list = []
        new_list = []
        for field_name, field_value in schema.fields.items():
            field_value.orig_ids = [
                i for i in field_value.orig_ids if i in self.var_mapping
            ]
            if not field_value.orig_ids:
                fix_list.append(field_name)

            existing_names = {
                self.var_mapping[i].created_attribute
                for i in field_value.orig_ids
                if self.var_mapping[i].created_attribute is not None
            }
            if len(existing_names):
                fix_list.append(field_name)
                for name in existing_names:
                    ids = [
                        i
                        for i in field_value.orig_ids
                        if self.var_mapping[i].created_attribute == name
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
        save=False,
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
        kwargs["n"] = 5

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
            },
        )
        task = OpenAITask(
            system_prompt=system,
            max_tokens=12000,
            model="gemini-1.5-flash",
            message=user,
            history=history,
            **kwargs,
        )

        candidates = self.caller(task)
        candidates = [c["content"] for c in candidates.candidates]

        valid_candidates = defaultdict(list)

        for candidate in candidates:
            if "//" in candidate:
                re.sub("//.*$", "", candidate)
            try:
                content = parse_json(
                    candidate, self.caller, task=deepcopy(task)
                )
            except ValueError:
                get_logger().warning("Failed to parse JSON")
                continue
            try:
                schema = self.fix_schema(Schema.from_json(content, template_id))
                assert schema is not None
            except:
                continue
            try:
                new_fields = tuple(
                    sorted(
                        [
                            field_name
                            for field_name, field in schema.fields.items()
                            if any(
                                i in unmapped_elements for i in field.orig_ids
                            )
                        ]
                    )
                )
            except:
                breakpoint()
            valid_candidates[new_fields].append(schema)

        # Get the most common schema
        try:
            schema = valid_candidates[
                max(
                    valid_candidates.keys(),
                    key=lambda x: (
                        len(valid_candidates[x]),
                        -len([f for f in x if existing_names]),
                    ),
                )
            ][0]
        except:
            return None
            breakpoint()

        self.schema_mapping[template_id] = schema

        self._write(
            query=(
                task.system_prompt
                + "\n\n__________\n\n"
                + "\n+++\n".join(v["content"] for v in task.history or [])
                + "\n\n__________\n\n"
                + task.message
            ),
            response="\n\n##########\n\n".join(candidates),
            template_id=template_id,
            save=False,
        )

        for field_name, field in schema.fields.items():
            for node_id in field.orig_ids:
                if not self.var_mapping[node_id].created_attribute:
                    self.var_mapping[node_id].created_attribute = field_name

        return schema

    def recluster_variables(self):
        """Merge clusters for which two variables share the same name in the same template"""
        new_clusters = {
            i: {
                i,
            }
            for i in self.clusters
        }
        for schema in self.schema_mapping.values():
            for field in schema.fields.values():
                if len(field.orig_ids) > 1:
                    merged_set = {
                        j for i in field.orig_ids for j in new_clusters[i]
                    }
                    for i in merged_set:
                        new_clusters[i] = merged_set

        self.clusters = ClusterVariables.merge_clusters(
            self.clusters, new_clusters
        )
        self.cluster_inverse_mapping = list(
            set([tuple(v) for v in self.clusters.values()])
        )

    @staticmethod
    def find_common_cluster_by_assigned_name(
        cluster_names, overlap_threshold=0.5
    ):
        # Merge clusters whose name histograms overlap
        clusters = list(cluster_names.keys())
        number_of_clusters = len(clusters)
        name_lists = [cluster_names[name] for name in clusters]
        histograms = []
        for names in name_lists:
            histogram = defaultdict(int)
            for name in names:
                histogram[name] += 1
            histograms.append({k: v / len(names) for k, v in histogram.items()})

        # Calculate overlap between all pairs of histograms
        overlap_matrix = [[0 for _ in clusters] for _ in clusters]

        for i in range(number_of_clusters):
            for j in range(i + 1, number_of_clusters):
                hist1 = histograms[i]
                hist2 = histograms[j]

                # Calculate overlap
                overlap = 0
                all_names = set(hist1.keys()) | set(hist2.keys())

                for name in all_names:
                    overlap += min(hist1.get(name, 0), hist2.get(name, 0))

                overlap_matrix[i][j] = overlap
                overlap_matrix[j][i] = overlap

        # Use a union-find algorithm to merge all clusters i,j for which overlap_matrix[i][j] > overlap_threshold
        # Initialize union-find data structure
        parent = list(range(number_of_clusters))
        rank = [0] * number_of_clusters

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1

        # Merge clusters based on overlap threshold
        for i in range(number_of_clusters):
            for j in range(i + 1, number_of_clusters):
                if overlap_matrix[i][j] > overlap_threshold:
                    union(i, j)

        # Create cluster dict using representatives from the original clusters
        new_clusters = {}
        for cl_idx, cl in enumerate(clusters):
            if find(cl_idx) != cl_idx:
                representative = list(clusters[find(cl_idx)])[0]
                for i in cl:
                    new_clusters[i] = (representative, i)

        return new_clusters

    def assign_popular_names(self, cluster_names):
        """Assign names to largest clusters that have an overlap

        Method: We recursively:
        * Find the largest cluster (in terms of number of templates involved)
        * Assign its name to the variables that are part of the cluster
        * Exclude clusters that share the type of the largest cluster and have no overlap (these clusters could potentially need to take the same name)
        * Repeat until no clusters are left
        """

        cluster_to_template = {
            cluster: {
                t for cl in cluster for t in self.tree.templates_per_node[cl]
            }
            for cluster in cluster_names
        }
        cluster_size = {
            cluster: len(templates)
            for cluster, templates in cluster_to_template.items()
        }

        while cluster_size:
            largest_cluster = max(cluster_size, key=cluster_size.get)
            name = max(
                cluster_names[largest_cluster],
                key=cluster_names[largest_cluster].count,
            )

            # Assign names
            for i in largest_cluster:
                self.var_mapping[i].created_attribute = name

            # Remove clusters that share the type of the largest cluster and have no overlap
            largest_cluster_templates = cluster_to_template[largest_cluster]
            to_be_removed = [
                cl
                for cl in cluster_size
                if not cluster_to_template[cl] & largest_cluster_templates
            ]
            for c in to_be_removed:
                del cluster_size[c]
                del cluster_names[c]

        # Cleanup - remove all schemas to start fresh.
        self.schemas = []
        self.schema_mapping = {}

    def combine_field_names(self):
        """List all the field names assigned to variables in the same cluster and assign the most common one to all of them"""
        self.recluster_variables()
        cluster_to_names = defaultdict(list)
        for schema in self.schemas:
            for name, field in schema.fields.items():
                for node_id in field.orig_ids:
                    cluster_to_names[self.clusters[node_id]].append(name)

        # Merge clusters whose names are most similar
        name_clusters = self.find_common_cluster_by_assigned_name(
            cluster_to_names
        )

        self.clusters = ClusterVariables.merge_clusters(
            self.clusters, name_clusters
        )
        self.cluster_inverse_mapping = list(
            set([tuple(v) for v in self.clusters.values()])
        )

        # Recompute cluster names
        cluster_to_names = defaultdict(list)
        for schema in self.schema_mapping.values():
            for name, field in schema.fields.items():
                for node_id in field.orig_ids:
                    cluster_to_names[self.clusters[node_id]].append(name)

        # Assign names to largest clusters
        self.assign_popular_names(cluster_to_names)

    def norm_schemas(self):
        for t_id in self.schema_mapping:
            template = self.tree.gen_template(t_id)
            field_names = defaultdict(lambda: {"description": None, "ids": []})
            for element in template.elements:
                if element.is_variable() and element.id in self.var_mapping:
                    field_name = self.var_mapping[element.id].created_attribute
                    field_names[field_name]["description"] = self.var_mapping[
                        element.id
                    ].field_description
                    field_names[field_name]["ids"].append(element.id)
            schema = Schema.from_json(field_names, orig_template=t_id)
            self.schema_mapping[t_id] = schema

    def process(self, **kwargs):

        ### Start with generating descriptions
        # Fix missing values (FIXME look into why this is happening in the first place)
        variable_ids = [
            node.id for node in self.tree.nodes if node and node.is_variable()
        ]
        for var_id in variable_ids:
            if var_id not in self.values:
                self.values[var_id] = Value(
                    element_id=var_id, values=[self.tree.nodes[var_id].value]
                )

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
                schema = self.schema_creation(template)

        self._save_dill()
