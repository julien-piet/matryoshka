import json
import os
import random
from collections import defaultdict

import dill
import torch

from ...tools.api import OpenAITask
from ...tools.classes import Parser
from ...tools.logging import get_logger
from ...tools.module import Module
from ...tools.OCSF import (
    OCSFSchemaClient,
)
from ...tools.prompts.attribute_description import gen_prompt
from .cluster_variables import ClusterVariables


class NormAttributes(Module):

    def __init__(
        self,
        caller,
        parser,
        log_description,
        few_shot_len=2,
        output_dir="output/",
        model="gemini-2.0-flash",
    ):
        super().__init__("NormAttributes", caller=caller)

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

    def _write(self, query, response, attribute_name):
        # Placeholder for writing logic
        os.makedirs(
            os.path.join(self.paths["prompts"], "creation"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.paths["outputs"], "creation"), exist_ok=True
        )
        with open(
            os.path.join(
                self.paths["prompts"], "creation", str(attribute_name)
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(
                self.paths["outputs"], "creation", str(attribute_name)
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

    def generate_description_and_embedding(
        self,
        attribute_names,
        node_ids_per_attribute,
        max_examples=20,
        tasks_per_attribute=5,
        **kwargs,
    ):

        tasks = []
        kwargs["temperature"] = 0.25
        kwargs["n"] = 1
        for attribute_name in attribute_names:
            # Get one example per node_id
            lines = []
            node_ids = node_ids_per_attribute[attribute_name]
            if (
                len(node_ids) == 1
                and self.var_mapping[list(node_ids)[0]].field_description
                and self.var_mapping[list(node_ids)[0]].embedding is not None
            ):
                continue
            if not node_ids:
                breakpoint()
            for node_id in node_ids:
                # Select random template
                template = random.choice(
                    list(self.tree.templates_per_node[node_id])
                )
                # Select random entry
                example = random.choice(self.entries_per_template[template])
                lines.append((template, example, node_id))

            for _ in range(tasks_per_attribute):
                # Select at most max_examples from
                local_lines = random.sample(
                    lines, k=min(max_examples, len(lines))
                )

                local_templates, local_entries, local_node_ids = tuple(
                    list(a) for a in zip(*local_lines)
                )
                local_templates = [
                    self.tree.gen_template(t) for t in local_templates
                ]

                user, history, system = gen_prompt(
                    local_templates,
                    local_entries,
                    local_node_ids,
                    field_name=attribute_name,
                    log_desc=self.log_description,
                )
                tasks.append(
                    (
                        OpenAITask(
                            system_prompt=system,
                            max_tokens=256,
                            history=history,
                            model="gemini-2.0-flash",
                            message=user,
                            **kwargs,
                        ),
                        attribute_name,
                    )
                )

        tasks_array = [t[0] for t in tasks]
        attribute_array = [t[1] for t in tasks]

        response = self.caller(
            tasks_array,
            use_tqdm=True,
        )

        # Parse outputs
        descriptions = defaultdict(list)
        for task_id, raw_llm_response in enumerate(response):
            task = tasks_array[task_id]
            attribute_name = attribute_array[task_id]
            self._write(
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
                attribute_name=attribute_name,
            )
            for candidate in raw_llm_response.candidates:
                try:
                    content = json.loads(candidate["content"].strip())
                except json.JSONDecodeError:
                    content = candidate["content"].replace("\"'", "").strip()
                descriptions[attribute_name].append(content)
                get_logger().info(f"{attribute_name}: {content}")

        # Embed the descriptions
        attribute_order = sorted(descriptions.keys())
        orig_attributes = [
            attr for attr in attribute_order for _ in descriptions[attr]
        ]
        all_descriptions = [
            desc for attr in attribute_order for desc in descriptions[attr]
        ]

        get_logger().info(f"Embedding {len(all_descriptions)} descriptions")
        task = OpenAITask(
            message=all_descriptions,
            query_type="embedding",
            model="models/text-embedding-004",
        )
        response = self.caller(
            task, distribute_parallel_requests=False, use_tqdm=False
        )
        get_logger().info(f"Embedded {len(all_descriptions)} descriptions")
        all_embeddings = [torch.tensor(emb) for emb in response]
        embeddings = {attr: [] for attr in attribute_order}
        for attr, emb in zip(orig_attributes, all_embeddings):
            embeddings[attr].append(emb)

        for attr, local_descriptions in descriptions.items():
            # Get the embedding that is closest to the average of all embeddings
            local_embeddings = embeddings[attr]
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
            for node_id in node_ids_per_attribute[attr]:
                self.var_mapping[node_id].field_description = description
                self.var_mapping[node_id].embedding = embedding

    def process(self, **kwargs):
        # List all attributes
        if "log_file" in kwargs:
            del kwargs["log_file"]

        attributes = {
            var.created_attribute: set()
            for var in self.var_mapping.values()
            if var.created_attribute
        }
        for node_idx, var in self.var_mapping.items():
            if var.created_attribute:
                attributes[var.created_attribute].add(node_idx)

        # For each attribute, create a description and embedding
        self.generate_description_and_embedding(
            attribute_names=list(attributes.keys()),
            node_ids_per_attribute=attributes,
            max_examples=self.few_shot_len,
            tasks_per_attribute=5,
            **kwargs,
        )

        self._save_dill()
