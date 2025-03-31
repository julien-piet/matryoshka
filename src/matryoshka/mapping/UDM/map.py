import json
import os
import random
import re
from collections import Counter, defaultdict

import dill
import torch
from tqdm import tqdm

from matryoshka.classes import Mapping, Module, Parser, VariableSemantics
from matryoshka.genai_api.api import LLMTask
from matryoshka.utils.embedding import DescriptionDistance
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.mapping.UDM.map import (
    gen_prompt as gen_confirm_prompt,
)
from matryoshka.utils.UDM import UDMSchemaClient
from matryoshka.validation.udm import MappingValidator


class MapToAttributes(Module):

    def __init__(
        self,
        caller,
        parser,
        log_description,
        few_shot_len=2,
        output_dir="output/",
        model="gemini-2.5-flash",
        validation_model="gemini-2.5-pro",
        udm_client=None,
        cache_dir=".cache/",
        save_contents=False,
        max_udm_attributes=1,
    ):
        super().__init__("MapToAttributes", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.few_shot_len = few_shot_len
        self.cache_dir = cache_dir
        self.save_contents = save_contents
        self.log_description = log_description

        self.schema_mapping = (
            parser.schema_mapping if parser.schema_mapping else {}
        )
        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.schema_mapping = parser.schema_mapping
        self.embedding = parser.embedding
        self.description_distance = None
        self.client = (
            UDMSchemaClient(caller, os.path.join(cache_dir, "UDM"))
            if udm_client is None
            else udm_client
        )

        self.clusters = {}

        self.paths = {
            "prompts": os.path.join(self.output_dir, "prompts/"),
            "outputs": os.path.join(self.output_dir, "outputs/"),
            "results": os.path.join(self.output_dir, "results/"),
        }
        os.makedirs(self.paths["prompts"], exist_ok=True)
        os.makedirs(self.paths["outputs"], exist_ok=True)
        os.makedirs(self.paths["results"], exist_ok=True)

        self.validate_heuristic = MappingValidator(
            caller,
            udm_client=udm_client,
            tree=self.tree,
            model=validation_model,
            var_mapping=self.var_mapping,
            entries_per_template=self.entries_per_template,
            values=self.values,
            save_path=self.output_dir,
            max_udm_attributes=max_udm_attributes,
        )

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
                        embedding=self.embedding,
                        var_mapping=self.var_mapping,
                        schema_mapping=self.schema_mapping,
                    ),
                    f,
                )

    def _write(self, query, response, element, mapping, description, save=True):
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

        with open(
            os.path.join(self.paths["results"], "mapping", str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{element.id}\n{element.value}\n{description}\n{mapping}")

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

    def filtering(
        self,
        element_ids,
        k=50,
        **kwargs,
    ):
        """This function uses similarity metrics to rank possible target attributes for a given element"""
        if isinstance(element_ids, tuple):
            element_ids = list(element_ids)
        if not isinstance(element_ids, list):
            element_ids = [element_ids]

        orig_element_ids = element_ids
        element_ids = [
            eid
            for eid in element_ids
            if eid in self.var_mapping
            and self.var_mapping[eid].embedding is not None
        ]

        source_embeddings = [
            self.var_mapping[elt].embedding for elt in element_ids
        ]
        source_embedding = torch.stack(source_embeddings).mean(dim=0)

        text_description = self.client.get_description()
        targets = [
            (k, self.client.attributes[k].embedding)
            for k in text_description.keys()
            if self.client.attributes[k].embedding
        ]
        dest_embeddings = torch.tensor([t[1] for t in targets])

        sim = torch.nn.CosineSimilarity()(source_embedding, dest_embeddings)
        closest = sim.topk(k=min(len(targets), k), largest=True).indices

        filtered_target_attributes = [
            (targets[i][0], text_description[targets[i][0]], sim[i])
            for i in closest
        ]

        for elt_id in orig_element_ids:
            self.var_mapping[elt_id].mapping = Mapping(
                candidates=filtered_target_attributes,
            )

        self._get_sibling_nodes(
            orig_element_ids,
            source_embedding=source_embedding,
        )

        return filtered_target_attributes

    def _get_sibling_nodes(self, element_ids, source_embedding):
        # Get sibling types
        sibling_nodes = {
            i.id
            for element_id in element_ids
            for t in self.tree.templates_per_node[element_id]
            for i in self.tree.gen_template(t).elements
            if i.is_variable()
        }
        sibling_fields = []
        for id in sibling_nodes:
            if id not in self.var_mapping:
                continue
            if not self.var_mapping[id].mapping:
                continue
            if not self.var_mapping[id].mapping.field_list:
                continue
            for field in self.var_mapping[id].mapping.field_list:
                sibling_fields.append(field)

        sibling_mapping = self.client.get_siblings(sibling_fields)

        # Get the embeddings for each sibling attribute
        candidate_sibling_list = []
        existing_candidates = set()
        for element_id in element_ids:
            if element_id not in self.var_mapping:
                continue
            if not self.var_mapping[element_id].mapping:
                continue
            for r in self.var_mapping[element_id].mapping.candidates:
                existing_candidates.add(r[0])

        for sibling in sibling_mapping:
            if sibling in existing_candidates:
                continue
            if sibling not in self.client.attributes:
                continue
            if not self.client.attributes[sibling].embedding:
                continue
            candidate_sibling_list.append(
                (
                    sibling,
                    self.client.attributes[sibling].embedding,
                )
            )

        if not candidate_sibling_list:
            return

        target_embeddings = torch.tensor([t[1] for t in candidate_sibling_list])
        sim = torch.nn.CosineSimilarity()(source_embedding, target_embeddings)
        closest = sim.topk(
            k=min(len(candidate_sibling_list), 25), largest=True
        ).indices

        for i in closest:
            for element_id in element_ids:
                self.var_mapping[element_id].mapping.candidates.append(
                    (
                        candidate_sibling_list[i][0],
                        self.client.attributes[
                            candidate_sibling_list[i][0]
                        ].global_description,
                        sim[i],
                    )
                )

    def get_matching_fewshot_examples(self, element_ids):
        """Get few-shot examples for a given cluster of elements"""
        if isinstance(element_ids, tuple):
            element_ids = list(element_ids)
        if not isinstance(element_ids, list):
            element_ids = [element_ids]

        element_ids = [
            eid
            for eid in element_ids
            if eid in self.var_mapping
            and self.var_mapping[eid].embedding is not None
        ]

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
            if self.var_mapping[cluster[0]].mapping
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
            and self.tree.nodes[cluster[0]].type
            == self.tree.nodes[element_ids[0]].type
            and self.var_mapping[cluster[0]].mapping
        }
        few_shot_nodes_tree = sorted(
            list(degrees_of_separation.keys()),
            key=lambda x: degrees_of_separation[x],
            reverse=True,
        )

        # Get list of few-shot examples.
        few_shot_nodes = few_shot_nodes_embeddings[: self.few_shot_len // 2]
        for node in few_shot_nodes_tree:
            if len(few_shot_nodes) >= self.few_shot_len:
                break
            if node not in few_shot_nodes:
                few_shot_nodes.append(node)

        # Get relevant info for each few-shot example: original event, elt info, and solution
        few_shot_examples = []
        for cluster in few_shot_nodes:
            fs_templates, _, fs_element, fs_matches = (
                self._get_prompt_parameters(cluster)
            )

            node = cluster[0]
            mapping = self.var_mapping[node].mapping
            fs_mapping = {
                f: self.client.attributes[f].global_description
                for f in mapping.field_list
                if f in self.client.attributes
            }

            fs_demonstration = mapping.demonstration
            fs_description = self.var_mapping[node].field_description
            fs_fields = {r[0]: r[1] for r in mapping.candidates}
            few_shot_examples.append(
                (
                    fs_templates,
                    fs_element,
                    fs_matches,
                    fs_description,
                    fs_fields,
                    fs_demonstration,
                    fs_mapping,
                )
            )

        return few_shot_examples[: self.few_shot_len]

    def matching(
        self,
        element_ids,
        **kwargs,
    ):
        if isinstance(element_ids, tuple):
            element_ids = list(element_ids)
        if not isinstance(element_ids, list):
            element_ids = [element_ids]

        # Get embedding mapping
        mapping = {
            r[0]: r[1]
            for elt_id in element_ids
            for r in self.var_mapping[elt_id].mapping.candidates
        }

        # Get element info
        templates, indices, elements, matches = self._get_prompt_parameters(
            element_ids
        )

        # Get few-shot examples
        few_shot_examples = self.get_matching_fewshot_examples(element_ids)

        # Map element to fields
        kwargs["temperature"] = 0.2
        kwargs["n"] = 1

        tasks = []
        mapping_list = list(mapping.items())
        for _ in range(3):
            random.shuffle(mapping_list)
            chosen_elt = random.choice(element_ids)
            user, history, system = gen_confirm_prompt(
                (templates, indices, elements, matches),
                few_shot_examples,
                dict(mapping_list),
                self.var_mapping[chosen_elt].field_description,
            )
            tasks.append(
                LLMTask(
                    system_prompt=system,
                    model=self.model,
                    thinking_budget=2048,
                    message=user,
                    history=history,
                    **kwargs,
                )
            )

        candidates = self.caller(tasks)
        valid_candidates = []
        if not isinstance(candidates, list):
            candidates = [candidates]

        result_pattern = re.compile(
            r"###\s+mapping[^`]*```json\s+(.*?)\s+```",
            flags=re.IGNORECASE | re.DOTALL,
        )
        demo_pattern = re.compile(
            r"###\s+explanation(.*?)\s+###\s+mapping",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for llm_response in candidates:
            for candidate in llm_response.candidates:
                mapping_match = result_pattern.search(candidate)
                demo_match = demo_pattern.search(candidate)
                if not mapping_match or not demo_match:
                    breakpoint()
                    continue
                try:
                    proposal = json.loads(mapping_match.group(1))
                except json.JSONDecodeError:
                    proposal = {}
                try:
                    proposal = list(proposal.keys())
                except AttributeError:
                    proposal = list(proposal)

                proposal = tuple(sorted([p for p in proposal if p in mapping]))

                # Description
                demonstration = demo_match.group(1).strip()

                valid_candidates.append((proposal, demonstration))

        field_counts = Counter(
            [proposal for p in valid_candidates for proposal in p[0]]
        )
        selected_fields = [p for p in field_counts if field_counts[p] > 1]
        field_demonstrations = {}
        for field in mapping:
            demo_regex = re.compile(f".*{field}.*", flags=re.IGNORECASE)
            included = [
                idx
                for idx, (proposal, _) in enumerate(valid_candidates)
                if (field in selected_fields) == (field in proposal)
            ]
            demonstrations = [
                d.group(0)
                for d in [
                    demo_regex.search(d)
                    for idx, (_, d) in enumerate(valid_candidates)
                    if idx in included
                ]
                if d
            ]
            field_demonstrations[field] = (
                demonstrations[0] if demonstrations else None
            )
            if not field_demonstrations[field]:
                get_logger().warning(
                    "No demonstration found for field %s",
                    field,
                )

        demo = "\n".join(
            f"{field}: {field_demonstrations[field]}" for field in mapping
        )
        for element_id in element_ids:
            self.var_mapping[element_id].mapping.demonstration = demo
            self.var_mapping[element_id].mapping.field_list = tuple(
                sorted(selected_fields)
            )

        task = tasks[0]
        for element_id in element_ids:
            self._write(
                query=(
                    task.system_prompt
                    + "\n\n__________\n\n"
                    + "\n+++\n".join(v["content"] for v in task.history or [])
                    + "\n\n__________\n\n"
                    + task.message
                ),
                response="\n\n###########\n\n".join(
                    [
                        c
                        for candidate in candidates
                        for c in candidate.candidates
                    ]
                ),
                element=self.tree.nodes[element_id],
                mapping=self.var_mapping[element_id].mapping.field_list,
                description=demo,
                save=False,
            )

        return bool(selected_fields)

    def cluster_by_name(self):
        """Group nodes by name"""
        name_to_id = defaultdict(list)
        for element_id, mapping in self.var_mapping.items():
            name_to_id[mapping.created_attribute].append(element_id)
        self.clusters = {}
        for name, ids in name_to_id.items():
            cluster = tuple(sorted(ids))
            for i in ids:
                self.clusters[cluster] = ids

        self.cluster_inverse_mapping = list(
            set([tuple(v) for v in self.clusters.values()])
        )

        # Merge descriptions
        for cluster in self.cluster_inverse_mapping:
            orig_cluster = cluster
            cluster = [
                c
                for c in cluster
                if c in self.var_mapping
                and self.var_mapping[c].embedding is not None
            ]
            if not cluster:
                continue

            embeddings = torch.stack(
                [self.var_mapping[i].embedding for i in cluster]
            )

            # Calculate pairwise distances between embeddings
            pairwise_distances = torch.cdist(embeddings, embeddings)

            # Find the index of the medoid (point with minimum sum of distances to all other points)
            medoid_idx = torch.argmin(pairwise_distances.sum(dim=1)).item()

            # Set this description as the description for the entire cluster
            for i in orig_cluster:
                self.var_mapping[i].cluster_description = self.var_mapping[
                    cluster[medoid_idx]
                ].field_description
                self.var_mapping[i].cluster_embedding = self.var_mapping[
                    cluster[medoid_idx]
                ].embedding

    def validate(self):

        new_var_mapping = self.validate_heuristic.run()
        for node_id, new_semantics in new_var_mapping.items():
            self.var_mapping[node_id] = new_semantics

        return True

    def process(self, log_file=None, only_validate=False, **kwargs):

        if only_validate:
            if self.validate():
                self._save_dill()
            return

        # Erase current mappings
        for element_id in self.var_mapping:
            self.var_mapping[element_id].mapping = None

        # Remove uninformative fields from mapping
        to_be_removed = []
        for element_id, mapping in self.var_mapping.items():
            if not mapping.created_attribute:
                to_be_removed.append(element_id)
                continue
            if (
                mapping.created_attribute == "SYNTAX"
                or mapping.created_attribute.endswith("_KEY")
            ):
                to_be_removed.append(element_id)
        for element_id in to_be_removed:
            del self.var_mapping[element_id]

        # Descriptions are generated from schema creation step. Now we just cluster variables based on them having the same name.
        self.cluster_by_name()
        self.description_distance = DescriptionDistance(
            self.clusters,
            {k: v.cluster_embedding for k, v in self.var_mapping.items()},
        )

        # Compute number of seen values per cluster, to determine assignment order
        volumes = {
            cluster: sum(
                len(self.values[x].values if x in self.values else [])
                for x in cluster
            )
            for cluster in self.cluster_inverse_mapping
        }

        determination_order = sorted(
            volumes.keys(),
            key=lambda x: volumes[x],
            reverse=True,
        )

        requires_save = False
        # Process each element
        for cluster in tqdm(
            determination_order,
            total=len(determination_order),
            desc="Generating Mappings",
            unit="element",
        ):

            # Get the list of associated events, continue if there are none
            element_ids = list(cluster)
            if all(
                self.var_mapping[element_id].mapping is not None
                for element_id in element_ids
            ):
                continue

            requires_save = True

            self.filtering(element_ids)
            self.matching(element_ids)

            get_logger().info(
                "%s ==> %s",
                self.var_mapping[element_ids[0]].created_attribute,
                str(self.var_mapping[element_ids[0]].mapping),
            )

        if requires_save:
            self._save_dill("_no_validation")

        if self.validate():
            self._save_dill()
