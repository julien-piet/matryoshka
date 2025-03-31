import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import dill
import torch
from tqdm import tqdm

from ...tools.api import OpenAITask
from ...tools.classes import Parser, Value
from ...tools.embedding import DescriptionDistance
from ...tools.logging import get_logger
from ...tools.module import Module
from ...tools.OCSF import (
    OCSFMapping,
    OCSFSchemaClient,
    VariableSemantics,
)
from ...tools.prompts.label_variables import gen_prompt as gen_label_prompt
from ...tools.prompts.map_variables import gen_prompt as gen_confirm_prompt
from .cluster_variables import ClusterVariables


class MapVariables(Module):

    def __init__(
        self,
        caller,
        parser,
        log_description,
        few_shot_len=2,
        output_dir="output/",
        model="gemini-1.5-flash",
    ):
        super().__init__("MapVariables", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.few_shot_len = few_shot_len

        self.log_description = log_description

        self.event_types = parser.event_types if parser.event_types else {}
        self.schema_mapping = (
            parser.schema_mapping if parser.schema_mapping else {}
        )
        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.schema_mapping = parser.schema_mapping
        self.embedding = parser.embedding
        self.description_distance = None
        self.client = OCSFSchemaClient(caller)

        self.clusters = {}

        self.paths = {
            "prompts": os.path.join(self.output_dir, "prompts/"),
            "outputs": os.path.join(self.output_dir, "outputs/"),
            "results": os.path.join(self.output_dir, "results/"),
        }
        os.makedirs(self.paths["prompts"], exist_ok=True)
        os.makedirs(self.paths["outputs"], exist_ok=True)
        os.makedirs(self.paths["results"], exist_ok=True)

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
                    event_types=self.event_types,
                    embedding=self.embedding,
                    var_mapping=self.var_mapping,
                    schema_mapping=self.schema_mapping,
                ),
                f,
            )

    def _write(
        self, query, response, element, event, mapping, description, save=True
    ):
        # Placeholder for writing logic
        os.makedirs(
            os.path.join(self.paths["prompts"], "mapping", event), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.paths["outputs"], "mapping", event), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.paths["results"], "mapping", event), exist_ok=True
        )
        with open(
            os.path.join(
                self.paths["prompts"], "mapping", event, str(element.id)
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(
                self.paths["outputs"], "mapping", event, str(element.id)
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response)

        with open(
            os.path.join(
                self.paths["results"], "mapping", event, str(element.id)
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{element.id}\n{element.value}\n{description}\n{mapping}")

        if save:
            self._save_dill()

    def _get_prompt_parameters(self, element_ids, event=None):
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
                if not event or event in self.event_types.get(t, [])
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

    def filtering(
        self,
        element_ids,
        event,
        fuzzy=0,
        k=25,
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

        text_description = self.client.get_description(
            event,
            (
                self.tree.nodes[element_ids[0]].type
                if self.tree.nodes[element_ids[0]].type
                not in ["user_t", "composite_t"]
                else None
            ),
            fuzzy=fuzzy,
        )
        if len(text_description) < k:
            rtn = [(k, v, 1) for k, v in text_description.items()]
            text_description = {
                k: v
                for k, v in self.client.get_description(
                    event,
                    (
                        self.tree.nodes[element_ids[0]].type
                        if self.tree.nodes[element_ids[0]].type
                        not in ["user_t", "composite_t"]
                        else None
                    ),
                    fuzzy=max(2, fuzzy + 1),
                ).items()
                if k not in text_description
            }
        else:
            rtn = []

        if text_description:
            targets = [
                (k, self.client.generated_descriptions[k].embedding)
                for k in text_description.keys()
            ]
            dest_embeddings = torch.tensor([t[1] for t in targets])

            sim = torch.nn.CosineSimilarity()(source_embedding, dest_embeddings)
            closest = sim.topk(k=min(len(targets), k), largest=True).indices

            rtn += [
                (targets[i][0], text_description[targets[i][0]], sim[i])
                for i in closest
            ]

        for elt_id in orig_element_ids:
            self.var_mapping[elt_id].mappings[event] = OCSFMapping(
                candidates=rtn,
                event=event,
            )
        self._get_sibling_nodes(
            orig_element_ids, event, source_embedding=source_embedding
        )

        return rtn

    def _get_sibling_nodes(self, element_ids, event, source_embedding):
        # Get sibling types
        sibling_nodes = {
            i.id
            for element_id in element_ids
            for t in self.tree.templates_per_node[element_id]
            for i in self.tree.gen_template(t).elements
            if i.is_variable()
        }
        sibling_fields = [
            f
            for id in sibling_nodes
            for f in self.var_mapping.get(id, VariableSemantics())
            .mappings.get(event, OCSFMapping())
            .field_list
            if id in self.var_mapping and event in self.var_mapping[id].mappings
        ]
        sibling_mapping = self.client.get_siblings(
            sibling_fields, self.tree.nodes[element_ids[0]].type
        )
        candidate_sibling_list = []
        for sibling in sibling_mapping:
            if (
                sibling
                not in {
                    r[0]
                    for element_id in element_ids
                    for r in self.var_mapping[element_id]
                    .mappings[event]
                    .candidates
                }
                and sibling in self.client.generated_descriptions
                and self.client.generated_descriptions[sibling].embedding
                is not None
            ):
                candidate_sibling_list.append(
                    (
                        sibling,
                        self.client.generated_descriptions[sibling].embedding,
                    )
                )

        if not candidate_sibling_list:
            return

        target_embeddings = torch.tensor([t[1] for t in candidate_sibling_list])
        sim = torch.nn.CosineSimilarity()(source_embedding, target_embeddings)
        closest = sim.topk(
            k=min(len(candidate_sibling_list), 5), largest=True
        ).indices

        for i in closest:
            for element_id in element_ids:
                self.var_mapping[element_id].mappings[event].candidates.append(
                    (
                        candidate_sibling_list[i][0],
                        self.client.generated_descriptions[
                            candidate_sibling_list[i][0]
                        ].generated_description,
                        sim[i],
                    )
                )

    def hierarchical_shuffle(self, mapping):
        """Shuffle the mapping in a hierarchical way"""
        random.shuffle(mapping)
        return mapping

    def get_matching_fewshot_examples(self, element_ids, event):
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
            if event in self.var_mapping[cluster[0]].mappings
            and self.var_mapping[cluster[0]].mappings[event].mapped
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
                1 if event in self.var_mapping[cluster[0]].mappings else 0,
            )
            for cluster in self.cluster_inverse_mapping
            if all(elt_id not in cluster for elt_id in element_ids)
            and self.tree.nodes[cluster[0]].type
            == self.tree.nodes[element_ids[0]].type
            and any(
                mapping.mapped
                for mapping in self.var_mapping[cluster[0]].mappings.values()
            )
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

            chosen_event, node = event, None
            for candidate_node in cluster:
                if event in self.var_mapping[candidate_node].mappings:
                    chosen_event = event
                    node = candidate_node
                    break

            if not node:
                events = [
                    (event, c)
                    for c in cluster
                    for event in self.var_mapping[c].mappings
                ]
                chosen_event, node = random.sample(events, 1)[0]

            mapping = self.var_mapping[node].mappings[chosen_event]

            fs_mapping = {
                f: self.client.generated_descriptions[f].generated_description
                for f in mapping.field_list
                if f in self.client.generated_descriptions
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
        event,
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
            for r in self.var_mapping[elt_id].mappings[event].candidates
        }

        try:
            # Get element info
            templates, indices, elements, matches = self._get_prompt_parameters(
                element_ids, event
            )
        except:
            breakpoint()

        # Get few-shot examples
        few_shot_examples = self.get_matching_fewshot_examples(
            element_ids, event
        )

        # Map element to fields
        kwargs["temperature"] = 0.2
        kwargs["n"] = 1

        tasks = []
        mapping_list = list(mapping.items())
        for _ in range(3):
            mapping_list = self.hierarchical_shuffle(mapping_list)
            chosen_elt = random.choice(element_ids)
            user, history, system = gen_confirm_prompt(
                (templates, indices, elements, matches),
                few_shot_examples,
                dict(mapping_list),
                self.var_mapping[chosen_elt].field_description,
            )
            tasks.append(
                OpenAITask(
                    system_prompt=system,
                    max_tokens=12000,
                    model="gemini-1.5-flash",
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
            r"###\s+mapping\s+###\s+```json\s+(.*?)\s+```",
            flags=re.IGNORECASE | re.DOTALL,
        )
        demo_pattern = re.compile(
            r"###\s+explanation\s+###\s+(.*?)\s+###",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for llm_response in candidates:
            for candidate_resp in llm_response.candidates:
                candidate = candidate_resp["content"]
                mapping_match = result_pattern.search(candidate)
                demo_match = demo_pattern.search(candidate)
                if not mapping_match or not demo_match:
                    continue
                try:
                    proposal = json.loads(mapping_match.group(1))
                except json.JSONDecodeError:
                    proposal = {}
                fixed_items = []
                for p in proposal:
                    if p not in mapping and p.count("."):
                        p = ".".join(p.split(".")[1:])
                        fixed_items += [
                            v
                            for v in mapping
                            if p in v and p.count(".") + 1 == v.count(".")
                        ]
                try:
                    proposal = list(proposal.keys()) + fixed_items
                except AttributeError:
                    proposal = list(proposal) + fixed_items

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
            self.var_mapping[element_id].mappings[event].demonstration = demo
            self.var_mapping[element_id].mappings[event].field_list = tuple(
                sorted(selected_fields)
            )
            self.var_mapping[element_id].mappings[event].mapped = True

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
                        c["content"]
                        for candidate in candidates
                        for c in candidate.candidates
                    ]
                ),
                element=self.tree.nodes[element_id],
                event=event,
                mapping=self.var_mapping[element_id].mappings[event].field_list,
                description=demo,
                save=False,
            )

        return bool(selected_fields)

    def normalize(self, element_id):
        """Make sure common fields are included in all relevant events"""
        if isinstance(element_id, list):
            for elt_id in element_id:
                self.normalize(elt_id)
            return

        source_mapping = self.client.get_source_to_event_mapping()
        mappings = self.var_mapping[element_id].mappings
        all_sources = defaultdict(int)
        for event, mapping in mappings.items():
            class_desc = self.client.get_class_objects(event)
            for field in mapping.field_list:
                field_name_root = field.split(".")[1]
                source = (
                    class_desc[field_name_root].source
                    if field_name_root not in {"actor"}
                    else "base_event"
                )
                mod_field_name = source + "." + ".".join(field.split(".")[1:])
                all_sources[mod_field_name] += 1

        # Only keep fields that have been assigned to a majority of the events they can be assigned to.
        presence_count = {
            k: len(
                [
                    event
                    for event in mappings
                    if k in source_mapping and event in source_mapping[k]
                ]
            )
            for k in all_sources
        }
        all_sources_keep = {
            k: v
            for k, v in all_sources.items()
            if not presence_count[k] or v / presence_count[k] >= 0.5
        }

        for event, mapping in mappings.items():
            new_field_list = [
                source_mapping[f][event]
                for f in all_sources_keep
                if f in source_mapping and event in source_mapping[f]
            ]
            if set(mapping.field_list) != set(new_field_list):
                get_logger().info(
                    "Normalizing mapping from %s (%s) to event %s: %s",
                    element_id,
                    self.tree.nodes[element_id].value,
                    event,
                    str(new_field_list),
                )
                mapping.field_list = tuple(new_field_list)

    def cluster_by_name(self):
        """Group nodes by name"""
        name_to_id = defaultdict(list)
        to_be_removed = []
        for element_id, mapping in self.var_mapping.items():
            if not self.tree.nodes[element_id].is_variable():
                to_be_removed.append(element_id)
                continue
            name_to_id[mapping.created_attribute].append(element_id)
        for element_id in to_be_removed:
            del self.var_mapping[element_id]

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
            if not len(cluster):
                breakpoint()

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

    def process(self, **kwargs):

        ### Erase current mappings
        for element_id in self.var_mapping:
            self.var_mapping[element_id].mappings = {}

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

        # Descriptions are generated from schema creation step. Now we just cluster variables based on them having the same name.
        self.cluster_by_name()
        self.description_distance = DescriptionDistance(
            self.clusters,
            {k: v.cluster_embedding for k, v in self.var_mapping.items()},
        )

        # Compute number of seen values per cluster, to determine assignment order
        try:
            volumes = {
                cluster: sum(len(self.values[x].values) for x in cluster)
                for cluster in self.cluster_inverse_mapping
            }
        except:
            breakpoint()
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
            associated_events = list(
                {
                    e
                    for element_id in element_ids
                    for template in self.tree.templates_per_node[element_id]
                    for e in self.event_types.get(template, [])
                    if e != "unsure"
                }
            )
            if not associated_events:
                continue

            for event in associated_events:
                if all(
                    event in self.var_mapping[element_id].mappings
                    and self.var_mapping[element_id].mappings[event].mapped
                    for element_id in element_ids
                ):
                    continue

                requires_save = True

                self.filtering(element_ids, event)
                self.matching(element_ids, event)

            for tgt_element_id in element_ids:
                get_logger().info(
                    "Mapping %s (%s) to event %s: %s",
                    tgt_element_id,
                    self.tree.nodes[tgt_element_id],
                    event,
                    str(self.var_mapping[tgt_element_id]),
                )

            self.normalize(element_ids)

        if requires_save:
            self._save_dill()
