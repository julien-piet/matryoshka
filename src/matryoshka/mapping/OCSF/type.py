import json
import os
import re
from collections import Counter, defaultdict

import dill
from tqdm import tqdm

from matryoshka.classes import Parser, Template
from matryoshka.genai_api.api import LLMTask
from matryoshka.utils.embedding import NaiveDistance
from matryoshka.utils.logging import get_logger
from matryoshka.utils.OCSF import OCSFSchemaClient
from matryoshka.utils.prompts.mapping.OCSF.typing import gen_prompt


class Typer:

    def __init__(
        self,
        caller,
        parser,
        few_shot_len=3,
        output_dir="output/",
        model="gemini-2.5-flash",
        ocsf_client=None,
        cache_dir=".cache/",
        ablation_fewshot=False,
    ):
        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.types = set()
        self.few_shot_len = few_shot_len
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.ablation_fewshot = ablation_fewshot

        self.client = (
            OCSFSchemaClient(
                self.caller, saved_path=os.path.join(self.cache_dir, "OCSF")
            )
            if ocsf_client is None
            else ocsf_client
        )
        self.var_mapping = parser.var_mapping
        self.schema_mapping = parser.schema_mapping

        self.node_to_cluster = {}
        self.cluster_to_nodes = {}

        if self.output_dir:
            self.paths = {
                "prompts": os.path.join(self.output_dir, "prompts/"),
                "outputs": os.path.join(self.output_dir, "outputs/"),
                "results": os.path.join(self.output_dir, "results/"),
            }
            os.makedirs(self.paths["prompts"], exist_ok=True)
            os.makedirs(self.paths["outputs"], exist_ok=True)
            os.makedirs(self.paths["results"], exist_ok=True)
        else:
            self.paths = {
                "prompts": None,
                "outputs": None,
                "results": None,
            }

        # Build embedding
        if not parser.embedding:
            self.embedding = NaiveDistance()
            for template_id, entries in self.entries_per_template.items():
                template = self.tree.gen_template(template_id)
                self.embedding.add_template(template_id, template)
                for entry in entries:
                    match, match_obj = template.match(entry)
                    if not match:
                        get_logger().warning(
                            "Entry %s does not match template %s",
                            entry,
                            template,
                        )
                    else:
                        self.embedding.update(match_obj)

            for template_id in self.entries_per_template:
                self.embedding.template_embeddings[template_id].simplify()

        else:
            self.embedding = parser.embedding

    def _get_typing_prompt_variables(self, element_ids):
        if not isinstance(element_ids, list):
            element_ids = [element_ids]

        matches = []
        for node in element_ids:
            if node in self.values:
                matches.extend(self.values[node].value_counts.keys())
            else:
                matches.append(self.tree.nodes[node].value)
        matches = list(set(matches))

        templates = [
            [
                self.tree.gen_template(t)
                for t in self.tree.templates_per_node[node]
            ]
            for node in element_ids
        ]
        elements = [self.tree.nodes[element_id] for element_id in element_ids]
        all_matches = matches + [element.value for element in elements]
        matches = list(set(all_matches))

        return templates, elements, matches

    def _write(self, query, response, element, determined_type, selected_regex):
        # Placeholder for writing logic
        if not self.output_dir:
            return
        with open(
            os.path.join(self.paths["prompts"], str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(query)

        with open(
            os.path.join(self.paths["outputs"], str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response)

        with open(
            os.path.join(self.paths["results"], str(element.id)),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{element}\n{determined_type}\n{selected_regex}")

    def get_closest_nodes(self, element_ids, k=3):
        """
        Get the k closest nodes to the given element ids based on their values.
        """
        if not isinstance(element_ids, list):
            element_ids = [element_ids]
        element_id_set = set(element_ids)

        typed_variable_values = [
            val
            for id, val in self.values.items()
            if self.tree.nodes[id].is_variable()
            and self.tree.nodes[id].type is not None
            and id not in element_id_set
        ]
        few_shot_nodes_per_element = [
            {
                val.element_id: val_idx
                for val_idx, val in enumerate(
                    self.values[element_id].get_closest(
                        typed_variable_values,
                        k=self.few_shot_len,
                        tree=self.tree,
                        embedding=self.embedding,
                    )
                )
            }
            for element_id in element_ids
            if element_id in self.values
        ]
        all_possible_element_ids = {
            key for d in few_shot_nodes_per_element for key in d
        }

        candidate_few_shot_clusters = [
            self.node_to_cluster[k[0]]
            for k in sorted(
                [
                    (
                        key,
                        sum(
                            d.get(key, len(d))
                            for d in few_shot_nodes_per_element
                        ),
                    )
                    for key in all_possible_element_ids
                ],
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        chosen = set()
        while len(chosen) < self.few_shot_len and len(
            candidate_few_shot_clusters
        ):
            cluster = candidate_few_shot_clusters.pop(0)
            if cluster not in chosen:
                chosen.add(cluster)

        return list(chosen)

    def typing(self, element_ids, **kwargs):

        kwargs["n"] = 5
        kwargs["temperature"] = 0.5

        if not isinstance(element_ids, list):
            element_ids = [element_ids]

        # Get relevant variables and generate prompts
        templates, elements, matches = self._get_typing_prompt_variables(
            element_ids
        )
        few_shot_examples = []

        few_shot_clusters = self.get_closest_nodes(element_ids)
        for cluster in few_shot_clusters:
            nodes = self.cluster_to_nodes[cluster]
            fs_templates = [
                [
                    self.tree.gen_template(t)
                    for t in self.tree.templates_per_node[node]
                ]
                for node in nodes
            ]
            fs_values = []
            for node in nodes:
                if node in self.values:
                    fs_values.extend(self.values[node].value_counts.keys())
                else:
                    fs_values.append(self.tree.nodes[node].value)
            fs_values = list(set(fs_values))
            few_shot_examples.append(
                (fs_templates, fs_values, nodes, self.tree.nodes[nodes[0]].type)
            )

        while True:
            user, system = gen_prompt(
                templates,
                matches,
                element_ids,
                few_shot_examples if not self.ablation_fewshot else None,
                self.client,
            )
            task = LLMTask(
                system_prompt=system,
                model=self.model,
                message=user,
                thinking_budget=128,
                **kwargs,
            )

            try:
                candidates = self.caller([task])[0].candidates
                break
            except ValueError as e:
                get_logger().warning(
                    "Error in typing %s (%s): %s",
                    element_ids,
                    elements[0],
                    e,
                )
                if len(few_shot_examples) == 1:
                    few_shot_examples = None
                elif len(few_shot_examples) > 1:
                    few_shot_examples = few_shot_examples[:-1]
                else:
                    raise e

        valid_candidates = []
        valid_regexes = []
        if isinstance(candidates, str):
            candidates = [candidates]

        regex_pattern = re.compile(
            r"\* regex(?:.(?!\* final answer))*?```(.*?)```",
            flags=re.IGNORECASE | re.DOTALL,
        )
        type_pattern = re.compile(
            r"final answer.*```(.*?)```.*?$",
            flags=re.IGNORECASE | re.DOTALL,
        )
        alt_pattern = re.compile(
            r"\s*```([^`]*?)```",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for candidate in candidates:
            type_match = type_pattern.search(candidate)
            if not type_match:
                type_match = alt_pattern.search(candidate)
                if not type_match:
                    continue
            proposed_type = type_match.group(1).strip()

            if (
                proposed_type.lower() not in self.client.get_basic_types()
                and proposed_type.lower() not in ["none_t", "composite_t"]
            ):
                continue

            valid_candidates.append(proposed_type.lower())

            regex_match = regex_pattern.search(candidate)
            if regex_match:
                try:
                    valid_regexes.append(
                        Template.unescape_string(regex_match.group(1).strip())
                    )
                except json.decoder.JSONDecodeError:
                    valid_regexes.append(regex_match.group(1).strip())

        if not len(valid_candidates):
            get_logger().warning(
                "No valid candidate types found for element %s (%s)",
                element_ids,
                elements[0],
            )
            breakpoint()
            return None, None

        count_per_candidate = Counter(valid_candidates)
        max_count = max(count_per_candidate.values())
        valid_candidates = list(
            {v for v in valid_candidates if count_per_candidate[v] == max_count}
        )

        # Heuristic: if many candidates are found, prefer 1/NONE 2/MSG or COMPOSITE, 3/ A type in DEFAULT_TYPES 4/ a type in self.add_types 5/ a new type
        valid_candidates_tagged = []
        for c in valid_candidates:
            if c == "none_t":
                valid_candidates_tagged.append(c)
            elif c == "composite_t":
                valid_candidates_tagged.append(c)
            else:
                valid_candidates_tagged.append("DEFAULT")

        counts = Counter(valid_candidates_tagged)
        max_count = max(counts.values())
        filtered_tagged_candidates = [
            c for c in valid_candidates_tagged if counts[c] == max_count
        ]
        if "none_t" in filtered_tagged_candidates:
            determined_tag = "none_t"
        elif "composite_t" in filtered_tagged_candidates:
            determined_tag = "composite_t"
        elif "DEFAULT" in filtered_tagged_candidates:
            determined_tag = "DEFAULT"

        determined_type = next(
            c
            for c, t in zip(valid_candidates, valid_candidates_tagged)
            if t == determined_tag
        )

        for element_id in element_ids:
            self.tree.nodes[element_id].type = determined_type

        get_logger().info(
            "Assigned type %s to %s (%s)",
            determined_type,
            element_ids,
            elements[0],
        )

        self._write(
            query=system + "\n\n##########\n\n" + user,
            response="\n\n###########\n\n".join(candidates),
            element=elements[0],
            determined_type=determined_type,
            selected_regex=None,
        )

        return determined_type, None

    def run(self):
        ### Group by var_mapping
        name_clusters = defaultdict(list)
        clusters = []
        for k, n in enumerate(self.tree.nodes):
            if n and k in self.var_mapping:
                created_attribute = self.var_mapping[k].created_attribute
                if not created_attribute:
                    continue
                if created_attribute.endswith("_KEY"):
                    continue
                if created_attribute == "SYNTAX":
                    continue
                name_clusters[created_attribute].append(k)
            elif n and n.is_variable():
                clusters.append([k])

        for v in name_clusters.values():
            clusters.append(v)

        self.node_to_cluster = {
            node: cluster_id
            for cluster_id, cluster in enumerate(clusters)
            for node in cluster
        }
        self.cluster_to_nodes = {
            cluster_id: cluster for cluster_id, cluster in enumerate(clusters)
        }
        volumes_per_cluster = {
            k: sum(
                len(self.values[x].values) if x in self.values else 0 for x in v
            )
            for k, v in self.cluster_to_nodes.items()
        }

        determination_order = sorted(
            volumes_per_cluster.keys(),
            key=lambda x: volumes_per_cluster[x],
            reverse=True,
        )

        for cluster_id in tqdm(
            determination_order,
            total=len(determination_order),
            desc="Typing Elements",
            unit="element",
        ):
            element_ids = self.cluster_to_nodes[cluster_id]
            if any(
                self.tree.nodes[element_id].type is None
                for element_id in element_ids
            ):
                self.typing(element_ids)

        parser = Parser(
            self.tree,
            self.values,
            self.entries_per_template,
            self.embedding,
            var_mapping=self.var_mapping,
            schema_mapping=self.schema_mapping,
        )
        if self.output_dir:
            with open(os.path.join(self.output_dir, "parser.dill"), "wb") as f:
                dill.dump(parser, f)

        return parser
