import json
import os
import random
import re
from collections import Counter, defaultdict

import dill
import torch
from tqdm import tqdm

from ..classes import Module, OCSFTemplateMapping, Parser
from ..genai_api.api import LLMTask
from ..utils.logging import get_logger
from ..utils.OCSF import OCSFSchemaClient
from ..utils.prompts.converter.map_templates import (
    gen_prompt as gen_mapping_prompt,
)
from ..utils.prompts.converter.template_description import (
    gen_prompt as gen_description_prompt,
)


class FillEvent(Module):

    def __init__(
        self,
        caller,
        parser,
        few_shot_len=3,
        output_dir="output/",
        model="gemini-2.5-flash",
        ocsf_client=None,
        cache_dir=".cache/",
        save_contents=False,
    ):
        super().__init__("FillEvent", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.few_shot_len = few_shot_len
        self.save_contents = save_contents

        self.event_types = parser.event_types if parser.event_types else {}
        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.template_mapping = (
            parser.template_mapping
            if parser.template_mapping
            else defaultdict(OCSFTemplateMapping)
        )
        self.embedding = parser.embedding
        self.client = (
            OCSFSchemaClient(caller, saved_path=os.path.join(cache_dir, "OCSF"))
            if ocsf_client is None
            else ocsf_client
        )

        self.clusters = {}

        self.cosine_threshold_for_identical = 0.98

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
            os.path.join(self.output_dir, "parser.dill"),
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
                        template_mapping=self.template_mapping,
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
                        template_mapping=self.template_mapping,
                    ),
                    f,
                )

    def _get_fewshot_examples_naive(self, template_id, templates):
        """Find few-shot examples"""
        if not templates:
            return []
        target_template = self.tree.gen_template(template_id)
        examples = [
            k
            for k in self.tree.get_ordered_templates(
                target_template.elements[-1].id
            )
            if k[1] in templates
        ]

        if not examples:
            return None

        closest_naive = self.embedding.template_distance(
            template_id, templates=templates
        )

        order_naive = {k: -v for (k, v) in closest_naive}
        order_tree = {t_idx: k for (k, t_idx) in examples}

        closest_tree = sorted(
            examples,
            key=lambda x: (
                order_tree[x[1]],
                order_naive[x[1]],
            ),
        )
        closest_tree = [i[1] for i in closest_tree]
        closest_naive = [i[0] for i in closest_naive]

        count = self.few_shot_len
        examples = closest_tree[: count // 2]
        for e in closest_naive:
            if e not in examples:
                examples.append(e)
            if len(examples) == count:
                break

        return examples

    def _get_fewshot_examples(self, template_id, templates):
        if not templates:
            return []

        target_embeddings = [
            self.template_mapping[t].embedding for t in templates
        ]
        target_embeddings = torch.cat(target_embeddings, dim=1)
        source_embeddings = self.template_mapping[template_id].embedding

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cos(source_embeddings, target_embeddings)
        closest = sim.topk(
            k=min(len(templates), self.few_shot_len), largest=True
        ).indices
        return [templates[i] for i in closest]

    def _get_prompt_parameters(self, template_id):

        examples = random.sample(
            self.entries_per_template[template_id],
            k=min(5, len(self.entries_per_template[template_id])),
        )

        template = self.tree.gen_template(
            template_id
        ).format_without_variables()

        return template, examples

    def generate_description(self, template_id, **kwargs):
        # Get relevant information for prompt
        template_params = self._get_prompt_parameters(template_id)

        # Get few shot examples
        templates = list(
            {
                t_id
                for t_id in range(len(self.tree.templates))
                if t_id in self.template_mapping
                and self.template_mapping[t_id].description
            }
        )
        fs_templates = (
            self._get_fewshot_examples_naive(template_id, templates) or []
        )
        fs_params = [
            self._get_prompt_parameters(t)
            + (self.template_mapping[t].description,)
            for t in fs_templates
        ]

        user, history, system = gen_description_prompt(
            template_params, fs_params
        )
        kwargs["temperature"], kwargs["n"] = 0, 1
        task = LLMTask(
            system_prompt=system,
            max_tokens=128,
            history=history,
            model="gemini-2.5-flash",
            message=user,
            **kwargs,
        )
        candidates = self.caller(task)
        if isinstance(candidates, list):
            candidates = candidates[0]
        try:
            description = re.sub(
                "\s+",
                " ",
                json.loads(candidates.candidates[0].strip()),
            )
        except json.JSONDecodeError:
            description = re.sub(
                '(^")|("$)', "", candidates.candidates[0].strip()
            )
        get_logger().info(
            "Description for template %s (%s): %s",
            template_id,
            template_params[0][0],
            description,
        )
        self.template_mapping[template_id].description = description

    def generate_embeddings(self):
        """Generate embeddings for all descriptions"""
        requires_save = False
        settings = [template_id for template_id in self.template_mapping]
        for template_id, mapping_obj in tqdm(
            self.template_mapping.items(),
            desc="Generating Embeddings",
            total=len(settings),
        ):
            if not mapping_obj.description:
                continue

            if mapping_obj.embedding is not None:
                continue

            task = LLMTask(
                message=mapping_obj.description,
                query_type="embedding",
                model="text-embedding-005",
            )
            mapping_obj.embedding = torch.tensor(
                self.caller(task, distribute_parallel_requests=False)
            ).unsqueeze(1)
            requires_save = True

        return requires_save

    def _filter_attributes(self, template_id, event, attributes, k=25):
        """Filter attributes based on the mapping"""
        source_embedding = self.template_mapping[template_id].embedding

        text_description = {
            k: v
            for k, v in self.client.get_description(
                event,
            ).items()
            if k in attributes
        }
        targets = [
            (k, self.client.generated_descriptions[k].embedding)
            for k in text_description.keys()
        ]
        dest_embeddings = torch.tensor([t[1] for t in targets]).t()

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        sim = cos(source_embedding, dest_embeddings)
        closest = sim.topk(k=min(len(targets), k), largest=True).indices
        selected = [targets[i][0] for i in closest]
        return {k: v for k, v in attributes.items() if k in selected}

    def generate_default_mapping(self, template_id, event, **kwargs):
        # Get relevant information for prompt
        template_params = self._get_prompt_parameters(template_id) + (
            self.template_mapping[template_id].description,
        )

        # Get few shot examples
        templates = list(
            {
                t_id
                for t_id in range(len(self.tree.templates))
                if t_id in self.template_mapping
                and event in self.template_mapping[t_id].assignment
            }
        )
        fs_templates = self._get_fewshot_examples(template_id, templates) or []
        fs_params = [
            self._get_prompt_parameters(t)
            + (
                self.template_mapping[t].description,
                self.template_mapping[t].demonstrations[event],
                self.template_mapping[t].candidates[event],
                self.template_mapping[t].assignment[event],
            )
            for t in fs_templates
        ]

        # Get list of possible attributes
        flattened_event, event_description = self.client.flatten_event(
            event
        ), self.client.get_event_description(event)

        # Remove attributes that are not top level
        flattened_event = {
            k: v for k, v in flattened_event.items() if k.count(".") < 2
        }

        # Remove attributes that are not objects
        flattened_event = {
            k: v for k, v in flattened_event.items() if not v.is_object
        }

        # Remove mapped attributes
        mapped_attributes = {
            f
            for elt in self.tree.templates[template_id]
            if elt in self.var_mapping
            for f in self.var_mapping[elt][event].field_list
        }
        flattened_event = {
            k: v
            for k, v in flattened_event.items()
            if k not in mapped_attributes
        }

        # Select closest attributes by cosine similarity
        flattened_event = self._filter_attributes(
            template_id, event, flattened_event
        )

        # Simplify attributes
        flattened_event = {
            k: v.to_dict_simple() for k, v in flattened_event.items()
        }

        kwargs["temperature"] = 0
        kwargs["n"] = 1

        tasks = []
        attribute_list = list(flattened_event.items())
        for _ in range(3):
            random.shuffle(attribute_list)
            template_params_full = template_params + (attribute_list,)
            user, history, system = gen_mapping_prompt(
                template_params_full, fs_params, event_description
            )
            tasks.append(
                LLMTask(
                    system_prompt=system,
                    model=self.model,
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
            for candidate in llm_response.candidates:
                mapping_match = result_pattern.search(candidate)
                demo_match = demo_pattern.search(candidate)
                if not mapping_match or not demo_match:
                    continue
                try:
                    proposal = json.loads(mapping_match.group(1))
                except json.JSONDecodeError:
                    proposal = {}

                # Description
                demonstration = demo_match.group(1).strip()

                # Fix demo of items with a variable
                variable_tags = {
                    val
                    for val in re.findall(
                        "\[[a-zA-Z_]+_t\]", template_params[0]
                    )
                }
                fields_mapped_to_variable = {
                    k
                    for k in proposal
                    if any(vt in str(proposal[k]) for vt in variable_tags)
                }
                for field in fields_mapped_to_variable:
                    demonstration = re.sub(f".*{field}.*", "", demonstration)

                # Remove values containing variables or not in the original attributes
                proposal = {
                    k: v
                    for k, v in proposal.items()
                    if k not in fields_mapped_to_variable
                    and k in flattened_event
                }

                valid_candidates.append(
                    ({p for p in proposal}, proposal, demonstration)
                )

        field_counts = Counter(
            [mapped_field for p in valid_candidates for mapped_field in p[0]]
        )
        selected_fields = [p for p in field_counts if field_counts[p] > 1]
        field_demonstrations = {}
        for field in flattened_event:
            demo_regex = re.compile(f".*{field}.*", flags=re.IGNORECASE)
            included = [
                idx
                for idx, (proposal, _, _) in enumerate(valid_candidates)
                if (field in selected_fields) == (field in proposal)
            ]
            demonstrations = [
                d.group(0)
                for d in [
                    demo_regex.search(d)
                    for idx, (_, _, d) in enumerate(valid_candidates)
                    if idx in included
                ]
                if d
            ]
            field_demonstrations[field] = (
                demonstrations[0] if demonstrations else None
            )

        # Get single most common value for each mapped field
        values = {
            field: Counter(
                [
                    proposal.get(field, None)
                    for (_, proposal, _) in valid_candidates
                    if field in proposal
                ]
            )
            for field in selected_fields
        }
        values = {field: values[field].most_common(1)[0][0] for field in values}

        demo = "\n".join(
            f"{field}: {field_demonstrations[field]}"
            for field, _ in attribute_list
        )

        self.template_mapping[template_id].assignment[event] = {
            field: values[field] for field in selected_fields
        }
        self.template_mapping[template_id].demonstrations[event] = demo
        self.template_mapping[template_id].candidates[event] = attribute_list

    def process(self, **kwargs):
        ### Assign types by order of most seen value
        determination_order = sorted(
            list(range(len(self.tree.templates))),
            key=lambda x: len(self.entries_per_template[x]),
            reverse=True,
        )

        # Generate Descriptions
        requires_save = False
        for template_id in tqdm(
            determination_order,
            total=len(determination_order),
            desc="Generating Template Descriptions",
            unit="template",
        ):
            if template_id not in self.template_mapping:
                self.template_mapping[template_id] = OCSFTemplateMapping(
                    id=template_id
                )

            if self.template_mapping[template_id].description:
                continue

            self.generate_description(template_id)
            requires_save = True

        # Save
        if requires_save:
            self._save_dill()

        # Generate Embeddings
        requires_save = self.generate_embeddings()

        # Save
        if requires_save:
            self._save_dill()

        # Generate Default Mappings
        for template_id in tqdm(
            determination_order,
            total=len(determination_order),
            desc="Generating Default Mappings",
            unit="template",
        ):

            for event in self.event_types[template_id]:
                if event in self.template_mapping[template_id].assignment:
                    continue
                if event == "unsure":
                    continue

                self.generate_default_mapping(template_id, event)

            def print_template(t_id):
                print("Template ID: ", t_id)
                print("Template: ", self.tree.gen_template(t_id))
                print("Descrpition: ", self.template_mapping[t_id].description)
                print("Mappings:")
                print(self.template_mapping[t_id].assignment)
                for k, _ in self.template_mapping[t_id].assignment.items():
                    print(f"* Event {k}:")
                    for key, val in (
                        self.template_mapping[t_id].assignment[k].items()
                    ):
                        print(f"{key}: {json.dumps(val, indent=2)}")

            print_template(template_id)
            self._save_dill()

        # Save
        self._save_dill()

        return Parser(
            tree=self.tree,
            values=self.values,
            entries_per_template=self.entries_per_template,
            event_types=self.event_types,
            embedding=self.embedding,
            var_mapping=self.var_mapping,
            template_mapping=self.template_mapping,
        )
