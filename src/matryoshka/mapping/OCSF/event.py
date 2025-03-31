import os
import random
import re
from collections import Counter

import dill
from tqdm import tqdm

from matryoshka.classes import Module, Parser
from matryoshka.genai_api.api import LLMTask
from matryoshka.utils.logging import get_logger
from matryoshka.utils.OCSF import OCSFSchemaClient
from matryoshka.utils.prompts.mapping.OCSF.event import gen_prompt


class MapToEvents(Module):

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
        super().__init__("MapToEvents", caller=caller)

        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir
        self.few_shot_len = few_shot_len
        self.cache_dir = cache_dir
        self.ablation_fewshot = ablation_fewshot

        self.var_mapping = parser.var_mapping
        self.schema_mapping = parser.schema_mapping

        self.event_types = parser.event_types if parser.event_types else {}
        self.embedding = parser.embedding
        self.client = (
            OCSFSchemaClient(
                self.caller, saved_path=os.path.join(self.cache_dir, "OCSF")
            )
            if ocsf_client is None
            else ocsf_client
        )

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
            self.paths = {}

    def _write(
        self, query, response, template, template_example, determined_class
    ):
        if self.output_dir:
            with open(
                os.path.join(self.paths["prompts"], str(template)),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(query)

            with open(
                os.path.join(self.paths["outputs"], str(template)),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(response)

            with open(
                os.path.join(self.paths["results"], str(template)),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"{template}\n{template_example}\n{determined_class}")

    def get_fewshot_examples(self, template_id, templates):
        """Find few-shot examples"""
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

    def parse_event_desc(self, event_desc):
        parsed_desc = {}
        if "attributes" not in event_desc:
            return None

        for desc_item in event_desc["attributes"]:
            for key, val in desc_item.items():
                parsed_desc[key] = val["description"]

        return parsed_desc

    def matching(self, template_id, do_few_shot=True, **kwargs):

        kwargs["n"] = 5
        kwargs["temperature"] = 0.33

        # Get relevant information for prompt
        entry_examples = random.sample(
            self.entries_per_template[template_id],
            k=min(5, len(self.entries_per_template[template_id])),
        )

        template_example = self.tree.gen_template(
            template_id
        ).format_as_example(
            force_match_with_entry=False,
            entry=entry_examples[0],
            regex=True,
            types=True,
        )

        specs = self.client.get_classes()
        specs = {s: v["description"] for s, v in specs.items()}

        detailed_specs = {
            name: self.parse_event_desc(self.client.get_class_details(name))
            for name in specs
        }
        detailed_specs = {
            k: {"fields": v, "description": specs[k]}
            for k, v in detailed_specs.items()
            if v and k != "base_event"
        }

        # Get few-shot examples
        assigned_templates = [
            t
            for t in range(len(self.tree.templates))
            if t in self.event_types and self.event_types[t]
        ]
        if assigned_templates:
            fs = (
                self.get_fewshot_examples(
                    template_id, templates=assigned_templates
                )
                or []
            )
        else:
            fs = []
        fs_template, fs_examples, fs_events = [], [], []

        if do_few_shot:
            for t_id in fs:
                examples = random.sample(
                    self.entries_per_template[t_id],
                    k=min(5, len(self.entries_per_template[t_id])),
                )
                template_example = self.tree.gen_template(
                    t_id
                ).format_as_example(
                    force_match_with_entry=False,
                    entry=examples[0],
                    regex=True,
                    types=True,
                )
                event = self.event_types[t_id]
                fs_template.append(template_example)
                fs_examples.append(examples)
                fs_events.append(event)

        user, system = gen_prompt(
            template_example,
            detailed_specs,
            entry_examples,
            (
                (fs_template, fs_examples, fs_events)
                if not self.ablation_fewshot
                else None
            ),
        )

        task = LLMTask(
            system_prompt=system,
            model=self.model,
            message=user,
            thinking_budget=128,
            **kwargs,
        )

        candidates = self.caller(task).candidates
        valid_candidates = []
        if isinstance(candidates, str):
            candidates = [candidates]

        filter_regex = re.compile(r"[^a-zA-Z0-9_]")
        for candidate in candidates:
            list_of_values = candidate.lower().strip().split("\n")

            # remove potential descriptions:
            clean = []
            for c in list_of_values:
                c = c.strip()
                if not c:
                    continue
                if ":" in c:
                    c = c.split(":")[0].strip()
                elif " " in c or "\t" in c:
                    c = c.split()[0].strip()
                c = filter_regex.sub("", c).lower().strip()
                clean.append(c)
            clean = [c for c in clean if c in specs or c == "unsure"][:3]

            if not clean:
                continue

            if len(clean) > 1 and "unsure" in clean:
                clean.remove("unsure")

            valid_candidates.append(tuple(clean))

        if not len(valid_candidates):
            get_logger().warning(
                "No valid event type found for template %s", template_example
            )
            return []

        count_per_candidate = {}
        for candidate in valid_candidates:
            for i, c in enumerate(candidate):
                count_per_candidate[c] = count_per_candidate.get(c, 0) + 3 - i

        if do_few_shot:
            solutions_without_fs = self.matching(
                template_id, do_few_shot=False, **kwargs
            )

            for solution, c in solutions_without_fs:
                count_per_candidate[solution] = (
                    count_per_candidate.get(solution, 0) + c
                )

        max_count = max(count_per_candidate.values())
        if count_per_candidate.get("unsure", 0) == max_count:
            rslt = []
        elif "unsure" in count_per_candidate:
            del count_per_candidate["unsure"]

        rslt = [v for v in Counter(count_per_candidate).most_common(3)]

        self._write(
            query=system + "\n\n##########\n\n" + user,
            response="\n\n###########\n\n".join(candidates),
            template=template_id,
            template_example=entry_examples[0],
            determined_class=rslt,
        )

        return rslt

    def process(self, **kwargs):
        ### Assign types by order of most seen value
        template_list = [i for i, t in enumerate(self.tree.templates) if t]
        determination_order = sorted(
            template_list,
            key=lambda x: len(self.entries_per_template[x]),
            reverse=True,
        )

        for template_id in tqdm(
            determination_order,
            total=len(determination_order),
            desc="Matching Templates to OCSF Events",
            unit="template",
        ):
            if template_id not in self.event_types:
                if not self.entries_per_template[template_id]:
                    continue
                self.event_types[template_id] = tuple(
                    v[0] for v in self.matching(template_id)
                )
            get_logger().info(
                "%s ==> %s",
                self.tree.gen_template(template_id),
                self.event_types[template_id],
            )

        parser = Parser(
            tree=self.tree,
            values=self.values,
            entries_per_template=self.entries_per_template,
            embedding=self.embedding,
            event_types=self.event_types,
            var_mapping=self.var_mapping,
            schema_mapping=self.schema_mapping,
        )
        if self.output_dir:
            with open(os.path.join(self.output_dir, "parser.dill"), "wb") as f:
                dill.dump(parser, f)

        return parser
