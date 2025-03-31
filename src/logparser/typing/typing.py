import json
import os
import re
from collections import Counter

import dill
from tqdm import tqdm

from ..tools.api import OpenAITask
from ..tools.classes import Parser, Template
from ..tools.embedding import NaiveDistance
from ..tools.logging import get_logger
from ..tools.module import Module
from ..tools.prompts.typing import DEFAULT_TYPES, gen_prompt


class TemplateTyper:

    def __init__(
        self,
        caller,
        parser,
        few_shot_len=3,
        output_dir="output/",
        model="gemini-1.5-flash",
    ):
        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.types = set()
        self.add_types = []
        self.add_types_with_context = []
        self.few_shot_len = few_shot_len
        self.caller = caller
        self.model = model
        self.output_dir = output_dir

        # Fix potential value id issues
        del_list = set()
        for id, val in self.values.items():
            val.element_id = id
            if not len(val.values):
                del_list.add(id)
            elif not self.tree.nodes[id]:
                del_list.add(id)

        for id in del_list:
            del self.values[id]

        self.paths = {
            "prompts": os.path.join(self.output_dir, "prompts/"),
            "outputs": os.path.join(self.output_dir, "outputs/"),
            "results": os.path.join(self.output_dir, "results/"),
        }
        os.makedirs(self.paths["prompts"], exist_ok=True)
        os.makedirs(self.paths["outputs"], exist_ok=True)
        os.makedirs(self.paths["results"], exist_ok=True)

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

    def _get_typing_prompt_variables(self, element_id):
        matches = list(self.values[element_id].value_counts.keys())
        templates = [
            self.tree.gen_template(t)
            for t in self.tree.templates_per_node[element_id]
        ]
        element = self.tree.nodes[element_id]

        elt_index = 0
        for elt_id, element in enumerate(templates[0].elements):
            if element.id == element_id:
                elt_index = elt_id
                break

        all_matches = matches + [element.value]
        matches = list(set(all_matches))

        return templates, elt_index, element, matches

    def _write(self, query, response, element, determined_type, selected_regex):
        # Placeholder for writing logic
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
                ),
                f,
            )

    def typing(self, element_id, **kwargs):

        kwargs["n"] = 5
        kwargs["temperature"] = 0.5

        # Get relevant variables and generate prompts
        templates, elt_index, element, matches = (
            self._get_typing_prompt_variables(element_id)
        )

        typed_variable_values = [
            val
            for id, val in self.values.items()
            if self.tree.nodes[id].is_variable()
            and self.tree.nodes[id].type is not None
            and id != element_id
        ]
        few_shot_nodes = [
            val.element_id
            for val in self.values[element_id].get_closest(
                typed_variable_values,
                k=self.few_shot_len,
                tree=self.tree,
                embedding=self.embedding,
            )
        ]

        few_shot_examples = []
        for node in few_shot_nodes:
            fs_templates = [
                self.tree.gen_template(t)
                for t in self.tree.templates_per_node[node]
            ]
            fs_values = self.values[node].value_counts.keys()
            few_shot_examples.append(
                (fs_templates, fs_values, node, self.tree.nodes[node].type)
            )

        while True:

            user, system = gen_prompt(
                templates,
                matches,
                element_id,
                few_shot_examples,
                self.add_types_with_context,
            )
            task = OpenAITask(
                system_prompt=system,
                max_tokens=2048,
                model=self.model,
                message=user,
                **kwargs,
            )

            try:
                candidates = self.caller([task])[0].candidates
                break
            except ValueError as e:
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
        for candidate in candidates:
            candidate = candidate['content']
            type_match = type_pattern.search(candidate)
            if not type_match:
                continue
            proposed_type = type_match.group(1).strip()
            valid_candidates.append(proposed_type.upper())

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
                element_id,
                element,
            )
            return None, None

        count_per_candidate = Counter(valid_candidates)
        max_count = max(count_per_candidate.values())
        valid_candidates = list(
            {v for v in valid_candidates if count_per_candidate[v] == max_count}
        )

        # Heuristic: if many candidates are found, prefer 1/NONE 2/MSG or COMPOSITE, 3/ A type in DEFAULT_TYPES 4/ a type in self.add_types 5/ a new type
        valid_candidates_tagged = []
        for c in valid_candidates:
            if c == "NONE":
                valid_candidates_tagged.append(c)
            elif c == "MSG":
                valid_candidates_tagged.append(c)
            elif c == "COMPOSITE":
                valid_candidates_tagged.append(c)
            elif c in DEFAULT_TYPES:
                valid_candidates_tagged.append("DEFAULT")
            elif c in self.add_types:
                valid_candidates_tagged.append("ADD")
            else:
                valid_candidates_tagged.append("NEW")

        counts = Counter(valid_candidates_tagged)
        max_count = max(counts.values())
        filtered_tagged_candidates = [
            c for c in valid_candidates_tagged if counts[c] == max_count
        ]
        new_type = False
        if "NONE" in filtered_tagged_candidates:
            determined_tag = "NONE"
        elif "MSG" in filtered_tagged_candidates:
            determined_tag = "MSG"
        elif "COMPOSITE" in filtered_tagged_candidates:
            determined_tag = "COMPOSITE"
        elif "DEFAULT" in filtered_tagged_candidates:
            determined_tag = "DEFAULT"
        elif "ADD" in filtered_tagged_candidates:
            determined_tag = "ADD"
        else:
            determined_tag = "NEW"
            new_type = True

        determined_type = next(
            c
            for c, t in zip(valid_candidates, valid_candidates_tagged)
            if t == determined_tag
        )

        self.tree.nodes[element_id].type = determined_type
        if new_type:
            self.add_types.append(determined_type)
            example_template = templates[0]
            example_value = (
                templates[0]
                .match(templates[0].example_entry)[1][elt_index]
                .value
            )
            self.add_types_with_context.append(
                (determined_type, example_template, element_id, example_value)
            )
            get_logger().info(
                "Added new type %s for %s", determined_type, element_id
            )

        get_logger().info(
            "Assigned type %s to %s (%s)", determined_type, element_id, element
        )

        # Assign regex if applicable
        selected_regex = None
        for regex in valid_regexes:
            try:
                r = re.compile(regex)
            except:
                continue
            if all(r.match(v) is not None for v in matches):
                get_logger().info(
                    "Assigned regex %s to %s (%s)", regex, element_id, element
                )
                # self.tree.nodes[element_id].regex = regex
                selected_regex = regex
                break

        self._write(
            query=system + "\n\n##########\n\n" + user,
            response="\n\n###########\n\n".join(candidates),
            element=element,
            determined_type=determined_type,
            selected_regex=selected_regex,
        )

        return determined_type, selected_regex

    def run(self):
        ### Assign types by order of most seen value
        determination_order = sorted(
            self.values.keys(),
            key=lambda x: len(self.values[x].values),
            reverse=True,
        )

        for element_id in tqdm(
            determination_order,
            total=len(determination_order),
            desc="Typing Elements",
            unit="element",
        ):
            if self.tree.nodes[element_id].type is None:
                self.typing(element_id)
