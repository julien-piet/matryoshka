import json
import os
import random
import re
from copy import deepcopy
from difflib import SequenceMatcher

import dill
import numpy as np
from tqdm import tqdm

from .tools.api import OpenAITask
from .tools.classes import (
    Element,
    ElementType,
    Match,
    Template,
    TemplateTree,
    Value,
)
from .tools.embedding import NeuralEmbedding
from .tools.module import Module
from .tools.prompts import (
    DEFAULT_TYPES,
    c2a_system_prompt,
    checklist_with_matches,
    checklist_without_matches,
    generate_adjust_separation_prompt,
    generate_c2a_prompt,
    generate_fewshot_examples_separation,
    generate_known_prefix_prompt,
    generate_regex_update_prompt,
    generate_submatches_prompt,
    generate_typing_prompt,
    generate_typing_prompt_example,
    generate_typing_system_prompt,
    generate_variable_confirmation_prompt,
    regex_update_system_prompt,
    system_prompt_separation,
    user_prompt_separation,
    variable_confirmation_system_prompt,
)


class VariableParser(Module):

    def __init__(
        self,
        caller=None,
        unit_regex=re.compile("\n"),
        parallel_attempts=5,
        few_shot_len=5,
        init_templates=None,
        debug_folder=None,
        **kwargs,
    ) -> None:

        super().__init__("variableparser", caller=caller)
        self.parallel_attempts = parallel_attempts
        self.unit_regex = unit_regex
        self.few_shot_len = few_shot_len

        self.embedding = NeuralEmbedding(filt=True)

        self.tree = TemplateTree(embedding_model=self.embedding)

        self.matches = []
        self.llm_outputs = []
        self.llm_inputs = []
        self.types = DEFAULT_TYPES[:]
        self.add_types = []
        self.add_types_with_context = []

        self.number_varia_queries = 0
        self.number_regex_queries = 0

        if init_templates:
            with open(init_templates, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            for seed in seeds:
                self.tree.add_template(
                    Template.load_from_json(
                        seed["template"], seed["example"].strip()
                    ),
                    seed["example"].strip(),
                    fixed=True,
                )

        self.entries_per_template = {
            t: [] for t in range(len(self.tree.templates))
        }
        self.values = {
            v: Value(v, self.embedding)
            for v in range(len(self.tree.nodes))
            if v > 0 and self.tree.nodes[v].is_variable()
        }

        self.start_index = 0
        self.debug_folder = debug_folder

    def init_caller(self, caller):
        self.caller = caller

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["embedding"]
        del state["caller"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.embedding = NeuralEmbedding(filt=True)
        self.tree.init_embedding(self.embedding)
        for v in self.values.values():
            v.init_embedding(self.embedding)
        self.caller = None
        if "add_types" not in self.__dict__:
            self.add_types = []
            self.add_types_with_context = []

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

    def typing(self, element_id, model="gemini-1.5-flash", **kwargs):

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
                typed_variable_values, k=self.few_shot_len, tree=self.tree
            )
        ]

        if few_shot_nodes:
            few_shot_examples = []
            for node in few_shot_nodes:
                few_shot_examples.append(
                    generate_typing_prompt_example(
                        *self._get_typing_prompt_variables(node)
                    )
                )
        else:
            few_shot_examples = None

        prompt, system_prompt = generate_typing_prompt(
            templates, elt_index, element, matches, few_shot_examples
        ), generate_typing_system_prompt(
            self.types,
            self.add_types_with_context if len(self.add_types) else None,
        )

        # Query the LLM
        kwargs["n"] = 1
        kwargs["temperature"] = 0
        while True:
            task = OpenAITask(
                system_prompt=system_prompt,
                max_tokens=2048,
                model=model,
                message=prompt,
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
                prompt = generate_typing_prompt(
                    templates, elt_index, element, matches, few_shot_examples
                )

        valid_candidates = []
        if isinstance(candidates, str):
            candidates = [candidates]
        for candidate in candidates:
            candidate = candidate["content"]
            if "### result" not in candidate.lower():
                continue
            split_regex = re.compile(r"### result", flags=re.IGNORECASE)
            proposed_type = re.split(split_regex, candidate)[-1].strip()
            valid_candidates.append(proposed_type.upper())

        if not len(valid_candidates):
            raise ValueError("No valid types found")

        print(f"Candidates are {valid_candidates}")

        # Heuristic: If over half the answers are None, return None. If not, return most common answer.
        counts = {v: 0 for v in valid_candidates}
        for v in valid_candidates:
            counts[v] += 1

        determined_type = None
        if "NONE" in counts and counts["NONE"] / len(valid_candidates) > 0.5:
            determined_type = "NONE"
        else:
            counts["NONE"] = 0
            max_count = max(counts.values())
            most_common_types = [k for k, v in counts.items() if v == max_count]
            determined_type = sorted(
                most_common_types, key=lambda x: 0 if x in self.types else 1
            )[0]

        self.tree.nodes[element_id].type = determined_type
        if (
            determined_type not in self.types
            and determined_type not in self.add_types
            and determined_type not in ["NONE", "COMPOSITE", "MSG"]
        ):
            self.add_types.append(determined_type)
            example_template = templates[0]
            example_value = (
                templates[0]
                .match(templates[0].example_entry)[1][elt_index]
                .value
            )
            self.add_types_with_context.append(
                (determined_type, example_template, elt_index, example_value)
            )

        print(f"Assigned type {determined_type}")
        return determined_type, prompt, system_prompt

    def separate(
        self,
        entry,
        examples=None,
        prefix_elements=None,
        unmatched_suffix=None,
        model="gemini-1.5-flash",
        **kwargs,
    ):

        if len(self.tree.templates) < 5:
            model = "gemini-1.5-flash"

        # First, we identify parts of the entry that match constant elements in the examples that are not just punctuation
        if examples:
            all_elements = [
                (elt, elt_idx, example, example_idx)
                for example_idx, example in enumerate(examples)
                for elt_idx, elt in enumerate(example.elements)
                if not elt.is_variable()
                and re.search(r"[a-zA-Z0-9]", elt.value)
            ]
            full_matches = []
            partial_matches = []
            seen = set()
            unmatched = unmatched_suffix if unmatched_suffix else entry.strip()
            for elt, elt_idx, example, example_idx in all_elements:
                mt = unmatched.find(elt.value)
                if mt != -1:
                    match_string = unmatched[mt : mt + len(elt.value)]
                    if match_string in seen:
                        continue
                    full_matches.append(
                        (
                            match_string,
                            elt.value,
                            elt_idx,
                            example.format_short(color=False),
                            example_idx,
                        )
                    )
                    seen.add(match_string)
                else:
                    mt = SequenceMatcher(
                        None, elt.value, unmatched
                    ).find_longest_match()
                    if mt and mt.size > 10:
                        match_string = unmatched[mt.b : mt.b + mt.size]
                        if match_string in seen:
                            continue
                        partial_matches.append(
                            (
                                (
                                    match_string,
                                    elt.value,
                                    elt_idx,
                                    example.format_short(color=False),
                                    example_idx,
                                ),
                                (mt.b, mt.b + mt.size),
                            )
                        )
                        seen.add(match_string)

            partial_matches = sorted(
                partial_matches, key=lambda x: -len(x[0][0])
            )
            full_matches = sorted(full_matches, key=lambda x: -len(x[0]))

            filtered_partial_matches = []
            accounted_for = [False for _ in range(len(unmatched))]
            for match, indices in partial_matches:
                if all(accounted_for[i] for i in range(*indices)):
                    continue
                filtered_partial_matches.append(match)
                for i in range(*indices):
                    accounted_for[i] = True
            partial_matches = filtered_partial_matches

        kwargs["n"] = self.parallel_attempts
        kwargs["temperature"] = 0.5
        while True:
            # Filter examples according to which elements are included:
            if examples:
                full_matches = [
                    f for f in full_matches if f[-1] < len(examples)
                ]
                partial_matches = [
                    p for p in partial_matches if p[-1] < len(examples)
                ]
            task = OpenAITask(
                system_prompt=system_prompt_separation,
                max_tokens=4096,
                model=model,
                message=user_prompt_separation.format(
                    generate_fewshot_examples_separation(examples),
                    entry.strip(),
                    (
                        generate_known_prefix_prompt(
                            Template(prefix_elements), unmatched_suffix
                        )
                        if prefix_elements
                        else ""
                    ),
                    (
                        generate_submatches_prompt(
                            full_matches, partial_matches
                        )
                        if examples
                        and (len(full_matches) or len(partial_matches))
                        else ""
                    ),
                    (
                        checklist_with_matches
                        if examples
                        and (len(full_matches) or len(partial_matches))
                        else checklist_without_matches
                    ),
                ),
                **kwargs,
            )

            try:
                candidates = self.caller([task])[0].candidates
                break
            except ValueError as e:
                if len(examples) == 1:
                    examples = None
                elif len(examples) > 1:
                    examples = examples[:-1]
                else:
                    raise e

        valid_candidates = []
        backup_candidates = []
        for candidate in candidates:
            candidate = candidate["content"]
            if "### result" not in candidate.lower():
                continue
            split_regex = re.compile(r"### result", flags=re.IGNORECASE)
            json_separation = re.split(split_regex, candidate)[-1].strip()
            if "```json" in json_separation:
                json_separation = json_separation.split("```json")[1].split(
                    "```"
                )[0]
            elif "```" in json_separation:
                json_separation = json_separation.split("```")[1]
            elif "[" in json_separation:
                json_separation = json_separation.split("[")[1]

            try:
                json_separation = json.loads(json_separation)
            except ValueError as e:
                print(f"Error parsing JSON: {e}")
                continue

            try:
                template = Template.load_from_json(
                    json_separation,
                    (
                        unmatched_suffix.strip()
                        if unmatched_suffix
                        else entry.strip()
                    ),
                )
            except ValueError as e:
                print(f"Error loading JSON: {e}")
                continue

            if unmatched_suffix:
                template.elements = prefix_elements + template.elements
                template.example_entry = entry.strip()

            orig_regex = True
            try:
                if not template.match(entry.strip())[0]:
                    # Try replacing the regex of each element until it matches
                    matches = False
                    for elt_id, elt in enumerate(template.elements):
                        if elt.is_variable():
                            original_regex = elt.regexp
                            elt.regexp = (
                                ".+?" if " " in elt.value.strip() else "\S+?"
                            )
                            template.generate_regex()
                            if template.match(entry.strip())[0]:
                                matches = True
                                break
                            elt.regexp = original_regex
                    if matches:
                        orig_regex = False
                    else:
                        print(
                            f"Error: created template {template} does not match {entry}"
                        )
                        continue
            except KeyError as e:
                print(f"Error matching template: {e}")
                continue

            # Make sure it does not match an example. If it does, still add it to the least, but only select if nothing better is found.
            overcaptures = False
            for ex in examples:
                match, _ = template.match(ex.example_entry)
                if match:
                    backup_candidates.append(
                        (template, candidate, task.message)
                    )
                    overcaptures = True
                    break

            if not overcaptures:
                valid_candidates.append((template, candidate, task.message))

        if not len(valid_candidates) and not len(backup_candidates):
            raise ValueError("No valid tokenizations found")

        if not len(valid_candidates):
            valid_candidates = backup_candidates

        # Heuristic: Return options with most votes, and within that those that reuse the most regexes.
        counts = {
            (
                len(e[0].elements),
                len([v for v in e[0].elements if v.is_variable()]),
            ): 0
            for e in valid_candidates
        }
        for e, _, _ in valid_candidates:
            counts[
                (
                    len(e.elements),
                    len([v for v in e.elements if v.is_variable()]),
                )
            ] += 1

        max_ct = max(counts.values())
        valid_candidates = [
            e
            for e in valid_candidates
            if counts[
                (
                    len(e[0].elements),
                    len([v for v in e[0].elements if v.is_variable()]),
                )
            ]
            == max_ct
        ]
        if examples is not None:
            existing_regexes = set()
            for example in examples:
                for elt in example.elements:
                    if elt.is_variable():
                        existing_regexes.add(elt.regexp)
            reuse_count = []
            for candidate in valid_candidates:
                reuse_count.append(
                    len(
                        [
                            v
                            for v in candidate[0].elements
                            if v.is_variable() and v.regexp in existing_regexes
                        ]
                    )
                )
            max_reuse = max(reuse_count)
            valid_candidates = [
                v
                for i, v in enumerate(valid_candidates)
                if reuse_count[i] == max_reuse
            ]

        return random.sample(valid_candidates, 1)[0]

    def adjust_template(
        self,
        template_1,
        template_2,
        entries_1,
        entries_2,
        raw_template_1,
        model="gemini-1.5-flash",
        **kwargs,
    ):
        kwargs["n"] = self.parallel_attempts
        kwargs["temperature"] = 0.5
        task = OpenAITask(
            system_prompt=system_prompt_separation,
            max_tokens=4096,
            model=model,
            message=generate_adjust_separation_prompt(
                template_1, template_2, entries_1, entries_2
            ),
            **kwargs,
        )

        candidates = self.caller([task])[0]

        valid_candidates = []
        votes = []
        for candidate in candidates:
            candidate = candidate["content"]
            if "### result" not in candidate.lower():
                continue
            split_regex = re.compile(r"### result", flags=re.IGNORECASE)
            result = re.split(split_regex, candidate)[-1].strip()
            decision = result.split("\n")[0]
            if decision == "(3)":
                votes.append(3)
                continue
            elif decision == "(2)":
                votes.append(2)
                continue
            else:
                votes.append(1)

            try:
                json_separation = result.split("\n", maxsplit=1)[1]
            except IndexError:
                continue

            if "```json" in json_separation:
                json_separation = json_separation.split("```json")[1].split(
                    "```"
                )[0]
            elif "```" in json_separation:
                json_separation = json_separation.split("```")[1]
            elif "[" in json_separation:
                json_separation = json_separation.split("[")[1]

            try:
                json_separation = json.loads(json_separation)
            except ValueError as e:
                print(f"Error parsing JSON: {e}")
                continue

            try:
                template = Template.load_from_json(
                    json_separation,
                    entries_1[0].strip(),
                )
            except ValueError as e:
                print(f"Error loading JSON: {e}")
                continue

            try:
                if not template.match(entries_1[0].strip())[0]:
                    print(
                        f"Error: created template {template} does not match {entries_1[0].strip()}"
                    )
                    continue
            except KeyError as e:
                print(f"Error matching template: {e}")
                continue

            # Confirm the prefix is correct
            correct_prefix = True
            for elt_idx, elt_val in enumerate(raw_template_1.elements):
                if (
                    "_" in str(elt_val.id)
                    or len(self.tree.templates_per_node[elt_val.id]) == 1
                ):
                    break

                if template.elements[elt_idx] != elt_val:
                    print("Error: new template does not share prefix")
                    correct_prefix = False
                    break
            if not correct_prefix:
                continue

            valid_candidates.append((template, candidate, task.message))

        votes_ctn = {1: 0, 2: 0, 3: 0}
        for vote in votes:
            votes_ctn[vote] += 1

        max_ctn = max(votes_ctn.values())
        decision = sorted(
            [v for v, c in votes_ctn.items() if c == max_ctn], reverse=True
        )[0]

        if decision != 1:
            return decision, None

        if not len(valid_candidates):
            print(f"No valid adjustments found for template {template_1}")
            return 3, None

        # Heuristic: Return most common answer. If there is a tie, return the one with the least underspecified variables. If there is still a tie, return that with most constants.
        counts = {
            (
                len(e[0].elements),
                len([v for v in e[0].elements if v.is_variable()]),
            ): 0
            for e in valid_candidates
        }
        for e, _, _ in valid_candidates:
            counts[
                (
                    len(e.elements),
                    len([v for v in e.elements if v.is_variable()]),
                )
            ] += 1

        max_ct = max(counts.values())
        valid_candidates = [
            e
            for e in valid_candidates
            if counts[
                (
                    len(e[0].elements),
                    len([v for v in e[0].elements if v.is_variable()]),
                )
            ]
            == max_ct
        ]

        valid_candidates = sorted(
            valid_candidates,
            key=lambda x: (
                sum(
                    [
                        1
                        for v in x[0].elements
                        if v.is_variable() and ".*" in v.regexp
                    ]
                ),
                -sum([1 for v in x[0].elements if not v.is_variable()]),
            ),
        )

        return 1, valid_candidates[0]

    def confirm_variable(
        self, value, templates, element_idx, matches, model, **kwargs
    ):
        # Ask the LLM if this field should be a variable or a constant
        kwargs["n"] = self.parallel_attempts
        kwargs["temperature"] = 0.5
        task = OpenAITask(
            system_prompt=variable_confirmation_system_prompt,
            max_tokens=4096,
            model=model,
            message=generate_variable_confirmation_prompt(
                templates, value, matches, element_idx
            ),
            **kwargs,
        )
        candidates = self.caller([task])[0]
        valid_candidates = []
        for candidate in candidates:
            candidate = candidate["content"]
            if "### result" not in candidate.lower():
                continue
            split_regex = re.compile(r"### result", flags=re.IGNORECASE)
            rslt = re.split(split_regex, candidate)[-1].strip()
            if rslt.lower() == "true":
                valid_candidates.append(1)
            else:
                valid_candidates.append(0)

        if sum(valid_candidates) / len(valid_candidates) > 0.5:
            self._write_variable_update(
                task.message,
                sum(valid_candidates) / len(valid_candidates),
                True,
            )
            return True

        self._write_variable_update(
            task.message,
            sum(valid_candidates) / len(valid_candidates),
            True,
        )

        return False

    def c2a(
        self, template_a, template_b, element_idx, model="gemini-1.5-flash", **kwargs
    ):
        # Ask the LLM if this field should be a variable or a constant
        kwargs["n"] = self.parallel_attempts
        kwargs["temperature"] = 0.5
        task = OpenAITask(
            system_prompt=c2a_system_prompt,
            max_tokens=4096,
            model=model,
            message=generate_c2a_prompt(template_a, template_b, element_idx),
            **kwargs,
        )
        candidates = self.caller([task])[0].candidates
        valid_candidates = []
        for candidate in candidates:
            candidate = candidate["content"]
            if "### result" not in candidate.lower():
                continue
            split_regex = re.compile(r"### result", flags=re.IGNORECASE)
            rslt = re.split(split_regex, candidate)[-1].strip()
            choice = rslt.split("\n")[0]
            if choice.lower() == "b":
                valid_candidates.append((1, None))
            else:
                separation = rslt.split("\n", maxsplit=1)[1].strip()
                if "```json" in separation:
                    separation = separation.split("```json")[1].split("```")[0]
                elif "```" in separation:
                    separation = separation.split("```")[1]
                elif "[" in separation:
                    separation = separation.split("[")[1]

                try:
                    separation = json.loads(separation)
                except ValueError as e:
                    continue

                if isinstance(separation, dict):
                    separation = [separation]
                try:
                    new_elements = Template.load_from_json(
                        separation, template_a.elements[element_idx].value
                    )
                except ValueError as e:
                    continue

                updated_template = deepcopy(template_a)
                updated_template.elements[element_idx : element_idx + 1] = (
                    new_elements.elements
                )
                updated_template.generate_regex()

                if (
                    not updated_template.match(template_a.example_entry)[0]
                    or not updated_template.match(template_b.example_entry)[0]
                ):
                    print(
                        f"Error: created template {updated_template} does not match {template_a.example_entry}"
                    )
                    continue

                valid_candidates.append(
                    (0 if choice.lower() == "a" else 2, new_elements.elements)
                )

        if any(v[0] == 1 for v in valid_candidates):
            return False, None

        breakpoint()

        if not len(valid_candidates):
            # None of the LLM outputs where correct, but we know we need to capture this new variable
            # TODO: use regex finding tool.
            element_values = [template_a.elements[element_idx].value] + [
                template_b.elements[element_idx].value
            ]
            regexp = (
                "(" + "|".join([re.escape(v) for v in element_values]) + ")"
            )
            print(
                f"LLM chose to merge these constants, but could not come up with valid solution. Using {regexp}"
            )
            elt = Element(
                id=template_a.elements[element_idx].id,
                entity=ElementType.VARIABLE,
                value=template_a.elements[element_idx].value,
                regexp=regexp,
            )
            return True, [elt]

        elif 2 * len([v for v in valid_candidates if v[0] == 2]) > len(
            valid_candidates
        ):
            valid_candidates = [v[1] for v in valid_candidates if v[0] == 2]
        else:
            valid_candidates = [v[1] for v in valid_candidates if v[0] == 0]

        return True, random.sample(valid_candidates, 1)[0]

    def _update_other_values(self, elt_id):
        """
        Update other variables to constants in templates that have not yet been updated
        """
        value = self.values[elt_id].values[0]

        for other_elt_id, values in self.values.items():

            other_elt = self.tree.nodes[other_elt_id]
            if not other_elt.is_variable():
                continue

            unique_values = set(values.values)
            if len(unique_values) == 1 and values.values[0].strip() == value:
                self.tree.nodes[other_elt_id].entity = ElementType.CONSTANT
                self.tree.nodes[other_elt_id].value = value

    def _check_other_values(self, elt_id):
        """
        Check if this value shares a unique value with another field that is fixed
        """
        value = self.values[elt_id].values[0]
        if self.tree.nodes[elt_id].fixed:
            return True

        for other_elt_id, values in self.values.items():

            other_elt = self.tree.nodes[other_elt_id]
            if not other_elt.is_variable():
                continue

            unique_values = set(values.values)
            if (
                len(unique_values) == 1
                and values.values[0].strip() == value
                and other_elt.fixed
            ):
                return True

        return False

    def _write_regex_update(self, message, percentage, solution):
        with open(
            f"saved_queries/regex/{self.number_regex_queries}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(message)
        with open(
            f"saved_queries/regex/{self.number_regex_queries}_rslt.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Percentage: {percentage}\nSolution: {solution}")

        self.number_regex_queries += 1

    def _write_variable_update(self, message, percentage, solution):
        with open(
            f"saved_queries/variable/{self.number_varia_queries}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(message)
        with open(
            f"saved_queries/variable/{self.number_varia_queries}_rslt.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"Percentage: {percentage}\nSolution: {solution}")

        self.number_varia_queries += 1

    def _update_regexp(
        self,
        template,
        matches,
        mismatch_entry,
        mismatch_index,
        mismatch_value,
        values,
        model,
        **kwargs,
    ):

        # Ask the LLM if this field should be a variable or a constant
        kwargs["n"] = self.parallel_attempts
        kwargs["temperature"] = 0.5
        matches = matches[: self.few_shot_len]
        while True:
            task = OpenAITask(
                system_prompt=regex_update_system_prompt,
                max_tokens=4096,
                model=model,
                message=generate_regex_update_prompt(
                    template,
                    matches,
                    mismatch_entry,
                    mismatch_value,
                    mismatch_index,
                ),
                **kwargs,
            )

            try:
                candidates = self.caller([task])[0].candidates
                break
            except ValueError:
                if len(matches) < 2:
                    model = "gemini-1.5-flash"
                else:
                    matches = matches[:-1]

        valid_candidates = []
        for candidate in candidates:
            candidate = candidate["content"]
            if "### result" not in candidate.lower():
                continue
            split_regex = re.compile(r"### result", flags=re.IGNORECASE)
            rslt = re.split(split_regex, candidate)[-1].strip()
            if rslt.lower() == "false":
                valid_candidates.append(None)
            else:
                valid_candidates.append(
                    re.sub(
                        "((^['\"`]+)|(['\"`]+$))",
                        "",
                        rslt.replace("```", "").strip(),
                    )
                )

        percentage = len([v for v in valid_candidates if v is None]) / len(
            valid_candidates
        )
        if percentage > 0.5:
            self._write_regex_update(task.message, percentage, False)
            return False

        valid_candidates = sorted(
            [v for v in valid_candidates if v is not None],
            key=lambda x: -len(x),
        )
        for candidate in valid_candidates:
            # Test the regex
            try:
                candidate = Element.normalize_regex(
                    json.loads('["' + candidate + '"]')[0]
                )
            except json.decoder.JSONDecodeError:
                candidate = Element.normalize_regex(candidate)

            try:
                if not re.match(candidate, mismatch_value):
                    continue
            except re.error:
                continue

            if any(
                not re.match("^" + candidate + "$", v)
                for v in values.values[:100]
            ):
                continue

            self._write_regex_update(task.message, percentage, candidate)
            return candidate

        print("Warning - Regex should be extended, but no candidate was valid.")
        if " " in mismatch_value.strip() or any(
            " " in v for v in values.values
        ):
            self._write_regex_update(task.message, percentage, ".*?")
            print("Defaulting to .*?")
            return ".*?"
        else:
            self._write_regex_update(task.message, percentage, "\S+?")
            print("Defaulting to \S+?")
            return "\S+?"

    def parse(self, log_file, model="gemini-1.5-flash", **kwargs) -> None:
        lines = [
            line
            for line in self.load_and_split_log(log_file, self.unit_regex)
            if len(line)
        ]

        for line_idx, line in tqdm(
            enumerate(lines),
            desc="Processing log file",
            unit="lines",
            total=len(lines),
        ):

            if line_idx < self.start_index:
                continue

            line = line.strip()
            if len(line) > 1000:
                continue

            match, candidates = self.tree.match(line)

            if match:
                t_id, matches = candidates[0].template_id, candidates[0].matches
                self.matches.append((line, t_id, matches))
                self.entries_per_template[t_id].append(line)

                variable_checks = set()
                for elt in matches.elements:
                    value, elt_id = elt.value, elt.id
                    if elt.is_variable():
                        self.values[elt_id].append(value)
                        if len(self.values[elt_id].values) == 100:
                            variable_checks.add(elt_id)

                for elt_id in variable_checks:
                    # Check if the variable should be a constant
                    if (
                        len(set(self.values[elt_id].values)) == 1
                        and self.tree.nodes[elt_id].is_variable()
                        and not self._check_other_values(elt_id)
                    ):
                        templates = [
                            self.tree.gen_template(t)
                            for t in self.tree.templates_per_node[elt_id]
                        ][: self.few_shot_len]
                        elt_idx = len(
                            self.tree.node_to_tree[elt_id].get_lineage()
                        )
                        example_matches = [
                            self.entries_per_template[t][: self.few_shot_len]
                            for t in self.tree.templates_per_node[elt_id]
                        ][: self.few_shot_len]
                        if not self.confirm_variable(
                            self.values[elt_id].values[0],
                            templates,
                            elt_idx,
                            example_matches,
                            model,
                            **kwargs,
                        ):
                            print(
                                f"Updating variable {self.tree.nodes[elt_id]}"
                            )
                            self.tree.nodes[elt_id].entity = (
                                ElementType.CONSTANT
                            )
                            self.tree.nodes[elt_id].value = self.values[
                                elt_id
                            ].values[0]

                            self._update_other_values(elt_id)

                continue

            if self.debug_folder:
                self.start_index = line_idx
                with open(
                    os.path.join(
                        self.debug_folder,
                        f"saved_states_{len(self.tree.templates)}.pkl",
                    ),
                    "wb",
                ) as f:
                    dill.dump(self, f)

            # Check if the template exists
            if not candidates[0].suffix.strip():

                t_id = self.tree.add_template(candidates[0].trail, line)
                self.entries_per_template[t_id] = [line]
                for elt in candidates[0].matches.elements:
                    if elt.is_variable():
                        if elt.id not in self.values:
                            self.values[elt.id] = Value(
                                elt.id, model=self.embedding, values=[elt.value]
                            )
                        else:
                            self.values[elt.id].append(elt.value)

                self.matches.append((line, t_id, candidates[0].matches))
                print(
                    f"Entry {line} did not match any template, but matches a branch. Added new template {self.tree.gen_template(t_id)}"
                )
                continue

            # Check for parsing errors
            last_node = (
                self.tree.node_to_tree[candidates[0].trail[-1]]
                if len(candidates[0].trail)
                else self.tree.tree
            )
            mismatch_index = 1 + (
                0 if last_node == self.tree.tree else len(candidates[0].trail)
            )
            added = False
            for branch_id in last_node.branches:
                if (
                    self.tree.nodes[branch_id].is_variable()
                    and not self.tree.nodes[branch_id].fixed
                ):
                    original_regex = self.tree.nodes[branch_id].regexp
                    self.tree.nodes[branch_id].regexp = ".*?"

                    hypothetical_match, hypothetical_candidates = (
                        self.tree.match(
                            candidates[0].suffix, start_node=last_node.node
                        )
                    )

                    self.tree.nodes[branch_id].regexp = original_regex

                    if not hypothetical_match:
                        continue
                    else:

                        print(
                            f"Line {line} mismatched template by a single variable."
                        )
                        template_id = hypothetical_candidates[0].template_id
                        matches = self.entries_per_template[template_id]
                        template = self.tree.gen_template(template_id)
                        mismatch_value = (
                            hypothetical_candidates[0].matches.elements[0].value
                        )

                        new_regex = self._update_regexp(
                            template,
                            matches,
                            line,
                            mismatch_index,
                            mismatch_value,
                            self.values[branch_id],
                            model,
                            **kwargs,
                        )
                        if new_regex:
                            print(
                                f"Updated regex of element {self.tree.nodes[branch_id]}:\n\t Old regex: {self.tree.nodes[branch_id].regexp}\n\t New regex: {new_regex}"
                            )
                            self.tree.nodes[branch_id].regexp = new_regex
                            self.matches.append(
                                (
                                    line,
                                    template_id,
                                    Match(
                                        candidates[0].matches.elements
                                        + hypothetical_candidates[
                                            0
                                        ].matches.elements
                                    ),
                                )
                            )
                            self.entries_per_template[template_id].append(line)
                            for elt in self.matches[-1][-1].elements:
                                if elt.is_variable():
                                    self.values[elt.id].append(elt.value)
                            added = True
                            break

            if added:
                continue

            # Find few-shot examples
            if len(candidates[0].trail):
                examples = self.tree.get_ordered_templates(
                    candidates[0].trail[-1]
                )
            else:
                examples = self.tree.get_ordered_templates()

            if len(examples):
                if self.embedding is not None:
                    current_embed = self.embedding(line)
                    examples_a = sorted(
                        examples,
                        key=lambda x: (
                            x[0],
                            np.linalg.norm(current_embed - (x[-3])),
                        ),
                    )
                    examples_b = sorted(
                        examples,
                        key=lambda x: (
                            np.linalg.norm(current_embed - (x[-3])),
                        ),
                    )

                    for e_idx, (_, _, _, _, full, part) in enumerate(examples):
                        fc, pc = 0, 0
                        for elt in full:
                            if elt in line:
                                fc += 1
                        for elt in part:
                            if elt in line:
                                pc += 1
                        if not len(full):
                            fc = 0
                        else:
                            fc /= len(full)

                        if not len(part):
                            pc = 0
                        else:
                            pc /= len(part)

                        examples[e_idx] = examples[e_idx] + ((fc + pc) / 2,)

                    examples_c = sorted(
                        examples,
                        key=lambda x: x[-1],
                        reverse=True,
                    )

                    ### New metric: for each template, remove variables, and list constants (both full and space or non-alphanumeric separated). Then, count how many of these match the current entry. Take first those that match most full constants, then those that match most partial constants.
                    examples = examples_a[: self.few_shot_len // 2]
                    for e in examples_c:
                        if e[1] not in {i[1] for i in examples}:
                            examples.append(e)
                        if len(examples) == self.few_shot_len:
                            break

                    # distances = [
                    #     np.linalg.norm(current_embed - (e[-1]))
                    #     for e in examples
                    # ]
                    example_matches = [
                        self.tree.gen_template(i[1]).example_entry
                        for i in examples
                    ]
                few_shot = [
                    self.tree.gen_template(e[1])
                    for e in examples[: self.few_shot_len]
                ]
            else:
                few_shot = None

            if len(candidates[0].trail):
                prefix_elements = [
                    deepcopy(self.tree.nodes[e]) for e in candidates[0].trail
                ]
                match_to_value = {
                    m.id: m.value for m in candidates[0].matches.elements
                }
                for elt in prefix_elements:
                    if elt.is_variable():
                        elt.value = match_to_value[elt.id]
                unmatched_suffix = candidates[0].suffix
            else:
                prefix_elements, unmatched_suffix = None, None

            print(f"Generating new template for entry {line}")
            try:
                template, llm_output, llm_input = self.separate(
                    line,
                    examples=few_shot,
                    prefix_elements=prefix_elements,
                    unmatched_suffix=unmatched_suffix,
                    model=model,
                    **kwargs,
                )
            except ValueError as e:
                print(f"Error processing line: {line} - {e}")
                continue

            # # Make sure new entry is not too broad
            # while True:
            #     current_examples = [
            #         self.tree.gen_template(t).example_entry
            #         for t in range(len(self.tree.templates))
            #     ]
            #     matches = [template.match(e) for e in current_examples]
            #     remove_indices = []
            #     if any([m[0] for m in matches]):
            #         start_again = False
            #         for idx, m in enumerate(matches):
            #             if not m[0]:
            #                 continue

            #             mismatch_template = self.tree.gen_template(idx)
            #             entries_2 = list(
            #                 set(self.entries_per_template[idx][:50])
            #             )[:5]
            #             entries_1 = [line.strip()]

            #             decision, candidate_template = self.adjust_template(
            #                 template.format_as_example(
            #                     force_match_with_entry=True,
            #                     prefix=(
            #                         len(prefix_elements)
            #                         if prefix_elements
            #                         else 0
            #                     ),
            #                 ),
            #                 mismatch_template.format_as_example(
            #                     force_match_with_entry=True
            #                 ),
            #                 entries_1,
            #                 entries_2,
            #                 template,
            #                 model,
            #             )

            #             if decision == 3:
            #                 LLM_decision = "Keep both (3)."
            #             elif decision == 2:
            #                 LLM_decision = "Remove the old template (2)."
            #             else:
            #                 LLM_decision = f"Update the new template (1).\nUpdated template: \n\n{candidate_template[0].format_as_example()}"

            #             print(
            #                 f"The new template intersects an old template:\nNew template: {template.format_as_example()}\n\nOld template: {mismatch_template.format_as_example()}\n\nLLM's Decision: {LLM_decision}.\nYour decision (1) update (2) remove (3) keep both (4) debugger:"
            #             )
            #             user_decision = input()
            #             try:
            #                 user_decision = int(user_decision)
            #                 assert user_decision in [1, 2, 3]
            #                 decision = user_decision
            #             except Exception as e:
            #                 print("Entering debugger...")
            #                 breakpoint()

            #             if decision == 3:
            #                 continue
            #             if decision == 2:
            #                 remove_indices.append(idx)
            #             if decision == 1:
            #                 template = candidate_template[0]
            #                 start_again = True
            #                 break

            #         if start_again and len(remove_indices):
            #             breakpoint()
            #         elif start_again:
            #             continue
            #         else:
            #             break
            #     else:
            #         break

            # Check if the new entry is one away from an existing template
            first_new_constants = []
            for idx in range(
                len(prefix_elements) if prefix_elements else 0,
                len(template.elements),
            ):
                if not template.elements[idx].is_variable():
                    first_new_constants.append((template.elements[idx], idx))
                else:
                    break
            added = False
            orig_template = template
            if first_new_constants:
                if len(candidates[0].trail):
                    all_templates = self.tree.get_ordered_templates(
                        candidates[0].trail[-1]
                    )
                else:
                    all_templates = self.tree.get_ordered_templates()

                close_templates = [
                    (self.tree.gen_template(t), t, part, full)
                    for _, t, _, _, full, part in all_templates
                    if not candidates[0].trail
                    or t
                    in self.tree.templates_per_node[candidates[0].trail[-1]]
                ]
                close_templates = [
                    (template, t_idx, part, full)
                    for template, t_idx, part, full in close_templates
                    if not template.elements[mismatch_index - 1].is_variable()
                    and not template.elements[mismatch_index - 1].fixed
                ]

                for e_idx, (_, _, full, part) in enumerate(close_templates):
                    fc, pc = 0, 0
                    for elt in full:
                        if elt in line:
                            fc += 1
                    for elt in part:
                        if elt in line:
                            pc += 1
                    if not len(full):
                        fc = 0
                    else:
                        fc /= len(full)

                    if not len(part):
                        pc = 0
                    else:
                        pc /= len(part)

                    close_templates[e_idx] = close_templates[e_idx] + (
                        (fc + pc) / 2,
                    )

                # Order close template to look at closest ones first
                close_templates = sorted(
                    close_templates,
                    key=lambda x: x[-1],
                    reverse=True,
                )
                suffix = (
                    unmatched_suffix.strip()
                    if unmatched_suffix
                    else line.strip()
                )
                template = deepcopy(template)
                for elt_idx, (elt, orig_idx) in enumerate(first_new_constants):
                    try:
                        suffix = suffix[
                            len(elt.value) + elt.trailing_whitespace :
                        ]
                    except IndexError:
                        suffix = ""

                    if elt_idx > 0:
                        template.elements[orig_idx - elt_idx].merge(
                            template.elements[orig_idx - elt_idx + 1]
                        )
                        template.elements.pop(orig_idx - elt_idx + 1)

                    for (
                        close_template,
                        close_template_idx,
                        _,
                        _,
                        _,
                    ) in close_templates:
                        tail = close_template.elements[mismatch_index:]
                        if (not len(tail) and not suffix) or Template(
                            tail
                        ).match(suffix.strip())[0]:
                            print(
                                f"Entry {line} is one variable away from template {close_template}"
                            )
                            template_a, template_b = close_template, template
                            element_idx = mismatch_index - 1
                            decision, new_elements = self.c2a(
                                template_a,
                                template_b,
                                element_idx,
                                model,
                                **kwargs,
                            )
                            if decision:
                                print(f"Updating template {close_template}")
                                _, added, t_id = (
                                    self.tree.splice_elements(
                                        close_template.elements[
                                            mismatch_index - 1
                                        ].id,
                                        new_elements,
                                    ),
                                    True,
                                    close_template_idx,
                                )
                                elements = self.tree.templates[t_id]
                                print(
                                    f"Updated to {self.tree.gen_template(t_id)}"
                                )
                                breakpoint()
                                break

            if not added:
                template = orig_template
                t_id = self.tree.add_template(template, line)
                elements = self.tree.templates[t_id]

            match, entries = self.tree.match(line)

            correct_entry = None
            for t in entries:
                if t.template_id == t_id:
                    correct_entry = t
                    break

            if correct_entry is None:
                # Change all new regexes to non greedy versions
                print("Expanding regex")
                for position, e_idx in enumerate(elements):
                    if (
                        self.tree.nodes[e_idx].is_variable()
                        and len(self.tree.templates_per_node[e_idx]) == 1
                        and position != len(elements) - 1
                        and any(
                            e.is_variable() for e in elements[position + 1 :]
                        )
                    ):
                        self.tree.nodes[e_idx].regexp = re.sub(
                            r"(?<!\\)\*(?!\?)",
                            "*?",
                            self.tree.nodes[e_idx].regexp,
                        )
                        self.tree.nodes[e_idx].regexp = re.sub(
                            r"(?<!\\)\+(?!\?)",
                            "+?",
                            self.tree.nodes[e_idx].regexp,
                        )

                # Try again
                match, entries = self.tree.match(line)
                correct_entry = None
                for t in entries:
                    if t.template_id == t_id:
                        correct_entry = t
                        break

                if correct_entry is None:
                    continue

            if t_id not in self.entries_per_template:
                self.entries_per_template[t_id] = []
            self.entries_per_template[t_id].append(line)

            for elt in correct_entry.matches.elements:
                if elt.is_variable():
                    if elt.id not in self.values:
                        self.values[elt.id] = Value(
                            elt.id, model=self.embedding, values=[elt.value]
                        )
                    else:
                        self.values[elt.id].append(elt.value)
            self.matches.append((line, t_id, correct_entry.matches))

            if not added:
                self.llm_outputs.append(llm_output)
                self.llm_inputs.append(llm_input)
                print(
                    f"New template #{len(self.tree.templates)}: \n\n{template}"
                )

            # if len(remove_indices):
            #     for idx in remove_indices:
            #         self.tree.remove_template(idx)

    def type(self, model="gemini-1.5-flash", **kwargs) -> None:
        # Typing
        orig_type_set = set(self.types)
        self.typing_prompts = {}
        # First pass
        for key in self.values:
            if (
                self.tree.nodes[key].is_variable()
                and self.tree.nodes[key].type is None
            ):
                print(f"Typing node #{key} ({self.tree.nodes[key].value})")
                _, prompt, sys_prompt = self.typing(key, model=model, **kwargs)
                self.typing_prompts[key] = (prompt, sys_prompt)

        # Cluster none types

        # Find sub-templates in non-standard types

        # Use TAG to build final parse graph.

    def process(self, log_file, model="gemini-1.5-flash", **kwargs) -> None:
        self.parse(log_file, model=model, **kwargs)
        self.type(model=model, **kwargs)
