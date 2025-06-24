import json
import random
import re
from copy import deepcopy

from ...classes import Element, Template
from ...genai_api.api import LLMTask
from ...utils.json import parse_json
from ...utils.logging import get_logger
from ...utils.prompts.syntax.regex_generation import (
    gen_prompt,
    get_assistant_msg_prefix,
)
from .heuristics import Heuristic


class GenerateRegex(Heuristic):

    def __init__(
        self,
        tree,
        caller,
        model="gemini-2.5-flash",
        parallel_attempts=4,
        temperature=0.5,
        top_p=0.8,
        path="./saved_queries/",
        few_shot_length=4,
        values=None,  # List of values for each element
        entries_per_template=None,
    ) -> None:
        super().__init__(
            tree,
            "GenerateRegex",
            caller,
            model,
            parallel_attempts,
            temperature,
            top_p,
            path,
            few_shot_length,
            values,
            entries_per_template,
        )

        self.cache = {}

    def _parse_answer(self, candidate, entries, orig_template, force=False):
        """
        Parse the answer from the model and run heuristics on it.
        """

        # Syntax check: Find the explanation and json response
        explanation_regex = re.compile(
            r"### Explanation ###\s+(.*?)\s+###\s+mapping",
            re.DOTALL | re.IGNORECASE,
        )
        explanation_match = explanation_regex.search(candidate)

        if not explanation_match:
            raise ValueError(
                "Invalid response from the model - no explanation or the explanation is properly delimited."
            )
        demo = explanation_match.group(1).strip()

        json_body_matches = list(
            re.finditer(r"```json\n(.*?)```", candidate, re.DOTALL)
        )
        if not json_body_matches:
            raise ValueError("Invalid response from the model - no JSON block.")

        try:
            last_match = json_body_matches[-1]
            json_content = last_match.group(1)
        except:
            raise ValueError(
                "Invalid response from the model - invalid JSON block."
            )

        # Update the template using the regexes in the response
        candidate_template = deepcopy(orig_template)
        relative_id_to_element = {}
        for element in candidate_template.elements:
            if element.is_variable():
                relative_id_to_element[len(relative_id_to_element)] = element

        # Check if the response can be parsed as JSON
        try:
            mapping = json.loads(json_content)
        except json.JSONDecodeError:
            try:
                mapping = json.loads(json_content.replace("\\", "\\\\"))
            except json.JSONDecodeError:
                mapping = parse_json(
                    json_content, self.caller, response_schema=None
                )

        try:
            mapping = {int(k): v for k, v in mapping.items()}
        except:
            breakpoint()

        # Make sure all keys are accounted for
        missing_keys = {
            k
            for k, elt in relative_id_to_element.items()
            if (
                k not in mapping
                or "regex" not in mapping[k]
                or not mapping[k]["regex"].strip()
            )
            and not elt.regexp
        }
        if missing_keys:
            raise ValueError(
                f"Invalid response from the model - the response is missing the regexes for the following variable element ids: {missing_keys}"
            )

        # Make sure each regex matches the value of the element
        incorrect_regexes = []
        for k, v in mapping.items():
            if k in relative_id_to_element and str(
                relative_id_to_element[k].id
            ).endswith("_"):
                try:
                    compiled_regex = re.compile(
                        Element.normalize_regex(v["regex"])
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid response from the model - the regex for variable element id {k} is invalid: {str(e)}"
                    )
                if not compiled_regex.fullmatch(v["value"]):
                    incorrect_regexes.append(k)

        if incorrect_regexes and not force:
            raise ValueError(
                f"Invalid response from the model - the regexes for the following variable element ids do not match the values: {incorrect_regexes}"
            )

        # Load up the regular expressions
        for k, v in mapping.items():
            if k in relative_id_to_element and str(
                relative_id_to_element[k].id
            ).endswith("_"):
                relative_id_to_element[k].regexp = Element.normalize_regex(
                    v["regex"]
                )

        # Check how many lines are matched by the template
        for element in candidate_template.elements:
            element.erase_regex_cache()

        candidate_template.generate_regex()
        rtn_values = []
        matched_lines = [
            i
            for i, entry in enumerate(entries)
            if candidate_template.match(entry.strip())[0]
        ]
        orig_matched_lines_length = len(matched_lines)

        if orig_matched_lines_length:
            rtn_values.append((candidate_template, matched_lines, demo))

        if orig_matched_lines_length == len(entries):
            return rtn_values

        # If the template does not match all lines, try replacing the regex of each element INDEPENDENTLY until it matches
        # The intuition here is that changing one regex is not changing the model output much, and hopefully will not overcapture.
        # This is in contrast with the next technique that changes all the regexes needed until it matches, which is more likely to overcapture, and thus only done as a last resort.
        candidate_template = deepcopy(candidate_template)
        for _, elt in enumerate(candidate_template.elements):
            if elt.is_variable() and "_" in str(elt.id):
                original_regex = elt.regexp
                if not force:
                    elt.regexp = r"\S+?"
                else:
                    elt.regexp = r".*?" if " " in elt.value else r"\S+?"
                elt.erase_regex_cache()
                candidate_template.generate_regex()
                if self._match_examples(candidate_template, entries):
                    rtn_values.append(
                        (
                            candidate_template,
                            [i for i, _ in enumerate(entries)],
                            demo,
                        )
                    )
                    return rtn_values
                elt.regexp = original_regex
                elt.erase_regex_cache()

        if not force and len(rtn_values) > 0:
            return rtn_values
        elif not force:
            raise ValueError(
                "Invalid response from the model - The regular expressions don't match the values of the fields."
            )

        # Hail mary: replace the regex of each element in turn if necessary
        # Find longest matching prefix
        last_elt_idx = 0
        for elt_idx in reversed(range(0, len(candidate_template.elements))):
            last_elt_idx = elt_idx
            prefix_template = Template(candidate_template.elements[:elt_idx])
            prefix_template.generate_regex(
                add_end=elt_idx == len(candidate_template.elements) - 1
            )
            try:
                if any(
                    re.match(prefix_template.regex, entry.strip())
                    for entry in entries
                ):
                    break
            except re.error:
                continue

        # Now go through each element and try to replace the regex
        for new_elt_idx in range(
            last_elt_idx, len(candidate_template.elements)
        ):
            prefix_template = Template(
                candidate_template.elements[:new_elt_idx]
            )
            prefix_template.generate_regex(
                add_end=new_elt_idx == len(candidate_template.elements) - 1
            )

            try:
                if any(
                    re.match(prefix_template.regex, entry.strip())
                    for entry in entries
                ):
                    continue
            except re.error:
                pass

            # Get last variable element and replace its regex
            last_var_elt_idx = new_elt_idx - 1
            while not candidate_template.elements[
                last_var_elt_idx
            ].is_variable():
                last_var_elt_idx -= 1

            # If it's an existing element, stop
            if "_" not in str(candidate_template.elements[last_var_elt_idx].id):
                break

            # Change the regex
            elt = candidate_template.elements[last_var_elt_idx]
            candidate_template.elements[last_var_elt_idx].regexp = (
                r".*?" if " " in elt.value else r"\S+?"
            )

            # Recompute the regex
            prefix_template = Template(
                candidate_template.elements[:new_elt_idx]
            )
            for element in candidate_template.elements[new_elt_idx:]:
                element.erase_regex_cache()
            prefix_template.generate_regex(
                add_end=new_elt_idx == len(candidate_template.elements) - 1
            )
            if any(
                re.match(prefix_template.regex, entry.strip())
                for entry in entries
            ):
                continue

            break

        # Final test
        matched_lines = [
            i
            for i, entry in enumerate(entries)
            if candidate_template.match(entry.strip())[0]
        ]
        if len(matched_lines) > orig_matched_lines_length:
            rtn_values.append(
                (
                    candidate_template,
                    matched_lines,
                    demo,
                )
            )

        if rtn_values:
            return rtn_values
        else:
            raise ValueError(
                "Invalid response from the model - The regular expressions don't match the values of the fields."
            )

    def generate_example(self, template_id):
        """
        Generate explanation and prompt for the given template id
        """
        template = self.tree.gen_template(template_id)
        entries = random.sample(
            self.entries_per_template[template_id],
            k=min(5, len(self.entries_per_template[template_id])),
        )
        explanation = get_assistant_msg_prefix(entries, template)["content"]
        prompt = gen_prompt(None, entries, template, force_match=True)[0][-1][
            "content"
        ]
        result = template.get_regex_mapping(
            force_match_with_entry=True,
            relative_ids=True,
            entry=entries[0],
        )
        return result, explanation, prompt

    def generate_examples(self, example_template_idx, **kwargs):
        """
        Generate examples for the given template indices
        """
        examples = []
        for idx in example_template_idx:
            examples.append(self.generate_example(idx))
        return examples

    def self_correct(self, llm_response, entries, candidate_template, task):
        # Correction: if any response cannot be parsed, ask the model to fix itself
        to_be_fixed = []
        tasks = []
        task.n = 1
        for idx, group_of_responses in enumerate(llm_response):
            for idx_2, candidate_resp in enumerate(
                group_of_responses.candidates
            ):
                candidate = candidate_resp
                try:
                    self._parse_answer(
                        candidate, entries, candidate_template, force=False
                    )
                except ValueError as e:
                    to_be_fixed.append((idx, idx_2, len(tasks), str(e)))
                tasks.append(deepcopy(task))

        for idx, idx_2, task_idx, err in to_be_fixed:
            get_logger().info(
                "Trying self correction for response #%s: %s", idx, err
            )
            orig_task = tasks[task_idx]
            orig_task.update_conversation(
                llm_response[idx].candidates[idx_2],
                f"Your response is invalid: we encountered the following error: {err}.\nPlease fix your response and try again. Please return a full explanation and response.",
            )

        if to_be_fixed:
            get_logger().info("Fixing invalid regex candidates...")
            corrected_candidates = self.caller(
                [tasks[i] for _, _, i, _ in to_be_fixed]
            )
            for idx, idx_2, _, _ in to_be_fixed:
                llm_response[idx].candidates[idx_2] = corrected_candidates.pop(
                    0
                ).candidates[0]

        return llm_response, tasks

    def _query(
        self,
        example_template_idx,
        entries,
        candidate_template,
        force=False,
        **kwargs,
    ):
        kwargs = self._prepare_kwargs(**kwargs)
        examples = self.generate_examples(example_template_idx, **kwargs)
        history, system = gen_prompt(examples, entries, candidate_template)
        if "model" not in kwargs:
            kwargs["model"] = self.model
        task = LLMTask(
            system_prompt=system,
            history=history,
            **kwargs,
        )

        response = self.caller([task])
        response, tasks = self.self_correct(
            response, entries, candidate_template, task
        )
        valid_candidates = []

        for idx, model_response_obj in enumerate(response):
            for candidate_resp in model_response_obj.candidates:
                candidate = candidate_resp
                try:
                    rtn_values = self._parse_answer(
                        candidate, entries, candidate_template, force=force
                    )
                except ValueError as e:
                    get_logger().warning(
                        "Invalid response from the model: %s", str(e)
                    )
                    continue
                for candidate_template, matched_lines, demo in rtn_values:
                    valid_candidates.append(
                        (candidate_template, matched_lines, demo, idx)
                    )

        # Count the number of overlapping matches for each candidate
        candidates = []
        for candidate, matched_lines, demo, idx in valid_candidates:
            match, list_of_matches = self.match_previous_entries(candidate)
            candidates.append(
                (
                    candidate,
                    tuple(matched_lines),
                    (
                        tuple(list_of_matches)
                        if match and list_of_matches
                        else None
                    ),
                    demo,
                    idx,
                )
            )

        if not candidates:
            get_logger().error(
                "No valid candidates found for entries %s",
                entries,
            )
            return None, [], None

        # Filter candidates based on the ones with the least number of previous matches
        min_previous_matches = min(
            len(list_of_matches) if list_of_matches else 0
            for _, _, _, list_of_matches, _ in candidates
        )
        candidates = [
            candidate
            for candidate in candidates
            if (len(candidate[3]) if candidate[3] else 0)
            == min_previous_matches
        ]

        # Filter candidates based on the ones with the most number of matches among the current entries
        max_entry_match = max(
            len(matched_lines) for _, matched_lines, _, _, _ in candidates
        )
        candidates = [
            candidate
            for candidate in candidates
            if len(candidate[1]) == max_entry_match
        ]

        (
            template,
            matched_entries,
            matched_others,
            demo,
            original_candidate_index,
        ) = candidates[0]

        self._write(
            len(self.tree.templates),
            self._print_history(
                tasks[original_candidate_index].history,
                tasks[original_candidate_index].message,
            ),
            "".join(
                [
                    f"Response #{c_idx+1}:\n\n" + c + "\n\n##########\n\n"
                    for c_idx, c in enumerate(response[0].candidates)
                ]
            ),
            template.format_as_example(regex=True),
        )

        return (
            template,
            matched_entries,
            matched_others,
        )

    def run(
        self,
        example_template_idx,
        entries,
        candidate_template,
        force=False,
        **kwargs,
    ):

        return self._query(
            example_template_idx,
            entries,
            candidate_template,
            force=force,
            **kwargs,
        )

    def ingest_fewshot(self, templates):
        """Ingest fewshot data into cache"""

        entries_per_template = [
            self.entries_per_template[t] for t in range(len(templates))
        ]

        for template, entries in zip(templates, entries_per_template):
            explanation = get_assistant_msg_prefix(entries, template)["content"]
            self.cache[template.id] = (
                template.format_as_example(
                    regex=True, force_match_with_entry=True, entry=entries[0]
                ),
                explanation,
                gen_prompt(None, entries, template)[0][-1]["content"],
            )
