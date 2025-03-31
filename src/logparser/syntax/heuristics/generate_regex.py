import random
import re
from collections import Counter
from copy import deepcopy

from ...tools.api import OpenAITask
from ...tools.classes import Template
from ...tools.logging import get_logger
from ...tools.prompts.regex_generation import (
    gen_prompt,
    get_assistant_msg_prefix,
    response_schema,
)
from .heuristics import Heuristic


class GenerateRegex(Heuristic):

    def __init__(
        self,
        tree,
        caller,
        model="gemini-1.5-flash",
        parallel_attempts=4,
        temperature=0.5,
        top_p=0.8,
        path="./saved_queries/",
        few_shot_length=4,
        values=None,  # List of values for each element
        entries_per_template=None,
        naive_distance=None,
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
            naive_distance,
        )

        self.cache = {}

    def _parse_answer(self, candidate, entries, orig_template, force=False):
        """
        Parse the answer from the model and run heuristics on it.
        """

        # Syntax check: Find tthe explanation and json response
        explanation_regex = re.compile(
            "### Explanation ###\s+(.*?)\s+###\s+(final)",
            re.DOTALL | re.IGNORECASE,
        )
        explanation_match = explanation_regex.search(candidate)

        if not explanation_match:
            raise ValueError(
                "Invalid response from the model - no explanation or the explanation is properly delimited."
            )
        demo = explanation_match.group(1).strip()

        json_body_matches = re.finditer(
            r"```json\n(.*?)```", candidate, re.DOTALL
        )
        if not json_body_matches:
            raise ValueError("Invalid response from the model - no JSON block.")

        try:
            last_match = list(json_body_matches)[-1]
            json_content = last_match.group(1)
        except:
            raise ValueError(
                "Invalid response from the model - invalid JSON block."
            )

        # Load the template
        candidate_template, errors = None, []
        for entry in entries:
            try:
                candidate_template = Template.load_array_from_response(
                    json_content,
                    entry,
                    caller=self.caller,
                    response_schema=response_schema,
                )
                break
            except ValueError as e:
                errors.append(e)
                continue
        if candidate_template is None:
            raise ValueError(
                f"Invalid response from the model - {str(errors[0])}"
            )

        # Check the template is consistent with the original template
        if len(candidate_template.elements) != len(orig_template.elements):
            raise ValueError(
                "Invalid response from the model - The new template does not have the same number of elements are the original one."
            )

        for elt_idx, (orig_elt, new_elt) in enumerate(
            zip(orig_template.elements, candidate_template.elements)
        ):
            if "_" not in str(orig_elt.id):
                candidate_template.elements[elt_idx] = orig_elt
                continue

            if (
                not orig_elt.is_variable()
                and new_elt.is_variable()
                and orig_elt.value.strip() == new_elt.value.strip()
            ):
                candidate_template.elements[elt_idx] = orig_elt
                continue

            if orig_elt.is_variable() and not new_elt.is_variable():
                raise ValueError(
                    f"Invalid response from the model - Element at index #{elt_idx} in the new template is not a variable, but was in the original template."
                )
            if not orig_elt.is_variable() and new_elt.is_variable():
                raise ValueError(
                    f"Invalid response from the model - Element at index #{elt_idx} in the new template is a variable, but was not in the original template."
                )
            if orig_elt.value.strip() != new_elt.value.strip():
                raise ValueError(
                    f'Invalid response from the model - Element at index #{elt_idx} in the new template has value "{new_elt.value.strip()}", whereas the original element had value "{orig_elt.value.strip()}".'
                )

        # Check how many lines are matched by the template
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

        if not force and len(rtn_values) > 0:
            return rtn_values
        elif not force:
            raise ValueError(
                "Invalid response from the model - The new template does not match any of the entries entries."
            )

        # Hail mary: replace the regex of each element in turn if necessary
        # Find longest matching prefix
        for elt_idx in reversed(range(0, len(candidate_template.elements))):
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
        for new_elt_idx in range(elt_idx, len(candidate_template.elements)):
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
                "Invalid response from the model - The new template does not match any of the entries entries."
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
        prompt = gen_prompt(None, entries, template)[0][-1]["content"]
        result = template.format_as_example(
            force_match_with_entry=True,
            regex=True,
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
        for idx, group_of_responses in enumerate(llm_response):
            for idx_2, candidate_resp in enumerate(
                group_of_responses.candidates
            ):
                candidate = candidate_resp["content"]
                try:
                    self._parse_answer(
                        candidate, entries, candidate_template, force=False
                    )
                except ValueError as e:
                    to_be_fixed.append((idx, str(e)))

        new_tasks = []
        for idx, err in to_be_fixed:
            orig_task = deepcopy(task)
            orig_task.update_conversation(
                llm_response[idx].candidates[0]["content"],
                f"Your response is invalid: we encountered the following error: {err}.\nPlease fix your response and try again. Please return a full explanation and response.",
            )
            new_tasks.append(orig_task)

        if new_tasks:
            get_logger().info("Fixing invalid regex candidates...")
            corrected_candidates = self.caller(new_tasks)
            for idx, _ in to_be_fixed:
                llm_response[idx] = corrected_candidates.pop(0)

        return llm_response

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
        task = OpenAITask(
            system_prompt=system,
            max_tokens=8192,
            history=history,
            **kwargs,
        )

        response = self.caller([task])
        response = self.self_correct(
            response, entries, candidate_template, task
        )
        valid_candidates = []

        for model_response_obj in response:
            for candidate_resp in model_response_obj.candidates:
                candidate = candidate_resp["content"]
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
                        (candidate_template, matched_lines, demo)
                    )

        # Count the number of overlapping matches for each candidate
        candidates = []
        for candidate, matched_lines, demo in valid_candidates:
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
            for _, _, _, list_of_matches in candidates
        )
        candidates = [
            candidate
            for candidate in candidates
            if (len(candidate[3]) if candidate[3] else 0)
            == min_previous_matches
        ]

        # Filter candidates based on the ones with the most number of matches among the current entries
        max_entry_match = max(
            len(matched_lines) for _, matched_lines, _, _ in candidates
        )
        candidates = [
            candidate
            for candidate in candidates
            if len(candidate[1]) == max_entry_match
        ]

        template, matched_entries, matched_others, demo = candidates[0]

        self._write(
            len(self.tree.templates),
            self._print_history(task.history),
            "".join(
                [
                    f"Response #{c_idx+1}:\n\n"
                    + c["content"]
                    + "\n\n##########\n\n"
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
