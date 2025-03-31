import json
import os
import random
import re
import time
from collections import Counter
from copy import deepcopy

from ...classes import Template
from ...genai_api.api import Caller, LLMTask
from ...utils.logging import get_logger
from ...utils.prompts.syntax.generate_explanation import gen_prompt


class GenerateExplanation:

    def __init__(
        self,
        tree,
        caller,
        separation_heuristic,
        cluster_heuristic,
        model="gemini-2.5-flash",
        temperature=1.0,
        top_p=1.0,
        path="./saved_queries/",
        values=None,  # List of values for each element
        entries_per_template=None,
    ) -> None:
        self.tree = tree

        # self.model = model
        # self.caller = caller
        self.model = "gemini-2.5-pro"
        self.caller = Caller(parallelism=32, distribute_parallel_requests=True)
        self.temperature = temperature
        self.top_p = top_p
        self.values = values
        self.entries_per_template = entries_per_template
        self.save_path = path
        self.separation_heuristic = separation_heuristic
        self.cluster_heuristic = cluster_heuristic

        self.save_path = os.path.join(self.save_path, "GenerateExplanation")
        os.makedirs(self.save_path, exist_ok=True)

    def _write(self, name, input, ouptut):
        bp = self.save_path
        name = str(name)
        inputs_path = os.path.join(bp, "inputs", name)
        outputs_path = os.path.join(bp, "outputs", name)
        os.makedirs(os.path.join(bp, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(bp, "outputs"), exist_ok=True)

        with open(inputs_path, "w") as f:
            f.write(input)
        with open(outputs_path, "w") as f:
            f.write(ouptut)

    def _print_history(self, hist, message):
        rtn = ""
        for elt in hist:
            if elt["content"]:
                rtn += f"*** {elt['role']} ***\n{elt['content']}\n\n"
        if message:
            rtn += f"*** User ***\n{message}\n\n"
        return rtn

    def _parse_answer(
        self,
        response_text,
        entries,
        template,
        raise_errors=True,
    ):

        def error(msg, new_task=None):
            if raise_errors:
                raise ValueError(msg)
            else:
                return None, None, msg, new_task

        # Extract fields from response
        explanation_regex = r"<explanation>\s+(.*?)\s+</explanation>\s+"
        json_body_matches = re.finditer(
            r"```json\n(.*?)```", response_text, re.DOTALL
        )
        if not json_body_matches:
            return error("Invalid response from the model - no JSON block.")
        try:
            last_match = list(json_body_matches)[-1]
            json_content = last_match.group(1)
        except:
            return error(
                "Invalid response from the model - invalid JSON block."
            )

        explanation_match = re.search(
            explanation_regex, response_text, re.DOTALL
        )
        if not explanation_match:
            return error(
                "Invalid response from the model - no explanation or the explanation is properly delimited."
            )
        demo = explanation_match.group(1).strip()

        template_raw, candidate_template = (
            json_content,
            None,
        )
        try:
            candidate_template = Template.load_array_from_response(
                template_raw,
                entries[0],
                full_entry=entries[0],
                caller=self.caller,
                model=self.model,
            )
        except Exception as e:
            return error(f"Invalid response from the model - {str(e)}")

        # Make sure we have the same template as the one provided
        if len(template.elements) != len(candidate_template.elements):
            return error(
                "Invalid response from the model - the template does not match the expected template."
            )
        if not all(
            e1.is_variable() == e2.is_variable()
            for e1, e2 in zip(template.elements, candidate_template.elements)
        ):
            return error(
                "Invalid response from the model - the template does not match the expected template."
            )

        if not all(
            e1.regexp == e2.regexp
            for e1, e2 in zip(template.elements, candidate_template.elements)
            if e1.is_variable() and e2.is_variable()
        ):
            offending_elements = [
                e1.id
                for e1, e2 in zip(
                    template.elements, candidate_template.elements
                )
                if e1.is_variable()
                and e2.is_variable()
                and e1.regexp != e2.regexp
            ]
            return error(
                f"Invalid response from the model - some elements in the template have different regexes than the original template."
            )

        # Make sure placeholders are correct
        placeholder_section = re.split(
            "placeholders", demo, flags=re.IGNORECASE
        )[-1]
        placeholder_section = re.split(
            "Key Value Pairs", placeholder_section, flags=re.IGNORECASE
        )[0]

        missing_constants = []
        for elt in json.loads(
            template.format_as_example(
                force_match_with_entry=True, entry=entries[0]
            )
        ):
            if (
                not elt["is_variable"]
                and elt["value"].strip() not in placeholder_section
            ):
                missing_constants.append(elt["value"].strip())
        if missing_constants:
            return error(
                f"Invalid response from the model - the following constants are missing in the placeholder string: {', '.join(missing_constants)}",
            )

        demo_placeholders = set(
            [
                v.lower().replace("<", "").replace(">", "").strip()
                for v in re.findall(r"<[a-zA-Z0-9\_\- ]+>", placeholder_section)
            ]
        )
        demo_placeholders = {
            p
            for p in demo_placeholders
            if all(p not in e.lower() for e in entries)
        }
        template_placeholders = set()
        for elt in candidate_template.elements:
            placeholder = elt.placeholder
            if placeholder:
                template_placeholders.add(
                    placeholder.lower()
                    .replace("<", "")
                    .replace(">", "")
                    .strip()
                )
        missing_placeholders = demo_placeholders - template_placeholders
        if missing_placeholders:
            return error(
                f"Invalid response from the model - the following placeholders are in the explanation but not in the template: {', '.join(missing_placeholders)}",
            )

        # If we get here, the template is valid and can be returned
        if not raise_errors:
            return demo, candidate_template, None, None
        else:
            return demo, candidate_template

    def self_correct(self, tasks, llm_tasks, response):

        def create_new_task(idx, content, message):
            new_task = llm_tasks[idx]
            new_task.update_conversation(content, message)
            return new_task

        error_messages = {
            "parsing_error": "Your response is invalid: we encountered the following error: {err}.\nPlease fix your response and try again. Remember, the template you return must be the exact same as the original, with the addition of the relevant placeholders. Please return a full explanation and response.",
        }

        # Correction: if any response contains "<MESSAGE>", ask the model to fix itself
        to_be_fixed = []
        for task_idx, (
            (template_id, entries, template),
            response_text,
        ) in enumerate(zip(tasks, response)):
            _, _, err, new_task = self._parse_answer(
                response_text, entries, template, raise_errors=False
            )
            if err is not None and new_task is not None:
                to_be_fixed.append((task_idx, new_task, err))
            elif err is not None:
                to_be_fixed.append(
                    (
                        task_idx,
                        create_new_task(
                            task_idx,
                            response_text,
                            error_messages["parsing_error"].format(err=err),
                        ),
                        err,
                    )
                )

        if not to_be_fixed:
            return response, llm_tasks

        for task_idx, new_task, msg in to_be_fixed:
            get_logger().info(
                "Trying self correction for response #%s: %s", task_idx, msg
            )

        new_tasks = [task for _, task, _ in to_be_fixed]
        if new_tasks:
            corrected_candidates = self.caller(new_tasks)
            for fix_index, (task_idx, _, _) in enumerate(to_be_fixed):
                response[task_idx] = corrected_candidates[fix_index].candidates[
                    0
                ]
                llm_tasks[task_idx] = new_tasks[fix_index]

        return response, llm_tasks

    def _query(self, tasks):
        llm_tasks = []
        for task in tasks:
            _, entries, template = task
            history, system = gen_prompt(entries, template)
            llm_task = LLMTask(
                system_prompt=system,
                history=history,
                n=1,
                model=self.model,
                temperature=self.temperature,
                thinking_budget=128,
                top_p=self.top_p,
            )
            llm_tasks.append(llm_task)

        get_logger().debug(
            "\t\tGenerating explanations at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        response = self.caller(llm_tasks)
        response = [val for resp in response for val in resp.candidates]

        for retry_idx in range(5):
            get_logger().debug(
                "\t\tSelf correction attempt #%s at time %s",
                retry_idx + 1,
                time.strftime("%H:%M:%S", time.localtime()),
            )
            response, llm_tasks = self.self_correct(tasks, llm_tasks, response)

        # Parse the responses
        results = []
        for task_idx, (
            (template_id, entries, template),
            response_text,
        ) in enumerate(zip(tasks, response)):
            self._write(
                template_id,
                self._print_history(
                    llm_tasks[task_idx].history, llm_tasks[task_idx].message
                ),
                response_text,
            )
            try:
                explanation, filled_template = self._parse_answer(
                    response_text, entries, template
                )
                for elt_idx, elt in enumerate(filled_template.elements):
                    elt.id = f"{template_id}_{elt_idx}"
                results.append(
                    (template_id, explanation, filled_template, entries)
                )
            except ValueError:
                get_logger().error(
                    "Failed to parse explanation response for template #%s",
                    template_id,
                )
                continue

        return results

    def run(self, template_ids=None):
        # Assume the file has already been parsed using the current tree
        if not self.entries_per_template:
            raise ValueError(
                "The log file must have been parsed using the current template to be able to generate explanations."
            )

        tasks = []
        if not template_ids:
            template_ids = range(len(self.tree.templates))
        for template_id in template_ids:
            if not self.tree.templates[template_id]:
                continue
            if template_id not in self.entries_per_template:
                get_logger().warning(
                    "Template #%s not found in entries_per_template. Skipping.",
                    template_id,
                )
                continue
            if not self.entries_per_template[template_id]:
                get_logger().warning(
                    "No entries found for template #%s. Skipping.",
                    template_id,
                )
                continue

            entries = self.entries_per_template[template_id]
            entries = random.sample(entries, k=min(len(entries), 5))
            template = self.tree.gen_template(template_id)

            tasks.append((template_id, entries, template))

        results = self._query(tasks)

        for template_id, explanation, filled_template, entries in results:
            self.separation_heuristic.save_to_cache(
                template_id,
                entries,
                None,
                explanation,
                "",
                filled_template.format_as_example(
                    force_match_with_entry=True,
                    entry=entries[0],
                    regex=True,
                    placeholder=True,
                ),
            )
            self.cluster_heuristic.save_to_cache(template_id, entries, "", "")
