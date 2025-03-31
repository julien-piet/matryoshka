import re
import time
from collections import Counter
from copy import deepcopy

from ...tools.api import OpenAITask
from ...tools.classes import Template
from ...tools.logging import get_logger
from ...tools.prompts.separation import (
    explanation_format,
    fix_prompt_overcapture,
    fix_prompt_undercapture,
    gen_prompt,
    response_schema,
)
from .heuristics import Heuristic


class GenerateSeparation(Heuristic):

    def __init__(
        self,
        tree,
        caller,
        model="gemini-1.5-flash",
        parallel_attempts=5,
        temperature=1.0,
        top_p=1.0,
        path="./saved_queries/",
        few_shot_length=4,
        values=None,  # List of values for each element
        entries_per_template=None,
        naive_distance=None,
    ) -> None:
        super().__init__(
            tree,
            "GenerateSeparation",
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

    def generate_examples(self, example_template_idx, **kwargs):
        examples = []
        for template_idx in example_template_idx:
            entries, suffix, demo, desc, result = self.cache[template_idx]
            examples.append((result, entries, suffix, demo, desc))
        return examples

    def _parse_answer(
        self,
        candidate,
        entries,
        suffixes,
        prefix_template,
        task,
        blocklist=None,
        force=False,
        raise_errors=True,
    ):

        def error(msg, new_task=None):
            if raise_errors:
                raise ValueError(msg)
            else:
                return None, msg, new_task

        def fix_prompt(orig_template, new_user_message):
            if raise_errors:
                return None

            fixed_template = orig_template.format_as_example(regex=False)
            last_match = list(
                re.finditer(r"```json\n(.*?)```", candidate, re.DOTALL)
            )[-1]
            response = (
                candidate[: last_match.start()]
                + f"```json\n{fixed_template}```"
                + candidate[last_match.end() :]
            )
            task.update_conversation(response, new_user_message)
            task.n = 1
            return task

        # Extract fields from response
        explanation_regex = r"<explanation>\s+(.*?)\s+</explanation>\s+"
        json_body_matches = re.finditer(
            r"```json\n(.*?)```", candidate, re.DOTALL
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

        explanation_match = re.search(explanation_regex, candidate, re.DOTALL)
        if not explanation_match:
            return error(
                "Invalid response from the model - no explanation or the explanation is properly delimited."
            )
        demo = explanation_match.group(1).strip()

        template_raw, candidate_template = (
            json_content,
            None,
        )
        errors = []
        for suffix, entry in zip(suffixes, entries):
            try:
                candidate_template = Template.load_array_from_response(
                    template_raw,
                    suffix,
                    full_entry=entry,
                    caller=self.caller,
                    response_schema=response_schema,
                )
                break
            except Exception as e:
                errors.append(e)
                continue
        if candidate_template is None:
            return error(f"Invalid response from the model - {str(errors[0])}")

        orig_candidate_template = deepcopy(candidate_template)

        # Check if we need to add a whitespace after the last element in the prefix
        prefix_template.generate_regex(add_end=False)
        prefixes = [
            (re.search(prefix_template.regex, entry), entry)
            for entry in entries
        ]
        prefixes = [(v[0].group(0), v[1]) for v in prefixes if v[0]]
        if prefixes:
            entry_prefix, entry = prefixes[0]
            suffix_entry = entry[len(entry_prefix) :]
            gap = len(suffix_entry) - len(suffix_entry.lstrip())
            if gap and not prefix_template.elements[-1].trailing_whitespace:
                prefix_template.elements[-1] = deepcopy(
                    prefix_template.elements[-1]
                )
                prefix_template.elements[-1].trailing_whitespace = gap
                prefix_template.elements[-1].id = "__0__"

        for elt in reversed(prefix_template.elements):
            candidate_template.elements.insert(0, elt)

        # Check how many of the suffixes are matched with a wildcard version of the template
        candidate_template_with_regex = deepcopy(candidate_template)
        for elt in candidate_template_with_regex.elements:
            if elt.is_variable() and not elt.regexp:
                elt.regexp = r".+?"

        # If this does not capture any of the lines, error out
        if all(
            not candidate_template_with_regex.match(entry.strip())[0]
            for entry in entries
        ):
            return error(
                "Invalid response from the model - no entry matches the template."
            )

        # If this only captures part of the lines, and we don't want to force the parsing, error out
        if (
            not all(
                candidate_template_with_regex.match(entry.strip())[0]
                for entry in entries
            )
            and not force
        ):
            missed_indices = [
                i
                for i, entry in enumerate(entries)
                if not candidate_template_with_regex.match(entry.strip())[0]
            ]
            missed_suffix_str = "\n".join([suffixes[i] for i in missed_indices])

            msg = f"Invalid response from the model - not all entries match the template: the model missed the following suffixes:\n{missed_suffix_str}"
            new_task = fix_prompt(
                orig_candidate_template,
                fix_prompt_undercapture.format(undercapture=missed_suffix_str),
            )
            return error(msg, new_task)

        # If this matches entries in the blocklist, error out
        if blocklist:
            matched_blocklist = [
                v
                for v in blocklist
                if candidate_template_with_regex.match(v)[0]
            ]

            if matched_blocklist:
                msg = "Invalid response from the model - the template overcaptures and matches these lines it should not match: \n{matched_blocklist}"
                return error(
                    msg,
                    fix_prompt(
                        orig_candidate_template,
                        fix_prompt_overcapture.format(
                            overcaptured="\n".join(matched_blocklist)
                        ),
                    ),
                )

        # If we get here, the template is valid and can be returned
        return (
            candidate_template,
            [
                i
                for i, entry in enumerate(entries)
                if candidate_template_with_regex.match(entry.strip())[0]
            ],
            demo,
        )

    def self_correct(
        self,
        entries,
        suffixes,
        prefix_template,
        tasks,
        response,
        blocklist=None,
        run_parsing=True,
        run_message_correction=False,
    ):

        def create_new_task(idx, content, message):
            new_task = tasks[idx]
            new_task.update_conversation(content, message)
            return new_task

        error_messages = {
            "overcapture_suspicion": "Please do not use placeholders for fields that are messages, statuses, or any other free form text, such as <Message>, <ErrorMessage>, <StatusMessage> or others. These placeholders are not allowed in the response, unless preceded by a key in a key_value pair. The values they represent are constants. Please fix your response if necessary and try again, by outputing your full explanation and response.",
            "parsing_error": "Your response is invalid: we encountered the following error: {err}.\nPlease fix your response and try again. Remeber, you must create a template that matches the suffixes. It must match exactly what is in the suffix. Only use previous examples as a reference. Please return a full explanation and response.",
        }

        # Correction: if any response contains "<MESSAGE>", ask the model to fix itself
        to_be_fixed = []
        for idx, llm_response in enumerate(response):
            if run_message_correction:
                # First check: if the response contains any placeholders that looks like a message field, try to self correct
                if re.search(
                    r"<[^>]*message[^>]*>",
                    llm_response.lower(),
                    re.IGNORECASE,
                ) or re.search(
                    r"<[^>]*error[^>]*>",
                    llm_response.lower(),
                    re.IGNORECASE,
                ):
                    to_be_fixed.append(
                        (
                            idx,
                            create_new_task(
                                idx,
                                llm_response,
                                error_messages["overcapture_suspicion"],
                            ),
                            error_messages["overcapture_suspicion"],
                        )
                    )
                    continue

            if run_parsing:
                # Second check: parse the response and see if it is valid
                template, err, new_task = self._parse_answer(
                    llm_response,
                    entries,
                    suffixes,
                    prefix_template,
                    tasks[idx],
                    blocklist=blocklist,
                    raise_errors=False,
                )

                if template is None:
                    if new_task is not None:
                        to_be_fixed.append((idx, new_task, err))
                    else:
                        to_be_fixed.append(
                            (
                                idx,
                                create_new_task(
                                    idx,
                                    llm_response,
                                    error_messages["parsing_error"].format(
                                        err=err
                                    ),
                                ),
                                err,
                            )
                        )
                    continue

        for idx, new_task, msg in to_be_fixed:
            get_logger().info(
                "Trying self correction for response #%s: %s", idx, msg
            )

        new_tasks = [task for _, task, _ in to_be_fixed]
        if new_tasks:
            corrected_candidates = self.caller(new_tasks)
            for fix_index, (idx, _, _) in enumerate(to_be_fixed):
                response[idx] = corrected_candidates[fix_index].candidates[0][
                    "content"
                ]
                tasks[idx] = new_tasks[fix_index]

        return response, tasks

    def _query(
        self,
        few_shot_ids,
        entries,
        prefix,
        description,
        blocklist=None,
        **kwargs,
    ):

        get_logger().info(
            "\t\tGenerating prompt at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        kwargs = self._prepare_kwargs(**kwargs)
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if blocklist is None:
            blocklist = []
        else:
            blocklist = list(set(blocklist))

        suffixes = self.get_suffix(entries)
        examples = self.generate_examples(few_shot_ids, **kwargs)
        history, system = gen_prompt(examples, entries, description, suffixes)
        task = OpenAITask(
            system_prompt=system,
            max_tokens=4096,
            history=history,
            **kwargs,
        )

        get_logger().info(
            "\t\tCalling LLM at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        response = self.caller([task])
        response = [
            val["content"] for resp in response for val in resp.candidates
        ]

        # Make copies of the tasks in case we need to rerun some
        tasks = [deepcopy(task) for _ in range(task.n)]
        for task in tasks:
            task.n = 1

        get_logger().info(
            "\t\tFirst round of self-correction at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        # First run a self-correction for potential overcaptures
        response, tasks = self.self_correct(
            entries,
            suffixes,
            prefix,
            tasks,
            response,
            run_parsing=False,
            run_message_correction=True,
        )

        get_logger().info(
            "\t\tSecond round of self-correction at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        # Then run a self-correction for parsing errors
        response, tasks = self.self_correct(
            entries,
            suffixes,
            prefix,
            tasks,
            response,
            blocklist=blocklist,
            run_parsing=True,
            run_message_correction=False,
        )

        # Parse the response
        task_idx = 0
        parsed_candidates = []
        for candidate in response:
            try:
                candidate_template, matches, demo = self._parse_answer(
                    candidate,
                    entries,
                    suffixes,
                    prefix,
                    tasks[task_idx],
                    blocklist=blocklist,
                    force=True,
                    raise_errors=True,
                )
            except ValueError:
                continue

            if candidate_template is not None:
                parsed_candidates.append(
                    (candidate_template, matches, demo, task_idx)
                )

        if not parsed_candidates:
            get_logger().error(
                "No valid candidates found for entries %s",
                entries,
            )
            return None, [], None, None, None, None

        # Filter the candidates with the most occurences
        all_candidates_str = [str(p[0]).strip() for p in parsed_candidates]
        candidate_str_counter = Counter(all_candidates_str)
        max_count = max(candidate_str_counter.values())
        majority_candidates_str = [
            k for k, v in candidate_str_counter.items() if v == max_count
        ]
        parsed_candidates = [
            p
            for p in parsed_candidates
            if str(p[0]).strip() in majority_candidates_str
        ]

        # Filter the candidates that capture most lines
        max_lines = max([len(p[1]) for p in parsed_candidates])
        parsed_candidates = [
            p for p in parsed_candidates if len(p[1]) == max_lines
        ]

        solution, matched_entries, demo, task_idx = parsed_candidates[0]

        self._write(
            len(self.tree.templates),
            self._print_history(tasks[task_idx].history),
            "".join(
                [
                    f"Response #{c_idx+1}:\n\n" + c + "\n\n##########\n\n"
                    for c_idx, c in enumerate(response)
                ]
            ),
            solution.format_as_example(regex=False),
        )

        get_logger().info(
            "Generated template without regex %s",
            solution,
        )

        if len(matched_entries) < len(entries):
            get_logger().warning(
                "Generated template %s does not match all entries %s",
                solution,
                entries,
            )

        # Compute the formatted suffix template for the few shot example
        if matched_entries:
            suffix_template = deepcopy(solution)
            suffix_template.elements = suffix_template.elements[
                len(prefix.elements) :
            ]
            suffix_template.generate_regex()
            formatted_template = suffix_template.format_as_example(
                force_match_with_entry=True,
                entry=suffixes[matched_entries[0]],
                regex=False,
            )
        else:
            formatted_template = None

        return (
            solution,
            matched_entries,
            demo,
            description,
            suffixes,
            formatted_template,
        )

    def run(
        self,
        few_shot_ids,
        entries,
        prefix,
        description,
        **kwargs,
    ):

        return self._query(
            few_shot_ids,
            entries,
            prefix,
            description,
            **kwargs,
        )

    def save_to_cache(
        self, template_id, entries, suffixes, demo, description, result
    ):
        self.cache[template_id] = (
            list(entries),
            list(suffixes),
            demo,
            description,
            result,
        )

    def ingest_fewshot(self, templates, seeds, descriptions):
        """Ingest fewshot data into cache"""

        entries_per_template = [
            self.entries_per_template[t] for t in range(len(templates))
        ]

        for template, entries, seed, description in zip(
            templates,
            entries_per_template,
            seeds,
            descriptions,
        ):
            explanation = explanation_format.format(
                desc=description,
                placeholder=seed["format"],
                constants="\n".join(seed["constants"]),
                variables="\n".join(seed["variables"]),
                kvp="\n".join(seed["key_values"]),
            )
            self.cache[template.id] = (
                list(entries),
                list(entries),
                explanation,
                description,
                template.format_as_example(
                    force_match_with_entry=True, entry=entries[0], regex=False
                ),
            )
