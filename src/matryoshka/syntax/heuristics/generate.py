import re
import time
from collections import Counter
from copy import deepcopy

from ...classes import Template
from ...genai_api.api import LLMTask
from ...utils.logging import get_logger
from ...utils.prompts.syntax.separation import (
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
        model="gemini-2.5-pro",
        parallel_attempts=5,
        temperature=1.0,
        top_p=1.0,
        path="./saved_queries/",
        few_shot_length=4,
        values=None,  # List of values for each element
        entries_per_template=None,
        ablation_fewshot=False,
        ablation_self_correction=False,
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
        )
        self.ablation_fewshot = ablation_fewshot
        self.ablation_self_correction = ablation_self_correction

        self.cache = {}

    def generate_examples(self, example_template_idx, **kwargs):
        examples = []
        for template_idx in example_template_idx:
            if template_idx not in self.cache:
                get_logger().warning(
                    "Template %s not in cache, skipping", template_idx
                )
                breakpoint()
                continue
            else:
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
        for entry_id, entry in enumerate(entries):
            try:
                suffix = suffixes[entry_id] if suffixes else entry
                candidate_template = Template.load_array_from_response(
                    template_raw,
                    suffix,
                    full_entry=entry,
                    caller=self.caller,
                    response_schema=response_schema,
                    model=self.model,
                    strict_regex=True,
                )
                break
            except Exception as e:
                errors.append(e)
                continue
        if candidate_template is None:
            return error(f"Invalid response from the model - {str(errors[0])}")

        # Check that every variable has a regexp
        for elt in candidate_template.elements:
            if (
                elt.is_variable()
                and not elt.regexp
                and not self.ablation_self_correction
            ):
                return error(
                    f"Invalid response from the model - variable {elt.value} has no regexp."
                )

        # Make sure no placeholder corresponds to an empty value
        empty_placeholder_regex = re.compile(
            r"\"value\":\s+\"\s*\"[^}]*\"placeholder\":\s+\"(\<?[a-zA-Z0-9\_\- ]*\>?)\"",
            re.DOTALL,
        )
        empty_placeholder_matches = empty_placeholder_regex.findall(
            json_content
        )
        if not force and any(
            elt.placeholder in empty_placeholder_matches
            for elt in candidate_template.elements
            if elt.is_variable() and not elt.value
        ):
            return error(
                f"Invalid response from the model - the explanation contains placeholders that are empty in the log lines: {empty_placeholder_matches}. Variables cannot be empty. Please rewrite your response omitting these placeholders and elements in the template."
            )

        # Check every regex captures its value
        regex_errors = []
        for elt_idx, elt in enumerate(candidate_template.elements):
            if elt.is_variable():
                try:
                    re.compile(elt.regexp)
                except re.error as e:
                    if self.ablation_self_correction:
                        elt.regexp = r".+?"
                    else:
                        regex_errors.append(
                            f"Invalid regexp {elt.regexp} for token #{elt_idx}: {str(e)}"
                        )
                    continue

                if not re.fullmatch(elt.regexp, elt.value):
                    if self.ablation_self_correction:
                        elt.regexp = r".+?"
                    else:
                        regex_errors.append(
                            f"Regexp {elt.regexp} does not capture value {elt.value} for token #{elt_idx}."
                        )
        if regex_errors:
            return error(
                "Invalid response from the model - some regexps do not capture their values: "
                + "; ".join(regex_errors)
            )

        # Make sure every placeholder in the explanation is in the template
        placeholder_section = re.split(
            "placeholders", demo, flags=re.IGNORECASE
        )[-1]
        placeholder_section = re.split(
            "Key Value Pairs", placeholder_section, flags=re.IGNORECASE
        )[0]
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
        for empty_placeholder in empty_placeholder_matches:
            template_placeholders.add(
                empty_placeholder.lower().replace("<", "").replace(">", "")
            )
        missing_placeholders = demo_placeholders - template_placeholders
        if missing_placeholders and not force:
            return error(
                f"Invalid response from the model - the following placeholders are in the explanation but not in the template: {', '.join(missing_placeholders)}",
            )

        orig_candidate_template = deepcopy(candidate_template)

        # Check if we need to add a whitespace after the last element in the prefix
        if prefix_template:
            prefix_template.generate_regex(add_end=False)
            prefixes = [
                (re.search(prefix_template.regex, entry), entry)
                for entry in entries
            ]
            filtered_prefixes = []
            for prefix, entry in prefixes:
                if prefix:
                    filtered_prefixes.append((prefix.group(0), entry))
            prefixes = filtered_prefixes

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

        try:
            candidate_template.generate_regex()
        except Exception as e:
            return error(
                f"Invalid response from the model - cannot generate regex from template: {str(e)}"
            )

        # If this does not capture any of the lines, error out
        if (
            all(
                not candidate_template.match(entry.strip())[0]
                for entry in entries
            )
            and not force
        ):
            return error(
                "Invalid response from the model - no entry matches the template."
            )

        # If this only captures part of the lines, and we don't want to force the parsing, error out
        if (
            not all(
                candidate_template.match(entry.strip())[0] for entry in entries
            )
            and not force
        ):
            missed_indices = [
                i
                for i, entry in enumerate(entries)
                if not candidate_template.match(entry.strip())[0]
            ]
            missed_suffix_str = "\n".join(
                [
                    suffixes[i] if suffixes else entries[i]
                    for i in missed_indices
                ]
            )

            msg = f"Invalid response from the model - not all entries match the template: the model missed the following lines:\n{missed_suffix_str}"
            new_task = fix_prompt(
                orig_candidate_template,
                fix_prompt_undercapture.format(undercapture=missed_suffix_str),
            )
            return error(msg, new_task)

        # Fix regexes if necessary
        elif not all(
            candidate_template.match(entry.strip())[0] for entry in entries
        ):
            candidate_template = deepcopy(candidate_template)
            for _, elt in enumerate(candidate_template.elements):
                if elt.is_variable() and "_" in str(elt.id):
                    original_regexp = elt.regexp
                    elt.regexp = r".+?" if " " in elt.value else r"\S+?"
                    candidate_template.generate_regex()
                    if self._match_examples(candidate_template, entries):
                        break
                    elt.regexp = original_regexp

        candidate_template.generate_regex()
        # If this matches entries in the blocklist, error out
        if blocklist:
            matched_blocklist = [
                v for v in blocklist if candidate_template.match(v)[0]
            ]

            if matched_blocklist:
                msg = f"Invalid response from the model - the template overcaptures and matches these lines it should not match: \n{matched_blocklist}"
                return error(
                    msg,
                    fix_prompt(
                        orig_candidate_template,
                        fix_prompt_overcapture.format(
                            overcaptured="\n".join(matched_blocklist)
                        ),
                    ),
                )

        # If this still does not capture any of the lines, error out
        if all(
            not candidate_template.match(entry.strip())[0] for entry in entries
        ):
            return error(
                "Invalid response from the model - no entry matches the template."
            )

        # If we get here, the template is valid and can be returned
        return (
            candidate_template,
            [
                i
                for i, entry in enumerate(entries)
                if candidate_template.match(entry.strip())[0]
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
            "parsing_error": "Your response is invalid: we encountered the following error: {err}.\nPlease fix your response and try again. Remeber, you must create a template that matches the entries. It must match exactly what is in the entries. Double check your answer, and make sure the regular expressions you are using are correct. Only use previous examples as a reference. Please explain your mistake, then return a full explanation and response.",
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
                response[idx] = corrected_candidates[fix_index].candidates[0]
                tasks[idx] = new_tasks[fix_index]

        return response, tasks

    def _query(
        self,
        few_shot_ids,
        entries,
        prefix,
        description,
        blocklist=None,
        ignore_prefix=True,
        **kwargs,
    ):

        get_logger().debug(
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

        if ignore_prefix:
            suffixes = None
            prefix = None
        else:
            suffixes = self.get_suffix(entries)

        examples = self.generate_examples(few_shot_ids, **kwargs)
        history, system = gen_prompt(examples, entries, description, suffixes)
        if self.ablation_fewshot:
            history = [history[-1]]
        task = LLMTask(
            system_prompt=system,
            history=history,
            thinking_budget=2048,
            **kwargs,
        )

        get_logger().debug(
            "\t\tCalling LLM at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        response = self.caller([task])
        response = [val for resp in response for val in resp.candidates]

        # Make copies of the tasks in case we need to rerun some
        tasks = [deepcopy(task) for _ in range(task.n)]
        for task in tasks:
            task.n = 1

        if not self.ablation_self_correction:
            for retry_idx in range(2):
                get_logger().debug(
                    "\t\tSelf correction attempt #%s at time %s",
                    retry_idx + 1,
                    time.strftime("%H:%M:%S", time.localtime()),
                )
                response, tasks = self.self_correct(
                    entries,
                    suffixes,
                    prefix,
                    tasks,
                    response,
                    run_parsing=True,
                    run_message_correction=False,
                    blocklist=blocklist,
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

        # Count the number of overlapping matches for each candidate
        candidates = []
        for candidate, matched_lines, demo, idx in parsed_candidates:
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
            return None, [], None, None, None, None, []

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
                    for c_idx, c in enumerate(response)
                ]
            ),
            template.format_as_example(regex=True),
        )

        get_logger().info(
            "Generated template %s",
            template,
        )

        if len(matched_entries) < len(entries):
            get_logger().warning(
                "Generated template %s does not match all entries %s",
                template,
                entries,
            )

        # Compute the formatted suffix template for the few shot example
        if matched_entries and prefix:
            suffix_template = deepcopy(template)
            suffix_template.elements = suffix_template.elements[
                len(prefix.elements) :
            ]
            suffix_template.generate_regex()
            formatted_template = suffix_template.format_as_example(
                force_match_with_entry=True,
                entry=suffixes[matched_entries[0]],
                regex=True,
                placeholder=True,
            )
        elif matched_entries and not prefix:
            formatted_template = template.format_as_example(
                force_match_with_entry=True,
                entry=entries[matched_entries[0]],
                regex=True,
                placeholder=True,
            )
        else:
            formatted_template = None

        return (
            template,
            matched_entries,
            demo,
            description,
            suffixes,
            formatted_template,
            matched_others,
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
            list(suffixes) if suffixes else None,
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
                    force_match_with_entry=True, entry=entries[0], regex=True
                ),
            )
