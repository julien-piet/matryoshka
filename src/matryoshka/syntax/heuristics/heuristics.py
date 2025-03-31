import os
import random
from abc import ABC

from tqdm import tqdm

from ...classes import Template


class Heuristic(ABC):
    """
    Abstract class for heuristic models
    Contains some helpful basic heuristics and tools used by multiple methods.
    """

    def __init__(
        self,
        tree,
        name,
        caller,
        model="gemini-2.5-flash",
        parallel_attempts=4,
        temperature=0.5,
        top_p=0.8,
        save_path="./saved_queries/",
        few_shot_length=4,
        values=None,
        entries_per_template=None,
        line_to_match=None,
    ) -> None:
        self.tree = tree
        self.name = name
        self.model = model
        self.caller = caller
        self.parallel_attempts = parallel_attempts
        self.temperature = temperature
        self.top_p = top_p
        self.few_shot_len = few_shot_length
        self.values = values
        self.entries_per_template = entries_per_template
        self.save_path = save_path
        self.write_count = 0
        self.line_to_match = line_to_match

        self.save_path = os.path.join(self.save_path, self.name)
        os.makedirs(self.save_path, exist_ok=True)

    def _prepare_kwargs(self, **kwargs):
        kwargs["n"] = (
            self.parallel_attempts
            if "parallel_attempts" not in kwargs
            else kwargs["parallel_attempts"]
        )
        kwargs["temperature"] = (
            self.temperature
            if "temperature" not in kwargs
            else kwargs["temperature"]
        )
        kwargs["top_p"] = self.top_p
        if "parallel_attempts" in kwargs:
            del kwargs["parallel_attempts"]
        return kwargs

    def init_caller(self, caller):
        self.caller = caller

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["caller"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caller = None

    def _match_examples(self, candidate_template, examples):
        """Tests a set of examples to see if they match the template. Make sure no variable is empty."""
        matches = [
            candidate_template.match(entry.strip()) for entry in examples
        ]
        issues = any(
            not m[0]
            or any(
                not v.value.strip() for v in m[1].elements if v.is_variable()
            )
            for m in matches
        )
        return not issues

    def _get_match_examples(self, candidate_template, examples):
        """Tests a set of examples to see if they match the template. Make sure no variable is empty."""
        matches = [
            candidate_template.match(entry.strip()) for entry in examples
        ]
        issues = [
            i
            for i, m in enumerate(matches)
            if not m[0]
            or any(
                not v.value.strip() for v in m[1].elements if v.is_variable()
            )
        ]
        return issues

    def match_previous_entries(
        self, candidate_template, ignored_templates=None
    ):
        """Tests a random subset of entries to see if they match the template"""
        if ignored_templates is None:
            ignored_templates = []

        matches = []
        for template_id, entries in self.entries_per_template.items():
            if (
                not entries
                or template_id in ignored_templates
                or not self.tree.templates[template_id]
            ):
                continue
            test_set = random.sample(entries, min(5, len(entries)))
            results = [candidate_template.match(entry)[0] for entry in test_set]
            for r_id, r in enumerate(results):
                if r:
                    matches.append((template_id, test_set[r_id]))
        if not matches:
            return False, None
        else:
            return True, matches

    def _ask_user_for_confirmation(self, message):
        print(f"An action needs validation by a human: {message}")
        print("Please confirm if the action is correct (y/n)")
        response = input()
        if response.lower() == "y":
            return True
        elif response.lower() == "n":
            return False
        else:
            print("Invalid response, please type 'y' or 'n'")
            return self._ask_user_for_confirmation(message)

    def _write(self, name, input, ouptut, result, prefix=""):
        if prefix:
            bp = os.path.join(self.save_path, prefix)
        else:
            bp = self.save_path
        name = str(name)
        inputs_path = os.path.join(bp, "inputs", name)
        outputs_path = os.path.join(bp, "outputs", name)
        results_path = os.path.join(bp, "results", name)
        os.makedirs(os.path.join(bp, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(bp, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(bp, "results"), exist_ok=True)

        with open(inputs_path, "w") as f:
            f.write(input)
        with open(outputs_path, "w") as f:
            f.write(ouptut)
        with open(results_path, "w") as f:
            f.write(result)

    def _print_history(self, hist, message):
        rtn = ""
        for elt in hist:
            if elt["content"]:
                rtn += f"*** {elt['role']} ***\n{elt['content']}\n\n"
        if message:
            rtn += f"*** User ***\n{message}\n\n"
        return rtn

    def get_suffix(self, entries):
        suffixes = []
        for entry in entries:
            if self.line_to_match and entry in self.line_to_match:
                # If we have a cached prefix, use it.
                suffixes.append(self.line_to_match[entry].suffix)
                continue
            _, candidates = self.tree.match(entry)
            suffixes.append(
                candidates[0].suffix
                if candidates and candidates[0].suffix
                else entry
            )
        return suffixes

    def get_prefix(self, entries):
        if not isinstance(entries, list):
            entries = [entries]
            return_list = False
        else:
            return_list = True

        prefixes = []
        for _, entry in tqdm(
            enumerate(entries), desc="Getting prefixes", total=len(entries)
        ):
            if self.line_to_match and entry in self.line_to_match:
                candidates = [self.line_to_match[entry]]
                match = candidates[0].template_id is not None
            else:
                match, candidates = self.tree.match(entry)

            if match:
                prefixes.append(
                    self.tree.gen_template(candidates[0].template_id)
                )
                continue

            matched_prefix_trail, suffix = (
                candidates[0].trail,
                candidates[0].suffix,
            )
            matched_prefix = entry[: -len(suffix)]
            prefixes.append(
                Template(
                    [self.tree.nodes[i] for i in matched_prefix_trail],
                    matched_prefix,
                )
            )

        if not return_list:
            return prefixes[0]
        return prefixes
