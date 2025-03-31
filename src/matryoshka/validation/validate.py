import os
import re
import time
from abc import ABC, abstractmethod

import dill

from matryoshka.classes.parser import Parser
from matryoshka.utils.json import parse_json
from matryoshka.utils.logging import get_logger


class Validator(ABC):
    """Abstract class for validation methods"""

    def __init__(
        self,
        name,
        caller,
        parser=None,
        tree=None,
        model="gemini-2.5-pro",
        save_path="./saved_queries/",
        values=None,
        entries_per_template=None,
        var_mapping=None,
    ) -> None:
        if not tree and not parser:
            raise ValueError("Either tree or parser must be provided.")

        if entries_per_template is None and not parser:
            raise ValueError("Entries per template must be provided.")

        self.name = re.sub(r"\s+", "_", name.lower())
        self.model = model
        self.caller = caller

        if tree:
            self.tree = tree
        else:
            self.tree = parser.tree

        if values is not None:
            self.values = values
        elif parser:
            self.values = parser.values

        if entries_per_template is not None:
            self.entries_per_template = entries_per_template
        elif parser:
            self.entries_per_template = parser.entries_per_template

        if var_mapping is not None:
            self.var_mapping = var_mapping
        elif parser:
            self.var_mapping = parser.var_mapping

        self.parser = parser
        self.save_path = os.path.join(save_path, self.name)
        os.makedirs(self.save_path, exist_ok=True)

        self.run_counter = 0

    def _prepare_kwargs(self, **kwargs):
        kwargs["n"] = 1
        kwargs["temperature"] = 1.0
        if "top_p" in kwargs:
            del kwargs["top_p"]
        return kwargs

    def _write(self, input, output, changes=None, prefix="", name=None):
        if prefix:
            bp = os.path.join(self.save_path, prefix)
        else:
            bp = self.save_path
        if name:
            name = str(name)
        else:
            name = f"run_{self.run_counter}"
        inputs_path = os.path.join(bp, "inputs", name)
        outputs_path = os.path.join(bp, "outputs", name)
        os.makedirs(os.path.join(bp, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(bp, "outputs"), exist_ok=True)

        with open(inputs_path, "w") as f:
            f.write(input)
        with open(outputs_path, "w") as f:
            f.write(output)

        if changes:
            changes_path = os.path.join(bp, "changes", name)
            os.makedirs(os.path.join(bp, "changes"), exist_ok=True)
            with open(changes_path, "w") as f:
                f.write(changes)

    def _write_llm_call(self, task, response, changes=None):
        input = self._print_history(
            task.history, task.message, task.system_prompt
        )
        output = response
        self._write(input, output, changes)

    def _print_history(self, hist, message=None, system=None):
        rtn = ""
        if system:
            rtn += f"*** System ***\n{system}\n\n"
        for elt in hist:
            if elt["content"]:
                rtn += f"*** {elt['role']} ***\n{elt['content']}\n\n"
        if message:
            rtn += f"*** User ***\n{message}\n\n"
        return rtn

    def _extract_answer(self, response, schema=None):
        return parse_json(
            response, self.caller, response_schema=schema, model=self.model
        )

    @abstractmethod
    def _parse_answer(
        self, response, original_template_ids, lines, force=False
    ):
        pass

    def _self_correct(
        self,
        response,
        original_template_ids,
        task,
        lines,
        schema=None,
        retries=15,
    ):
        """
        Test the model's output, return it if correct, or retry if not.
        """
        return_value = None
        self.run_counter += 1

        def rerun(error_message):
            task.update_conversation(response, error_message)
            return self.caller(task).candidates[0]

        for retry_count in range(retries):
            get_logger().debug(
                f"Running self-correction attempt #{retry_count} at time {time.strftime('%H:%M:%S', time.localtime())}"
            )
            try:
                json_response = self._extract_answer(response, schema=schema)
            except Exception as e:
                response = rerun(
                    f"Your answer is invalid, we cannot parse the JSON. Specifically, we got the following error(s): {str(e)}. Please fix your answer and try again."
                )
                continue

            try:
                return_value = self._parse_answer(
                    json_response,
                    original_template_ids,
                    lines,
                    force=retry_count == retries - 1,
                )
            except Exception as e:
                print(f"Error: {e}")
                response = rerun(
                    f"Your answer does not conform to the rules. Specifically, we got the following error(s):\n{str(e)}. Please fix your answer and try again. After outputting your explanation, before the code, print and explanation of the changes you made since the previous attempt."
                )
                continue

            break

        return return_value, response

    def _save(self):
        with open(os.path.join(self.save_path, f"parser.dill"), "wb") as f:
            new_parser = Parser(
                tree=self.tree,
                values=self.values,
                entries_per_template=self.entries_per_template,
            )
            if self.var_mapping:
                new_parser.var_mapping = self.var_mapping
            elif self.parser:
                new_parser.var_mapping = self.parser.var_mapping

            if self.parser:
                new_parser.embedding = self.parser.embedding
                new_parser.schema_mapping = self.parser.schema_mapping
                new_parser.clusters = self.parser.clusters

            dill.dump(new_parser, f)
