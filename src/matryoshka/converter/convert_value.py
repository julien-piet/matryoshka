import json
import re
from dataclasses import dataclass, field

# Imports for the generated parser
from datetime import datetime
from typing import Any, Callable, List

from ..classes import Element, Template
from ..genai_api.api import Caller, LLMTask
from ..utils.OCSF import OCSFObject, OCSFSchemaClient
from ..utils.prompts.converter import parse_variables


class OcsfDataTypeParser:
    def __init__(
        self,
        schema_client: OCSFSchemaClient,
        caller: Caller,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.caller = caller
        self.schema_client = schema_client
        self.model = model
        self.data_type_parsers = {}
        self.data_types = self.schema_client.get_data_types()
        self.get_data_types()

    def get_data_types(self):
        for data_type, data_type_dict in self.data_types.items():
            if "type_name" not in data_type_dict:
                # This is a base type (integer, long, string, etc.)
                self.data_type_parsers[data_type] = (
                    self.get_parser_for_base_type(data_type_dict["caption"])
                )
            else:
                self.data_type_parsers[data_type] = self.get_parser_for_type(
                    data_type, data_type_dict
                )

    def get_parser_for_base_type(self, data_type: str):
        """
        Gets a parser for a base data type that has no constraints.
        """
        data_type = data_type.lower()
        if data_type == "string":
            return str
        if data_type == "integer" or data_type == "long":
            return int
        elif data_type == "float":
            return float
        elif data_type == "json":
            return json.loads
        elif data_type == "boolean":
            return lambda x: x.lower() == "true"
        else:
            raise ValueError(f"Base data type {data_type} is not supported.")

    def get_parser_for_type(self, data_type: str, data_type_dict: dict):
        """
        Gets a parser for a data type that has constraints.
        """
        data_type = data_type.lower()
        base_type_parser = self.get_parser_for_base_type(
            data_type_dict["type_name"]
        )

        def parser(to_parse: str):
            parsed_to_base_type = base_type_parser(to_parse)
            if "regex" in data_type_dict:
                if not re.match(data_type_dict["regex"], parsed_to_base_type):
                    raise ValueError(
                        f"Data {parsed_to_base_type} does not match regex {data_type_dict['regex']}."
                    )
            if "max_len" in data_type_dict:
                if len(parsed_to_base_type) > data_type_dict["max_len"]:
                    raise ValueError(
                        f"Data {parsed_to_base_type} is longer than the maximum length {data_type_dict['max_len']}."
                    )
            if "range" in data_type_dict:
                if (
                    parsed_to_base_type < data_type_dict["range"][0]
                    or parsed_to_base_type > data_type_dict["range"][1]
                ):
                    raise ValueError(
                        f"Data {parsed_to_base_type} is not in the range {data_type_dict['range']}."
                    )
            return parsed_to_base_type

        return parser

    def get_parser_for_variable_path(
        self,
        variable_path: str,
        node: Element,
        templates: List[Template],
        examples: List[str],
    ):
        objects = self.schema_client.get_objects_from_path(variable_path)
        leaf_object = objects[-1]
        leaf_object_parser = self.data_type_parsers[leaf_object.type]

        # template_element_idx = template.elements.index(template_value)
        # print("idx", template_element_idx)
        # example_value = template.match(examples[0])[1][template_element_idx].value
        example_value = examples[0]

        try:
            example_value_parsed = leaf_object_parser(example_value)
        except ValueError:
            leaf_object_parser = self.get_model_generated_parser(
                objects, node, templates, examples[:1]
            )
            example_value_parsed = leaf_object_parser(example_value)
        print(
            f"Example value: {example_value} parsed to {example_value_parsed}"
        )
        examples_for_prompt = examples[:1]
        for example in examples:
            try:
                example_value_parsed = leaf_object_parser(example)
            except:
                print(f"Parsing {example} failed. Refining.")
                # print("Failed. refining.")
                examples_for_prompt.append(example)
                leaf_object_parser = self.get_model_generated_parser(
                    objects, node, templates, examples_for_prompt
                )
                example_value_parsed = leaf_object_parser(example)
        return leaf_object_parser

    def get_data_type_description_for_variable_path(
        self, variable_path_objects: List[OCSFObject]
    ):
        variable = variable_path_objects[-1]
        data_types = self.schema_client.get_data_types()
        data_type_dict = data_types[variable.type]
        description = ""
        if "regex" in data_type_dict:
            description += (
                f"- It must match the regex {data_type_dict['regex']}.\n"
            )
        if "max_len" in data_type_dict:
            description += f"- It must be at most {data_type_dict['max_len']} characters long.\n"
        if "range" in data_type_dict:
            description += (
                f"- It must be in the range {data_type_dict['range']}.\n"
            )
        description += f"- {data_type_dict['description']}\n"
        description += f"""Additionally, the data type is ultimately being parsed as a {variable.name}, with the following description:
- {variable.description}"""
        return description

    def get_model_generated_parser(
        self,
        variable_path_objects: List[OCSFObject],
        node: Element,
        templates: List[Template],
        examples: List[str],
    ):
        """
        Call this when initial, hard-coded parsers fail.
        """
        # template_element_idx = template.elements.index(template_value)
        # example_value = template.match(examples[0])[1][
        #     template_element_idx
        # ].value

        example_value = examples[-1]

        data_type = variable_path_objects[-1].type
        data_type_dict = self.data_types[data_type]
        data_type_str = data_type
        if "type_name" in data_type_dict:
            data_type_str += (
                f", with underlying type {data_type_dict['type_name']}"
            )

        user_message = parse_variables.user.format(
            examples=parse_variables.get_example_str(examples),
            data_type=data_type_str,
            data_type_description=self.get_data_type_description_for_variable_path(
                variable_path_objects
            ),
        )
        print("#####")
        print("Prompt", user_message)
        print("#####")
        task = LLMTask(
            system_prompt=parse_variables.system,
            message=user_message,
            model=self.model,
            max_tokens=2048,
            temperature=0.33,
        )
        candidates = self.caller(task)
        if isinstance(candidates, list):
            candidates = candidates[0]
        response = candidates.candidates[0].strip()
        parser = ModelGeneratedParser(response)
        for e in examples:
            try:
                parsed_value = parser(e)
                self.validate_parsed_value(
                    parsed_value, variable_path_objects[-1]
                )
            except (ValueError, AssertionError) as err:
                print("Failed with error", err, "retrying.")
                task.update_conversation(
                    response,
                    f"The function you wrote failed parsed {e} as {parsed_value} which is invalid because {err}. Please fix it.",
                )
                candidates = self.caller(task)
                if isinstance(candidates, list):
                    candidates = candidates[0]
                response = candidates.candidates[0].strip()
                parser = ModelGeneratedParser(response)

        return parser

    def validate_parsed_value(self, parsed_value, ocsf_obj: OCSFObject):
        data_type = ocsf_obj.type
        data_type_dict = self.data_types[data_type]
        if "type_name" in data_type_dict:
            type_name = data_type_dict["type_name"].lower()
            if type_name == "string":
                # If the parsed value is None, that means the value probably
                # failed a regex check. We should flag this elsewhere.
                assert parsed_value is None or isinstance(
                    parsed_value, str
                ), "Parsed value must be a string."
            elif type_name == "long" or type_name == "integer":
                assert isinstance(
                    parsed_value, int
                ), "Parsed value must be an integer."
        if data_type == "timestamp_t":
            assert (
                parsed_value > 0
            ), f"Timestamp must be positive. If there is missing information, right now it is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

        # TODO: sibling checks?


@dataclass
class ModelGeneratedParser:
    response: str
    function_body: str = field(init=False)
    parser: Callable[[str], Any] = field(init=False, repr=False)

    def __post_init__(self):
        function_body_match = re.match(
            r"```python(.*?)```", self.response, re.DOTALL
        )
        if function_body_match is None:
            raise ValueError(
                f"Could not parse the function body from the response: {self.response}"
            )
        self.function_body = function_body_match.group(1).strip("\n")

        print(self.response)

        def parser(to_parse: str):
            """
            This is a horrible hack to get around the fact that exec() doesn't work with nested functions.
            """
            parser_function_call = f"""def _parse(to_parse):
{self.function_body}
ret_val = _parse(to_parse)
"""
            # print(parser_function_call)
            code = compile(parser_function_call, "<string>", "exec")
            loc = {"to_parse": to_parse}
            exec(code, globals(), loc)
            return loc["ret_val"]

        self.parser = parser

    def __call__(self, to_parse: str) -> Any:
        return self.parser(to_parse)
