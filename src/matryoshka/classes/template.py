from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..utils.json import parse_json
from ..utils.logging import get_logger
from .element import Element, ElementType, alternatives
from .match import Match


@dataclass
class Template:

    elements: List[Element] = field(default_factory=list)
    example_entry: str = ""
    regex: Optional[str] = None
    keywords: Tuple[List[str], List[str]] = ([], [])
    id: int = -1

    def generate_regex(self, add_start=True, add_end=True) -> str:
        """
        Generate a regular expression from the template
        """
        regex = ""

        for _, element in enumerate(self.elements):
            current_element = element.get_regex().pattern
            regex += current_element
        if add_end:
            regex += "$"
        if add_start:
            regex = "^" + regex
        self.regex = regex

    def get_keywords(self):
        if not self.keywords[0]:
            full, part = [], []
            replace_regex = re.compile(r"[^a-zA-Z0-9]+")
            for element in self.elements:
                if not element.is_variable():
                    full.append(element.value.strip())
                    for kw in re.split(replace_regex, element.value):
                        if kw.strip():
                            part.append(kw.strip())
            self.keywords = (full, part)
        return self.keywords

    @staticmethod
    def escape_string(string):
        return json.dumps(string)

    @staticmethod
    def unescape_string(string):
        return json.loads(string)

    def get_regex_mapping(
        self,
        relative_ids=True,
        force_match_with_entry=False,
        partial=False,
        entry=None,
    ):
        regexes = {}
        if force_match_with_entry:
            entry = entry if entry is not None else self.example_entry
            match, matches = (
                self.match(entry) if not partial else self.partial_match(entry)
            )
            if not match:
                breakpoint()
                get_logger().error(
                    "Could not match entry %s with template %s", entry, self
                )
                force_match_with_entry = False
        else:
            matches = None

        for entry_idx, entry in enumerate(self.elements):
            if not entry.is_variable():
                continue

            value = entry.value
            if (
                matches
                and force_match_with_entry
                and entry_idx < len(matches.elements)
            ):
                value = matches.elements[entry_idx].value

            if relative_ids:
                entry_id = len(regexes)
            else:
                try:
                    entry_id = int(entry.id)
                except ValueError:
                    entry_id = entry.id

            regexes[entry_id] = {"regex": entry.regexp, "value": value}

        return json.dumps(regexes, indent=2)

    def format_as_example(
        self,
        force_match_with_entry=False,
        partial=False,
        entry=None,
        regex=False,
        types=False,
        ids=False,
        relative_ids=False,
        field_names=None,
        descriptions=None,
        whitespace=False,
        ignore_pending=False,
        placeholder=False,
    ):
        entries = []
        if ignore_pending:
            pending_var_ids = [
                i
                for i, e in enumerate(self.elements)
                if e.is_variable() and str(e.id).endswith("_")
            ]
            if not pending_var_ids:
                ignore_pending = False
        if force_match_with_entry and not ignore_pending:
            entry = entry if entry is not None else self.example_entry
            match, matches = (
                self.match(entry) if not partial else self.partial_match(entry)
            )
            if not match:
                breakpoint()
                get_logger().error(
                    "Could not match entry %s with template %s", entry, self
                )
                force_match_with_entry = False
        elif force_match_with_entry and ignore_pending:
            cutoff_index = min(
                i
                for i, e in enumerate(self.elements)
                if e.is_variable() and str(e.id).endswith("_")
            )
            prefix_template = Template(self.elements[:cutoff_index])
            entry = entry if entry is not None else self.example_entry
            match, matches = prefix_template.partial_match(entry)
            if not match:
                breakpoint()
                get_logger().error(
                    "Could not match entry %s with template %s", entry, self
                )

        else:
            matches = None

        variable_count = 0
        for entry_idx, entry in enumerate(self.elements):
            value = entry.value
            if (
                matches
                and force_match_with_entry
                and entry_idx < len(matches.elements)
            ):
                value = matches.elements[entry_idx].value
            line_value = {
                "is_variable": entry.is_variable(),
                "value": value,
            }
            if regex and entry.is_variable():
                line_value["regex"] = entry.regexp
            if placeholder and entry.placeholder:
                line_value["placeholder"] = entry.placeholder
            if types and entry.type is not None:
                line_value["type"] = entry.type
            if ids:
                line_value["id"] = entry.id
            elif relative_ids and entry.is_variable():
                line_value["id"] = variable_count
                variable_count += 1
            if field_names and entry.id in field_names:
                line_value["field_name"] = field_names[entry.id]
            if descriptions and entry.id in descriptions:
                line_value["description"] = descriptions[entry.id]
            if whitespace:
                line_value["trailing_whitespace"] = entry.trailing_whitespace

            entries.append(line_value)

        return json.dumps(entries, indent=2)

    def format_without_variables(self):
        rtn = ""
        for entry in self.elements:
            if entry.is_variable():
                value = f"[{entry.type}]"
            else:
                value = entry.value
            rtn += value + " " * entry.trailing_whitespace

        return rtn.strip()

    def print_example_from_values(self):
        rtn = ""
        for entry in self.elements:
            rtn += entry.value + " " * entry.trailing_whitespace
        return rtn.strip()

    def format_short(
        self,
        highlight=-1,
        color=True,
        force_match_with_entry=False,
        print_types=False,
        entry=None,
        field_name=None,
        start_separator="***",
        end_separator="***",
    ):

        if force_match_with_entry:
            match, matches = (
                self.match(entry)
                if entry is not None
                else self.match(self.example_entry)
            )
        else:
            matches = None

        rtn = ""
        for entry_idx, entry in enumerate(self.elements):
            value = str(
                matches.elements[entry_idx].value if matches else entry.value
            )
            if print_types and entry.type is not None:
                value += f"_{entry.type}"
            if highlight >= 0 and entry_idx == highlight:
                if not field_name:
                    rtn += (
                        f"{start_separator}{value}{end_separator}"
                        + " " * entry.trailing_whitespace
                    )
                else:
                    rtn += (
                        f"{start_separator}{field_name}{end_separator}"
                        + " " * entry.trailing_whitespace
                    )

            elif highlight >= 0:
                rtn += str(value) + " " * entry.trailing_whitespace
            elif color:
                rtn += entry.entity.color() + str(value) + "\033[0m"
                if entry_idx < len(self.elements) - 1:
                    rtn += " " * entry.trailing_whitespace + "\033[91m|"
            else:
                rtn += str(value)
                if entry_idx < len(self.elements) - 1:
                    rtn += " " * entry.trailing_whitespace + "|"

        return rtn

    def highlight(
        self,
        id,
        entry=None,
        start_separator="***",
        end_separator="***",
        field_name=None,
    ):
        elt_idx = next(i for i, elt in enumerate(self.elements) if elt.id == id)
        return self.format_short(
            highlight=elt_idx,
            color=False,
            print_types=False,
            force_match_with_entry=True,
            entry=entry,
            field_name=field_name,
            start_separator=start_separator,
            end_separator=end_separator,
        )

    def convert_to_wildcard_template(
        self, name=False, var_mapping=None, placeholder=False
    ):
        """Return a string in which each variable is replaced with a wildcard"""
        if name and not var_mapping and not placeholder:
            raise ValueError(
                "If name is True, var_mapping must be provided to map variable names."
            )
        rtn = ""
        for entry in self.elements:
            if entry.is_variable():
                if not name:
                    value = "<*>"
                else:
                    if placeholder and entry.placeholder:
                        value = f"<{entry.placeholder.upper()}>"
                    elif (
                        var_mapping
                        and entry.id in var_mapping
                        and var_mapping[entry.id].created_attribute
                    ):
                        value = f"<{var_mapping[entry.id].created_attribute.upper()}>"
                    else:
                        value = f"<var_{entry.id}>"
            else:
                value = entry.value
            rtn += value + " " * entry.trailing_whitespace
        return rtn

    def match(self, entry: str, exact=True):
        """
        Match a list of tokens against the template.
        """
        self.generate_regex()

        matched_entry = deepcopy(self.elements)

        try:
            match = re.fullmatch(self.regex, entry)
        except re.error:
            return False, None

        if match:
            groups = match.groupdict()
            for _, element in enumerate(matched_entry):
                if element.is_variable():
                    element.value = groups[f"var_{element.id}"]
            return True, Match(matched_entry)

        return False, None

    def partial_match(self, entry: str) -> Tuple[bool, Optional[Match]]:
        """
        Find matches where the template matches the start of a line.

        Args:
            entry (str): The text to match against

        Returns:
            Tuple[bool, Optional[Match]]: A tuple containing:
                - Boolean indicating if there was a partial match
                - Match object if there was a match, None otherwise
        """
        self.generate_regex(
            add_end=False
        )  # Don't add $ anchor since we want partial matches

        matched_entry = deepcopy(self.elements)

        try:
            match = re.match(
                self.regex, entry
            )  # Use match() instead of fullmatch()
        except re.error:
            return False, None

        if match:
            groups = match.groupdict()
            for element in matched_entry:
                if element.is_variable():
                    element.value = groups[f"var_{element.id}"]

            return True, Match(matched_entry)

        return False, None

    @staticmethod
    def load_from_json(tags, entry=None):
        # Sanity checks
        if any(
            "value" not in item
            or ("type" not in item and "is_variable" not in item)
            for item in tags
        ):
            raise ValueError(
                f"Invalid tags provided (missing 'value' or 'type' fields): {tags}"
            )

        elements = []
        entry_idx = 0
        # Create template

        for tag in tags:
            element = Element()
            element.value = tag["value"].strip()

            if "is_variable" in tag:
                tag["type"] = (
                    "CONSTANT" if not tag["is_variable"] else "VARIABLE"
                )

            if element.value == "":
                continue

            if entry is None:
                element.placeholder = tag.get("placeholder", None)
                element.id = tag.get("id", str(len(elements)) + "_")
                element.trailing_whitespace = tag.get("trailing_whitespace", 0)
                element.regexp = tag.get("regex", None)
                element.placeholder = tag.get("placeholder", None)
                element.entity = (
                    ElementType.VARIABLE
                    if tag["type"] == "VARIABLE"
                    else ElementType.CONSTANT
                )
                elements.append(element)
                continue

            old_entry_idx = entry_idx
            entry_idx = entry.find(element.value, entry_idx)
            if entry_idx < 0 and "regex" in tag:
                # Try to find the expression using a regex
                search_result = re.search(
                    Element.normalize_regex(tag["regex"]), entry[old_entry_idx:]
                )
                if not search_result:
                    raise ValueError(
                        f"Could not parse the JSON: value {element.value} not found in entry {entry}"
                    )
                else:
                    entry_idx = old_entry_idx + search_result.start()
                    element.value = search_result.group()
            elif entry_idx < 0:
                raise ValueError(
                    f"Could not parse the JSON: value {element.value} not found in entry {entry}"
                )

            if entry_idx - old_entry_idx > 0:
                gap = entry[old_entry_idx:entry_idx]
                if gap.strip() != "":
                    trailing_whitespace = len(gap) - len(gap.rstrip())
                    elements.append(
                        Element(
                            value=gap.strip(),
                            entity=(ElementType.CONSTANT),
                            trailing_whitespace=trailing_whitespace,
                            id=str(len(elements)) + "_",
                        )
                    )

            entry_idx += len(element.value)

            if isinstance(tag["type"], bool):
                element.entity = (
                    ElementType.VARIABLE
                    if tag["type"]
                    else ElementType.CONSTANT
                )
            else:
                element.entity = ElementType.from_str(tag["type"])
            if "regex" in tag:
                element.regexp = Element.normalize_regex(tag["regex"])

            element.trailing_whitespace = 0
            while entry[entry_idx:].startswith(" "):
                element.trailing_whitespace += 1
                entry_idx += 1

            element.id = tag.get("id", str(len(elements)) + "_")
            element.placeholder = tag.get("placeholder", None)
            elements.append(element)

        # Remove empty elements
        elements_fixed = []
        for e in elements:
            val = e.value
            if val is not None:
                if val.strip() != "" or e.trailing_whitespace > 0:
                    elements_fixed.append(e)
        elements = elements_fixed

        if entry is None:
            entry = ""
            for elt in elements:
                entry += elt.value + " " * elt.trailing_whitespace
        rtn = Template(elements, example_entry=entry)

        return rtn

    @staticmethod
    def load_array_from_response(
        tags,
        entry: str,
        caller=None,
        response_schema=None,
        full_entry=None,
        model=None,
        strict_regex=False,
    ):
        autofix = False
        try:
            elements = json.loads(tags)
        except json.JSONDecodeError:
            try:
                elements = json.loads(tags.replace("\\", "\\\\"))
            except json.JSONDecodeError as e:
                if caller:
                    elements = parse_json(
                        tags,
                        caller,
                        response_schema=response_schema,
                        model=model,
                    )
                    autofix = True
                else:
                    raise e
        ret_elements = []
        entry_idx = 0
        entry = entry.strip()
        for element_dict_idx, element_dict in enumerate(elements):
            if ("is_variable" not in element_dict) or (
                "value" not in element_dict
            ):
                raise ValueError(
                    f"Could not parse the array: element {element_dict_idx} ({element_dict}) is missing 'is_variable' or 'value' fields"
                )
            element = Element()
            if element_dict.get("value", None) is None:
                continue
            element.value = element_dict["value"].strip()
            if element_dict["is_variable"]:
                element.entity = ElementType.VARIABLE
                if "regex" not in element_dict and strict_regex:
                    raise ValueError(
                        f"Could not parse the array: Some variable fields are missing regex attributes"
                    )
                if "regex" in element_dict:
                    try:
                        element.regexp = Element.normalize_regex(
                            element_dict["regex"]
                        )
                    except:
                        raise ValueError(
                            f"Could not parse the array: invalid regex {element_dict['regex']}"
                        )
                if "placeholder" in element_dict:
                    element.placeholder = element_dict["placeholder"]
            else:
                element.entity = ElementType.CONSTANT

            if not element_dict["value"]:
                continue

            # In some cases the LLM parses a bit too much.
            # If that's the case, we can skip the first elements until we reach an unparsed part.
            if (
                not ret_elements
                and full_entry
                and element.value in full_entry
                and not entry.startswith(element.value)
            ):
                # First, see if any suffix of element.value is a prefix of entry
                valid_suffix = False
                for i in range(len(element.value)):
                    suffix = element.value[i:]
                    if entry.startswith(suffix.strip()):
                        element.value = suffix.strip()
                        valid_suffix = True
                        break

                # If that did not work, look for the next elements in the entry, and set the prefix of that as the current value.
                if not valid_suffix and len(elements) > element_dict_idx + 1:
                    # Combine next 3 elements without spaces
                    next_elements_combined = "".join(
                        elements[i]["value"].strip()
                        for i in range(
                            element_dict_idx + 1,
                            min(element_dict_idx + 4, len(elements)),
                        )
                    )
                    next_elements_combined = re.sub(
                        r"\s+", "", next_elements_combined
                    )
                    entry_no_spaces = re.sub(r"\s+", "", entry)

                    # Find first match of combined elements in entry without spaces
                    next_entry_start_index = entry_no_spaces.find(
                        next_elements_combined
                    )

                    if next_entry_start_index > 0:
                        # Get prefix up to match
                        missing_prefix_no_spaces = entry_no_spaces[
                            :next_entry_start_index
                        ]

                        # Get original entry up to where no-spaces version matched
                        original_length = 0
                        current_no_spaces_length = 0
                        for i, c in enumerate(entry):
                            if not c.isspace():
                                current_no_spaces_length += 1
                            if current_no_spaces_length > len(
                                missing_prefix_no_spaces
                            ):
                                original_length = i
                                break

                        missing_prefix = entry[:original_length].strip()

                        element.value = missing_prefix
                        valid_suffix = True

                if not valid_suffix:
                    raise ValueError(
                        "Could not parse the array: the template captures more than just the suffix."
                    )

            old_entry_idx = entry_idx
            entry_idx = entry.find(element.value, entry_idx)
            if entry_idx < 0 and "regex" in element_dict:
                # Try to find the expression using a regex
                search_result = re.search(element.regexp, entry[old_entry_idx:])
                if not search_result:
                    raise ValueError(
                        f"Could not parse the array: value {element.value} not found in suffix {entry}"
                    )
                else:
                    entry_idx = old_entry_idx + search_result.start()
                    element.value = search_result.group()
            elif entry_idx < 0 and element.entity != ElementType.VARIABLE:
                # First attempt to fix: There sometime are too many / not enough space in the variable.
                element_value_regex = re.escape(element.value)
                element_value_regex = re.sub(
                    "(\\\\( )+)+", "\\\\s+", element_value_regex
                )
                search_result = re.search(
                    element_value_regex, entry[old_entry_idx:]
                )
                if search_result:
                    entry_idx = old_entry_idx + search_result.start()
                    element.value = search_result.group()
                else:
                    # Try to see if a substring of the value is present at this location
                    new_value = None
                    for approx_match in range(1, len(entry) - old_entry_idx):
                        approx_val = entry[
                            old_entry_idx : old_entry_idx + approx_match
                        ]
                        if approx_val in element.value:
                            new_value = approx_val
                        else:
                            break
                    if (
                        new_value
                        and abs(1 - len(new_value) / len(element.value)) < 0.25
                    ):
                        element.value = new_value
                        entry_idx = entry.find(element.value, old_entry_idx)

                    if entry_idx < 0:
                        # Third fix: look for alternatives (quotes can get mistaken for one another)
                        alternate_entry = entry
                        alternate_value = element_dict["value"].strip()
                        for alt in alternatives:
                            if any(a in element.value for a in alt):
                                for a in alt[1:]:
                                    alternate_entry = alternate_entry.replace(
                                        a, alt[0]
                                    )
                                    alternate_value = alternate_value.replace(
                                        a, alt[0]
                                    )

                        entry_idx = alternate_entry.find(
                            alternate_value, old_entry_idx
                        )
                        if entry_idx > 0:
                            element.value = entry[
                                entry_idx : entry_idx + len(element.value)
                            ]
                            get_logger().warning(
                                "Automatic fix applied to value, changed from %s to %s",
                                element_dict["value"],
                                element.value,
                            )
                        else:
                            if element_dict_idx == len(elements) - 1:
                                # If this is the last element, and it is a constant, just append the constant to the end
                                entry_idx = old_entry_idx
                                break
                            else:
                                raise ValueError(
                                    f"Could not parse the array: value {element.value} not found in suffix {entry}"
                                )
                    else:
                        get_logger().warning(
                            "Automatic fix applied to value, changed from %s to %s",
                            element_dict["value"],
                            element.value,
                        )

            elif entry_idx < 0:
                raise ValueError(
                    f"Could not parse the array: value {element.value} not found in suffix {entry}"
                )

            if entry_idx - old_entry_idx > 0:
                gap = entry[old_entry_idx:entry_idx]
                if len(gap.strip()) > 5:
                    raise ValueError(
                        f'Could not parse the array: substring from character #{old_entry_idx} to #{entry_idx} with value "{gap}" is missing from the template'
                    )
                elif gap.strip():
                    trailing_whitespace = len(gap) - len(gap.rstrip())
                    ret_elements.append(
                        Element(
                            value=gap.strip(),
                            entity=(ElementType.CONSTANT),
                            trailing_whitespace=trailing_whitespace,
                            id=str(len(ret_elements))
                            + str(random.randint(0, 100))
                            + "_",
                        )
                    )

            entry_idx += len(element.value)
            element.trailing_whitespace = 0
            while entry_idx < len(entry) and entry[entry_idx].isspace():
                element.trailing_whitespace += 1
                entry_idx += 1

            element.id = (
                str(len(ret_elements)) + str(random.randint(0, 100)) + "_"
            )
            ret_elements.append(element)

        # Remove empty elements
        ret_elements = [
            e
            for e in ret_elements
            if e.value and (e.value.strip() or e.trailing_whitespace)
        ]

        # Add missing parts at the end if the last part is a constant
        if entry[entry_idx:].strip() and len(entry[entry_idx:].strip()) < 5:
            get_logger().warning(
                "Adding missing constant part at the end of the template"
            )
            if not ret_elements:
                elt = Element(
                    value=entry[entry_idx:].strip(),
                    entity=ElementType.CONSTANT,
                    trailing_whitespace=0,
                    id="0_",
                )
                ret_elements.append(elt)
            elif not ret_elements[-1].is_variable():
                ret_elements[-1].value += entry[entry_idx:].strip()
                ret_elements[-1].trailing_whitespace = 0
            else:
                missing_part = entry[entry_idx:].strip()
                ret_elements[-1].trailing_whitespace += len(missing_part) - len(
                    missing_part.rstrip()
                )
                ret_elements.append(
                    Element(
                        value=entry[entry_idx:].strip(),
                        entity=(ElementType.CONSTANT),
                        trailing_whitespace=0,
                        id=str(len(ret_elements)) + "_",
                    )
                )
        elif entry[entry_idx:].strip():
            raise ValueError(
                f"Could not parse the array: The template does not match suffix {entry} and is likely missing elements or contains too many elements."
            )

        rtn = Template(ret_elements, example_entry=entry)
        return rtn

    def __str__(self):
        return self.format_short()

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Template):
            return False
        if not len(self.elements) == len(other.elements):
            return False
        return all(i == j for i, j in zip(self.elements, other.elements))

    def __hash__(self) -> int:
        return hash(tuple(self.elements))

    @staticmethod
    def match_templates(templates, entry):
        matches = []
        for t_idx, template in enumerate(templates):
            match, matched_entries = template.match(entry, exact=True)
            if match:
                matches.append((t_idx, matched_entries))

        if matches:
            return matches, None

        # If not, look for close templates to use as few-shot examples
        for t_idx, template in enumerate(templates):
            _, matched_entries = template.match(entry, exact=False)
            if matched_entries is not None:
                for idx, m in matched_entries:
                    matches.append((t_idx, idx, m))

        return None, matches
