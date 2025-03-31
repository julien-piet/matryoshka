from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import regex as re

from .logging import get_logger
from .utils import parse_json

alternatives = [["'", '"', "`"], ["/", "\\"]]


class ElementType(Enum):
    VARIABLE = 1
    CONSTANT = 2

    @staticmethod
    def from_str(value: str) -> ElementType:
        if value.upper() == "VARIABLE" or value.upper() == "VAR":
            return ElementType.VARIABLE
        if value.upper() == "CONSTANT" or value.upper() == "CST":
            return ElementType.CONSTANT
        raise ValueError(f"Unknown ElementType: {value}")

    def color(self):
        if self == ElementType.VARIABLE:
            return "\033[94m"
        if self == ElementType.CONSTANT:
            return "\033[93m"

    def to_str(self):
        if self == ElementType.VARIABLE:
            return "VAR"
        if self == ElementType.CONSTANT:
            return "CST"

    def __hash__(self):
        return hash(self.to_str())


@dataclass
class Element:

    entity: ElementType = ElementType.CONSTANT
    type: Optional[str] = None  # OCSF type
    value: Optional[str] = None
    regexp: Optional[str] = ".*?"
    trailing_whitespace: int = 0
    description: Optional[str] = None

    id: int = -1
    fixed: bool = False

    def is_variable(self):
        return self.entity == ElementType.VARIABLE

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Element):
            return False

        if self.entity != value.entity:
            return False

        if self.is_variable():
            return self.regexp == value.regexp and (
                (self.trailing_whitespace > 0 and value.trailing_whitespace > 0)
                or (
                    self.trailing_whitespace == 0
                    and value.trailing_whitespace == 0
                )
            )
        else:
            return self.value == value.value

    def __hash__(self) -> int:
        if self.is_variable():
            return hash(
                (
                    self.entity,
                    self.regexp,
                    self.trailing_whitespace > 0,
                )
            )
        else:
            return hash(
                (
                    self.value,
                    self.entity,
                    self.trailing_whitespace > 0,
                )
            )

    def format_as_example(self):
        rtn_value = {"value": self.value, "type": self.entity.to_str()}
        if self.is_variable():
            rtn_value["regex"] = self.regexp

        return json.dumps(rtn_value, separators=(",", ": "), sort_keys=False)

    @staticmethod
    def generalize_regex(pattern: str) -> str:
        stripped = pattern.strip("^$")
        stripped_spaces = re.sub(r"(\\s| |\\\\.)(\+|\*|\?)*", "", stripped)
        if any(c in stripped_spaces for c in "*+?{}[]().|\\"):
            return pattern
        return (
            r"\d+?"
            if stripped.isdigit()
            else (
                r"\S+?"
                if (" " not in stripped and "\\s" not in stripped)
                else r".+?"
            )
        )

    @staticmethod
    def normalize_regex(r):
        r = re.sub(r" +(?=[^*+?{])", r"\\s+", r)
        r = re.sub(r"\\s(?=[^*+?{])", r"\\s+", r)
        r = re.sub(r"((?<!\[)\^|\\$|\\b|\$)", "", r)
        r = re.sub(
            r"(^\\s([+?*]|\{\d+(,\d+)?\})?)|(\\s([+?*]|\{\d+(,\d+)?\})?$)",
            "",
            r,
        )
        return Element.generalize_regex(re.compile(r).pattern)

    def get_regex(self, append_whitespace=False, forced=False):
        regex = ""
        if forced:
            regex = f"(?P<var_{self.id}>.*?)"
        elif self.is_variable():
            regex = f"(?P<var_{self.id}>{self.regexp})"
        else:
            regex = re.escape(self.value).replace("\\ ", "\\s+")
        if append_whitespace or self.trailing_whitespace:
            regex += r"\s+"
        return regex

    def match_tree(self, suffix, pending, match_var=False, terminal=False):
        # Accept simple regexes without any quantifiers or large character classes, since these are unlikely to overcapture\
        if self.regexp:
            match_var = match_var or all(
                t not in self.regexp for t in ["*", "+", "?", ".", "\\w", "\\S"]
            )
        if self.is_variable() and not match_var:
            return True, suffix, True, None
        else:
            suffix = suffix.strip()
            match_regex = "^"
            for key in pending:
                match_regex += key.get_regex()
            match_regex += self.get_regex()

            if terminal:
                match_regex += "$"

            mtch = re.match(match_regex, suffix)
            if mtch and len(suffix[len(mtch.group()) :]) == len(
                suffix[len(mtch.group()) :].lstrip()
            ):
                groups = mtch.groupdict()
                matches = deepcopy(pending + [self])
                for elt in matches:
                    if elt.is_variable():
                        elt.value = groups[f"var_{elt.id}"]

                return (
                    True,
                    suffix[len(mtch.group()) :].strip(),
                    False,
                    Match(matches) if len(matches) > 0 else None,
                )
            else:
                return False, suffix, False, None

    def merge(self, other):
        if not isinstance(other, Element):
            raise ValueError("Cannot merge different types of elements")
        if self.is_variable() != other.is_variable():
            raise ValueError("Cannot merge different types of elements")
        if self.is_variable():
            self.regexp += r"\s+" if self.trailing_whitespace else ""
            self.regexp += other.regexp
            self.trailing_whitespace = other.trailing_whitespace
            self.type = None if self.type != other.type else self.type
            self.value += " " * self.trailing_whitespace + other.value
            self.fixed = self.fixed or other.fixed
        else:
            self.value += " " * self.trailing_whitespace + other.value
            self.trailing_whitespace = other.trailing_whitespace
            self.fixed = self.fixed or other.fixed

    def __str__(self):
        return f"{self.entity.to_str()}({self.value})"


@dataclass(frozen=True)
class Match:
    elements: List["Element"] = field(default_factory=list)

    def __add__(self, other: Match) -> Match:
        if other is None:
            return self

        if not isinstance(other, Match):
            raise NotImplementedError()
        combined_elements = self.elements + other.elements
        return Match(elements=combined_elements)

    def __getitem__(self, index: int) -> "Element":
        return self.elements[index]


class Value:
    def __init__(self, element_id, values=None):
        self.element_id = element_id

        self.values = []
        self.value_counts = {}
        self.running_sum = None

        if values is not None:
            for v in values:
                self.append(v)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def append(self, value: str):
        self.values.append(value)
        if value not in self.value_counts:
            self.value_counts[value] = 1
        else:
            self.value_counts[value] += 1

    def __len__(self):
        return len(self.values)

    def get_closest(
        self,
        values: List[Value],
        metrics: Optional[Union[List[str], str]] = None,
        k=5,
        tree=None,
        embedding=None,
    ):
        if not values:
            return []

        overall = False
        if metrics is None:
            metrics = [
                "intersection",
                "cousins",
                "template_embedding",
            ]
            overall = True

        if not isinstance(metrics, list):
            metrics = [metrics]

        # Compute all the metrics
        distances = [[0] * len(metrics) for _ in values]
        for m_idx, metric in enumerate(metrics):
            if metric not in [
                "intersection",
                "cousins",
                "template_embedding",
            ]:
                raise ValueError(f"Unknown value closeness metric: {metric}")
            elif metric == "intersection":
                dist = self.get_most_common(values)
            elif metric == "cousins":
                dist = self.get_cousins(values, tree)
            else:
                dist = self.get_closest_template_embedding(
                    values, tree, embedding
                )

            for i, (_, d) in enumerate(dist):
                distances[i][m_idx] = d

        # Augment values
        values = [(v, d) for v, d in zip(values, distances)]

        # Rank and return
        if not overall:
            values = sorted(values, key=lambda x: x[1])
            return [v[0] for v in (values[:k] if k > 0 else values)]
        else:
            # Custom ranking
            selections = [
                (0,),
                (2,),
                (1, 2),
            ]
            selected, selected_idx = [], []

            for s_idx, s in enumerate(selections):
                ct = (
                    k // len(selections)
                    if s_idx >= k % len(selections)
                    else k // len(selections) + 1
                )
                ranked = sorted(
                    values,
                    key=lambda x: tuple(x[1][i] for i in s),  # noqa: E741
                )
                if 1 in s:
                    ranked = [r for r in ranked if r[1][1] != math.inf]
                ranked = [
                    r[0] for r in ranked if r[0].element_id not in selected_idx
                ][:ct]
                for r in ranked:
                    selected.append(r)
                    selected_idx.append(r.element_id)

            return selected

    def get_most_common(self, values: List[Value]):
        augmented_values = []
        for v in values:
            intersection_keys = self.value_counts.keys() & v.value_counts.keys()
            if len(intersection_keys) > 0:
                weight = sum(
                    self.value_counts[k] for k in intersection_keys
                ) + sum(v.value_counts[k] for k in intersection_keys)
                weight /= len(self) + len(v)
            else:
                weight = -math.inf
            augmented_values.append((v, -weight))

        return augmented_values

    def get_cousins(self, values: List[Value], tree: TemplateTree):
        """Order other variables by degree of separation on tree"""
        return [
            (v, -tree.degree_of_separation(self.element_id, v.element_id))
            for v in values
        ]

    def get_closest_template_embedding(
        self, values: List[Value], tree: TemplateTree, embedding: Any
    ):
        """Order other variables by template embedding distance"""

        relevant_templates = [
            t for v in values for t in tree.templates_per_node[v.element_id]
        ]

        distances = {t: 0 for t in relevant_templates}
        for t in tree.templates_per_node[self.element_id]:
            for t_id, dist in embedding.template_distance(
                t, templates=relevant_templates
            ):
                distances[t_id] += dist

        distance_per_value = {v.element_id: [0, 0] for v in values}

        for v in values:
            for t in tree.templates_per_node[v.element_id]:
                distance_per_value[v.element_id][0] += distances[t]
                distance_per_value[v.element_id][1] += 1

        distance_per_value = {
            v: d[0] / d[1] for v, d in distance_per_value.items() if d[1] > 0
        }

        return [
            (
                v,
                (
                    -distance_per_value[v.element_id]
                    if v.element_id in distance_per_value
                    else 0
                ),
            )
            for v in values
        ]


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
            current_element = element.get_regex()
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

    def format_as_example(
        self,
        force_match_with_entry=False,
        partial=False,
        entry=None,
        regex=False,
        types=False,
        ids=False,
        field_names=None,
        descriptions=None,
        whitespace=False,
    ):
        entries = []
        if force_match_with_entry:
            entry = entry if entry is not None else self.example_entry
            match, matches = (
                self.match(entry) if not partial else self.partial_match(entry)
            )
            if not match:
                get_logger().error(
                    "Could not match entry %s with template %s", entry, self
                )
                force_match_with_entry = False
        else:
            matches = None

        for entry_idx, entry in enumerate(self.elements):
            value = (
                entry.value
                if not force_match_with_entry
                else matches.elements[entry_idx].value
            )
            line_value = {
                "is_variable": entry.is_variable(),
                "value": value,
            }
            if regex and entry.is_variable():
                line_value["regex"] = entry.regexp
            if types and entry.type is not None:
                line_value["type"] = entry.type
            if ids:
                line_value["id"] = entry.id
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

    def format_short(
        self,
        highlight=-1,
        color=True,
        force_match_with_entry=False,
        print_types=False,
        entry=None,
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
                val = value.replace("*", "\\*")
                rtn += f"***{val}***" + " " * entry.trailing_whitespace
            elif highlight >= 0:
                rtn += (
                    str(value.replace("*", "\\*"))
                    + " " * entry.trailing_whitespace
                )
            elif color:
                rtn += entry.entity.color() + str(value) + "\033[0m"
                if entry_idx < len(self.elements) - 1:
                    rtn += " " * entry.trailing_whitespace + "\033[91m|"
            else:
                rtn += str(value)
                if entry_idx < len(self.elements) - 1:
                    rtn += " " * entry.trailing_whitespace + "|"

        return rtn

    def highlight(self, id):
        elt_idx = next(i for i, elt in enumerate(self.elements) if elt.id == id)
        return self.format_short(
            highlight=elt_idx,
            color=False,
            print_types=False,
            force_match_with_entry=True,
        )

    def convert_to_wildcard_template(self):
        """Return a string in which each variable is replaced with a wildcard"""
        rtn = ""
        for entry in self.elements:
            if entry.is_variable():
                value = "<*>"
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

    def merge_consecutive_messages(self):
        """
        Merges consecutive constant message elements in the template.

        This method is useful for simplifying the template, especially when there are repetitive constant message elements.
        """
        if not self.elements:
            return  # Early exit if there are no elements

        # Temporary list to hold merged elements
        merged_elements = [deepcopy(self.elements[0])]

        for element in deepcopy(self.elements[1:]):
            last_element = merged_elements[-1]

            # Check if the current element can be merged with the last one
            if (
                element.entity == ElementType.CONSTANT
                and last_element.entity == ElementType.CONSTANT
            ):
                # Merge with the last element
                last_element.value += (
                    last_element.trailing_whitespace * " " + element.value
                )
                last_element.trailing_whitespace = element.trailing_whitespace
            else:
                # Otherwise, just append the current element to the list
                merged_elements.append(element)

        # Update self.elements with the merged elements list
        self.elements = merged_elements

    @staticmethod
    def load_from_json(tags, entry):
        # Sanity checks
        if any("value" not in item or "type" not in item for item in tags):
            raise ValueError(
                f"Invalid tags provided (missing 'value' or 'type' fields): {tags}"
            )

        elements = []
        entry_idx = 0
        # Create template

        for tag in tags:
            element = Element()
            element.value = tag["value"].strip()

            if element.value == "":
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

            element.id = str(len(elements)) + "_"
            elements.append(element)

        # Remove empty elements
        elements = [
            e
            for e in elements
            if e.value.strip() != "" or e.trailing_whitespace > 0
        ]
        rtn = Template(elements, example_entry=entry)

        # rtn.merge_consecutive_messages()
        return rtn

    @staticmethod
    def load_array_from_response(
        tags, entry: str, caller=None, response_schema=None, full_entry=None
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
                        tags, caller, response_schema=response_schema
                    )
                    autofix = True
                else:
                    raise e
        ret_elements = []
        entry_idx = 0
        if autofix:
            get_logger().warning(
                "Automatic fix applied to parse the json array %s.",
                json.dumps(elements),
            )
        entry = entry.strip()
        for element_dict_idx, element_dict in enumerate(elements):
            element = Element()
            if element_dict.get("value", None) is None:
                continue
            element.value = element_dict["value"].strip()
            if element_dict["is_variable"]:
                element.entity = ElementType.VARIABLE
                if "regex" in element_dict:
                    try:
                        element.regexp = Element.normalize_regex(
                            element_dict["regex"]
                        )
                    except:
                        raise ValueError(
                            f"Could not parse the array: invalid regex {element_dict['regex']}"
                        )
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
                        "\s+", "", next_elements_combined
                    )
                    entry_no_spaces = re.sub("\s+", "", entry)

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
                elif gap.strip() != "":
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
            if e.value.strip() != "" or e.trailing_whitespace > 0
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


@dataclass
class Tree:
    node: int
    parent: Optional[Tree] = None
    branches: dict = field(default_factory=dict)
    terminal: bool = False
    template_id: Optional[int] = None

    def __getitem__(self, item) -> Tree:
        if isinstance(item, int) and item in self.branches:
            return self.branches[item]
        elif isinstance(item, int):
            raise KeyError(f"Key {item} not found in tree")
        else:
            raise ValueError("Invalid key type")

    def __setitem__(self, node: int, value: Tree) -> None:
        if not isinstance(node, int) or not isinstance(value, Tree):
            raise ValueError("Invalid assignment")

        self.branches[node] = value

    def __delitem__(self, item) -> None:
        if isinstance(item, int):
            del self.branches[item]
        else:
            raise ValueError("Invalid key type")

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.branches
        else:
            return False

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["parent"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.parent = None

    def get_element(self, node_to_element: list, element: Element):
        branch_elements = {node_to_element[i]: i for i in self.branches}
        if element in branch_elements:
            return branch_elements[element]
        else:
            return None

    def get_lineage(self):
        lineage = []
        current_node = self
        while current_node.parent is not None:
            lineage.append(current_node.node)
            current_node = current_node.parent
        return lineage[::-1]


@dataclass
class ExplorationNode:
    node: int
    suffix: str
    pending: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    matches: Match = None


@dataclass
class TerminalNode:
    suffix: str
    trail: list = field(default_factory=list)
    template_id: Optional[int] = None
    matches: Match = None

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TerminalNode):
            return False
        return (
            self.suffix == value.suffix
            and self.template_id == value.template_id
        )


@dataclass
class TemplateTree:

    tree: Tree
    nodes: list = field(default_factory=list)
    templates: list = field(default_factory=list)
    examples: list = field(default_factory=list)

    templates_per_node: dict = field(default_factory=dict)
    node_to_tree: dict = field(default_factory=dict)
    overlap_map: dict = field(default_factory=dict)

    def __init__(self, distances=None):
        self.tree = Tree(0, parent=None, terminal=False)
        self.nodes = [None]
        self.templates_per_node = {0: set()}
        self.node_to_tree = {0: self.tree}

        self.templates = []
        self.examples = []

        self.distances = distances if distances is not None else []
        self.overlap_map = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild parents
        for _, tree in self.node_to_tree.items():
            for branch in tree.branches.values():
                branch.parent = tree

    def add_template(self, elements, example=None, fixed=False):
        if isinstance(elements, Template):
            elements = elements.elements

        added_new_node = False

        current_node = self.tree
        for element in elements:
            next_node_id = current_node.get_element(self.nodes, element)
            if next_node_id is None:
                added_new_node = True
                next_node_id = len(self.nodes)
                current_node[next_node_id] = Tree(
                    next_node_id, parent=current_node, terminal=False
                )
                element.id = next_node_id
                element.fixed = fixed
                self.nodes.append(element)
                self.templates_per_node[next_node_id] = set()
                self.node_to_tree[next_node_id] = current_node[next_node_id]
            else:
                self.nodes[next_node_id].fixed = (
                    self.nodes[next_node_id].fixed or fixed
                )

            current_node = current_node[next_node_id]

        if added_new_node:
            current_node.terminal = True
            new_template_id = len(self.templates)
            current_node.template_id = new_template_id
            self.templates.append(current_node.get_lineage())
            for node in self.templates[-1]:
                self.templates_per_node[node].add(new_template_id)
            self.templates_per_node[0].add(new_template_id)
            self.examples.append([])
            if example:
                self.examples[-1].append(example)

            for distance in self.distances:
                distance.add_template(
                    new_template_id, self.gen_template(new_template_id)
                )

            return new_template_id
        else:
            # Template already exists
            return current_node.template_id

    def remove_template(self, t_id):
        """Remove a template from the tree."""
        elements = self.templates[t_id]

        # Remove terminal flag on last element
        if elements[-1] in self.node_to_tree:
            self.node_to_tree[elements[-1]].terminal = False
            self.node_to_tree[elements[-1]].template_id = None

        # Update templates per element
        for elt in elements[::-1]:
            if elt in self.templates_per_node:
                self.templates_per_node[elt] = set(
                    [t for t in self.templates_per_node[elt] if t != t_id]
                )
                # Remove node if it only was in this template
                if not len(self.templates_per_node[elt]):
                    del self.templates_per_node[elt]
                    parent = self.node_to_tree[elt].parent
                    if parent:
                        del parent[elt]
                    del self.node_to_tree[elt]
                    self.nodes[elt] = None
            elif (
                elt in self.node_to_tree
                and self.node_to_tree[elt].parent
                and elt in self.node_to_tree[elt].parent.branches
            ):
                del self.node_to_tree[elt].parent.branches[elt]
                del self.node_to_tree[elt]
                self.nodes[elt] = None
            elif elt in self.node_to_tree:
                del self.node_to_tree[elt]
                self.nodes[elt] = None

        self.templates[t_id] = []
        self.examples[t_id] = []

        if t_id in self.overlap_map:
            del self.overlap_map[t_id]

        del_list = []
        for m in self.overlap_map:
            self.overlap_map[m] = [v for v in self.overlap_map[m] if v != t_id]
            if not self.overlap_map[m]:
                del_list.append(m)

        for m in del_list:
            if m in self.overlap_map:
                del self.overlap_map[m]

        # Remove template from root list of templates
        self.templates_per_node[0].remove(t_id)

    def replace_template(self, target_id, elements, example):
        """
        Replace the target template with a new template
        """
        if isinstance(elements, Template):
            elements = elements.elements

        # Unmark elements in original template
        target = self.templates[target_id]
        for elt in target:
            self.templates_per_node[elt] = set(
                [t for t in self.templates_per_node[elt] if t != target_id]
            )
        self.node_to_tree[target[-1]].terminal = False
        self.node_to_tree[target[-1]].template_id = None

        # Add new template
        added_new_node = False
        current_node = self.tree
        for element in elements:
            next_node_id = current_node.get_element(self.nodes, element)
            if next_node_id is None:
                added_new_node = True
                next_node_id = len(self.nodes)
                current_node[next_node_id] = Tree(
                    next_node_id, parent=current_node, terminal=False
                )
                element.id = next_node_id
                element.fixed = False
                self.nodes.append(element)
                self.templates_per_node[next_node_id] = set()
                self.templates_per_node[next_node_id].add(target_id)
                self.node_to_tree[next_node_id] = current_node[next_node_id]
            else:
                self.nodes[next_node_id].fixed = (
                    self.nodes[next_node_id].fixed or False
                )
                self.templates_per_node[next_node_id].add(target_id)

            current_node = current_node[next_node_id]

        if added_new_node:
            current_node.terminal = True
            current_node.template_id = target_id
            self.templates[target_id] = current_node.get_lineage()
            self.templates_per_node[0].add(target_id)
            self.examples[target_id].append(example)
            for distance in self.distances:
                distance.add_template(target_id, self.gen_template(target_id))

        # Remove old nodes
        for elt in target[::-1]:
            if not self.templates_per_node[elt]:
                del self.templates_per_node[elt]
                parent = self.node_to_tree[elt].parent
                del parent[elt]
                del self.node_to_tree[elt]
                self.nodes[elt] = None

        return target_id

    def splice_elements(self, target, new_elements):
        """
        Replace the target element in all templates with new_elements
        """
        new_ids = []
        tree = self.node_to_tree[target].parent
        if target not in tree.branches:
            raise ValueError(f"Target element {target} not found in tree")
        original_tree = tree[target]

        for elt_idx, elt in enumerate(new_elements):
            if elt_idx:
                elt.id = len(self.nodes)
                self.nodes.append(elt)
            else:
                elt.id = target
                self.nodes[target] = elt

            new_ids.append(elt.id)

            if elt_idx < len(new_elements) - 1:
                new_tree = Tree(elt.id, tree, terminal=False)
                tree[elt.id] = new_tree
                tree = new_tree
                self.node_to_tree[elt.id] = tree
            else:
                original_tree.node = elt.id
                original_tree.parent = tree
                tree[elt.id] = original_tree
                self.node_to_tree[elt.id] = original_tree

        # Update templates
        for t_id, trail in enumerate(self.templates):
            if target in trail:
                index = trail.index(target)
                self.templates[t_id][index : index + 1] = new_ids

        # Update templates per node
        for elt in new_ids:
            if elt not in self.templates_per_node:
                self.templates_per_node[elt] = deepcopy(
                    self.templates_per_node[target]
                )

        return new_elements

    def get_deepest_node_elements(self, elements: List[Element]):
        if isinstance(elements, Template):
            elements = elements.elements

        current_node = self.tree
        for element in elements:
            next_node_id = current_node.get_element(self.nodes, element)
            if next_node_id is None:
                return current_node
            else:
                current_node = current_node[next_node_id]

        return current_node

    def get_ordered_templates(self, node: int = 0, compute_percentage=False):
        if node > 0:
            lineage = self.node_to_tree[node].get_lineage()[::-1] + [0]
            seen = set()
            rslt = []
            for idx, node in enumerate(lineage):
                for t_idx in self.templates_per_node[node]:
                    if t_idx not in seen:
                        # Compute the percentage of length shared in terms of number of elements
                        percent = len(
                            [e for e in self.templates[t_idx] if e in lineage]
                        ) / len(self.templates[t_idx])
                        if compute_percentage:
                            rslt.append(
                                (
                                    idx + 1 - percent,
                                    t_idx,
                                )
                            )
                        else:
                            rslt.append(
                                (
                                    idx,
                                    t_idx,
                                )
                            )
                        seen.add(t_idx)
        else:
            rslt = [
                (
                    0,
                    t_idx,
                )
                for t_idx in range(len(self.templates))
            ]
        return rslt

    def degree_of_separation(self, node_a, node_b, norm=False):
        """Return the degree of separation between node_a and node_b.
        The degree of separation is the height of the largest common prefix.
        If both are on the same template, return the depth of the first"""

        lineage_a = self.node_to_tree[node_a].get_lineage()
        lineage_b = self.node_to_tree[node_b].get_lineage()

        if lineage_a[0] != lineage_b[0]:
            return 0

        if node_a in lineage_b or node_b in lineage_a:
            return len(lineage_a)

        i = 0
        while lineage_b[i] == lineage_a[i]:
            i += 1

        if not norm:
            return i
        else:
            return i / min(len(lineage_a), len(lineage_b))

    def gen_template(self, t_id):
        trail = self.templates[t_id]
        elements = [self.nodes[n] for n in trail]
        return Template(
            elements,
            self.examples[t_id][0] if len(self.examples[t_id]) else "",
            id=t_id,
        )

    def match(self, entry: str, start_node: int = 0, debug=False):
        entry = entry.strip()
        if start_node > 0:
            current_nodes = [
                ExplorationNode(
                    start_node,
                    entry,
                    [],
                    self.node_to_tree[start_node].get_lineage(),
                    Match(),
                )
            ]
        else:
            current_nodes = [ExplorationNode(0, entry, [], [], Match())]
        terminal_nodes = []

        while len(current_nodes):
            updated_nodes = []
            for expl_node in current_nodes:
                node, suffix, pending, trail, matches = (
                    expl_node.node,
                    expl_node.suffix,
                    expl_node.pending,
                    expl_node.trail,
                    expl_node.matches,
                )
                local_tree, terminal = self.node_to_tree[node], True

                # If we have fininshed parsing the string, add it to terminal nodes
                if not suffix.strip():
                    terminal_nodes.append(
                        TerminalNode("", trail, local_tree.template_id, matches)
                    )
                    continue

                # If not, continue matching along current branches
                for branch_id in local_tree.branches:
                    element = self.nodes[branch_id]
                    result, new_suffix, is_pending, local_matches = (
                        element.match_tree(
                            suffix,
                            [self.nodes[p] for p in pending],
                        )
                    )

                    # If the current element matches the suffix, continue exploring the tree
                    terminal = terminal and (not result)
                    if result:
                        # If there are no pending elements, add the new node to the updated nodes
                        if not is_pending:
                            new_trail = trail + pending + [branch_id]
                            new_matches = matches + local_matches
                            updated_nodes.append(
                                ExplorationNode(
                                    branch_id,
                                    new_suffix,
                                    [],
                                    new_trail,
                                    new_matches,
                                )
                            )
                        # If there are pending elements, update pending and add to updated nodes
                        else:
                            new_pending = pending + [branch_id]
                            updated_nodes.append(
                                ExplorationNode(
                                    branch_id,
                                    new_suffix,
                                    new_pending,
                                    deepcopy(trail),
                                    deepcopy(matches),
                                )
                            )

                # Check if this is a terminal node:
                terminal = terminal or local_tree.terminal
                if terminal:
                    added = False
                    if len(pending) and local_tree.terminal:
                        tail_idx, pending = pending[-1], pending[:-1]
                        tail, tail_elt, local_pending = (
                            self.node_to_tree[tail_idx],
                            self.nodes[tail_idx],
                            (
                                [self.nodes[p] for p in pending]
                                if len(pending) > 0
                                else []
                            ),
                        )

                        result, new_suffix, _, local_matches = (
                            tail_elt.match_tree(
                                suffix, local_pending, True, terminal=True
                            )
                        )
                        if result:
                            trail += pending + [tail_idx]
                            new_matches = matches + local_matches
                            template_id = tail.template_id
                            terminal_nodes.append(
                                TerminalNode(
                                    new_suffix,
                                    trail,
                                    template_id,
                                    new_matches,
                                )
                            )
                            added = True

                    if not added:
                        terminal_nodes.append(
                            TerminalNode(
                                suffix, trail, template_id=None, matches=matches
                            )
                        )

            current_nodes = updated_nodes

        # Order candidates by specificity
        terminal_nodes = sorted(
            terminal_nodes,
            key=lambda x: (
                len([e for e in x.trail if self.nodes[e].is_variable()]),
                1
                / (
                    1
                    + sum(
                        [
                            len(self.nodes[e].regexp)
                            for e in x.trail
                            if self.nodes[e].is_variable()
                        ]
                    )
                ),
            ),
            reverse=True,
        )

        complete_matches = [
            t
            for t in terminal_nodes
            if not t.suffix.strip() and t.template_id is not None
        ]
        if len(complete_matches):
            return True, complete_matches

        # Fall back: if the tree doesn't match, try individual templates. Sometimes matching element per element can fail to match because of non greedy matching
        for template_id, nodes in enumerate(self.templates):
            if nodes is None or not len(nodes):
                continue
            template = self.gen_template(template_id)
            match, matches = template.match(entry)
            if match:
                complete_matches.append(
                    TerminalNode("", nodes, template_id, matches)
                )

        if len(complete_matches):
            return True, complete_matches

        possible_new_templates = [
            t for t in terminal_nodes if not t.suffix.strip()
        ]

        if len(possible_new_templates):
            return False, possible_new_templates

        candidates = sorted(terminal_nodes, key=lambda t: -len(t.trail))
        return False, candidates


class ParserStep(Flag):
    INIT = auto()
    SYNTAX = auto()
    TYPED = auto()
    EVENT_MAPPING = auto()
    VARIABLE_MAPPING = auto()
    TEMPLATE_FILLING = auto()
    DATA_CONVERSION = auto()
    DONE = auto()


@dataclass
class Parser:

    tree: TemplateTree
    values: Dict[int, Value] = field(default_factory=dict)
    entries_per_template: Dict[int, List[str]] = field(default_factory=dict)
    embedding: Any = None
    event_types: Optional[Dict[int, str]] = None
    var_mapping: Optional[Dict] = None
    schema_mapping: Optional[Dict] = None
    schemas: Optional[List] = None
    template_mapping: Optional[Dict] = None
    completed_steps: Optional[ParserStep] = field(default=ParserStep.INIT)
    clusters: Optional[Dict] = None

    def update_step(self, step: ParserStep):
        self.completed_steps |= step

    def has_completed(self, step: ParserStep) -> bool:
        return bool(self.completed_steps & step)
