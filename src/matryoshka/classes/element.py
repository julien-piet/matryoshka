from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import regex as re

alternatives = [["'", '"', "`"], ["/", "\\"]]


class ElementType(Enum):
    """
    Represents the syntax type of an element: variable or constant.
    """

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


class ElementMatchType(Enum):
    """
    Class used for tracking tree matching results.
    """

    NOMATCH = 0
    MATCH = 1
    PENDING = 2


@dataclass
class Element:
    """
    Represents a single element in a template
    """

    entity: ElementType = ElementType.CONSTANT
    type: Optional[str] = None  # OCSF type
    value: Optional[str] = None
    regexp: Optional[str] = ".*?"
    trailing_whitespace: int = 0
    description: Optional[str] = None
    placeholder: Optional[str] = None

    compiled_element_regex = None
    compiled_prefix_regex = None
    compiled_prefix_regex_terminal = None

    id: int | str = "-1"
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

    def erase_regex_cache(self):
        """Erase the compiled regex cache."""
        self.compiled_element_regex = None
        self.compiled_prefix_regex = None
        self.compiled_prefix_regex_terminal = None

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
        r = re.sub(r"\.\+(?!\?)", r".+?", r)
        r = re.sub(r"\.\*(?!\?)", r".*?", r)
        return Element.generalize_regex(re.compile(r).pattern)

    def get_regex(self):
        if self.compiled_element_regex:
            return self.compiled_element_regex
        else:
            regex = ""
            if self.is_variable():
                regex = f"(?P<var_{self.id}>{self.regexp})"
            else:
                regex = re.escape(self.value).replace("\\ ", "\\s+")
            if self.trailing_whitespace:
                regex += r"\s+"
            self.compiled_element_regex = re.compile(regex)
            return self.compiled_element_regex

    def match_tree(
        self, entry, trail, match_var=False, terminal=False, add_lookahead=False
    ):
        if self.regexp:
            match_var = match_var or all(
                t not in self.regexp for t in ["*", "+", "?", ".", "\\w", "\\S"]
            )
        if self.is_variable() and not match_var:
            return {
                "result": ElementMatchType.PENDING,
                "match_dict": {},
                "suffix_index": 0,
            }
        else:
            if self.compiled_prefix_regex and not terminal:
                match_regex = self.compiled_prefix_regex
            elif self.compiled_prefix_regex_terminal and terminal:
                match_regex = self.compiled_prefix_regex_terminal
            else:
                match_regex = "^"
                for key in trail:
                    match_regex += key.get_regex().pattern
                match_regex += self.get_regex().pattern

                if terminal:
                    match_regex += "$"

                # Compile the regex for the prefix
                match_regex = re.compile(match_regex)
                if terminal:
                    self.compiled_prefix_regex_terminal = match_regex
                else:
                    self.compiled_prefix_regex = match_regex

            # In some cases, partial matching can fail if we don't verify the absence of trailing whitespace.
            # In these cases, we use a lookahead assertion to ensure that the match is valid.
            if (
                add_lookahead
                and (not terminal)
                and (not self.trailing_whitespace)
            ):
                match_regex = re.compile(match_regex.pattern + r"(?!\s)")

            mtch = match_regex.match(entry)
            if mtch and len(entry[len(mtch.group()) :]) == len(
                entry[len(mtch.group()) :].lstrip()
            ):
                groups = mtch.groupdict()
                match_dict = {
                    key.split("_")[1]: groups[key]
                    for key in groups
                    if key.startswith("var_")
                }
                return {
                    "result": ElementMatchType.MATCH,
                    "match_dict": match_dict,
                    "suffix_index": len(mtch.group()),
                }
            elif (
                (not mtch)
                or terminal
                or self.trailing_whitespace > 0
                or add_lookahead
            ):
                # No match found
                return {
                    "result": ElementMatchType.NOMATCH,
                    "match_dict": None,
                    "suffix_index": None,
                }
            else:
                return self.match_tree(entry, trail, match_var, terminal, True)

    def __str__(self):
        return f"{self.entity.to_str()}({self.value})"
