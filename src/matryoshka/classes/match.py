from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .element import Element


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
