from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Dict, List, Optional

from .tree import TemplateTree
from .value import Value


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
