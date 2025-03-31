import json
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Dict, List, Optional

from .semantics import VariableSemantics
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

    @classmethod
    def load_from_json(cls, json_parser: dict) -> "Parser":
        # Load tree
        tree = TemplateTree.load_from_json(json_parser["templates"])

        # Load var mapping
        var_mapping = {
            int(k): VariableSemantics.from_dict(v)
            for k, v in json_parser["var_mapping"].items()
        }

        # Load event types
        event_types = {
            int(k): v for k, v in json_parser.get("event_types", {}).items()
        }

        return cls(
            tree=tree,
            var_mapping=var_mapping,
            event_types=event_types,
        )

    def as_json(self, include_embeddings=False):

        # Represent tree as JSON
        templates = []
        template_id_to_order = {}
        for template_id, template in enumerate(self.tree.templates):
            if template:
                json_template = json.loads(
                    self.tree.gen_template(template_id).format_as_example(
                        regex=True, placeholder=True, whitespace=True, ids=True
                    )
                )
                templates.append(json_template)
                template_id_to_order[template_id] = len(templates) - 1

        var_mapping = {k: v.to_dict() for k, v in self.var_mapping.items()}
        if not include_embeddings:
            for v in var_mapping.values():
                if "embedding" in v:
                    del v["embedding"]

        json_parser = {
            "templates": templates,
            "var_mapping": var_mapping,
            "event_types": {
                template_id_to_order[k]: list(v)
                for k, v in self.event_types.items()
            },
        }

        return json_parser
