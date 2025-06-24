import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class SchemaEntry:
    """Schema for a single log entry."""

    name: str
    description: str = ""
    orig_ids: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert the schema entry to a JSON string."""
        return json.dumps(
            {
                "description": self.description,
                "ids": self.orig_ids,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str, name: str) -> "SchemaEntry":
        """Load a schema entry from a JSON string."""
        if isinstance(json_str, str):
            data = json.loads(json_str)
        else:
            data = json_str
        return cls(
            name=name,
            description=data["description"],
            orig_ids=data["ids"],
        )


@dataclass
class Schema:
    """Schema for a single log entry."""

    orig_template: int = -1
    fields: Dict[str, SchemaEntry] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert the schema to a JSON string."""
        return json.dumps(
            {
                k: json.loads(v.to_json())
                for k, v in sorted(
                    list(self.fields.items()), key=lambda x: min(x[1].orig_ids)
                )
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str, orig_template) -> "Schema":
        """Load a schema from a JSON string."""
        if isinstance(json_str, str):
            data = json.loads(json_str)
        else:
            data = json_str
        fields = {
            k: SchemaEntry.from_json(v, name=k)
            for k, v in data.items()
            if v["ids"]
        }
        return cls(orig_template=orig_template, fields=fields)


@dataclass
class CreatedAttribute:
    name: str
    type: str
    description: str
    event: str
    embedding: Optional[torch.Tensor] = None


@dataclass
class OCSFMapping:
    field_list: List[str] = field(default_factory=list)
    demonstration: str = ""
    candidates: List[str] = field(default_factory=list)
    mapped: bool = False
    created_attribute_candidates: List[CreatedAttribute] = field(
        default_factory=list
    )
    created_attribute_demonstration: str = ""
    created_attribute: Optional[CreatedAttribute] = None
    event: str = ""

    def __str__(self):
        if not self.created_attribute:
            return f"""Field list for {self.event}: {json.dumps(self.field_list)}"""
        else:
            return f"""Created attribute for {self.event}: {self.created_attribute.name}"""


@dataclass
class VariableSemantics:
    field_description: str = ""
    cluster_description: str = ""
    orig_node: int = -1
    embedding: Optional[torch.Tensor] = None
    cluster_embedding: Optional[torch.Tensor] = None
    mappings: Dict[str, OCSFMapping] = field(default_factory=dict)
    created_attribute: Optional[str] = None

    def __str__(self):
        return (
            f"""\nVariable #{self.orig_node}: {self.field_description}\n"""
            + "\n".join(str(m) for m in self.mappings.values())
        )


@dataclass
class OCSFTemplateMapping:
    assignment: Dict[str, Dict[str, str]] = field(default_factory=dict)
    demonstrations: Dict[str, str] = field(default_factory=dict)
    candidates: Dict[str, Dict] = field(default_factory=dict)
    id: int = -1
    description: str = ""
    embedding: Optional[torch.Tensor] = None
