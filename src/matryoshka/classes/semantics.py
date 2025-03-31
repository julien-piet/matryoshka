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

    @classmethod
    def from_var_mapping(cls, template_id, template, var_mapping):
        """Load a schema from a variable mapping."""
        id_to_field = {}
        id_to_description = {}

        def add_description(id, description):
            if id not in id_to_description:
                id_to_description[id] = description
            elif id_to_description[id] != description:
                id_to_description[id] = ""

        for node in template.elements:
            if node.id in var_mapping:
                id_to_field[node.id] = var_mapping[node.id].created_attribute
                add_description(
                    var_mapping[node.id].created_attribute,
                    var_mapping[node.id].field_description,
                )
            elif not node.is_variable():
                id_to_field[node.id] = "SYNTAX"
                add_description("SYNTAX", "SYNTAX")

        fields = {
            v: {"description": None, "ids": []} for v in id_to_field.values()
        }
        for k, v in id_to_field.items():
            fields[v]["description"] = (
                id_to_description[v] if v in id_to_description else ""
            )
            fields[v]["ids"].append(k)

        return cls.from_json(
            fields,
            orig_template=template_id,
        )


@dataclass
class Mapping:
    field_list: List[str] = field(default_factory=list)
    demonstration: str = ""
    candidates: List[str] = field(default_factory=list)
    type: str = "OCSF"

    def __str__(self):
        return f"""{self.type} field list: {json.dumps(self.field_list)}"""

    def to_dict(self):
        return {"field_list": self.field_list, "type": self.type}


@dataclass
class VariableSemantics:
    field_description: str = ""
    cluster_description: str = ""
    orig_node: int = -1
    embedding: Optional[torch.Tensor] = None
    cluster_embedding: Optional[torch.Tensor] = None
    mapping: Optional[Mapping] = field(default_factory=dict)
    created_attribute: Optional[str] = None

    def __str__(self):
        base = f"""\nVariable #{self.orig_node}: {self.field_description}"""
        if self.mapping:
            base += "\n\t" + str(self.mapping)
        return base

    def to_dict(self):
        return {
            "field_description": self.field_description,
            "cluster_description": self.cluster_description,
            "orig_node": self.orig_node,
            "mapping": self.mapping.to_dict(),
            "created_attribute": self.created_attribute,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VariableSemantics":
        mapping_data = data.get("mapping", {})
        mapping = (
            Mapping(
                field_list=mapping_data.get("field_list", []),
                demonstration=mapping_data.get("demonstration", ""),
                candidates=mapping_data.get("candidates", []),
                type=mapping_data.get("type", "OCSF"),
            )
            if mapping_data
            else None
        )

        embedding = (
            torch.tensor(data["embedding"])
            if data.get("embedding") is not None
            else None
        )

        return cls(
            field_description=data.get("field_description", ""),
            cluster_description=data.get("cluster_description", ""),
            orig_node=data.get("orig_node", -1),
            embedding=embedding,
            mapping=mapping,
            created_attribute=data.get("created_attribute"),
        )


@dataclass
class OCSFTemplateMapping:
    assignment: Dict[str, Dict[str, str]] = field(default_factory=dict)
    demonstrations: Dict[str, str] = field(default_factory=dict)
    candidates: Dict[str, Dict] = field(default_factory=dict)
    id: int = -1
    description: str = ""
    embedding: Optional[torch.Tensor] = None
