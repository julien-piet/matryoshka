import json
from dataclasses import dataclass, field
from typing import Dict, List


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
