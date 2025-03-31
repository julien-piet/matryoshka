import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import networkx as nx
from torch import Tensor

import matryoshka.utils.UDM_extractor as extractor
from matryoshka.genai_api.api import Caller
from matryoshka.genai_api.classes import LLMTask
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.taxonomies.UDM.descriptions import (
    gen_prompt as gen_desc_prompt,
)


def digest(val):
    if val:
        return hashlib.md5(str(val).encode("utf-8")).hexdigest()
    else:
        return "0"


class UDMMetaType(Enum):
    OBJECT = 0
    STRING = 1
    NUMERIC = 2
    BOOL = 3
    ENUM = 4
    DATETIME = 5
    EXTERNAL = 6
    UNKNOWN = 7

    def __str__(self):
        if self == UDMMetaType.OBJECT:
            return "object"
        elif self == UDMMetaType.STRING:
            return "string"
        elif self == UDMMetaType.NUMERIC:
            return "numeric"
        elif self == UDMMetaType.BOOL:
            return "bool"
        elif self == UDMMetaType.ENUM:
            return "enum"
        elif self == UDMMetaType.EXTERNAL:
            return "external"
        elif self == UDMMetaType.DATETIME:
            return "datetime"
        else:
            return "unknown"

    @classmethod
    def from_string(cls, val):
        if "date" in val.lower() or "time" in val.lower():
            return UDMMetaType.DATETIME
        mapping = {
            "string": UDMMetaType.STRING,
            "bool": UDMMetaType.BOOL,
            "enum": UDMMetaType.ENUM,
            "object": UDMMetaType.OBJECT,
            "external": UDMMetaType.EXTERNAL,
            "numeric": UDMMetaType.NUMERIC,
            "int32": UDMMetaType.NUMERIC,
            "int64": UDMMetaType.NUMERIC,
            "uint32": UDMMetaType.NUMERIC,
            "uint64": UDMMetaType.NUMERIC,
            "double": UDMMetaType.NUMERIC,
            "float": UDMMetaType.NUMERIC,
            "bytes": UDMMetaType.STRING,
            "fixed32": UDMMetaType.NUMERIC,
            "sfixed32": UDMMetaType.NUMERIC,
            "fixed64": UDMMetaType.NUMERIC,
            "sfixed64": UDMMetaType.NUMERIC,
            "sint32": UDMMetaType.NUMERIC,
            "sint64": UDMMetaType.NUMERIC,
        }
        if val in mapping:
            return mapping[val]
        else:
            return UDMMetaType.UNKNOWN


@dataclass
class UDMAttributeDescription:
    description: str
    embedding: List[float] = field(default_factory=list)
    checksum: Optional[str] = None

    @classmethod
    def from_json(cls, json_data: str):
        try:
            data = json.loads(json_data)
        except:
            data = json_data
        return cls(**data)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class UDMAttribute:
    local_description: str
    name: str
    type: UDMMetaType = UDMMetaType.UNKNOWN
    is_object: bool = False
    is_array: bool = False
    enum_type: Optional[str] = None
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    description_list: List[str] = field(default_factory=list)
    global_description: Optional[str] = None
    embedding: Optional[List[float]] = None
    checksum: Optional[str] = None

    def __post_init__(self):
        if not self.description_list:
            self.description_list = [self.local_description]
        self.checksum = UDMAttribute.digest(self)

    def to_dict_simple(
        self,
        global_description=True,
        type=True,
        is_array=True,
        reduced_name=False,
    ):
        """Keep the most important attributes"""
        obj = {"description": self.local_description, "name": self.name}
        if reduced_name:
            obj["name"] = self.name.split(".")[-1]
        if not self.children and reduced_name:
            obj["full_name"] = self.name
        if global_description and self.global_description:
            obj["global_description"] = self.global_description
        if type and self.type and self.type != UDMMetaType.UNKNOWN:
            obj["type"] = str(self.type)
        if is_array and self.is_array:
            obj["is_array"] = True
        return obj

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def is_leaf(self):
        return not self.children

    def update_digest(self):
        self.checksum = UDMAttribute.digest(self)

    @classmethod
    def from_json(cls, json_data: str):
        """Deserializing from a JSON string back to a dataclass object"""
        data = json.loads(json_data)
        return cls(**data)

    @staticmethod
    def digest(obj):
        desc_digest = digest(
            "".join(digest(v.strip().lower()) for v in obj.description_list)
        )
        id_string = (
            desc_digest
            + digest(obj.type)
            + digest(obj.is_object)
            + digest(obj.is_array)
            + digest(obj.name)
        )
        return digest(id_string)

    @staticmethod
    def load_from_raw(raw_attributes, generated_descriptions=None):
        """
        Loads a list of UDMAttributes from a list of raw attributes.
        """
        if not generated_descriptions:
            generated_descriptions = {}

        # First load all the raw attributes without preserving relationships
        attributes = {}
        for attr, attr_dict in raw_attributes.items():
            if attr.startswith("event.additional"):
                continue
            enum_type = None
            attr_type = UDMMetaType.from_string(attr_dict["Type"])
            if attr_dict["Enum"]:
                attr_type = UDMMetaType.ENUM
                enum_type = attr_dict["Type"]
            elif attr_dict.get("Link", None):
                if (
                    "date" in attr_dict["Link"].lower()
                    or "time" in attr_dict["Link"].lower()
                ):
                    attr_type = UDMMetaType.DATETIME
                else:
                    attr_type = UDMMetaType.EXTERNAL
            attributes[attr] = UDMAttribute(
                local_description=attr_dict["Description"],
                name=attr,
                type=attr_type,
                is_object=attr_dict["Object"],
                is_array=attr_dict["Label"].lower() == "repeated",
                enum_type=enum_type,
            )

        # Add the parent relationships
        for attr_name in attributes:
            parent_path = None
            if "." in attr_name:
                parent_path = ".".join(attr_name.split(".")[:-1])
            if parent_path and parent_path in attributes:
                attributes[attr_name].parent = parent_path

        # Add the children relationships
        for attr_name, attr in attributes.items():
            if attr.parent:
                attributes[attr.parent].children.append(attr_name)

        # Generate the description lists
        queue = [
            attr_name
            for attr_name, attr in attributes.items()
            if not attr.parent
        ]
        while queue:
            attr_name = queue.pop()
            attr = attributes[attr_name]
            if not attr.parent:
                attr.description_list = [attr.local_description]
            else:
                attr.description_list = attributes[
                    attr.parent
                ].description_list[:] + [attr.local_description]
            attr.update_digest()
            queue.extend(attr.children)

        # Populate the global descriptions
        missing_descriptions = []
        for attr_name, attr in attributes.items():
            missing = False
            if not attr.is_leaf():
                continue
            if attr_name not in generated_descriptions:
                missing = True
            elif generated_descriptions[attr_name].checksum != attr.checksum:
                missing = True
            elif (
                not generated_descriptions[attr_name].description
                or not generated_descriptions[attr_name].embedding
            ):
                missing = True
            else:
                attr.global_description = generated_descriptions[
                    attr_name
                ].description
                attr.embedding = generated_descriptions[attr_name].embedding
            if missing:
                missing_descriptions.append(attr_name)

        get_logger().info(
            "Loaded %d attributes, %d missing descriptions",
            len(attributes),
            len(missing_descriptions),
        )
        return attributes, missing_descriptions

    @staticmethod
    def generate_description(
        fd,
        caller,
        tqdm=False,
        model="gemini-2.5-flash",
        save_to_cache=None,
        **kwargs,
    ):
        if not isinstance(fd, list):
            fd = [fd]

        fd = [f for f in fd if not f.global_description]

        # Build tasks
        kwargs["n"] = 1
        kwargs["temperature"] = 0.33
        tasks = []
        for f in fd:
            user, system = gen_desc_prompt(f)
            task = LLMTask(
                system_prompt=system,
                max_tokens=256,
                thinking_budget=0,
                model=model,
                message=user,
                **kwargs,
            )
            tasks.append(task)

        # Chunk tasks into chunks of 10000
        chunks = [[]]
        for task in tasks:
            if len(chunks[-1]) >= 10000:
                chunks.append([])
            chunks[-1].append(task)

        # Run tasks
        for chunk_id, chunk in enumerate(chunks):
            get_logger().info(
                "Running tasks %d out of %d", 1 + chunk_id, len(chunks)
            )
            responses = caller(chunk, use_tqdm=tqdm)
            for r_id_raw, resp_array in enumerate(responses):
                r_id = r_id_raw + chunk_id * 10000
                if resp_array:
                    resp = resp_array.candidates[0]
                    try:
                        desc = json.loads(resp)
                    except json.JSONDecodeError:
                        desc = resp
                    fd[r_id].global_description = desc
                else:
                    fd[r_id].global_description = ""

        if save_to_cache:
            save_to_cache()

    @staticmethod
    def build_attribute_embeddings(
        attributes, caller, tqdm=False, save_to_cache=None
    ):
        if not isinstance(attributes, list):
            attributes = [attributes]

        attributes = [
            f for f in attributes if not f.embedding and f.global_description
        ]
        if not attributes:
            return

        chunk_size = 100
        tasks = []
        for chunk_start in range(0, len(attributes), chunk_size):
            chunk = attributes[chunk_start : chunk_start + chunk_size]
            tasks.append(
                LLMTask(
                    message=[attr.global_description for attr in chunk],
                    query_type="embedding",
                    model="text-embedding-005",
                )
            )

        embeddings = caller(
            tasks, distribute_parallel_requests=False, use_tqdm=tqdm
        )
        embeddings = [
            emb for embedding_chunk in embeddings for emb in embedding_chunk
        ]
        for embedding_id, embedding in enumerate(embeddings):
            attributes[embedding_id].embedding = embedding

        if save_to_cache:
            save_to_cache()

    def gen(self, caller):
        self.generate_description(self, caller)

    def gen_embedding(self, caller):
        self.build_attribute_embeddings(self, caller)


class UDMAttributeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UDMAttribute) or isinstance(
            obj, UDMAttributeDescription
        ):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


class UDMSchemaClient:
    def __init__(self, caller, saved_path=".UDM_cache/"):
        # Init variables
        self.saved_path = saved_path
        self.generated_descriptions = {}
        os.makedirs(saved_path, exist_ok=True)
        self._load_from_cache()

        # Load UDM schema
        self.raw_attributes, self.enumerations, self.basic_types = (
            extractor.extract()
        )

        # Ingest UDM schema
        self.attributes, missing_descriptions = UDMAttribute.load_from_raw(
            self.raw_attributes, self.generated_descriptions
        )

        # Build descriptions
        if caller and missing_descriptions:
            print("Building descriptions")
            self.build_descriptions(caller, missing_descriptions)
            print("Building attribute embeddings")
            self.build_attribute_embeddings(caller, missing_descriptions)

    def _load_from_cache(self):
        path = f"{self.saved_path}/generated_descriptions.json"
        if os.path.exists(path):
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                self.generated_descriptions = json.load(f)
            self.generated_descriptions = {
                k: UDMAttributeDescription.from_json(v)
                for k, v in self.generated_descriptions.items()
            }

    def _save_to_cache(self, save_all=True):
        if save_all:
            _generated_descriptions = {
                k: UDMAttributeDescription(
                    description=v.global_description,
                    embedding=v.embedding,
                    checksum=UDMAttribute.digest(v),
                )
                for k, v in self.attributes.items()
                if v.global_description and v.embedding
            }
            with open(
                f"{self.saved_path}/generated_descriptions.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    _generated_descriptions,
                    f,
                    cls=UDMAttributeEncoder,
                    indent=2,
                )

    def build_descriptions(self, caller, missing_descriptions):
        """
        Generates the missing descriptions.
        """
        get_logger().info(
            "Building attribute descriptions for %d attributes",
            len(missing_descriptions),
        )
        UDMAttribute.generate_description(
            [self.attributes[k] for k in missing_descriptions],
            caller,
            tqdm=True,
            save_to_cache=lambda: self._save_to_cache(),
        )
        self._save_to_cache()

    def build_attribute_embeddings(self, caller, missing_descriptions):
        """
        Generates the missing embeddings.
        """
        get_logger().info("Building attribute embeddings...")
        UDMAttribute.build_attribute_embeddings(
            [self.attributes[k] for k in missing_descriptions],
            caller,
            tqdm=True,
            save_to_cache=lambda: self._save_to_cache(),
        )
        self._save_to_cache()

    def get_description(self, target_types=None, fuzzy=False):
        """
        Returns the descriptions of each field in a list of events.
        """

        def fuzzy_match(fuzzy):
            if target_types is None:
                return lambda x: True
            if fuzzy:
                return lambda x: True
            else:
                return lambda x: any(x == t for t in target_types)

        return {
            attr_name: attr.global_description
            for attr_name, attr in self.attributes.items()
            if attr.global_description
            and attr.embedding
            and fuzzy_match(fuzzy)(attr.type)
        }

    def get_siblings(self, attributes, target_types=None, fuzzy=False):
        """
        Returns the siblings of a given field that match a given type.
        """
        if target_types and not isinstance(target_types, list):
            target_types = [target_types]

        if attributes and not isinstance(attributes, list):
            attributes = [attributes]

        def fuzzy_match(fuzzy):
            if not target_types:
                return lambda x: True
            if fuzzy:
                return lambda x: True
            else:
                return lambda x: any(x == t for t in target_types)

        parents = set()
        for attr_name in attributes:
            if attr_name not in self.attributes:
                continue
            if self.attributes[attr_name].parent:
                parents.add(self.attributes[attr_name].parent)

        siblings = set()
        for parent in parents:
            for child in self.attributes[parent].children:
                child_attr = self.attributes[child]
                if not child_attr.is_leaf():
                    continue
                if not fuzzy_match(fuzzy)(child_attr.type):
                    continue
                siblings.add(child)
        return list(siblings)

    def attribute_similarity(self, attr_a, attr_b):
        """
        Compute attribute similarity based on their common prefix.
        """
        path_a, path_b = attr_a.split("."), attr_b.split(".")
        shortest_path, longest_path = min(len(path_a), len(path_b)), max(
            len(path_a), len(path_b)
        )
        i = 0
        for i in range(shortest_path):
            if path_a[i] != path_b[i]:
                break
        return max(0, i - 1) / max(1, longest_path - 1)

    def create_networkx_graph(self, field_subset=None, add_siblings=False):
        nodes, edges = {}, []

        def add_node(node_name):
            if node_name not in self.attributes:
                raise ValueError(f"Node {node_name} does not exist")
            nodes[node_name] = self.attributes[node_name].to_dict_simple(
                global_description=False, reduced_name=True
            )

        if not field_subset:
            field_subset = list(self.attributes.keys())

        nodes["event"] = {"name": "event"}

        for field in field_subset:
            field_ancestry = []
            for node in field.split("."):
                if not field_ancestry:
                    field_ancestry.append(node)
                else:
                    field_ancestry.append(field_ancestry[-1] + "." + node)

            if len(field_ancestry) < 2:
                continue

            parent = "event"
            for partial_field in field_ancestry[1:]:
                if partial_field in nodes:
                    parent = partial_field
                    continue
                add_node(partial_field)
                edges.append((parent, partial_field))
                parent = partial_field

            if add_siblings:
                additional_fields = self.get_siblings(field)
                parent = field_ancestry[-2]
                for field in additional_fields:
                    if field in nodes:
                        continue
                    add_node(field)
                    edges.append((parent, field))

        # Add enumeration information
        for key, value in nodes.items():
            if (
                key in self.attributes
                and self.attributes[key].type == UDMMetaType.ENUM
            ):
                value["is_enum"] = True

        # 1. Create a directed graph from the edge list
        graph = nx.DiGraph(edges)

        # 2. Add node attributes from the node attribute dictionary
        nx.set_node_attributes(graph, nodes)

        # 3. Return the populated graph object
        return graph

    def get_classes(self):
        return {"event": 0}
