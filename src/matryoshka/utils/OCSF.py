import hashlib
import json
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import dill
import networkx as nx
import requests
import torch
from torch import Tensor

from matryoshka.classes import Mapping
from matryoshka.genai_api.api import Caller
from matryoshka.genai_api.classes import LLMTask
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.taxonomies.OCSF.descriptions import (
    gen_prompt as gen_desc_prompt,
)
from matryoshka.utils.prompts.taxonomies.OCSF.top_level_descriptions import (
    gen_prompt as gen_top_level_desc_prompt,
)
from matryoshka.utils.prompts.taxonomies.OCSF.types import (
    gen_prompt as gen_type_prompt,
)


def digest(val: Any) -> str:
    """Computes the MD5 hash of a given value."""
    if val:
        return hashlib.md5(str(val).encode("utf-8")).hexdigest()
    return "0"


class OCSFCache:
    """A simple file-based cache for API requests to avoid repeated network calls."""

    def __init__(self, path: str = ".OCSF_cache/API_cache/"):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache: Dict[str, Any] = {}
        for file in os.listdir(path):
            try:
                with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                    content = json.load(f)
                    self.cache[content["key"]] = content["value"]
            except (json.JSONDecodeError, UnicodeDecodeError):
                get_logger().warning("Could not load cache file: %s" % file)

    def __call__(self, url: str) -> Any:
        """Fetches from cache, or makes an HTTP request if the URL is not in the cache."""
        if url not in self.cache:
            get_logger().debug("Issuing request to %s", url)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            self.cache[url] = response.json()
            self._save(url)
        return self.cache[url]

    def _save(self, key: str):
        """Saves a specific cache entry to a file.
        Use a cryptographic hash for a stable filename"""
        filename = hashlib.sha256(key.encode()).hexdigest()
        with open(
            os.path.join(self.path, f"{filename}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"key": key, "value": self.cache[key]}, f, indent=2)


@dataclass
class OCSFAttribute:
    """A comprehensive representation of a single OCSF attribute, including its hierarchy, descriptions, type, and semantic embeddings."""

    # Identity & Hierarchy
    name: str  # Full path, e.g., "process_activity.process.name"
    local_name: str  # The last part of the path, e.g., "name"
    events: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    object_path: Optional[str] = None

    # Descriptions & Semantics
    description_list: List[str] = field(default_factory=list)
    local_description: str = ""
    global_description: str = ""
    embedding: List[float] = field(default_factory=list)

    # Type Information
    type: Optional[str] = None
    is_object: bool = False
    is_array: bool = False
    enum: Optional[Dict] = None

    # Metadata
    source: Optional[str] = None
    checksum: Optional[str] = (
        None  # Hash of core fields to detect upstream changes
    )
    sibling: Optional[str] = None

    def __post_init__(self):
        """Calculates the checksum after initialization if not provided."""
        if self.checksum is None:
            self.update_digest()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the dataclass to a dictionary, omitting None values."""
        return {
            k: v for k, v in asdict(self).items() if v is not None and v != []
        }

    def to_dict_simple(
        self,
        global_description=True,
        type=True,
        is_array=True,
        reduced_name=False,
    ):
        """Only keep the most important attributes"""
        obj = {"description": self.local_description, "name": self.name}
        if reduced_name:
            obj["name"] = self.name.split(".")[-1]
        if not self.children and reduced_name:
            obj["full_name"] = self.name
        if global_description and self.global_description:
            obj["global_description"] = self.global_description
        if type and self.type:
            obj["type"] = str(self.type)
        if is_array and self.is_array:
            obj["is_array"] = True

        return obj

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "OCSFAttribute":
        """Deserializes a dictionary into an OCSFAttribute object."""
        return cls(**data)

    def is_leaf(self) -> bool:
        """Returns True if the attribute is not an object type."""
        return not self.is_object

    def update_digest(self):
        """Recalculates and updates the checksum for the attribute."""
        self.checksum = OCSFAttribute.digest(self)

    @staticmethod
    def digest(obj: "OCSFAttribute") -> str:
        """Creates a stable hash based on the attribute's primary defining fields."""
        desc_list_digest = "".join(
            digest(d.strip().lower()) for d in obj.description_list
        )
        id_string = (
            digest(desc_list_digest)
            + digest(obj.type)
            + digest(obj.is_object)
            + digest(obj.is_array)
            + digest(obj.name)
            + digest(obj.local_name)
            + digest(obj.sibling)
            + digest(json.dumps(obj.enum, sort_keys=True))
        )
        return digest(id_string)

    @staticmethod
    def generate_description(
        attributes,
        caller,
        tqdm=False,
        model="gemini-2.5-flash",
        save_to_cache=None,
        **kwargs,
    ):
        """Generates a description for the attribute using the LLM caller."""
        if not isinstance(attributes, list):
            attributes = [attributes]

        attributes = [
            attribute
            for attribute in attributes
            if not attribute.global_description
        ]

        # Build tasks
        kwargs = {"n": 1}
        kwargs["temperature"] = 0.33
        tasks = []
        for f in attributes:
            user, system = gen_desc_prompt(f)
            task = LLMTask(
                system_prompt=system,
                max_tokens=256,
                model=model,
                message=user,
                thinking_budget=1,
                **kwargs,
            )
            tasks.append(task)

        # Chunk tasks into chunks of 10000
        chunks = [[]]
        for task in tasks:
            if len(chunks[-1]) >= 100:
                chunks.append([])
            chunks[-1].append(task)

        for chunk_id, chunk in enumerate(chunks):
            # Run tasks
            get_logger().info(
                f"Running tasks %d out of %d ...", 1 + chunk_id, len(chunks)
            )
            responses = caller(chunk, use_tqdm=tqdm)
            for r_id_raw, resp_array in enumerate(responses):
                r_id = +chunk_id * 100
                if resp_array:
                    resp = resp_array.candidates[0]
                    try:
                        desc = json.loads(resp)
                    except json.JSONDecodeError:
                        desc = resp
                    attributes[r_id].global_description = desc
                else:
                    attributes[r_id].global_description = ""

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


class OCSFAttributeEncoder(json.JSONEncoder):
    """Custom JSON encoder for the OCSFAttribute dataclass."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, OCSFAttribute):
            return obj.to_dict()
        return super().default(obj)


class OCSFSchemaClient:
    BASE_URL = "https://schema.ocsf.io/api"

    def __init__(
        self,
        caller,
        saved_path=".OCSF_cache/",
        load_path=None,
        model="gemini-2.5-flash",
    ):
        self.basic_types = None
        self.classes = None
        self.objects = {}
        self.attributes = {}
        self.object_list = {}
        self.categories = {}
        self.class_details = {}
        self.class_objects = {}
        self.descriptions = {}
        self.source_to_event_mapping = {}
        self.model = model

        self.unmapped_attributes = {}

        # Cache for generated descriptions
        self.generated_descriptions = {}

        # Maps path to OCSFFieldDescriptor object
        self.mislabeled_categories = {
            "finding": "findings",
            "remediation_activity": "remediation",
        }

        # Cache for inferred types
        self.inferred_types = {}

        self.cache = OCSFCache(os.path.join(saved_path, "API_cache"))
        os.makedirs(saved_path, exist_ok=True)
        self.saved_path = saved_path
        if not load_path:
            load_path = saved_path
        self._load_from_cache(
            path=load_path, erase_embeddings=load_path != saved_path
        )

        if caller:
            if not self.attributes:
                print("Building attribute map")
                self.build_attribute_map(caller)

            print("Building descriptions")
            self.build_descriptions(caller)

            print("Building inferred types")
            self.build_inferred_types(caller)

            print("Building attribute embeddings")
            self.build_attribute_embeddings(caller)

    def _load_from_cache(self, path, erase_embeddings=False):
        attr_path = f"{path}/attributes.json"
        if os.path.exists(attr_path):
            with open(attr_path, "r", encoding="utf-8") as f:
                raw_attrs = json.load(f)
                self.attributes = {
                    k: OCSFAttribute.from_json(v) for k, v in raw_attrs.items()
                }
            get_logger().info(
                "Loaded %d attributes from cache.", len(self.attributes)
            )

        types_path = os.path.join(path, "inferred_types.json")
        if os.path.exists(types_path):
            with open(types_path, "r", encoding="utf-8") as f:
                self.inferred_types = json.load(f)
            get_logger().info(
                "Loaded %d inferred type entries from cache.",
                len(self.inferred_types),
            )

        if erase_embeddings:
            for attr in self.attributes.values():
                attr.embedding = []
            get_logger().info("Erased all attribute embeddings.")

    def _save_to_cache(self):
        """Saves the current state of attributes and inferred types to cache files."""
        attr_path = os.path.join(self.saved_path, "attributes.json")
        with open(attr_path, "w", encoding="utf-8") as f:
            json.dump(self.attributes, f, cls=OCSFAttributeEncoder, indent=2)

        types_path = os.path.join(self.saved_path, "inferred_types.json")
        with open(types_path, "w", encoding="utf-8") as f:
            json.dump(self.inferred_types, f, indent=2)
        get_logger().info("Saved OCSF caches to disk.")

    def get_data_types(self):
        url = f"{self.BASE_URL}/data_types"
        return self.cache(url)

    def get_basic_types(self):
        """Queries the data types and removes 'description' and 'regex' attributes"""
        if self.basic_types:
            return self.basic_types

        data = self.get_data_types()

        def remove_keys(obj):
            if isinstance(obj, dict):
                return {
                    k: remove_keys(v)
                    for k, v in obj.items()
                    if k not in ["regex"]
                }
            elif isinstance(obj, list):
                return [remove_keys(item) for item in obj]
            else:
                return obj

        cleaned_data = remove_keys(data)
        self.basic_types = cleaned_data
        return cleaned_data

    def get_object_list(self):
        """Queries the list of objects."""
        if self.object_list:
            return self.object_list
        url = f"{self.BASE_URL}/objects"
        response = self.cache(url)
        self.object_list = {v["name"]: v for v in response}
        return self.object_list

    def _build_class_path(self, resp):
        if "extension" not in resp:
            return resp["name"]
        return f"{resp['extension']}/{resp['name']}"

    def get_classes(self):
        """Queries and returns the list of classes."""
        if self.classes:
            return self.classes
        url = f"{self.BASE_URL}/classes"
        response = self.cache(url)
        self.classes = {v["name"]: v for v in response}
        return self.classes

    def get_categories(self):
        """Queries and returns the list of categories."""
        if self.categories:
            return self.categories
        url = f"{self.BASE_URL}/categories"
        response = self.cache(url)
        self.categories = response["attributes"]
        return self.categories

    def get_class_details(self, class_name):
        """Queries and returns details about a specific class."""
        if class_name in self.class_details:
            return self.class_details[class_name]
        class_details = self.get_classes().get(class_name, None)
        if class_details:
            class_name = self._build_class_path(class_details)
        url = f"{self.BASE_URL}/classes/{class_name}"
        response = self.cache(url)
        self.class_details[class_name] = response
        return self.class_details[class_name]

    def build_top_level_descriptions(self, caller, **kwargs):
        """Builds the top level descriptions for all classes."""
        events = self.get_classes()
        attributes = defaultdict(dict)
        source_descriptions = {}
        for event in events:
            raw_event_attributes = self.get_class_details(event).get(
                "attributes", {}
            )
            event_attributes = {}
            for attr in raw_event_attributes:
                for attr_name, attr_value in attr.items():
                    event_attributes[attr_name] = attr_value

            for key, value in event_attributes.items():
                source = value.get("_source", event)
                attributes[key][source] = value.get("description", "")
                source_descriptions[source] = self.get_event_description(source)
                if key not in self.attributes:
                    self.attributes[key] = OCSFAttribute(
                        local_name=key,
                        name=key,
                        events=[],
                        parent=None,
                        type=value.get("object_type", value.get("type", None)),
                        is_object="object_type" in value,
                        is_array=value.get("is_array", False),
                        enum=value.get("enum", None),
                        sibling=value.get("sibling", None),
                        object_path=key if "object_type" not in value else None,
                    )
                self.attributes[key].events.append(event)

        tasks = []
        attribute_order = sorted(attributes.keys())
        for key in attribute_order:
            value = attributes[key]
            message, history, system = gen_top_level_desc_prompt(
                key,
                value,
                source_descriptions,
            )
            task = LLMTask(
                system_prompt=system,
                max_tokens=256,
                model=self.model,
                message=message,
                history=history,
                thinking_budget=1,
                **kwargs,
            )
            tasks.append(task)

        responses = caller(tasks, use_tqdm=True)
        for key, response in zip(attribute_order, responses):
            resp = response.candidates[0]
            try:
                desc = json.loads(resp)
            except json.JSONDecodeError:
                desc = resp
            self.attributes[key].global_description = desc
            self.attributes[key].local_description = desc
            self.attributes[key].description_list = [desc]

    def build_attributes_recursive(
        self,
        object_name,
        path="",
        events=None,
        description_list=None,
        seen_types=None,
    ):
        """Queries and returns details about a specific object."""
        if not events:
            events = []
        if not description_list:
            description_list = []
        if not seen_types:
            seen_types = set()

        url = f"{self.BASE_URL}/objects/{object_name}"
        response = {
            k: v
            for attr in self.cache(url).get("attributes", [])
            for k, v in attr.items()
        }
        if "error" in response or "deprecated" in response:
            return

        for attr_name, attr in response.items():
            sibling = attr.get("sibling", None)
            if sibling and path:
                sibling = f"{path}.{sibling}"
            ocsf_attr = OCSFAttribute(
                local_name=attr_name,
                name=attr_name if not path else f"{path}.{attr_name}",
                events=events,
                parent=path if path else None,
                type=attr.get("object_type", attr.get("type", None)),
                is_object="object_type" in attr,
                is_array=attr.get("is_array", False),
                enum=attr.get("enum", None),
                sibling=sibling,
                description_list=description_list[:]
                + [attr.get("description", "")],
                local_description=attr.get("description", ""),
                object_path=(
                    f"{object_name}.{attr_name}"
                    if "object_type" not in attr
                    else None
                ),
            )
            self.attributes[ocsf_attr.name] = ocsf_attr
            if not ocsf_attr.is_object or ocsf_attr.type in seen_types:
                continue
            new_seen_types = seen_types.copy()
            new_seen_types.add(ocsf_attr.type)
            self.build_attributes_recursive(
                ocsf_attr.type,
                path=ocsf_attr.name,
                events=events,
                description_list=ocsf_attr.description_list,
                seen_types=new_seen_types,
            )

    def build_attribute_map(self, caller, **kwargs):
        """Build the top level descriptions"""
        self.attributes = {}
        self.build_top_level_descriptions(caller, **kwargs)

        # Iterate over all top level attributes
        top_level_attributes = list(self.attributes.keys())
        for attr_name in top_level_attributes:
            attr = self.attributes[attr_name]
            if not attr.is_object:
                continue

            self.build_attributes_recursive(
                attr.type,
                path=attr_name,
                events=attr.events,
                description_list=attr.description_list,
            )

        # Add children relationships
        for attr_name, attr in self.attributes.items():
            parent = attr.parent
            if not parent:
                continue
            parent_attr = self.attributes[parent]
            parent_attr.children.append(attr_name)

    def get_event_description(self, source):
        """
        Queries the description of a specific event or category.
        """
        classes, categories = self.get_classes(), self.get_categories()
        if source in classes:
            return classes[source]["description"]
        elif source in categories:
            return categories[source]["description"]
        else:
            return ""

    def source_type(self, type):
        """
        Returns the source type of the given type.
        """
        types = self.get_basic_types()
        if type not in types:
            return type

        return types[type].get("type", type)

    def build_descriptions(self, caller):
        """
        Builds the descriptions for all objects.
        """
        get_logger().info("Building attribute descriptions...")
        path_blocklist = ["category_name", "class_name", "observable"]
        paths = []
        for key, attr in self.attributes.items():
            if any(p in key for p in path_blocklist):
                continue
            if attr.is_object:
                continue
            if attr.global_description:
                continue
            paths.append(key)

        if paths:
            OCSFAttribute.generate_description(
                [self.attributes[p] for p in paths],
                caller,
                tqdm=True,
                save_to_cache=lambda: self._save_to_cache(),
                model=self.model,
            )

            # Save the descriptions
            self._save_to_cache()

    def get_description(self, events, target_class=None, fuzzy=0):
        """
        Returns the descriptions of each field in a list of events.
        """
        if isinstance(target_class, str):
            target_class = [target_class]
        if isinstance(events, str):
            events = [events]

        def fuzzy_match(fuzzy):
            if target_class is None:
                return lambda x: True
            if fuzzy == 2:
                return lambda x: True
            elif fuzzy == 1:
                return lambda x: any(
                    self.source_type(v) == self.source_type(t)
                    for t in target_class
                    for v in x
                )
            else:
                return lambda x: any(v == t for t in target_class for v in x)

        rtn = {}
        for key, attr in self.attributes.items():
            if not attr.global_description:
                continue
            if all(event not in attr.events for event in events):
                continue

            types = self.inferred_types.get(attr.object_path, [attr.type])
            if fuzzy_match(fuzzy)(types):
                rtn[key] = attr.global_description

        return rtn

    def build_inferred_types(self, caller, tqdm=True, **kwargs):
        """
        Builds the inferred types for all objects.
        """
        get_logger().info("Building inferred types...")
        # Get the list of objects, basic types and event classes
        basic_types = self.get_basic_types()
        all_object_paths = set()
        for attr in self.attributes.values():
            if attr.object_path is not None:
                all_object_paths.add(attr.object_path)

        new_paths = {}
        for object_path in all_object_paths:
            if not object_path:
                continue

            if any(
                p in object_path for p in ["observable", "object", "enrichment"]
            ):
                continue

            if object_path in new_paths or object_path in self.inferred_types:
                continue

            if "." not in object_path:
                new_paths[object_path] = (
                    object_path,
                    [None, self.attributes[object_path].local_description],
                    self.attributes[object_path].type,
                )
                continue

            object_name, attr_name = tuple(object_path.split("."))
            url = f"{self.BASE_URL}/objects/{object_name}"
            response = self.cache(url)
            if "error" in response or "deprecated" in response:
                continue

            obj_desc = response.get("description", "")

            attributes = {
                k: v
                for attr in response.get("attributes", [])
                for k, v in attr.items()
            }

            if attr_name not in attributes:
                continue

            attr = attributes[attr_name]
            attr_desc = attr.get("description", "")
            attr_type = attr.get("type", None)

            new_paths[object_path] = (
                object_path,
                [obj_desc, attr_desc],
                attr_type,
            )

        # Infer the types
        if new_paths:
            tasks = []
            kwargs["n"] = 1
            kwargs["temperature"] = 0.33
            new_path_list = []
            for new_path, value in new_paths.items():
                user, system = gen_type_prompt(*value, self)
                task = LLMTask(
                    system_prompt=system,
                    max_tokens=128,
                    model=self.model,
                    thinking_budget=1,
                    message=user,
                    **kwargs,
                )
                tasks.append(task)
                new_path_list.append(new_path)

            responses = caller(tasks, use_tqdm=tqdm)
            regexp = re.compile(r"\{.*\}")
            for r_id, resp_array in enumerate(responses):
                resp = resp_array.candidates[0]
                current_type = new_paths[new_path_list[r_id]][-1]
                try:
                    desc = json.loads(resp)
                except json.JSONDecodeError:
                    content = [l for l in resp.split("\n") if l.strip()]
                    if not content:
                        desc = [current_type]
                    else:
                        content = content[0]
                        match = regexp.search(content)
                        if not match:
                            desc = [current_type]
                        else:
                            content = match.group()
                            try:
                                desc = json.loads(content)
                            except json.JSONDecodeError:
                                desc = [current_type]

                if current_type == "string_t":
                    desc = [d for d in desc if d in basic_types]
                else:
                    desc = [
                        d for d in desc if d in basic_types and d != "string_t"
                    ]

                self.inferred_types[new_path_list[r_id]] = desc

            self._save_to_cache()

    def build_attribute_embeddings(self, caller):
        get_logger().info("Building attribute embeddings...")

        targets = []
        for key, attr in self.attributes.items():
            if attr.global_description and not attr.embedding:
                targets.append(key)

        if targets:
            OCSFAttribute.build_attribute_embeddings(
                [self.attributes[t] for t in targets],
                caller,
                tqdm=True,
                save_to_cache=lambda: self._save_to_cache(),
            )
            self._save_to_cache()

    def get_siblings(self, fields, target_type, fuzzy=0):
        if not isinstance(target_type, list):
            target_type = [target_type]

        if not isinstance(fields, list):
            fields = [fields]

        def fuzzy_match(fuzzy):
            if fuzzy == 2:
                return lambda x: True
            elif fuzzy == 1:
                return lambda x: any(
                    self.source_type(v) == self.source_type(t)
                    for t in target_type
                    for v in x
                )
            else:
                return lambda x: any(v == t for t in target_type for v in x)

        parents = list(".".join(f.split(".")[:-1]) for f in fields)
        siblings = []
        for parent in parents:
            if parent not in self.attributes:
                continue
            attr = self.attributes[parent]
            children = attr.children
            for child_path in children:
                if child_path not in self.attributes:
                    continue
                child = self.attributes[child_path]
                if child.is_object:
                    continue
                if not child.global_description:
                    continue
                if not child.object_path:
                    continue
                if not fuzzy_match(fuzzy)(
                    self.inferred_types.get(child.object_path, [child.type])
                ):
                    continue
                siblings.append(child_path)

        return siblings

    def attribute_similarity(self, attr_a, attr_b):
        """Compute attribute similarity"""
        path_a, path_b = attr_a.split("."), attr_b.split(".")
        shortest_path, longest_path = min(len(path_a), len(path_b)), max(
            len(path_a), len(path_b)
        )
        i = 0
        for i in range(shortest_path):
            if path_a[i] != path_b[i]:
                break
        return max(0, (i - 1)) / max(1, longest_path - 1)

    def create_networkx_graph(
        self, field_subset=None, add_siblings=False, max_sibling_depth=-1
    ):
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

        # Create nodes and edges
        for field in field_subset:
            field_ancestry = []
            for node in field.split("."):
                if not field_ancestry:
                    field_ancestry.append(node)
                else:
                    field_ancestry.append(field_ancestry[-1] + "." + node)

            parent = "event"
            for partial_field in field_ancestry:
                if partial_field in nodes:
                    parent = partial_field
                    continue
                add_node(partial_field)
                edges.append((parent, partial_field))
                parent = partial_field

            field_depth = len(field_ancestry) - 1
            if add_siblings and (
                max_sibling_depth == -1 or field_depth < max_sibling_depth
            ):
                additional_fields = self.get_siblings(
                    field, target_type="any", fuzzy=2
                )
                parent = (
                    field_ancestry[-2] if len(field_ancestry) > 1 else "event"
                )
                for field in additional_fields:
                    if field in nodes:
                        continue
                    add_node(field)
                    edges.append((parent, field))

        # 1. Create a directed graph from the edge list
        graph = nx.DiGraph(edges)

        # 2. Add node attributes from the node attribute dictionary
        nx.set_node_attributes(graph, nodes)

        # 3. Return the populated graph object
        return graph


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
        try:
            return {
                "field_description": self.field_description,
                "cluster_description": self.cluster_description,
                "orig_node": self.orig_node,
                "mapping": self.mapping.to_dict() if self.mapping else {},
                "created_attribute": self.created_attribute,
                "embedding": (
                    self.embedding.tolist()
                    if self.embedding is not None
                    else None
                ),
            }
        except Exception as e:
            breakpoint()


@dataclass
class OCSFMapping:
    field_list: List[str] = field(default_factory=list)
    demonstration: str = ""
    candidates: List[str] = field(default_factory=list)
    type: str = "OCSF"

    def __str__(self):
        return f"""{self.type} field list: {json.dumps(self.field_list)}"""

    def to_dict(self):
        return {"field_list": self.field_list, "type": self.type}
