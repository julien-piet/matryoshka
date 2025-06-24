import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
import torch

from ..genai_api.api import LLMTask
from .logging import get_logger
from .prompts.OCSF.descriptions import gen_prompt as gen_desc_prompt
from .prompts.OCSF.types import gen_prompt as gen_type_prompt


class OCSFCache:
    def __init__(self, path=".OCSF_cache/API_cache/"):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache = {}
        for file in os.listdir(path):
            with open(f"{path}/{file}", "r", encoding="utf-8") as f:
                content = json.load(f)
                self.cache[content["key"]] = content["value"]

    def __call__(self, url, debug=False):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if debug:
            breakpoint()

        if url not in self.cache:
            get_logger().debug("Issuing request to %s", url)
            response = requests.get(url, headers=headers)
            self.cache[url] = response.json()
            self.save(url)
        return self.cache[url]

    def save(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            with open(
                f"{self.path}/{hash(key)}.json", "w", encoding="utf-8"
            ) as f:
                json.dump({"key": key, "value": self.cache[key]}, f, indent=2)


def digest(val):
    if val:
        return hashlib.md5(str(val).encode("utf-8")).hexdigest()
    else:
        return "0"


@dataclass
class OCSFObject:
    description: str
    type: Optional[str] = None
    is_object: Optional[bool] = None
    is_array: Optional[bool] = None
    name: Optional[str] = None
    source: Optional[str] = None
    enum: Optional[Dict] = None
    sibling: Optional[str] = None
    attributes: Optional[Dict] = None
    path: Optional[str] = None
    description_list: Optional[List[str]] = None

    def to_dict_simple(self):
        # Only keep the most important attributes
        attr_list = ["description", "type", "is_array", "name", "enum"]
        return {k: v for k, v in self.__dict__.items() if k in attr_list}

    def to_dict(self):
        # Using asdict from dataclasses which will traverse the dataclass and ensure nested structures are also transformed into dictionaries/lists.
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_json(cls, json_data: str):
        # Deserializing from a JSON string back to a dataclass object
        data = json.loads(json_data)
        return cls(**data)

    def flattened_copy(self, depth=1):
        # Create a deep copy of the object and flatten the attributes
        obj = deepcopy(self)
        if obj.attributes and depth > 0:
            obj.attributes = {
                k: v.flattened_copy(depth - 1)
                for k, v in obj.attributes.items()
                if v is not None
            }
        elif obj.attributes:
            obj.attributes = None
        return obj

    def strip(self, remove_attributes=False):
        obj = deepcopy(self)
        obj.name = None
        obj.source = None
        obj.sibling = None
        obj.is_object = None
        obj.is_array = None
        obj.type = None
        if obj.attributes and not remove_attributes:
            obj.attributes = {
                k: v.strip() if isinstance(v, OCSFObject) else v
                for k, v in obj.attributes.items()
            }
        else:
            obj.attributes = None
        return obj

    def digest(self):
        id_string = (
            digest(self.description.strip().lower())
            + digest(self.type)
            + digest(self.is_object)
            + digest(self.is_array)
            + digest(self.name)
            + digest(self.source)
            + digest(self.sibling)
        )
        id_string += digest(json.dumps(self.enum))
        flat_hash = digest(id_string)

        if not self.attributes:
            return flat_hash
        else:
            attr_hash = sorted(
                [digest(k) + v.digest() for k, v in self.attributes.items()]
            )
            attr_hash = digest("".join(attr_hash))
            return digest(flat_hash + attr_hash)


class OCSFObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, OCSFObject) or isinstance(obj, OCSFFieldDescriptor):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


@dataclass
class OCSFFieldDescriptor:
    path: str
    original_descriptions: List[str] = field(default_factory=list)
    generated_description: str = ""
    checksum: str = ""
    leaf: bool = False
    parent_type: Optional[str] = None
    type: Optional[str] = None
    embedding: List[float] = field(default_factory=list)

    @staticmethod
    def digest(original_descriptions, obj):
        desc_digest = digest(
            "".join(digest(v.strip().lower()) for v in original_descriptions)
        )
        return digest(desc_digest + obj.digest())

    def __init__(
        self,
        path,
        original_descriptions,
        obj=None,
        checksum=None,
        generated_description="",
        leaf=False,
        embedding=None,
        parent_type=None,
        type=None,
    ):
        self.path = path
        self.original_descriptions = original_descriptions
        self.checksum = (
            self.digest(original_descriptions, obj) if obj else checksum
        )
        self.generated_description = generated_description
        if obj is not None:
            self.leaf = not obj.is_object
            self.type = obj.type
        else:
            self.leaf = leaf
            self.type = type
        self.embedding = embedding
        self.parent_type = parent_type

    def update(self, original_descriptions, obj, parent_type=None):
        new_digest = self.digest(original_descriptions, obj)
        if new_digest != self.checksum:
            self.original_descriptions = original_descriptions
            self.checksum = new_digest
            self.generated_description = ""
            self.leaf = not obj.is_object
            self.embedding = []
            self.parent_type = parent_type
            self.type = obj.type

    def is_equal(self, other):
        return self.checksum == other.checksum

    @staticmethod
    def generate_description(
        fd,
        caller,
        tqdm=False,
        model="gemini-2.5-flash",
        save_to_cache=None,
        chunk_size=100000,
        **kwargs,
    ):
        if not isinstance(fd, list):
            fd = [fd]

        fd = [f for f in fd if not f.generated_description]

        if not fd:
            return False

        # Build tasks
        kwargs["n"] = 1
        kwargs["temperature"] = 0.33
        tasks = []
        for f in fd:
            user, system = gen_desc_prompt(f)
            task = LLMTask(
                system_prompt=system,
                max_tokens=128,
                model=model,
                message=user,
                **kwargs,
            )
            tasks.append(task)

        # Run tasks
        chunks = [[]]
        for task in tasks:
            if len(chunks[-1]) >= chunk_size:
                chunks.append([])
            chunks[-1].append(task)

        for chunk_id, chunk in enumerate(chunks):
            get_logger().info(
                "Generating descriptions for chunk %d/%d",
                chunk_id + 1,
                len(chunks),
            )
            responses = caller(chunk, use_tqdm=tqdm)
            for r_id_raw, resp_array in enumerate(responses):
                r_id = chunk_id * chunk_size + r_id_raw
                if resp_array:
                    resp = resp_array.candidates[0]
                    desc = ""
                    try:
                        desc = json.loads(resp)
                    except json.JSONDecodeError:
                        desc = resp

                    fd[r_id].generated_description = desc
                else:
                    get_logger().error(
                        "Failed to generate description for %s", fd[r_id].path
                    )
                    fd[r_id].generated_description = "No description available."

            if save_to_cache:
                save_to_cache()

        return True

    def gen(self, caller):
        self.generate_description(self, caller)

    def to_dict(self):
        # Using asdict from dataclasses which will traverse the dataclass and ensure nested structures are also transformed into dictionaries/lists.
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_json(cls, json_data: str):
        # Deserializing from a JSON string back to a dataclass object
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        return cls(**data)


class OCSFSchemaClient:
    BASE_URL = "https://schema.ocsf.io/api"

    def __init__(self, caller, saved_path=".OCSF_cache/"):
        self.basic_types = None
        self.classes = None
        self.objects = {}
        self.object_list = {}
        self.categories = {}
        self.class_details = {}
        self.class_objects = {}
        self.descriptions = {}
        self.source_to_event_mapping = {}

        self.unmapped_attributes = {}

        # Cache for generated descriptions
        self.generated_descriptions = (
            {}
        )  # Maps path to OCSFFieldDescriptor object
        self.mislabeled_categories = {
            "finding": "findings",
            "remediation_activity": "remediation",
        }

        # Cache for inferred types
        self.inferred_types = {}

        self.cache = OCSFCache(os.path.join(saved_path, "API_cache"))

        os.makedirs(saved_path, exist_ok=True)
        self.saved_path = saved_path
        self._load_from_cache()
        if caller:
            self.build_descriptions(caller)
            self.build_inferred_types(caller)
            self.build_attribute_embeddings(caller)

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
                k: OCSFFieldDescriptor.from_json(v)
                for k, v in self.generated_descriptions.items()
            }

        path = f"{self.saved_path}/inferred_types.json"
        if os.path.exists(path):
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                self.inferred_types = json.load(f)

        path = f"{self.saved_path}/created_attributes.json"
        if os.path.exists(path):
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                self.unmapped_attributes = json.load(f)

    def _save_to_cache(self, save_all=True):

        if save_all:
            with open(
                f"{self.saved_path}/generated_descriptions.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.generated_descriptions,
                    f,
                    cls=OCSFObjectEncoder,
                    indent=2,
                )

            with open(
                f"{self.saved_path}/inferred_types.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.inferred_types, f, indent=2)

    def get_data_types(self):
        url = f"{self.BASE_URL}/data_types"
        return self.cache(url)

    def get_basic_types(self):
        """Queries the data types and removes 'description' and 'regex' attributes."""
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
        url = f"{self.BASE_URL}/classes/{class_name}"
        response = self.cache(url)
        self.class_details[class_name] = response
        return self.class_details[class_name]

    def get_object_description(self, object_name):
        """Queries and returns details about a specific object."""
        if object_name in self.descriptions:
            return self.descriptions[object_name]

        url = f"{self.BASE_URL}/objects/{object_name}"
        response = self.cache(url)

        if "error" in response or "@deprecated" in response:
            return None

        obj = OCSFObject(
            name=response.get("name", object_name),
            type=response.get("object_type", response.get("type", None)),
            is_object="object_type" in response,
            is_array=response.get("is_array", False),
            description=response.get("description", ""),
            source=response.get("_source", None),
            enum=response.get("enum", None),
            sibling=response.get("sibling", None),
        )

        attributes = {
            k: v
            for attr in response.get("attributes", [])
            for k, v in attr.items()
        }

        obj.attributes = {}
        for attr_name, attr in attributes.items():
            attr = OCSFObject(
                name=attr_name,
                type=attr.get("object_type", attr.get("type", None)),
                is_object="object_type" in attr,
                is_array=attr.get("is_array", False),
                description=attr.get("description", ""),
                source=attr.get("_source", None),
                enum=attr.get("enum", None),
                sibling=attr.get("sibling", None),
            )

            obj.attributes[attr_name] = attr

        if not obj.attributes:
            obj.attributes = None

        self.descriptions[object_name] = obj
        return obj if obj else None

    def get_object(self, object_name, seen_types=None):
        """Queries and returns details about a specific object."""
        if not seen_types:
            seen_types = set()

        url = f"{self.BASE_URL}/objects/{object_name}"
        response = {
            k: v
            for attr in self.cache(url).get("attributes", [])
            for k, v in attr.items()
        }

        if "error" in response or "@deprecated" in response:
            return None

        obj = {}
        for attr_name, attr in response.items():
            attr = OCSFObject(
                name=attr_name,
                type=attr.get("object_type", attr.get("type", None)),
                is_object="object_type" in attr,
                is_array=attr.get("is_array", False),
                description=attr.get("description", ""),
                source=attr.get("_source", None),
                enum=attr.get("enum", None),
                sibling=attr.get("sibling", None),
            )

            obj[attr_name] = attr
            if not attr.is_object or attr.type in seen_types:
                continue

            new_seen_types = seen_types.copy()
            new_seen_types.add(attr.type)
            attr.attributes = self.get_object(attr.type, new_seen_types)

        self.objects[object_name] = obj
        return obj if obj else None

    def get_class_objects(self, class_name):
        """
        Queries the list of objects and returns detailed descriptions of each object. Unrolls the attributes and adopts simpler but verbose structure.
        """
        if class_name in self.class_objects:
            return self.class_objects[class_name]

        class_details = {
            k: v
            for entry in self.get_class_details(class_name).get(
                "attributes", {}
            )
            for k, v in entry.items()
        }

        unrolled_attributes = {}

        for key, value in class_details.items():
            ocsf_obj = OCSFObject(
                name=key,
                type=value.get("object_type", value.get("type", None)),
                is_object="object_type" in value,
                is_array=value.get("is_array", False),
                description=value.get("description", ""),
                source=value.get("_source", None),
                enum=value.get("enum", None),
                sibling=value.get("sibling", None),
            )
            if ocsf_obj.is_object:
                ocsf_obj.attributes = self.get_object(ocsf_obj.type)
            unrolled_attributes[key] = ocsf_obj

        self.class_objects[class_name] = unrolled_attributes

        return unrolled_attributes

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

    def flatten_event(self, class_name, weights=False, depth=0):
        """
        Flattens the object types in the class attributes.
        """
        class_objects = self.get_class_objects(class_name)
        flattened_types = {}

        def flatten(obj, path="", description=None, source=None, weight=1):
            for key, value in obj.items():
                source = source or (
                    value.source
                    if value.source not in self.mislabeled_categories
                    else self.mislabeled_categories[value.source]
                )
                new_path = f"{path}.{key}" if path else f"{class_name}.{key}"
                new_weight = weight / (len(obj) if obj else 1)
                if description:
                    local_description = description.copy()
                else:
                    local_description = [self.get_event_description(class_name)]
                local_description.append(value.description)
                if value.is_object and value.attributes:
                    flatten(
                        value.attributes,
                        new_path,
                        description=local_description,
                        source=source,
                        weight=new_weight,
                    )
                local_value = value.flattened_copy(depth=depth)
                local_value.description_list = local_description
                local_value.source = source
                flattened_types[new_path] = (
                    (local_value, new_weight) if weights else local_value
                )

        flatten(class_objects)

        return flattened_types

    def get_objects_from_path(self, path: str) -> List[OCSFObject]:
        """
        Returns a list of OCSFObjects from a given path.
        """
        objects = []
        paths = path.split(".")
        class_name = paths[0]
        curr_objects = self.get_class_objects(class_name)
        for p in paths[1:]:
            objects.append(curr_objects[p])
            curr_objects = curr_objects[p].attributes
        return objects

    def filter_event(self, class_name, target_class, fuzzy=0):
        """
        Return a non flattened simplified version of the event, where only fields of a certain type are included, and only description and enum fields are kept.
        """

        if isinstance(target_class, str):
            target_class = [target_class]
        target_class = list(set(target_class))

        def fuzzy_match(fuzzy):
            if fuzzy == 2:
                return lambda x: True
            elif fuzzy == 1:
                return lambda x: any(
                    self.source_type(x) == self.source_type(t)
                    for t in target_class
                )
            else:
                return lambda x: any(x == t for t in target_class)

        event_object = deepcopy(self.get_class_objects(class_name))
        flattened_event = self.flatten_event(class_name)
        fields = [
            k for k, v in flattened_event.items() if fuzzy_match(fuzzy)(v.type)
        ]
        all_fields = set()
        for field in fields:
            paths = field.split(".")
            for idx, _ in enumerate(paths):
                all_fields.add(".".join(paths[: idx + 1]))

        def filter(obj, orig_obj, path=""):
            for key, value in orig_obj.items():
                if key == "unmapped":
                    continue
                if value.is_object and not value.attributes:
                    continue

                new_path = f"{path}.{key}" if path else key
                if new_path not in all_fields:
                    continue
                if not value.is_object:
                    obj[key] = value.strip()
                    obj[key].path = new_path
                else:
                    obj[key] = value.strip(remove_attributes=True)
                    obj[key].attributes = {}
                    filter(obj[key].attributes, value.attributes, new_path)

        filtered_event = {}
        filter(filtered_event, event_object)

        return filtered_event

    def factor_event(self, class_name, definitions=None):
        """
        Return a dict with the definition of all included objects.
        """
        if definitions is None:
            definitions = {}
        event_flattened_object = self.flatten_event(class_name)
        for value in event_flattened_object.values():
            if value.is_object and value.type not in definitions:
                obj = self.get_object_description(value.type)
                definitions[value.type] = obj.strip() if obj else value.strip()

        return definitions

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
        # Get the list of all paths
        events = self.get_classes()

        # Get the list of all paths
        path_blocklist = ["category_name", "class_name", "observable"]
        paths = self.generated_descriptions
        for event in events:
            new_paths = self.flatten_event(event)
            for path, value in new_paths.items():
                if any(p in path for p in path_blocklist):
                    continue
                if value.is_object:
                    continue
                parent_type = event
                if len(path.split(".")) > 1:
                    parent_path = ".".join(path.split(".")[:-1])
                    if parent_path in new_paths:
                        parent_type = new_paths[parent_path].type
                if path in paths:
                    paths[path].parent_type = parent_type
                    paths[path].type = value.type
                if path not in paths:
                    paths[path] = OCSFFieldDescriptor(
                        path,
                        value.description_list,
                        value,
                        parent_type=parent_type,
                    )
                else:
                    paths[path].update(
                        value.description_list, value, parent_type=parent_type
                    )

        # Generate the descriptions
        if OCSFFieldDescriptor.generate_description(
            list(paths.values()),
            caller,
            tqdm=True,
            save_to_cache=lambda: self._save_to_cache(),
        ):

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
                    self.source_type(y) == self.source_type(t)
                    for t in target_class
                    for y in x
                )
            else:
                return lambda x: any(y == t for t in target_class for y in x)

        fields = []
        for event in events:
            flattened_event = self.flatten_event(event)
            local_fields = [
                k for k, v in flattened_event.items() if not v.is_object
            ]
            object_paths = []
            for fd in local_fields:
                paths = fd.split(".")
                parent_path, leaf = ".".join(paths[:-1]), paths[-1]
                if parent_path in flattened_event:
                    object_path = flattened_event[parent_path].type + "." + leaf
                else:
                    object_path = fd
                object_paths.append(object_path)
            field_types = [
                self.inferred_types.get(p, [flattened_event[fd].type])
                for p, fd in zip(object_paths, local_fields)
            ]
            fields += [
                k
                for k, type in zip(local_fields, field_types)
                if fuzzy_match(fuzzy)(type)
            ]

        fields = list(set(fields))

        return {
            f: self.generated_descriptions[f].generated_description
            for f in fields
            if f in self.generated_descriptions
        }

    def build_inferred_types(
        self,
        caller,
        tqdm=True,
        model="gemini-2.5-flash",
        **kwargs,
    ):
        """
        Builds the inferred types for all objects.
        """
        get_logger().info("Building inferred types...")
        # Get the list of objects, basic types and event classes
        object_list = self.get_object_list()
        basic_types = self.get_basic_types()
        events = self.get_classes()

        # Get the list of all paths
        paths, new_paths = self.inferred_types, {}
        for object_name in object_list:
            if object_name in ["observable", "object", "enrichment"]:
                continue
            obj = self.get_object_description(object_name)
            if not obj or not obj.attributes:
                continue
            for attr_name, attr in obj.attributes.items():
                if attr.is_object or attr.enum:
                    continue
                attr_type = attr.type
                path = f"{object_name}.{attr_name}"
                if path in paths and attr_type not in paths[path]:
                    paths[path].append(attr_type)
                elif path not in paths:
                    new_paths[path] = (
                        path,
                        [obj.description, attr.description],
                        attr.type,
                    )

        for event in events:
            event_obj = self.get_class_objects(event)
            if not event_obj:
                continue
            for attr_name, attr in event_obj.items():
                if attr.is_object or attr.enum:
                    continue
                source = (
                    attr.source
                    if attr.source not in self.mislabeled_categories
                    else self.mislabeled_categories[attr.source]
                )
                path = f"{event}.{attr_name}"
                orig_path = f"{source}.{attr_name}"
                if path in paths and attr.type not in paths[path]:
                    paths[path].append(attr.type)
                elif path not in paths and orig_path in paths:
                    paths[path] = paths[orig_path]
                elif path not in paths:
                    new_paths[path] = (
                        path,
                        [self.get_event_description(event), attr.description],
                        attr.type,
                    )

        # Infer the types
        tasks = []
        kwargs["n"] = 1
        kwargs["temperature"] = 0.33
        new_path_list = []
        for new_path, value in new_paths.items():
            user, system = gen_type_prompt(*value, self)
            task = LLMTask(
                system_prompt=system,
                max_tokens=128,
                model=model,
                message=user,
                stop=["\n"],
                **kwargs,
            )
            tasks.append(task)
            new_path_list.append(new_path)

        responses = caller(tasks, use_tqdm=tqdm)
        regexp = re.compile(r"\[.*\]")
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
                desc = [d for d in desc if d in basic_types and d != "string_t"]

            paths[new_path_list[r_id]] = desc

        self._save_to_cache()

    def build_attribute_embeddings(self, caller, chunk_size=100000):
        get_logger().info("Building attribute embeddings...")

        targets = list(
            {
                k: d
                for k, d in self.generated_descriptions.items()
                if d.leaf and not d.embedding
            }.items()
        )

        gemini_embed_chunk_size = 200
        tasks = []
        for i in range(0, len(targets), gemini_embed_chunk_size):
            description_chunk = [
                t[1].generated_description
                for t in targets[i : i + gemini_embed_chunk_size]
            ]
            if not description_chunk:
                continue
            tasks.append(
                LLMTask(
                    message=description_chunk,
                    query_type="embedding",
                    model="text-embedding-005",
                )
            )

        if not tasks:
            return

        chunks = [[]]
        for task in tasks:
            if len(chunks[-1]) >= chunk_size:
                chunks.append([])
            chunks[-1].append(task)

        for chunk_id, chunk in enumerate(chunks):
            get_logger().info(
                "Generating embeddings for chunk %d/%d",
                chunk_id + 1,
                len(chunks),
            )
            embeddings = caller(
                chunk, distribute_parallel_requests=False, use_tqdm=True
            )
            embeddings = [emb for r in embeddings for emb in r]
            for embedding_id, embedding in enumerate(embeddings):
                self.generated_descriptions[
                    targets[embedding_id + chunk_id * chunk_size][0]
                ].embedding = embedding

            self._save_to_cache()

    def get_siblings(self, fields, target_type, fuzzy=0):

        if not isinstance(target_type, list):
            target_type = [target_type]

        def fuzzy_match(fuzzy):
            if fuzzy == 2:
                return lambda x: True
            elif fuzzy == 1:
                return lambda x: any(
                    self.source_type(y) == self.source_type(t)
                    for t in target_type
                    for y in x
                )
            else:
                return lambda x: any(y == t for t in target_type for y in x)

        parents = list({".".join(f.split(".")[:-1]) for f in fields})
        siblings = [
            (k, f)
            for k, f in self.generated_descriptions.items()
            if any(p in k and p.count(".") + 1 == k.count(".") for p in parents)
            and f.leaf
        ]

        field_types = [
            self.inferred_types.get(
                f.parent_type + "." + k.split(".")[-1], [f.type]
            )
            for k, f in siblings
        ]

        filtered_siblings = [
            f[0]
            for f, types in zip(siblings, field_types)
            if fuzzy_match(fuzzy)(types)
        ]

        return filtered_siblings

    def attribute_similarity(self, attr_a, attr_b):
        """Compute attribute similarity"""
        path_a, path_b = attr_a.split("."), attr_b.split(".")
        shortest_path, longest_path = min(len(path_a), len(path_b)), max(
            len(path_a), len(path_b)
        )
        for i in range(shortest_path):
            if path_a[i] != path_b[i]:
                break
        return max(0, (i - 1)) / max(1, longest_path - 1)

    def get_similar_variables(
        self, target_candidates, target_events, mappings, k=5
    ):
        """Returns the variables that matched to the closest attributes as the ones in the candidate list.
        The similarity between an attribute and a list of attributes is the max similarity between the attribute and each individual attribute.
        The similarity between two attributes is length of their common prefix divided by the length of the longest attribute, removing the root the of tree.
        """

        if target_events and not isinstance(target_events, list):
            target_events = [target_events]

        elt_list = []

        for element_id in mappings:
            for event, mapping in mappings[element_id].items():
                if target_events and event not in target_events:
                    continue
                if not mapping.field_list:
                    continue
                if not mapping.mapped:
                    continue

                similarities = [
                    max(
                        self.attribute_similarity(f[0], c)
                        for c in mapping.field_list
                    )
                    for f in target_candidates
                ]

                # Get the top 5 closest and compute their average
                top5 = sorted(similarities, reverse=True)[:5]
                sim = sum(top5) / len(top5) if top5 else 0
                elt_list.append((element_id, sim))

        elt_list = sorted(elt_list, key=lambda x: x[1], reverse=True)
        return elt_list[:k]

    def get_source_to_event_mapping(self):
        if not self.source_to_event_mapping:
            for event in self.get_classes():
                class_objects, flattened_event = self.get_class_objects(
                    event
                ), self.flatten_event(event)
                for path, attribute in flattened_event.items():
                    if attribute.is_object or len(path.split(".")) < 2:
                        continue

                    first_level_item = path.split(".")[1]
                    if first_level_item not in class_objects:
                        continue

                    if first_level_item in {"actor"}:
                        source = "base_event"
                    else:
                        source = class_objects[first_level_item].source
                    source_path = f"{source}.{'.'.join(path.split('.')[1:])}"

                    if source_path not in self.source_to_event_mapping:
                        self.source_to_event_mapping[source_path] = {
                            event: path
                        }
                    else:
                        self.source_to_event_mapping[source_path][event] = path

        return self.source_to_event_mapping

    def get_unmapped_attributes(self, event):
        """Get unmapped attributes from the cache"""
        return self.unmapped_attributes.get(event, [])

    def add_unmapped_attributes(self, event, unmapped_attribute):
        """Add unmapped attributes to the cache"""
        if event in self.unmapped_attributes:
            self.unmapped_attributes[event].append(unmapped_attribute)
        else:
            self.unmapped_attributes[event] = [unmapped_attribute]
        self._save_to_cache(save_all=False)

    def get_event_lineage(self, event):
        """Get the lineage of an event

            client = OCSFSchemaClient(Caller(
            4,
            backend="gemini"
            distribute_parallel_requests=True))
            client.get_event_lineage([EVENT_NAME])
        ))
        """
        lineage = []
        event_details = self.cache(f"{self.BASE_URL}/classes/{event}")
        try:
            description, category = (
                event_details["description"],
                event_details["category"],
            )
            lineage.append((event, description))
        except KeyError:
            category = event
        category_details = self.cache(f"{self.BASE_URL}/categories/{category}")
        description = category_details["description"]
        lineage.append((category, description))

        return lineage


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
