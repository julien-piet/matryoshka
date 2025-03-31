import json
import random

system = """You are a log analysis expert. Your role is to expand the list of types associated with OCSF log fields

*** OCSF Format Intro ***

The Open Cybersecurity Schema Framework is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

*** OCSF Overview ***

The framework is made up of a set of data types, an attribute tree, and the taxonomy. It is not restricted to the cybersecurity domain nor to events, however the initial focus of the framework has been a schema for cybersecurity events. OCSF is agnostic to storage format, data collection and ETL processes. The core schema for cybersecurity events is intended to be agnostic to implementations. The schema framework definition files and the resulting normative schema are written as JSON.

*** Fields and Types ***

OCSF contains a list of objects. Each object has a series of attributes. Fields are defined as the attributes, and are assigned data types from the following list: 
```
{standard_types}
```

*** Instructions ***

I will provide you with a field name, description, and assigned type. Each field is currently only assigned one data type: You should add any additional more specific types that values in the field could have. Do not assign additional types if no other types apply. The goal is to provide a comprehensive list of data types that could be taken by values of the field. For example, unique identier could be a string_t, but it could also be a uuid_t, or an integer_t, so all should be listed. Pay particular attention to the provided examples, as they will guide you in determining the most specific types for each field.
"""

input_template = """Field: {path}

Description: 
* {parent}: {parent_desc}
* {field}: {field_desc}

Assigned Type: {orig_type}

Additional Types:"""

user = """{fs}{input}"""

input_template_top_level = """Field: {path}

Description: {field_desc}

Assigned Type: {orig_type}

Additional Types:"""


def gen_fewshot():

    example_header = """### Example Descriptions ###\n\n"""
    example_separator = """\n\n##########\n\n"""
    example_footer = """\n\n### End of Examples ###\n\n"""

    example_1 = """Field: user.name
        
Description: 
* user: The User object describes the characteristics of a user/person or a security principal.
* name: The username. For example, <code>janedoe1</code>.

Assigned Type: "string_t"

Additional Types: '["string_t", "username_t"]' """

    example_2 = """Field: group.uid
    
Description:
* group: The Group object represents a collection or association of entities, such as users, policies, or devices. It serves as a logical grouping mechanism to organize and manage entities with similar characteristics or permissions within a system or organization.
* uid: The unique identifier of the group. For example, for Windows events this is the security identifier (SID) of the group.

Assigned Type: "string_t"

Additional Types: '["string_t", "uuid_t", "integer_t", "long_t"]' """

    example_3 = """Field: database.size
    
Description: 
* database: The database object is used for databases which are typically datastore services that contain an organized collection of structured and unstructured data or a types of data.
* type: The size of the database in bytes.

Assigned Type: "long_t"

Additional Types: '["long_t", "integer_t"]' """

    example_4 = """Field: actor.app_name
    
Description:
* actor: The Actor object contains details about the user, role, application, service, or process that initiated or performed a specific activity.
* app_name: The client application or service that initiated the activity. This can be in conjunction with the user if present. Note that app_name is distinct from the process if present.

Assigned Type: "string_t"

Additional Types: '["string_t"]' """

    example_5 = """Field: fingerprint.value
    
Description:
* fingerprint: The Fingerprint object provides detailed information about a digital fingerprint, which is a compact representation of data used to identify a longer piece of information, such as a public key or file content. It contains the algorithm and value of the fingerprint, enabling efficient and reliable identification of the associated data.
* value: The digital fingerprint value.

Assigned Type: "string_t"

Additional Types: '["string_t", "file_hash_t"]' """

    example_6 = """Field: file.name
    
Description:
* file: The File object represents the metadata associated with a file stored in a computer system. It encompasses information about the file itself, including its attributes, properties, and organizational details.
* name: The name of the file. For example: svchost.exe

Assigned Type: "string_t"

Additional Types: '["string_t", "file_name_t"]' """

    example_7 = """Field: kill_chain_phase.phase
    
Description:
* kill_chain_phase: The Kill Chain Phase object represents a single phase of a cyber attack, including the initial reconnaissance and planning stages up to the final objective of the attacker. It provides a detailed description of each phase and its associated activities within the broader context of a cyber attack.
* phase: The cyber kill chain phase.

Assigned Type: "string_t"

Additional Types: '["string_t"]' """

    example_8 = """Field: hassh.algorithm
    
Description:
* hassh: The HASSH object contains SSH network fingerprinting values for specific client/server implementations. It provides a standardized way of identifying and categorizing SSH connections based on their unique characteristics and behavior.
* algorithm: The concatenation of key exchange, encryption, authentication and compression algorithms (separated by ';'). NOTE: This is not the underlying algorithm for the hash implementation.

Assigned Type: "string_t"

Additional Types: '["string_t"]' """

    fs = (
        example_header
        + example_1
        + example_separator
        + example_2
        + example_separator
        + example_3
        + example_separator
        + example_4
        + example_separator
        + example_5
        + example_separator
        + example_6
        + example_separator
        + example_7
        + example_separator
        + example_8
        + example_footer
    )

    return fs


def gen_system(client):
    basic_types = client.get_basic_types()
    key_val_pairs = []
    for key, val in basic_types.items():
        val_copy = val.copy()
        if "type" in val:
            del val_copy["type"]
        if "observable" in val:
            del val_copy["observable"]
        if "type_name" in val:
            del val_copy["type_name"]
        key_val_pairs.append(json.dumps({key: val_copy}))
    standard_types = "\n".join(val for val in key_val_pairs)

    return system.format(
        standard_types=standard_types,
    )


def gen_prompt(path, descriptions, orig_type, client):
    if not path.count(".") == 1:
        ipt = input_template_top_level.format(
            path=path,
            field_desc=descriptions[0],
            orig_type=json.dumps(orig_type),
        )
        return user.format(fs=gen_fewshot(), input=ipt), gen_system(client)

    ipt = input_template.format(
        path=path,
        parent=path.split(".")[0],
        parent_desc=descriptions[0],
        field=path.split(".")[1],
        field_desc=descriptions[1],
        orig_type=json.dumps(orig_type),
    )
    return user.format(fs=gen_fewshot(), input=ipt), gen_system(client)
