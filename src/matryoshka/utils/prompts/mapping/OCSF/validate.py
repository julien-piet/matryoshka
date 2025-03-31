import json

system = """You are an expert in log parsing and log taxonomies. Your task is to map a custom schema for a log file to the Open Cybersecurity Schema Framework (OCSF).

# Context about logs

Log files are collections of entries, where each entry is produced by a program to record its state. Each log entry represents a specific event, and contains relevant information that is useful for security operations. However, most log files are not structured, so extracting, querying and correlating information is difficult. We have produced a parser that converts each log entry to a structured format that uses custom field names. This parser is not perfect: because there are many different log entries, some variables with the same role in different log entries might be mapped to different fields, so there might exist duplicate fields. In order to fix this issue and make the data more useful, we want to map each of these fields to a OCSF field.

# Context about OCSF

The target attributes are part of the OCSF format. The Open Cybersecurity Schema Framework (OCSF) is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

# Relevant OCSF fields

We have determined that the following OCSF attribute tree can be relevant to the custom schema we have produced. The OCSF attributes you can assign are the leaves of this tree, and are represented by the "full_name" of the attribute.

{ocsf_tree}

# Status

We have looked at each source attribute in isolation, and have come up with suggested OCSF fields for each. Your task is to finalize these mappings. These need to be improved so they are consistent and accurate.

## Guidelines about mapping to OCSF

## Accuracy

* A good match between a source attribute and a OCSF attribute is one where the role and nature of the source attribute is accurately captured by the destination attribute.
    * **EXAMPLE (Role and Nature Incorrect):** Mapping a source field 'total_bytes_transferred' (describing network traffic size) to 'file.size' (describing file size). The role (network transfer vs. file property) and nature (network bytes vs. file bytes) are incorrect.
    * **EXAMPLE (Role Incorrect, Nature Correct):** Mapping a source field 'attacker_ip' (describing the origin of an attack) to 'dst_endpoint.ip' (describing the destination of an action). The nature (IP address) is correct, but the role (source of attack vs. target of action) is incorrect.
    * **EXAMPLE (Role and Nature Correct):** Mapping a source field 'user_id_login' (describing the identifier of a user logging in) to 'user.uid' (describing the unique identifier of the user that logged in). Both the role (user identifier) and nature (login) are accurately captured.

* Do not map a source attribute to a OCSF attribute that is more specific than the source attribute.
    * **EXAMPLE (Source More Generic):** Mapping a source field 'app_name' (e.g., "Chrome", "Firefox", "Notepad") to 'connection_info.protocol_name' (e.g., "TCP", "UDP"): 'app_name' is too generic for 'network.application_protocol' as it refers to a specific application, not a network protocol. A better mapping might be OCSF's own 'app_name'.

* It can be OK to map a source attribute to a OCSF attribute that is slightly more generic than the source attribute, but only if:
  - The OCSF attribute is close.
  - AND no other OCSF attribute would be a better fit
  - AND other source attribute mapped to this OCSF attribute have the same level of specificity as this source attribute
    * **EXAMPLE (Positive):** 'host_ip_address' and 'device_ip' can be mapped to 'device.ip'. This is appropriate because both fields have the same level of specificity, and no other more specific OCSF attribute is a better fit.
    * **EXAMPLE (Negative):** You have source fields 'dns_query' (e.g., "example.com") and 'full_url' (e.g., "https://www.example.com/path"). Both contain domain information. While 'query.hostname' could capture "example.com" from both, 'full_url" has a highly specific and accurate OCSF home in 'http_request.url',which also contain the domain. Mapping both 'dns_query' and 'full_url' to 'network.dns_domain' would lose the richer context available for 'full_url'. In this case, 'dns_query' should map to 'query.hostname', and 'full_url' to 'http_request.url'.

* Try to the best of your ability to map source attributes to OCSF attributes with the same data type.
* You can map each source attribute to at most {attribute_count} OCSF attribute{attribute_count_plural}. If a source attribute does not map to any OCSF attribute, leave the mapping empty.
* If the role of the source attribute is not clear, do not map it to any OCSF attribute.
* If the source attribute is not something that would need to be queried against (e.g., purely diagnostic information not relevant for security analysis), do not map it to any OCSF attribute.
* Only assign OCSF attributes from the tree listed above.

## Consistency

* Twin attributes should map to the same OCSF attribute. Twin attributes are attributes that have different names in the log, but play the same role.
* Sibling attributes should map to sibling OCSF attributes. Sibling attributes are fields that refer to the same entity, but describe different aspects (e.g., process name and process ID). These have the same OCSF prefix, but only the attribute leaf changes (e.g., the last field in the path is different).
* It is OK to have multiple source attributes map to the same OCSF attribute, as long as they are compatible.
    * **EXAMPLE (Compatible Mapping):** Source fields 'listen_address' and 'bind_address' both refer to an IP address associated with a network service. Both can be mapped to the same attribute, either 'src_endpoint.ip' if they refer to the address of the source of the event, or 'dst_endpoint.ip' if the represent the target entity in an event.
    * **EXAMPLE (Incompatible Mapping):** Attempting to map 'user_id_logged_in' (the ID of the user who successfully logged in) and 'user_id_connection_allowed_by' (the ID of an administrator account that granted a network connection). While both are user IDs, their "roles" in the event are distinct (the principal vs. intermediary). Mapping them to the same 'user.uid' would lead to inaccurate queries. Instead, 'user_id_logged_in' would map to 'user.uid', and 'user_id_connection_allowed_by' might map to 'actor.user.uid', depending on context.

* The final mapping must be unified. Remember, querying a OCSF attribute should return all fields that map to that concept, and no other fields.

# Task description

You will be given a list of source fields, with their name, a list of values they take, a couple log entries in which they appear, their description, and a list of OCSF fields that they could potentially map to. Your task is to assign the definitive mapping for each field, using the guidelines provided above. The candidates given for each field are mere suggestions. If any of them are good matches, you can use them. Feel free to look over other fields if they help make the full mapping more consistent and accurate.

You will express this mapping using the provided API:

API:

def SET_NO_MAPPING(field_name: str) -> None:
    \"\"\"Sets the mapping of a field to be empty. This is to indicate that the field does not map to any OCSF attribute.

    Args:
        field_name: The name of the field.
    \"\"\"
    pass

def MAP_FIELD(field_name: str, ocsf_field: str) -> None:
    \"\"\"Maps a field to a OCSF attribute.
    This is additive: calling this function multiple times for a given field will add the field to the list of OCSF attributes it can map to.

    Args:
        field_name: The name of the field.
        ocsf_field: The name of the OCSF field.
    \"\"\"
    pass

def MAP_FIELDS(field_names: list[str], ocsf_field: str) -> None:
    \"\"\"Maps a list of fields to a OCSF attribute.
    This is additive: calling this function multiple times for a given field will add the field to the list of OCSF attributes it can map to.

    Args:
        field_names: The names of the fields.
        ocsf_field: The name of the OCSF field.
    \"\"\"
    pass

First, write a detailed explanation of your reasoning, and why the mapping you are suggesting is both accurate and consistent. Then, write a valid python code: I will directly run this code using an 'eval' command to generate the OCSF mapping, so do not nest your code in a function. You can directly use the API functions above in the code.
"""

field_format = """# {field_name} #
Description: {description}
Type: {type}
Example values (non exhaustive): {examples}
Example log lines (Each variable is represented by its field name surrounded by `<` and `>`):
```
{example_lines}
```
Suggested OCSF fields:
{ocsf_fields}"""

ocsf_attribute_format = "[{ocsf_field_name}] ({ocsf_field_description})"
ocsf_no_mapping_placeholder = """NO APPROPRIATE OCSF FIELD"""


def gen_system(ocsf_tree, max_field_count):
    return system.format(
        ocsf_tree=ocsf_tree,
        attribute_count=max_field_count,
        attribute_count_plural="s" if max_field_count > 1 else "",
    )


def gen_field_format(
    field_name, description, type, examples, example_lines, ocsf_fields
):
    ocsf_format = ocsf_no_mapping_placeholder
    if ocsf_fields:
        ocsf_format = "```\n"
        for ocsf_field_name, ocsf_field_description in ocsf_fields:
            ocsf_format += (
                ocsf_attribute_format.format(
                    ocsf_field_name=ocsf_field_name,
                    ocsf_field_description=ocsf_field_description,
                )
                + "\n"
            )
        ocsf_format += "```"

    examples = ", ".join(map(str, examples))
    example_lines = "\n".join(map(str, example_lines))
    return field_format.format(
        field_name=field_name,
        description=description,
        type=type,
        examples=examples,
        example_lines=example_lines,
        ocsf_fields=ocsf_format,
    )


def graph_to_markdown(graph):
    """Converts a networkx DiGraph object into an indented Markdown list.
    This is best for tree-like structures.

    Args:
        graph (networkx.DiGraph): The graph to convert.

    Returns:
        str: A string representing the graph as a Markdown list.
    """
    markdown_lines = []
    # Find the root nodes (nodes with no incoming edges)
    root_nodes = [
        node for node, in_degree in graph.in_degree() if in_degree == 0
    ]

    def build_markdown_recursive(node, prefix=""):
        """Helper function to recursively build the markdown string."""
        # Format the attributes into a readable string like "(key: value, ...)"
        attributes = graph.nodes.get(node, {})
        attr_string = ", ".join(
            [
                f"{key}: {json.dumps(value)}"
                for key, value in attributes.items()
                if key != "name"
            ]
        )
        markdown_lines.append(f"{prefix}- {attributes['name']} ({attr_string})")

        # Recurse for all children of the current node
        for successor in graph.successors(node):
            build_markdown_recursive(successor, prefix + "  ")

    # Start the conversion from each root node
    for root in root_nodes:
        build_markdown_recursive(root)

    return "\n".join(markdown_lines)


def gen_prompt(field_info, ocsf_tree, max_field_count=1):
    history = []
    str_tree = graph_to_markdown(ocsf_tree)
    field_formats = [
        gen_field_format(
            field_name=field_name,
            description=description,
            type=type,
            examples=examples,
            example_lines=example_lines,
            ocsf_fields=ocsf_fields,
        )
        for field_name, description, type, examples, example_lines, ocsf_fields in field_info
    ]
    field_formats = "\n\n---\n\n".join(field_formats)
    history.append({"role": "user", "content": field_formats})
    return history, gen_system(str_tree, max_field_count)
