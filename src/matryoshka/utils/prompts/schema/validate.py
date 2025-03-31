import json

system = """You are an expert systems analyst. Your role is to create log parsers. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. We have identified a set of candidate templates, along with some lines they match. These templates are structured in a parsing tree, so they share common tokens. Some nodes have a field name and description, which describe the information that is captured by the node. Your task is to edit these field names and descriptions to form a unified structure that can be queried agnostically of underlying template, by ensuring their names and descriptions are consistent with each other and the guidelines below.

Templates consist of static parts (fixed text from the formatting string) and variable parts (variable entries in the formatting string). Templates can be broken down into tokens, where each token represents a single semantic unit within the log entry. A token can be a single character, word, or multiple words. Templates aim at capturing as much information as possible from the log entries, so that the logs can be queried and analyzed.

# Schema creation guidelines

## Scope

### Variables
* Every variable must be assigned a field name and description

### Constants

#### Default naming
* By default, constants should be assigned the field name "SYNTAX"
* In particular, any separator between tokens should be assigned the field name "SYNTAX"
* This indicates the constant does not contain any information that is relevant to an analyst and not already contained in variables.

#### Semantic constants
* Some constants contain useful information that should be assigned a field name:
  - Contains queryable information
  - The information is relevant to the event
  - The information is not contained in any variable
  - The constant should be considered a variable
  - The meaning of the constant is obvious and without it the event would not be interpretable or would lack information.
* Constants that follow all the above criteria should be assigned a field name
* This name must follow the same rules as variables

### Key in key-value pairs
* Key-value pairs have specific naming rules
* For key-value pairs in which both the value and key are fields, the name of the key field must be the same as the name of the value field, with "_KEY" appended to the end of the name. The description of the key field must "KEY".
* For key-value pairs in which only the key is a field, or if the value is not a single field, the name and description of the key field must be "SYNTAX".

## Description Writing Instructions

### Core Principles
* **Scope**: Write descriptions that characterize the role of variables in log events, not their values.
* **Independence**: Descriptions must remain valid regardless of the specific values observed in any instance.

### Quality Requirements

#### Completeness
* Include all information necessary to understand the variable's role
* Cover the full scope of the variable's function in the context
* Explain any relevant relationships with other fields

#### Precision
* Use accurate technical terminology
* Avoid ambiguous or vague language
* Be explicit about the variable's purpose
* Maintain consistent technical depth

#### Soundness
* Ensure descriptions align with the technical reality of the variable's role
* Avoid misleading or incorrect characterizations
* Stay within the bounds of what is definitively known about the variable

#### Language Standards
* Use clear, professional technical writing
* Write in complete sentences
* Maintain consistent verb tense
* Use active voice

### What to Avoid
* Specific values or examples
* Assumptions about typical or possible values
* Instance-specific observations
* References to implementation details that may change
* Speculative functionality

## Field Naming Instructions

### Core Principles

**Abstract Role**: Names must represent the fundamental semantic role of the field in the event structure, independent of:
- Any specific values it might contain
- The specific system generating the log
- The specific type of event being logged

**Structural Context**: Names should reflect the field's position and purpose in the event's logical structure:
- What information does this field provide about the event?
- How does this field relate to other fields?
- What aspect of the event does this field describe?

**Universal Applicability**: Names must be valid across:
- All possible values the field could contain
- All types of events where this field appears
- All systems that might generate similar events
...

### Implementation Rules

#### Format:
* Use snake_case
* Use singular nouns
* Avoid vendor names, product names, or technology-specific terms
* Maximum 3 words joined by underscores

#### Field Grouping:
* Fields serving the same semantic role MUST use identical names
* When grouping fields, list all matching IDs in the schema
* Cross-reference previous schemas to maintain consistency

#### Naming Structure:
* If the original element included a field_name, please reuse it
* Start with the most general category (e.g., 'time', 'sequence', 'host')
* Add specificity only when needed to disambiguate
* Maintain parallel structure across related fields

### Validation Checklist

Before finalizing field names, verify:
1. Would the name still make sense if the field contained different values?
2. Does the name depend on any specific technology or system?
3. Is this name identical to names used for similar fields in other schemas?
4. Could this field appear in other types of events with the same meaning?
5. All fields that end with "_KEY" have a corresponding field without the suffix.

### What to Avoid

#### Technology-Specific Terms:
* No implementation-specific terminology

#### Value-Based Names:
* No names that describe specific values
* No names that assume specific formats or types
* No names that limit the field's interpretation

#### Overly Specific Names:
* No names that incorporate current use cases
* No names that assume specific contexts
* No names that limit future applications

#### Generic Names:
* Avoid names that are too generic and might apply to fields that represent different entities or play different roles in the event structure.
* In particular, be very careful about naming constants. Only do so if the constant could be considered a variable. If it is not, it should be named "SYNTAX".
* Do not reuse names for attributes that play different roles. Remember, all attributes with the same field name must have the same description.

Guidelines about schema consistency

We want to be able to run queries against the variables in these templates without needing to know the specifics about each individual template: The naming should be consistent across templates so that we can do this. For instance, all entities that represent a source IP field should have the same name, regardless of the specific template that contains them. Fields that are marked as frozen cannot be modified: you should align non frozen fields with the frozen fields.

## Naming

* Fields that have the same purpose or role in different templates must have the same name.
* Adopt a unified field name style as well and maintain parallel structure.
* Two nodes that represent different types of entities, roles or concepts must have different names.

## Descriptions

* Fields with the same name must have the same description.
* All descriptions should have the same technical depth and level of detail.

## Pitfalls

* Two nodes that have different roles must have different names.
* Two nodes with the same name must have the same description.
* Frozen fields cannot be modified. Instead, you should align the non-frozen fields with the frozen fields.
* Do not change field names if they are already correct and consistent.
* If you update any field names, make sure to update the corresponding key field names as well if any.

## Task description

You will be given a set of entries, as well as the parsing tree that matches them. Your task is to correct the field names and descriptions in the tree. You must align new fields with frozen ones. You can perform the following actions:
```python
node_type = {
    "type": "object",
    "properties": {
        "is_variable": {"type": "boolean"}, # True if the node is a variable, false if it is a static token
        "value": {"type": "string"}, # The observed value of the node. For a constant, this is the value of the constant. For a variable, it's an example value of the variable in one of the lines that matched the template.
        "regexp": {"type": "string"}, # The regular expression used to match the node.
        "id": {"type": "integer"},
        "field_name": {"type": "string"}, # The name of the field.
        "description": {"type": "string"}, # The description of the field.
    }
}

def SET_SYNTAX(node_id: int) -> None:
    \"\"\"
    Sets the field name of a node to "SYNTAX". If the node was assigned another field name, it will be cleared.
    
    Args:
        node_id: The ID of the target node.
    \"\"\"
    pass


def SET_KEY_FIELD(node_id: int, value_node_id: int) -> None:
    \"\"\"
    Sets the field name of a node as a key field, linked to a value node.
    The field name of the key node will be that of the value node, with "_KEY" appended to the end of the name.
    The description of the key field will be "KEY".

    This can only be used for key-value pairs in which the value node has been assigned a field name.
    This must be called after the value node has been assigned a field name.

    Args:
        node_id: The ID of the node that represents the key in the key-value pair.
        value_node_id: The ID of the node that contains the value in the key-value pair.
    \"\"\"
    pass


def SET_EXISTING_FIELD(node_id: int, field_name: str) -> None:
    \"\"\"
    Sets the field name of a node to an existing field name in the tree.
    The node will inherit the description from the existing field.
    If the description of the existing field does not accurately describe the node,
    you should use a new name with SET_NEW_FIELD_NAME instead.

    Args:
        node_id: The ID of the target node.
        field_name: The existing field name.
    \"\"\"
    pass


def SET_NEW_FIELD_NAME(node_id: int, field_name: str, description: str) -> None:
    \"\"\"
    Sets the field name of a node to a new field name and description.
    You can only call this with new field names that have not been used before in the tree.
    If you call this on an existing field name, it will error out.
    Do not call this to change a description.

    Args:
        node_id: The ID of the target node.
        field_name: The new field name.
        description: The new description.
    \"\"\"
    pass


def CHANGE_DESCRIPTION(field_name: str, description: str) -> None:
    \"\"\"
    Changes the description of a node.
    You can only call this with an existing field name.
    This will change the description of ALL the nodes with the given field name.
    If that is not what you want, you should create a new field name instead.
    Use this sparingly.

    Args:
        field_name: The existing field name.
        description: The new description.
    \"\"\"
    pass




def SET_SYNTAX(node_id: int) -> None:
    \"\"\"
    Sets the field name of a node to "SYNTAX". If the node was assigned another field name, it will be cleared.
    
    Args:
        node_id: The ID of the target node.
    \"\"\"
    pass
```

Please first explain what changed need to be made to the tree for it to be consistent. Then, write some python code in a markdown code block (delimited by ```python and ```) that uses these primitives to edit the field names and descriptions. Make sure the code can execute: do not write it inside a function, but instead make sure executing your code as is will work (e.g. running python's exec on your code block). Remember, frozen nodes names cannot be changed.
"""

user_input = """
Log lines:
```
{entries}
```

Parsing tree:
```
{tree}
```"""


def gen_system():
    return system


def graph_to_markdown(graph):
    """Converts a networkx DiGraph object into an indented Markdown list.

    This is best for tree-like structures.

    Args:
        graph (networkx.DiGraph): The graph to convert.

    Returns:
        str: A string representing the graph as a Markdown list.
    """
    markdown_lines = []
    # find the root nodes (nodes with no incoming edges)
    root_nodes = [
        node for node, in_degree in graph.in_degree() if in_degree == 0
    ]

    def build_markdown_recursive(node, prefix=""):
        """Helper function to recursively build the markdown string."""
        # Format the attributes into a readable string like "(key: value, ...)"
        attributes = graph.nodes.get(node, {})
        attr_string = ", ".join(
            [f"{key}: {json.dumps(value)}" for key, value in attributes.items()]
        )
        markdown_lines.append(f"{prefix}- {node} ({attr_string})")

        # Recurse for all children of the current node
        for successor in graph.successors(node):
            build_markdown_recursive(successor, prefix + "  ")

    # Start the conversion from each root node
    for root in root_nodes:
        build_markdown_recursive(root)

    return "\n".join(markdown_lines)


def graph_to_json_list(graph):
    """Converts a networkx DiGraph object into a JSON list representation.

    This function iterates through each node in the graph, determines its parent,
    and creates a list of dictionaries, where each dictionary represents a node
    with its ID and its parent's ID.

    Args:
        graph (networkx.DiGraph): The graph to convert, which is expected to be a tree.

    Returns:
        str: A JSON string representing the graph as a list of nodes.
        Returns an empty JSON array '[]' for an empty graph.
    """
    node_list = []
    for node in graph.nodes():
        # A node in a tree structure can have at most one parent (predecessor).
        parents = list(graph.predecessors(node))
        parent_id = parents[0] if parents else None
        node_list.append({"id": node, "parent": parent_id})
    return json.dumps(node_list, indent=2)


def gen_prompt(entries, tree, json_tree=False):
    history = []

    try:
        entries = "\n".join(entries)
    except:
        breakpoint()
    str_tree = graph_to_markdown(tree)
    if json_tree:
        str_tree = graph_to_json_list(tree)
    history.append(
        {
            "role": "user",
            "content": user_input.format(entries=entries, tree=str_tree),
        }
    )

    return history, gen_system()
