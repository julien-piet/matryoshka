import json

system = """You are an expert systems analyst. Your role is to create log parsers. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. We have identified a set of candidate templates, along with some lines they match. These templates are structured in a parsing tree, so they share common tokens. Your task is to correct this parsing tree so all templates are consistent with each other and with the guidelines below.

Templates consist of static parts (fixed text from the formatting string) and variable parts (variable entries in the formatting string). Templates can be broken down into tokens, where each token represents a single semantic unit within the log entry. A token can be a single character, word, or multiple words. Templates aim at capturing as much information as possible from the log entries, so that the logs can be queried and analyzed.

# Guidelines about parsing tree

## Variability

Trees are made of nodes, called tokens. There are two types of tokens:

* Variable tokens, or entities: These represent parameters from the original formatting string. They are parts of the template that can vary between log lines from the same group. Variable tokens must represent distinct entities like resources, devices, locations, dates, times, or users. In key-value pairs, the value is always a variable token. Variables cannot be messages, descriptions, actions, or complete sentences — they must be specific parameters. We also refer to these as entities. Values can be variables even if they do not change between log lines, as long as they represent entities or parameters. Examples of entities that should always be variable tokens are: timestamps, IP addresses, hostnames, ports, usernames, paths, URLs, usernames, email addresses, phone numbers, MAC addresses, UUIDs, values in quotes, unique identifiers, authentication methods, etc. Variables cannot be empty or optional.

* Static tokens: These are the fixed parts of the formatting string that appear identically in every line produced by this template. They typically include descriptions, punctuation marks, or structural elements that provide context and connect variable entities. Static tokens include verbs describing actions, descriptive messages, and keywords in key-value pairs. They encompass anything that is not strictly a variable. Entities that do not change between log lines should NOT be treated as static, but as variables.

## Tokenization

* Tokens can either be static or variable, but not both.
* Static tokens should be broken down into granular components: Tokens should break at punctuation or other separators. Queryable concepts should be isolated into their own tokens. Action verbs or nouns should be separate tokens. In short, make sure tokens are semantic groupings of related words into logical phrases.
* Variable tokens that contain multiple types of data must be broken down into their individual components: They can only include one parameter from the formatting message. For instance, variables that contain two non related data types that can vary independently should be split in different tokens. Pay attention to separators, these often indicate there are multiple variables.
* Punctuation should be kept separate from variables, except if that punctuation is part of the variable (such as punctuation used to separate multiple items in a single variable).
* Valid json data formats (dictionaries, arrays, lists, etc.) must be kept as one single token.
* If you encounter key-value pairs, the key (or keyword) should always be a separated token from the value. The key should always ve static, the value should always be variable (even if it does not change in the provided examples). Key value pairs are sets of tokens in which one represents the name of the field, and the other its value, such as "port 5836", "uid=0", or "retries: 1".
* If you encounter nested key-value pairs (key-value pairs that are part of the value of a larger key-value pair), the inner-most key-value pair should be separated. For instance, in "message='port:5836,uid=0'", "port" and "uid" should be separate constant tokens, and "5836" and "0" should be separate variable tokens.
* Units associated with variables should be included in the variable: for instance, time units, size units, or distance units, should be part of the entry they follow.
* Fields cannot be optional: variables cannot be empty. If two lines differ by the presence or absence of a field, each line should have its own template. Never treat a field as optional, instead create multiples templates.
* Spaces between variables are omitted from the JSON representation; you can ignore them.
* Some data types have specific rules:
  * Hostnames, IP addresses and port numbers should be in separate tokens. "127.0.0.1:53" should be broken down into two variable tokens: "127.0.0.1" and "53".
  * Paths and URLs should be kept as a single token: do not separate them into sub tokens. "https://google.com/search?q=log+parser" is a single variable token. So is "/var/log/sshd.log".
  * Dates and times should be merged into a single token if adjacent. "Mon Apr 7, 2025 12:30:00" is a single variable token.
  * Hex values should include the 0x prefix if present. "0x12345678" is a single variable token.

## Templates

Templates are branches in the tree. They represent a single formatting string, while the tree represents a set of formatting strings. The last node in a template is marked with the ID of the template. Nodes that are frozen cannot be changed. Use these as a reference, and only change nodes that are not frozen.

## Guidelines about tree consistency

The tree was build by adding templates one by one. Each of the templates was generated separately, and may have different formatting, tokenization, or variable definitions. This means the tree is not consistent. Your task is to make sure the tokens in the tree follow the guidelines above, but most importantly, that the templates that are part of the tree are consistent with each other.

## Separation consistency

Ensure tokenization is consistent across tokens. If a token is broken down into multiple tokens in one template, it should be broken down in the same way in all templates. Use parallel structures for tokens that play similar roles across templates. If you encounter different tokenization for the same entity across templates, choose the most granular and consistent tokenization across all templates.

## Variable consistency

* Ensure that variables are consistent across templates. If a token is a variable in one template, it should be a variable in all templates. If a token is a static token in one template, it should be a static token in all templates. If a token is inconsistently defined as a variable in one template and a static token in another, choose the most appropriate definition based on the guidelines above.

## Pitfalls

### Focus on inconsistencies between templates, not within templates

Your task is to enforce consistency among templates, not within templates. Any change must be motivated by the fact there are two or more templates with similar structures but inconsistent tokenizations.

### Inference from partial information

* Do not assume that the lines provided are fully representative of every possible line that can match the template. As such, do not change tokens from variables to constants unless you need to do so for consistency amongst templates and you are convinced the value is static and does not represent any entity or queryable parameter.

### Word-by-word tokenization

* Do not default to splitting constants on spaces. In some contexts, keeping a few constant words together is better than splitting them. Only split on spaces if you need to isolate one of the constant words because it represents a queryable concept.

### Missed variables

* In some cases, the original templates might miss the fact a token is a variable. This is noticable when you see all of the following:
  * Multiple templates with parallel structures, which share a common prefix that only differs by a single static token.
  * These templates represent the same type of event.
  * The token fits the definition of a variable, and changing it to a variable would not lead to an overcapturing template.
* If all of these are true, then merge these template prefixes into a single prefix in the tree by replacing the static token with an appropriate new tokenization, and moving the parallel branches into this new prefix.

### Trying to change frozen tokens


* Do not change tokens that are marked as frozen. These are part of the tree that is fixed, and should be assumed consistent. Use this part of the tree as a reference, and only change tokens that are not frozen. You might notice some inconsistencies within the frozen parts of the tree: these can be noted, but they should not be changed.

### Ignoring whitespaces

* Pay attention to whitespaces: some nodes are identical except for the presence or absence of a single space after.

## Task description

You will be given a set of entries, as well as the parsing tree that matches them. Your task is to correct the tree. You can perform the following actions:
```python
node_type = {
    "type": "object",
    "properties": {
        "is_variable": {"type": "boolean"}, # True if the node is a variable, false if it is a static token
        "value": {"type": "string"}, # The observed value of the node. For a constant, this is the value of the constant. For a variable, it's an example value of the variable in one of the lines that matched the template.
        "regexp": {"type": "string"}, # The regular expression used to match the node.
        "id": {"type": "integer"},
        "placeholder": {"type": "string"},
        "trailing_whitespace": {"type": "integer"} # Number of spaces between this token and the next token
    }
}

def CREATE(parent_id: int, value: node_type) -> int:
    \"\"\"
    Create a new node in the tree and connect it to the parent node.
    If parent node is 0, the new node should be a root node.
    If an ID value is provided in the value, it will be ignored; a new ID will be generated.

    Args:
        parent_id: The ID of the parent node.
        value: The value of the new node.

    Returns:
        The ID of the new node.
    \"\"\"
    pass

def EDIT(node_id: int, new_value: node_type) -> None:
    \"\"\"
    Update the value of an existing node in the tree.

    Args:
        node_id: The ID of the node to update.
        new_value: The new value of the node. Non specified fields will not be updated. IDs cannot be changed.
    \"\"\"
    pass
```

def DELETE(node_id: int) -> None:
    \"\"\"
    Delete an existing node in the tree.
    If it has children, they will be connected to the parent of the deleted node.

    Args:
        node_id: The ID of the node to delete.
    \"\"\"
    pass

def MOVE(node_id: int, new_parent_id: int) -> None:
    \"\"\"
    Move an existing node in the tree to a new parent.
    All templates that contain this node will be updated to connect to the new parent.
    This will throw an error if the this creates cycles or if the node_id is the same as the new_parent_id.

    Args:
        node_id: The ID of the node to move.
        new_parent_id: The ID of the new parent node.    
    \"\"\"
    pass

def REPLACE(node_id: int, values: list[node_type]) -> list[int]:
    \"\"\"
    Replace a node in the tree with a list of nodes.
    The parent of the current node will point to the first node in the list. 
    The nodes in the list will form a chain, and the last node in the list will point to the current node's children. 
    This will return a list of the new node ids for the new chain.
    The current node will be replaced with the first node in the list.

    Args:
        node_id: The ID of the node to replace.
        values: The values of the new nodes.

    Returns:
        The IDs of the new nodes.
    \"\"\"
    pass

def ADD_TEMPLATE(node_id: int) -> int:
    \"\"\"
    Mark a node in the tree as being the end of a new template.

    Args:
        node_id: The ID of the last node in the new template.

    Returns:
        The ID of the new template.
    \"\"\"
    pass

def DELETE_TEMPLATE(template_id: int) -> None:
    \"\"\"
    Delete a template from the tree.
    This will delete the template from the tree.
    All nodes that are part of the template and are not shared with other templates will be removed.

    Args:
        template_id: The ID of the template to delete.
    \"\"\"
    pass
```

Avoid using DELETE_TEMPLATE and ADD_TEMPLATE if you just need to edit a template and can use MOVE and REPLACE instead.

Please first explain what changed need to be made to the tree for it to be consistent. Then, write some python code in a markdown code block (delimited by ```python and ```) that uses these primitives to edit the tree. Make sure your changes lead to a valid tree, and that the tree still parses the set of entries. Also make sure the code can execute: do not write it inside a function, but instead make sure executing your code as is will work (e.g. running python's exec on your code block). Remember, frozen nodes cannot be changed.
"""

user_input = """
log_lines:
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
    # Find the root nodes (nodes with no incoming edges)
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
    # The 'indent' parameter formats the JSON string with indentation for readability.
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
