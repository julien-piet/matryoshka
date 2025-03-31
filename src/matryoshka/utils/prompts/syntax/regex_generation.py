import json

import typing_extensions as typing


class RegexAssignment(typing.TypedDict):
    value: str
    regex: typing.Optional[str]


regex_assign_schema_dict = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "regex": {"type": "string"},
    },
    "required": ["value", "regex"],
}

response_schema = {
    "type": "object",
    "patternProperties": {"^[0-9]+$": regex_assign_schema_dict},
}

system = """Your are an expert systems analyst. Your role is to map variables in log messages to regular expressions. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. Templates consist of constant parts (fixed text from the formatting string) and variable parts (variable entries in the formatting string). Templates can be broken down into tokens, where each token represents a single semantic unit within the log entry. A token can be a single character, word, or multiple words. Here are the key concepts about tokens:

### Variability ###

Templates contain two distinct types of tokens:

* Variable tokens: These represent parameters from the original formatting string. They are parts of the template that can vary between log lines from the same group. Variable tokens must represent distinct entities like resources, devices, locations, dates, times, or users. In key-value pairs, the value is always a variable token. Variables cannot be messages, descriptions, actions, or complete sentences - they must be specific parameters. 

* Constant tokens: These are the fixed parts of the formatting string that appear identically in every line produced by this template. They typically include descriptions, punctuation marks, or structural elements that provide context and connect variable entities. Constants include verbs describing actions, descriptive messages, and keywords in key-value pairs. They encompass anything that is not strictly a variable.

## Regular expressions ##

Variable tokens are associated with a regular expression. These help capture the expected syntax of the variable. A good regular expression should match all possible values the variable tokens of a given type can take. They should capture specifics about the expected data format - for example, a date, a time, a file path, or a user name. They should not be specific to observed subsets of values, but generalize to any possible value of the same type. Similar typed values accross templates should reuse the same regex. Do not use capturing groups in the regex. 

## Avoiding Overcapture ##

Avoid overcapture by ensuring the regular expresion captures the structure of the variable. For instance, json objects should be captured using  `\\{.*\\}` and arrays should be captured using `\\[.*\\]`.

### Representation ###

* We represent templates as a JSON list of tokens, which are objects with the following fields:
    * "is_variable": a boolean indicating if the token is a variable (true) or a constant (false)
    * "value": the value of the token
* For example, log line `update-alternatives 2022-10-04 22:32:23: run with --install git` is produced by template:
```json
[
    {"is_variable": false, "value": "update-alternatives"},
    {"is_variable": true, "value": "2022-10-04 22:32:23", "id": 0},
  
    {"is_variable": false, "value": ": run with --install"},
    {"is_variable": true, "value": "git", "id": 1}
]
```
The regular expressions for the variable tokens are:
```json
{
    0: {
        "value": "2022-10-04 22:32:23",
        "regex": "\\\\d+-\\\\d+-\\\\d+\\\\s+\\\\d+:\\\\d+:\\\\d+"}
    },
    1: {
        "value": "git",
        "regex": "\\S+"
    }
}
```

  
### Rules ###

I will give you log entries and their template missing some regular expressions. You will output a copy of the template that matches the entries with regular expressions. You must adhere to all guidelines, and ensure your responses are consistent with previous templates you have generated. In particular, make sure any part of the current log entry that is similar to a previous you encountered has the same regular expression. Do not change the number of elements in the template, only update the regular expressions of variable elements missing them! 

In order to complete this tasks, you must follow this algorithm:
```regex generation algorithm
print("### Explanation ###")
for each variable token T in the template:
    L <- values taken by T in the log entries
    print("Variable token T has values: ", L)
    if values like L have been seen in previous templates:
        R <- regex from previous template
    else:
        R <- infer a regular expression that captures all possible values of the same type as the values in L. As per the guidelines, it should be general, match values in L, and not overfit to specific observed values. 
        if R is a fixed value (matches an exact string):
            expand R to be more general
        if R does not match all possible values:
            expand R to be more general
        if R represents a json object:
            use the following regex for objects: \\{.*\\}. Use the following regex for arrays: \\[.*\\].
    print("Regular expression for token T: ", R)
    update the template to include R for T
    
print("### Mapping ###")
print(mapping) 
# Print a json object mapping each variable in the template to its value and regular expression.
# Make sure the values in the mapping are the ones extracted from the first entry!
# Make sure every variable in the template has a corresponding entry in the mapping!
```
"""

user_input = """
### Log entries (all belonging to the same template) ###
```
{entries}
```

### Matched template ###
```json
{template}
```
"""

output = """
### Explanation ###
{explanation}

### Mapping ###
```json
{mapping}
```
"""

partial_output = """
### Explanation ###
{partial_explanation}"""

# Updated template with IDs for variable tokens
oneshot_example = {
    "entries": [
        'Jun 18 15:42:17 srv-dc01.corporate.local TerminalServices-RemoteConnectionManager: Event ID 1149: User authentication succeeded for user "CORPORATE\\jdoe" from source network address 10.45.122.78 using authentication package "Negotiate" with logon type "RemoteInteractive" session created with ID 0x8A2F3 connection established successfully via RDP protocol version 10.0 client build 19041 operating system "Windows 10 Enterprise" computer name "LAPTOP-USER123" total session duration will be tracked under tracking ID TR_20250618154217_8A2F3',
        'Jun 18 16:15:23 srv-dc01.corporate.local TerminalServices-RemoteConnectionManager: Event ID 1149: User authentication succeeded for user "CORPORATE\\admin.smith" from source network address 192.168.50.145 using authentication package "Kerberos" with logon type "RemoteInteractive" session created with ID 0x9B4E7 connection established successfully via RDP protocol version 10.0 client build 22000 operating system "Windows 11 Pro" computer name "ADMIN-WORKSTATION" total session duration will be tracked under tracking ID TR_20250618161523_9B4E7',
        'Jun 18 17:33:41 srv-dc01.corporate.local TerminalServices-RemoteConnectionManager: Event ID 1149: User authentication succeeded for user "CORPORATE\\service.backup" from source network address 172.16.8.92 using authentication package "NTLM" with logon type "RemoteInteractive" session created with ID 0x7C1A9 connection established successfully via RDP protocol version 8.1 client build 17763 operating system "Windows Server 2019" computer name "SRV-BACKUP02" total session duration will be tracked under tracking ID TR_20250618173341_7C1A9',
    ],
    "template": [
        {"value": "Jun 18 15:42:17", "is_variable": True, "id": 0},
        {"value": "srv-dc01.corporate.local", "is_variable": True, "id": 1},
        {
            "value": "TerminalServices-RemoteConnectionManager:",
            "is_variable": False,
        },
        {"value": "Event ID", "is_variable": False},
        {"value": "1149:", "is_variable": False},
        {"value": "User", "is_variable": False},
        {"value": "authentication succeeded", "is_variable": False},
        {"value": "for", "is_variable": False},
        {"value": "user", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "CORPORATE", "is_variable": True, "id": 2},
        {"value": "\\", "is_variable": False},
        {"value": "jdoe", "is_variable": True, "id": 3},
        {"value": '"', "is_variable": False},
        {"value": "from", "is_variable": False},
        {"value": "source network address", "is_variable": False},
        {"value": "10.45.122.78", "is_variable": True, "id": 4},
        {"value": "using", "is_variable": False},
        {"value": "authentication package", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "Negotiate", "is_variable": True, "id": 5},
        {"value": '"', "is_variable": False},
        {"value": "with", "is_variable": False},
        {"value": "logon type", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "RemoteInteractive", "is_variable": True, "id": 6},
        {"value": '"', "is_variable": False},
        {"value": "session created", "is_variable": False},
        {"value": "with", "is_variable": False},
        {"value": "ID", "is_variable": False},
        {"value": "0x8A2F3", "is_variable": True, "id": 7},
        {"value": "connection established successfully", "is_variable": False},
        {"value": "via", "is_variable": False},
        {"value": "RDP", "is_variable": False},
        {"value": "protocol version", "is_variable": False},
        {"value": "10.0", "is_variable": True, "id": 8},
        {"value": "client build", "is_variable": False},
        {"value": "19041", "is_variable": True, "id": 9},
        {"value": "operating system", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "Windows 10 Enterprise", "is_variable": True, "id": 10},
        {"value": '"', "is_variable": False},
        {"value": "computer name", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "LAPTOP-USER123", "is_variable": True, "id": 11},
        {"value": '"', "is_variable": False},
        {
            "value": "total session duration will be tracked under",
            "is_variable": False,
        },
        {"value": "tracking ID", "is_variable": False},
        {"value": "TR_20250618154217_8A2F3", "is_variable": True, "id": 12},
    ],
}

oneshot_solution = {
    0: {
        "value": "Jun 18 15:42:17",
        "regex": "\\w{3}\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2}",
    },
    1: {"value": "srv-dc01.corporate.local", "regex": "\\S+"},
    2: {"value": "CORPORATE", "regex": "[A-Z0-9_]+"},
    3: {"value": "jdoe", "regex": "\\S+"},
    4: {
        "value": "10.45.122.78",
        "regex": "\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}",
    },
    5: {"value": "Negotiate", "regex": "\\S+"},
    6: {"value": "RemoteInteractive", "regex": "\\S+"},
    7: {"value": "0x8A2F3", "regex": "0x[A-F0-9]+"},
    8: {"value": "10.0", "regex": "\\d+(\\.\\d+)?"},
    9: {"value": "19041", "regex": "\\d+"},
    10: {
        "value": "Windows 10 Enterprise",
        "regex": '[^"]+',
    },
    11: {
        "value": "LAPTOP-USER123",
        "regex": '[^"]+',
    },
    12: {"value": "TR_20250618154217_8A2F3", "regex": "\\S+"},
}


def gen_fewshot(examples):
    fewshot_prompts = []
    if not examples:
        examples = []
    for fs_result, fs_explanation, fs_prompt in reversed(examples):
        mapping = fs_result
        fewshot_prompts.append({"role": "user", "content": fs_prompt})
        fewshot_prompts.append(
            {
                "role": "assistant",
                "content": output.format(
                    explanation=fs_explanation.replace(
                        "### Explanation ###", ""
                    ).strip(),
                    mapping=mapping,
                ),
            }
        )

    # Add standard example
    std_example = []
    std_example.append(
        {
            "role": "user",
            "content": user_input.format(
                entries="\n".join(oneshot_example["entries"]),
                template=json.dumps(
                    oneshot_example["template"],
                    indent=2,
                    ensure_ascii=False,
                ),
            ),
        }
    )
    std_explanation = get_assistant_msg_prefix(oneshot_solution)["content"]
    std_example.append(
        {
            "role": "assistant",
            "content": output.format(
                explanation=std_explanation.replace(
                    "### Explanation ###", ""
                ).strip(),
                mapping=json.dumps(
                    oneshot_solution, indent=2, ensure_ascii=False
                ),
            ),
        }
    )

    return std_example + fewshot_prompts


def gen_system():
    return system


def get_assistant_msg_prefix(entries, matched_prefix=None):
    if not matched_prefix and isinstance(entries, dict):
        explanation = "### Explanation ###\n"
        for token_id, assignment in entries.items():
            explanation += (
                f"Variable token #{token_id} has value: {assignment['value']}\n"
            )
            explanation += f"Regular expression for token #{token_id}: {assignment['regex']}\n"
        return {"role": "assistant", "content": explanation}
    else:
        templates = {
            token.id: []
            for token in matched_prefix.elements
            if token.is_variable()
        }
        regexes = {token.id: token.regexp for token in matched_prefix.elements}
        for entry in entries:
            partial_match, matches = matched_prefix.match(entry.strip())
            if not partial_match:
                partial_match, matches = matched_prefix.partial_match(
                    entry.strip()
                )
                if not partial_match:
                    raise ValueError(
                        f"Entry {entry} does not match the prefix {matched_prefix}"
                    )
            for tok in matches.elements:
                if tok.id in templates:
                    templates[tok.id].append(tok.value)

        explanation = "### Explanation ###\n"
        index = 0
        for token_id, values in templates.items():
            explanation += (
                f"Variable token #{index} has values: {', '.join(values)}\n"
            )
            explanation += (
                f"Regular expression for token #{index}: {regexes[token_id]}\n"
            )
            index += 1

        return {"role": "assistant", "content": explanation}


def gen_prompt(examples, entries, template, force_match=False):
    history = gen_fewshot(examples)

    template = template.format_as_example(
        force_match_with_entry=True,
        relative_ids=True,
        entry=entries[0],
        ignore_pending=True,
    )

    user = user_input.format(entries="\n".join(entries), template=template)
    history.append({"role": "user", "content": user})

    return history, gen_system()
