import re

import typing_extensions as typing

from ..classes import Template


class Token(typing.TypedDict):
    is_variable: bool
    value: str


response_schema = list[Token]

always_variable = [
    "timestamp",
    "IP address",
    "hostname",
    "port",
    "username",
    "command path",
    "file path",
    "resource path",
    "URL",
    "email address",
    "phone number",
    "MAC address",
    "UUID",
    "value in quotes",
    "unique identifier",
]

system = """Your are an expert systems analyst. Your role is to create log parsers. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. Since we only observe the log entries, we must infer these templates.

Templates consist of static parts (fixed text from the formatting string) and variable parts (variable entries in the formatting string). Templates can be broken down into tokens, where each token represents a single semantic unit within the log entry. A token can be a single character, word, or multiple words. Here are the key concepts about tokens:

### Variability ###

Templates contain two distinct types of tokens:

* Variable tokens, or entities: These represent parameters from the original formatting string. They are parts of the template that can vary between log lines from the same group. Variable tokens must represent distinct entities like resources, devices, locations, dates, times, or users. In key-value pairs, the value is always a variable token. Variables cannot be messages, descriptions, actions, or complete sentences - they must be specific parameters. We also refer to these as entities. Values can be variables even if they do not change between log lines, as long as they represent entities or parameters. Examples of entities that should always be variable tokens are: timestamps, IP addresses, hostnames, ports, usernames, paths, URLs, usernames, email addresses, phone numbers, MAC addresses, UUIDs, values in quotes, unique identifiers, etc. 

* Static tokens: These are the fixed parts of the formatting string that appear identically in every line produced by this template. They typically include descriptions, punctuation marks, or structural elements that provide context and connect variable entities. Static tokens include verbs describing actions, descriptive messages, and keywords in key-value pairs. They encompass anything that is not strictly a variable. Entities that do not change between log lines should NOT be treated as static, bus as variables.

### Tokenization ###

* Tokens can either be static or variable, but not both.
* Variable tokens that contain multiple types of data must be broken down into their individual components: They can only include one parameter from the formatting message. For instance, variables that contain two non related data types that can vary independently should be in different tokens. Pay attention to separators, these often indicate there are multiple variable.
* Punctuation should be kept separate from variables, except if that punctuation is part of the variable (such as punctuation used to separate multiple items in a single variable).
* Valid json data formats (dictionaries, arrays, lists, etc.) must be kept as one single token.
* If you encounter key-value pairs, the key (or keyword) should be in a separate token from the value. The key should always ve static, the value should always be variable (even if it does not change in the provided examples). Key value pairs are sets of tokens in which one represents the name of the field, and the other is the value, such as "port 5836", "uid=0", or "retries: 1". 
* Units associated with variables should be included in the variable: for instance, time units, size units, or distance units, should be part of the entry they follow.
* Fields cannot be optional: variables cannot be empty.
* Some data types have specific rules:
** Hostnames, IP addresses and ports should be in separate tokens.
** Paths and URLs should be kept as a single token: do not separate them into sub tokens.
** Dates and times should be merged into a single token if adjacent
** Hex values should include the 0x prefix if present

### Consistency ###

* When creating templates, ensure that the same entity is represented in the same way across all templates. For example, a timestamp including a date and time is parsed as a single entity in one template, it should be parsed the same way in all templates.
* If a variable is represented as a single token in one template, it should be represented as a single token in all templates.
* If a variable is broken down into multiple tokens in one template, it should be broken down in the same way in all templates.

### Representation ###

* We represent templates as a JSON list of tokens, which are objects with the following fields:
    * "is_variable": a boolean indicating if the token is an entity or a variable (true) or a static part of the formatting string (false)
    * "value": the value of the token
* For example, log line `update-alternatives 2022-10-04 22:32:23: uid=0 run with --install git` is produced by template:
```json
[
    {"is_variable": false, "value": "update-alternatives"},
    {"is_variable": true, "value": "2022-10-04 22:32:23"},
    {"is_variable": false, "value": ":"},
    {"is_variable": false, "value": "uid="},
    {"is_variable": true, "value": "0"},
    {"is_variable": false, "value": "run with --install"},
    {"is_variable": true, "value": "git"}
]
```

### Rules ###

"""

rules = """I will give you log entries. You will output the template that matches the entries. 
Make sure you take into account all parts of the entries, including any punctuation or special characters.
You must adhere to all guidelines about tokenization, and the definition of variables and static tokens.
Most importantly, ensure your responses are consistent with previous templates you have generated.
Entities in new templates should be parsed in the *exact* same way as in previous templates.

In order to complete this tasks, follow the following algorithm:
```algorithm
print("<explanation>")
First, look at prior templates and explanations to understand the expected format. Ensure entities are parsed in the same way as in previous templates.
Print a description of the log lines consistent with the previous descriptions you wrote. Do not include more a less information than necessary and that is provided in previous descriptions.
print("Key Value Pairs:")
What is the format of the string that produced these suffixes? Are there any key-value pairs? List all key value pairs. Remeber, keys are static, values are entities.

print("Static tokens:")
Which part is a static message, description, or action? List all the static tokens. 

print("Entities:")
Which part is an entity? List all the entities, including:
* those that do not change.
* any value in a key value pair.
* any value what is one of: {}
Remember, entities cannot contain messages, descriptions, actions, or complete sentences - they must be specific parameters.

print the first line where entities are replaced with placeholders. All entities must be present.

print("</explanation>")

template = Use the explanation and placeholder above when generating the template for this message.

template = set all variable values in the template to be equal to their values in the first log entry

print(template) IN JSON
```
"""

user_input = """### Full Log Lines ###
{entries}

### Suffix (only parse these!) ###
{suffixes}
"""
output = """<explanation>
{explanation}
</explanation>

```json
{template}
```"""

fix_prompt_overcapture = """Your solution overcaptures: it matches the following lines which it should not:
```
{overcaptured}
```
Please fix your solution so it does not overcapture."""

fix_prompt_undercapture = """Your solution undercaptures: it misses the following suffixes:
```
{undercapture}
```
If this was intentional because the non-matched suffixes do not represent the same log event, return your original response. 
If you believe these additional suffixes should be captured by the same template, fix your solution so it captures these lines as well. Don't overfit to these examples: imagine what other lines could be produced by the same template, and try to capture all of those. Make sure to return both an explanation and a template."""

explanation_format = """
Description: {desc}
Each line follows the format: {placeholder}

Key Value Pairs:
{kvp}

Static Tokens: 
{constants}

Entities:
{variables}

The first line with placeholders is: {placeholder}
"""


def gen_fewshot(examples):
    fewshot_prompts = []
    for (
        fs_template,
        fs_entries,
        fs_suffixes,
        fs_explanation,
        fs_description,
    ) in examples:
        if not isinstance(fs_entries, list):
            fs_entries = [fs_entries]

        entries = "\n".join(fs_entries)
        suffixes = "\n".join(fs_suffixes)
        template = fs_template

        fewshot_prompts.append(
            {
                "role": "user",
                "content": user_input.format(
                    entries=entries,
                    suffixes=suffixes,
                ),
            }
        )
        fewshot_prompts.append(
            {
                "role": "assistant",
                "content": output.format(
                    explanation=fs_explanation.strip(), template=template
                ),
            }
        )

    return fewshot_prompts


def gen_system():
    return system + rules.format(", ".join(always_variable))


def gen_prompt(examples, entries, description, suffixes):
    history = gen_fewshot(examples)

    user = user_input.format(
        entries="\n".join(entries),
        suffixes="\n".join(suffixes),
    )
    history.append(
        {"role": "user", "content": rules.format(", ".join(always_variable))}
    )
    history.append({"role": "user", "content": user})

    return history, gen_system()
