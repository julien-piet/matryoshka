import typing_extensions as typing


class Token(typing.TypedDict):
    is_variable: bool
    value: str
    regex: typing.Optional[str]


response_schema = list[Token]

system = """Your are an expert systems analyst. Your role is to create log parsers. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. Since we only observe the log entries, we must infer these templates.

Templates consist of constant parts (fixed text from the formatting string) and variable parts (variable entries in the formatting string). Templates can be broken down into tokens, where each token represents a single semantic unit within the log entry. A token can be a single character, word, or multiple words. Here are the key concepts about tokens:

### Variability ###

Templates contain two distinct types of tokens:

* Variable tokens: These represent parameters from the original formatting string. They are parts of the template that can vary between log lines from the same group. Variable tokens must represent distinct entities like resources, devices, locations, dates, times, or users. In key-value pairs, the value is always a variable token. Variables cannot be messages, descriptions, actions, or complete sentences - they must be specific parameters. 

* Constant tokens: These are the fixed parts of the formatting string that appear identically in every line produced by this template. They typically include descriptions, punctuation marks, or structural elements that provide context and connect variable entities. Constants include verbs describing actions, descriptive messages, and keywords in key-value pairs. They encompass anything that is not strictly a variable.

## Regular expressions ##

Variable tokens are associated with a regular expression. These help capture the expected syntax of the variable. A good regular expression should match all possible values the variable tokens of a given type can take. They should capture specifics about the expected data format - for example, a date, a time, a file path, or a user name. They should not be specific to observed subsets of values, but generalize to any possible value of the same type. Similar typed values accross templates should reuse the same regex. 

### Representation ###

* We represent templates as a JSON list of tokens, which are objects with the following fields:
    * "is_variable": a boolean indicating if the token is a variable (true) or a constant (false)
    * "value": the value of the token
* For example, log line `update-alternatives 2022-10-04 22:32:23: run with --install git` is produced by template:
```json
[
    {"is_variable": false, "value": "update-alternatives"},
    {"is_variable": true, "value": "2022-10-04 22:32:23", "regex": "\\\\d+-\\\d+-\\\\d+\\\\s+\\\\d+:\\\\d+:\\\\d+"},
    {"is_variable": false, "value": ": run with --install"},
    {"is_variable": true, "value": "git", "regex": "\\S+"}
]
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
    print("Regular expression for token T: ", R)
    update the template to include R for T
    
print("### Final Template ###")
print(template) 
# Print the full template, even if very long. 
# Make sure the values you print in the template are those from the first entry!
# Make sure the template contains the same number of elements as the original template, and that no element has changed type!
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

### Final Template ###
```json
{template}
```
"""

partial_output = """
### Explanation ###
{partial_explanation}"""


def gen_fewshot(examples):
    fewshot_prompts = []
    for fs_result, fs_explanation, fs_prompt in reversed(examples):
        template_after = fs_result
        fewshot_prompts.append({"role": "user", "content": fs_prompt})
        fewshot_prompts.append(
            {
                "role": "assistant",
                "content": output.format(
                    explanation=fs_explanation.replace(
                        "### Explanation ###", ""
                    ).strip(),
                    template=template_after,
                ),
            }
        )

    return fewshot_prompts


def gen_system():
    return system


def get_assistant_msg_prefix(entries, matched_prefix):
    templates = {
        token.id: [] for token in matched_prefix.elements if token.is_variable()
    }
    regexes = {token.id: token.regexp for token in matched_prefix.elements}
    for entry in entries:
        partial_match, matches = matched_prefix.match(entry.strip())
        if not partial_match:
            partial_match, matches = matched_prefix.partial_match(entry.strip())
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


def gen_prompt(examples, entries, template):
    if examples:
        history = gen_fewshot(examples)
    else:
        history = []
    user = user_input.format(
        entries="\n".join(entries),
        template=template.format_as_example(),
    )
    history.append({"role": "user", "content": user})

    return history, gen_system()
