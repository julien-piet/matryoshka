import json

response_schema = list[str]

system = """You are an expert systems analyst. Your role is to create log parsers. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. Since we only observe the log entries, we must infer these templates.

Templates consist of constant parts (messages from the formatting string) and variable parts (everything else).

### Rules ###
1. **Multiple Parsers:** A single set of log lines may require more than one parser. If you identify distinct formats that cannot be represented by a single template, you must define separate templates and output distinct parsers for them.
   If the set of lines includes:
   - Some lines with one format and
   - Other lines with a clearly distinct format,
   you must produce separate templates and parsers for these formats. Each line must map to the appropriate parser.
2. **Constant vs Variable Parts:** Identify the constant parts (messages from the formatting string) and variable parts (everything else).
3. **Grouping:** Do not assign the same formatting string to two lines with different formats or structures.
4. **Placeholders:** Replace variables with meaningful placeholders. Do not overgeneralize variables—retain constant parts and messages as constants.
5. **Message and Variable Distinction:** Messages cannot be variables and must remain part of the formatting string. Variables cannot be status or error messages. Variables can only be parameters from the original formatting string.
6. **Consistency:** Be consistent with previous answers: If a variable similar to one previously seen appears, use the same placeholder.


### Algorithm ###

Use the following algorithm to guide your process:
```mapping algorithm
print("### Explanation ###")
1. Print a description of the log lines, highlighting common patterns.
2. Identify the format of the string that produced these entries. Highlight constant messages and extract variables.
3. For variables that seem to be over-generalized:
   - Break down the variable further into their structured components.
   - Retain constant phrases as part of the template.
   - Replace only truly variable elements with placeholders.
4. Print each line with placeholders, ensuring that:
   - Lines with different formats have different placeholders.
   - Lines with different numbers of variables have different formats.

print("### Mapping ###")
5. Output the results as a JSON array, maintaining the same order as the input.
```"""

user_input = """### Lines ###
```json
{entries}
```

"""

output = """### Explanation ###
{explanation}

### Mapping ###
```json
{mapping}
```
"""

explanation_format = """
Description: {desc}
Each line follows the format: {placeholder}

The constant parts are: 
{constants}

The variable parts are: 
{variables}

The lines with placeholders are:
```
{placeholder_list}
```
"""


def gen_fewshot(examples):
    fewshot_prompts = []
    for (
        fs_entries,
        fs_explanation,
        fs_mapping,
    ) in examples:
        if not isinstance(fs_entries, list):
            fs_entries = [fs_entries]
        fs_entries = fs_entries[:5]

        entries = json.dumps(fs_entries, indent=2)
        mapping = json.dumps(fs_mapping, indent=2)

        prompt = user_input.format(entries=entries)
        response = output.format(
            explanation=fs_explanation.strip(),
            mapping=mapping,
        )

        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append({"role": "assistant", "content": response})

    return fewshot_prompts


def gen_system():
    return system


def gen_prompt(examples, entries):
    history = gen_fewshot(examples)
    if not isinstance(entries, list) or len(entries) == 1:
        raise ValueError("Cluster confirmation requires multiple entries.")

    user = user_input.format(entries=json.dumps(entries, indent=2))

    return user, history, gen_system()
