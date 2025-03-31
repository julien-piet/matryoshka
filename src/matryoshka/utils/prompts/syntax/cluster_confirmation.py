import json

response_schema = {"type": "array", "items": {"type": "string"}}

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
7. **Precision:** Be precise. Do not use overly generic placeholders. Do not use a single placeholder for a set of variables. Do not include empty variables. 
8. **Irreducibility:** If there are variables with nested components (for example, key-value pairs within a value of a key-value pair), always break these down into the most granular and irreducible possible formatting string, separating the inner-most components into their own variables.


### Algorithm ###

Use the following algorithm to guide your process:
```mapping algorithm
print("### Explanation ###")
1. Print a description of the log lines, highlighting common patterns.
2. Identify the format of the string that produced these entries. Highlight constant messages and extract variables.
3. Further separate variables into their atomic parts.
   - Key-value pairs must have a constant key and a variable value. If you find any, you can only replace the value with a placeholder, not the key.
   - Produce irreducible formatting strings. If you encounter nested fields, use the most granular formatting string possible, separating the inner-most components into their own variables and constants.
   - Variables cannot contain messages, descriptions, actions or sentences - they must refer to specific parameters. 
   - You cannot use empty variables. If a variable is not present in a line, it should not be included in the formatting string for that line. If the value of a key-value pair is not present, you should only include the key in the formatting string, and not include a placeholder for the value.
4. Print each line with placeholders, ensuring that:
   - Lines with different formats have different placeholders.
   - Lines with different numbers of variables have different formats.
      - In particular, lines that are missing variables found in other lines should have a different format.

print("### Mapping ###")
5. Output the results as a JSON array, maintaining the same order as the input, where each item is a string containing the formatting string for the line. Print all placeholders, do not omit any.
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
