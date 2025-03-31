import json
import random

system = """Your role is to create log parsers. Log files are collections of entries, each produced by some program to record state. Log lines can be grouped according to the formatting string used to produce them in the original program. These formatting strings define the template of this group of lines. However, we only observe the log entries, so we must infer the templates.

Templates are made of constant parts (fixed parts of the formatting string) and variable parts (variables in the formatting string). We can separate templates into tokens. Each token represents a single semantic unit within the log entry, and can be made up of individual characters, single words, or even multiple words. We now define some useful characteristics of tokens:

** Variability **

* Variable tokens: These are the variable parts of the template. Tokens are variable if they were a parameter of the original formatting string that produced them. These are any part of the templates that is susceptible to change in other log lines from the same group, even if rarely. Variable tokens represent entities. They are a distinct concept with independent existence: a resource, device, location, date, time, user, and others. If the template contains any key-value pairs, the value will always be a variable. We refer to these elements as VAR.

* Constant tokens: These are the fixed parts of the formatting string. They will always be the same in every line produced by this string. They are often descriptions, punctuations, or delimiters of the template. They can be verbs describing the main action of the log line, or other messages that are not parameters but rather descriptive sentences to contextualize and link variable, independent entities. These include keywords and delimiters in key-value pairs. We refer to these elements as CST.

** Types **

Variable tokens are assigned data types.
Types should convey information about the semantics of each variable. 
I will provide a list of standard types. There are two additional special types: 
* composite_t: for structured data formats (dictionaries, arrays, lists, etc.).
* none_t: for elements that do not fit any type.

** Consistency **

When determining the type of a variable, please check in order:
1/ If any of the provided examples contains variables of the same type: if so, use that type
2/ If the variable is of type composite_t
3/ The context of the value within the templates it is a part of.
4/ If the value is part of a key-value pair, use the key to determine the type.
5/ If multiple types fit the element, select the type that is most specific.
6/ If the variable contains fixed punctuation elements (e.g., underscores, hyphens, periods), ignore these and focus on the variable part.

** Representation **

I will provide examples of each templates, where the variable value to be typed is highlighted using three stars. If any stars are part of the actual template, they will be escaped with a backslash.

** Data types **

Here is a list of standard data types:
```
{standard_types}
```

** Instructions **

I will give you zero or more example templates, observed values, and types. Your task is to identify the type of another variable, using the previous guidelines, and relying on the examples for guidance and consistency."""

few_shot = """
TEMPLATES
```
{templates}
```

SUBSET OF OBSERVED VALUES
```
{observed_values}
```

TYPE
```
{type}
```"""


user = """TEMPLATES
```
{templates}
```

SUBSET OF OBSERVED VALUES
```
{observed_values}
```

Copy and fill out the following checklist to guide your answer. 


[[ CHECKLIST ]]
* First try
[ Output the type separated by triple quotes ```]

* Fixed parts
Is any part of this value fixed punctuation or unrelated to the value itself? If so, ignore these and focus on the variable part.
[ Update the type if needed and output it ]

* Semantics
What does this value represent? What role does it serve in the templates?
[ Update the type if needed and output it ]

* Consistency with examples
Is the element similar to one of the examples (meaning it plays the same role and has similar values)? If so, assign the same type.
[ Update the type if needed and output it ]

* Special types
Is the element a structured data format (dictionaries, arrays, lists, etc.)? If so, assign type composite_t.
[ Update the type if needed and output it ]

* Key-value pairs
Is the item part of a key-value pair? 
If so, what is the keyword in this key-value pair? 
Considering this, should you update your type?
[ Update the type if needed and output it ]

* Context
What is the context of the value?
Does a suitable type already exist in the list of standard types?
If not, assign type none_t
[ Update the type if needed and output it ]

* Regex
If you have determined a type, please provide a regular expression that matches this type.
[ Output the regex in triple quotes ```]

* Final answer
[ Output the final type in triple quotes ```]"""

DEFAULT_TYPES = [
    "DATETIME",
    "USERNAME",
    "HOSTNAME",
    "FILENAME",
    "PROCESS_NAME",
    "PROCESS_ID",
    "IP_ADDRESS",
    "PORT",
    "URL",
    "EMAIL_ADDRESS",
    "MAC_ADDRESS",
    "COUNTRY_CODE",
    "SESSION_ID",
    "LOG_LEVEL",
    "FILE_SIZE",
    "DURATION",
    "HASH_VALUE",
    "ALGORITHM_NAME",
]


def gen_fewshot(examples):
    fewshot_prompts = []
    for templates, values, ids, type in examples:
        values = random.sample(values, min(20, len(values)))
        highlighted_templates = [
            t.highlight(id)
            for template_list, id in zip(templates, ids)
            for t in template_list
        ]
        highlighted_templates = random.sample(
            highlighted_templates, min(10, len(highlighted_templates))
        )
        templates = "\n".join(highlighted_templates)
        values = ", ".join(values)
        prompt = few_shot.format(
            templates=templates,
            observed_values=values,
            type=type,
        )
        fewshot_prompts.append(prompt)
    fewshot_prompts = (
        "[ EXAMPLES ]\n"
        + "\n\n".join(fewshot_prompts)
        + "\n[ END OF EXAMPLES ]\n\n"
    )
    return fewshot_prompts


def gen_system(client=None):
    if client:
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
    else:
        standard_types = "\n".join(DEFAULT_TYPES)

    return system.format(
        standard_types=standard_types,
    )


def gen_prompt(templates, values, ids, examples=None, client=None):

    values = random.sample(values, min(20, len(values)))
    if examples:
        fewshot = gen_fewshot(examples)
    else:
        fewshot = ""

    highlighted_templates = [
        t.highlight(id)
        for template_list, id in zip(templates, ids)
        for t in template_list
    ]
    highlighted_templates = random.sample(
        highlighted_templates, min(10, len(highlighted_templates))
    )
    templates = "\n".join(highlighted_templates)
    values = ", ".join(values)
    prompt = user.format(
        templates=templates,
        observed_values=values,
    )
    return fewshot + prompt, gen_system(client=client)
