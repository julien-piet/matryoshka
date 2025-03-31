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
I will provide two lists of types: a list of standard types, and a list of additional types.
Please use standard types in priority. If no standard type fits, try to use an additional type. If still no type works, you can create new types.
There are three special types: 
* MSG: for descriptive messages, sentences, or free text.
* COMPOSITE: for structured data formats (dictionaries, arrays, lists, etc.).
* NONE: for elements that do not fit any type and do not have any strong commonalities between examples.

** Consistency **

When determining the type of a variable, please check in order:
1/ If any of the provided examples contains variables of the same type: if so, use that type
2/ If the variable is of type MSG or COMPOSITE
3/ The context of the value within the templates it is a part of.
4/ If the value is part of a key-value pair, use the key to determine the type.
5/ If multiple types fit the element, select the type that is most specific.

** Representation **

I will provide examples of each templates, where the variable value to be typed is highlighted using three stars. If any stars are part of the actual template, they will be escaped with a backslash.

** Data types **

Here is a list of standard data types:
```
{standard_types}
```

Here is the list of additional types, along with their context. Only assign these if the variable has the same semantics.
```
{additional_types}
```

** Instructions **

I will give you zero or more example templates, observed values, and types. Your task is to identify the type of another variable, using the previous guidelines, and relying on the examples for guidance and consistency. 
Remember, favor standard types over additional types, and favor already existing types over new types."""

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

* Semantics
What does this value represent? What role does it serve in the templates?
[ Update the type if needed and output it ]

* Consistency with examples
Is the element similar to one of the examples (meaning it plays the same role and has similar values)? If so, assign the same type.
[ Update the type if needed and output it ]

* Special types
Is the element a structured data format (dictionaries, arrays, lists, etc.)? If so, assign type COMPOSITE.
Is the element a descriptive message, a free text field, or a sentence? If so, assign type MSG.
[ Update the type if needed and output it ]

* Key-value pairs
Is the item part of a key-value pair? 
If so, what is the keyword in this key-value pair? 
Considering this, should you update your type?
[ Update the type if needed and output it ]

* Context
What is the context of the value?
Does a suitable type already exist in the list of standard types?
If not, does a suitable existing type exist in the list of additional types? If so, is the context for this type the same 
If no suitable existing type exists, what is a specific type that would capture these values but only these values?
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
    for templates, values, id, type in examples:
        templates = random.sample(templates, min(10, len(templates)))
        values = random.sample(values, min(20, len(values)))
        templates = "\n".join(t.highlight(id) for t in templates)
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


def gen_system(additional_types, client=None):
    if additional_types is None:
        additional_types = []
    prompt_addition_types = []
    for (
        determined_type,
        example_template,
        element_id,
        example_value,
    ) in additional_types:
        prompt_addition_types.append(
            f"{determined_type} was assigned to value ```{example_value}``` in ```{example_template.highlight(element_id)}```"
        )
    if len(prompt_addition_types) > 0:
        prompt_addition_types = "\n".join(prompt_addition_types)
    else:
        prompt_addition_types = "NO ADDITIONAL TYPES"

    if client:
        basic_types = client.get_basic_types()
        standard_types = "\n".join(
            {key: val} for key, val in basic_types.items()
        )
    else:
        standard_types = "\n".join(DEFAULT_TYPES)

    return system.format(
        standard_types=standard_types,
        additional_types=prompt_addition_types,
    )


def gen_prompt(
    templates, values, id, examples=None, add_types=None, client=None
):

    templates = random.sample(templates, min(10, len(templates)))
    values = random.sample(values, min(20, len(values)))
    if examples:
        fewshot = gen_fewshot(examples)
    else:
        fewshot = ""

    templates = "\n".join(t.highlight(id) for t in templates)
    values = ", ".join(values)
    prompt = user.format(
        templates=templates,
        observed_values=values,
    )
    return fewshot + prompt, gen_system(
        additional_types=add_types, client=client
    )
