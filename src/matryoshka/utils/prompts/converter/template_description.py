import json
import random

system = """Your role is to annotate log files. Log files are collections of entries produced by programs to record state. Log lines can be grouped according to the formatting string used to produce them in the original program. These formatting strings define the template of this group of lines. We have already parsed the log entries to templates. We now need to write descrpitons for the templates.

Templates are made up of into tokens. Each token represents a single semantic unit within the log entry, and can be made up of individual characters, single words, or even multiple words. Tokens can either be constant (fixed parts of the formatting string) or variable (variables in the formatting string). Variable tokens are assigned a regular expression that matches the token, and a data type that conveys information about the semantics of each variable.

*** Instructions ***
I will provide you with a template (a few example values of the template, as well as a format string where variables are replaced with their type)
I need you to write a description of the template token. The rules are the following:
* The description should provide detailed information about the what event the template refers to.
* The description should be complete: it should provide all the information needed to understand what the template refers to.
* The description should be sound: it should be consistent with the role of the template, and must not mislead into misrepresenting its meaning.
* The description should be precise: it should not be ambiguous, it should use the proper terminology, and should be clear about the meaning of the template.
* The description should be general: it should not reference specific values or examples of variable fields.

For example, the following template:
```
[datetime_t] TRACE :......rsvp_event_mapSession: Session=[ip_t]:[port_t]:[integer_t] does not exist
```
that takes the following example value:
```
03/22 08:52:50 TRACE :......rsvp_event_mapSession: Session=9.67.116.99:1047:6 does not exist
```
can be described as:
```
RSVP session lookup failure: Unable to find an existing RSVP (Resource Reservation Protocol) session matching the specified IP address and port/index identifiers
```

*** Consistency ***
If the template you are describing is identical to one you have already described, make sure you use the same description.
If the template you are describing plays a similar role to one you have already described, please write a description consistent with the previous one.
"""

user_input = """### Template

Consider the following template:
```
{template}
```

This template takes, among others, the following values:
```
{observed_values}
```

### Description
"""


def gen_fewshot(examples):
    fewshot_prompts = []
    for (
        fs_template,
        fs_examples,
        fs_description,
    ) in examples:
        examples = "\n".join(fs_examples)
        values = ", ".join(fs_examples)
        prompt = user_input.format(
            template=fs_template,
            observed_values=values,
        )
        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append(
            {"role": "assistant", "content": json.dumps(fs_description)}
        )

    return fewshot_prompts


def gen_system():
    return system


def gen_prompt(params, few_shot_examples):
    template, examples = params

    if few_shot_examples:
        fewshot = gen_fewshot(few_shot_examples)
    else:
        fewshot = []

    prompt = user_input.format(
        template=template,
        observed_values="\n".join(examples),
    )

    return prompt, fewshot, gen_system()
