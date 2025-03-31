import json
import random

system = """Your role is to annotate log files. Log files are collections of entries produced by programs to record state. Log lines can be grouped according to the formatting string used to produce them in the original program. These formatting strings define the template of this group of lines. We have already parsed the log entries to templates. We now need to write descrpitons for the variables in these templates.

Templates are made up of into tokens. Each token represents a single semantic unit within the log entry, and can be made up of individual characters, single words, or even multiple words. Tokens can either be constant (fixed parts of the formatting string) or variable (variables in the formatting string). Variable tokens are assigned a regular expression that matches the token.

*** Instructions ***
I will provide you with a variable field (given by the template it is part of, as well as some example values). 
I need you to write a description of the variable token. The rules are the following:
* The description should provide detailed information about the role of the variable in the templates it is part of.
* The description should be complete: it should provide all the information needed to understand what the variable refers to.
* The description should be sound: it should be consistent with the role of the variable, and must not mislead into misrepresenting the role of the field.
* The description should be precise: it should not be ambiguous, it should use the proper terminology, and should be clear about the role of the variable.
* The description should be general: it should not reference specific values or examples.

For example, for variable highlighted using three stars in the following template:
```
***03/22 08:52:50*** TRACE :......rsvp_event_mapSession: Session=9.67.116.99:1047:6 does not exist
```

A good description would be:
```
The timestamp of the log entry. It captures the date and time when the event described in the log line occurred. 
```

*** Consistency ***
If the attribute you are describing is identical to one you have already described, make sure you use the same description.
If the attribute you are describing plays a similar role to one you have already described, please write a description consistent with the previous one.

*** Context-Aware Information ***
All variables you are annotating are part of log templates for a specific log file. This log file is described as:
```
{log_desc}
```
"""

user_input = """### Variable

Consider the variable highlighted using three stars in the following templates:
```
{templates}
```

We've observed it take the following subset of values:
```
{observed_values}
```

### Description
"""


def gen_system(desc):
    return system.format(log_desc=desc)


def gen_fewshot(examples):
    fewshot_prompts = []
    for (
        fs_templates,
        fs_element,
        fs_matches,
        fs_description,
    ) in examples:
        templates = random.sample(fs_templates, min(10, len(fs_templates)))
        values = random.sample(fs_matches, min(20, len(fs_matches)))
        templates = "\n".join(t.highlight(fs_element.id) for t in templates)
        values = ", ".join(values)
        prompt = user_input.format(
            templates=templates,
            observed_values=values,
            type=fs_element.type,
        )
        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append(
            {"role": "assistant", "content": json.dumps(fs_description)}
        )

    return fewshot_prompts


def gen_prompt(elt, few_shot_examples, log_desc):
    templates, _, element, matches = elt
    templates = random.sample(templates, min(10, len(templates)))
    values = random.sample(matches, min(20, len(matches)))
    templates = "\n".join(t.highlight(element.id) for t in templates)
    values = ", ".join(values)

    random.shuffle(few_shot_history_grouped)
    fewshot = []
    for user, assistant in few_shot_history_grouped:
        fewshot.append(user)
        fewshot.append(assistant)

    prompt = user_input.format(
        templates=templates,
        observed_values=values,
        type=element.type,
    )

    # if few_shot_examples:
    #     fewshot += gen_fewshot(few_shot_examples)

    return prompt, fewshot, gen_system(log_desc)


fixed_few_shot_examples = []

user_input_1 = """### Variable

Consider the variable highlighted using three stars in the following templates:
```
[2024-01-10 15:23:45] Connection established from ***192.168.1.100***
[2024-01-10 15:23:46] Received request from ***10.0.0.50***
```

We've observed it take the following subset of values:
```
192.168.1.100
10.0.0.50
172.16.0.25
192.168.0.1
```

### Description"""

model_output_1 = """The source IP address involved in the network connection or request. This IP identifies the network host that initiated the connection."""

user_input_2 = """### Variable

Consider the variable highlighted using three stars in the following templates:
```
User ***johndoe*** successfully logged in from web interface
Failed login attempt for user ***admin*** - account locked
```

We've observed it take the following subset of values:
```
johndoe
admin
sysuser
webmaster
```

### Description"""

model_output_2 = """The username of the user account that was involved in the authentication operation."""

user_input_3 = """### Variable

Consider the variable highlighted using three stars in the following templates:
```
Backup operation started at ***2024-01-10T15:30:00.123Z***
Database checkpoint completed at ***2024-01-10T16:45:12.456Z***
```

We've observed it take the following subset of values:
```
2024-01-10T15:30:00.123Z
2024-01-10T16:45:12.456Z
2024-01-10T17:22:33.789Z
2024-01-10T18:15:00.000Z
```

### Description"""

model_output_3 = """The timestamp when the system operation was executed, including millisecond precision and UTC timezone indicator."""

few_shot_history_grouped = [
    (
        {"role": "user", "content": user_input_1},
        {"role": "assistant", "content": model_output_1},
    ),
    (
        {"role": "user", "content": user_input_2},
        {"role": "assistant", "content": model_output_2},
    ),
    (
        {"role": "user", "content": user_input_3},
        {"role": "assistant", "content": model_output_3},
    ),
]
