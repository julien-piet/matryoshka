import json
import random

system = """Your role is to map log entries to the standard OCSF format. Log files are collections of entries, each produced by some program to record state. Log lines can be grouped according to the formatting string used to produce them in the original program. These formatting strings define the template of this group of lines. We have already parsed the log entries to typed templates. We know want to map these template to standardized events in the OCSF format.

Templates are made of constant parts (fixed parts of the formatting string) and variable parts (variables in the formatting string). We can separate templates into tokens. Each token represents a single semantic unit within the log entry, and can be made up of individual characters, single words, or even multiple words. Variable tokens are assigned a regular expression that matches the token, and a data type that conveys information about the semantics of each variable.

*** OCSF Format Intro ***

The Open Cybersecurity Schema Framework is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

*** OCSF Overview ***

The framework is made up of a set of data types, an attribute dictionary, and the taxonomy. It is not restricted to the cybersecurity domain nor to events, however the initial focus of the framework has been a schema for cybersecurity events. OCSF is agnostic to storage format, data collection and ETL processes. The core schema for cybersecurity events is intended to be agnostic to implementations. The schema framework definition files and the resulting normative schema are written as JSON.

*** Standardized OCSF Events ***

{events}

** Instructions **

I will give you a log template along with some example values it takes. You must select the 3 most likely OCSF events. Rank them from the event that best captures this template to the one that worse captures the template. Please use the full description of the event as well as the fields it defines to guide your decision. If no event is a match, please return "unsure"."""

few_shot_header = """Here are some examples of what events prior templates have been mapped to:"""

few_shot_content = """Example #{}:

Template: {}

Observed values:
```
{}
```

Event(s): 
```
{}
```"""

user = """
```
{template}
```

Here are some observed values:
```
{observed_values}
```

Please return the 3 most likely OCSF events. Rank them from the event that best captures this template to the one that worse captures the template. Please use the full description of the event as well as the fields it defines to guide your decision. If no event is a match, please return "unsure". 

Expected Output Format:
```
event_1
event_2
event_3
```
OR
```
unsure
```
Return one event name per line, without its description, index, or rank number. Do not return anything else."""


def gen_system(events):

    formatting = "**{}**: {}"
    events = "\n\n".join(
        formatting.format(event, json.dumps(value, indent=2))
        for event, value in events.items()
    )
    return system.format(
        events=events,
    )


def few_shot_template(fs_templates, fs_examples, fs_events):
    if not len(fs_templates) == len(fs_examples) == len(fs_events):
        raise ValueError("Length of templates, examples, and events must match")

    if not fs_templates:
        return "Here is the template:"

    rtn = few_shot_header
    for i, (template, examples, event) in enumerate(
        zip(fs_templates, fs_examples, fs_events)
    ):
        rtn += "\n\n" + few_shot_content.format(
            i + 1, template, "\n".join(examples), "\n".join(event)
        )
    rtn += "\n\nEnd of examples. Now here is the template you need to label:"

    return rtn


def gen_prompt(template_example, specs, entry_examples, fs):

    entry_examples = "\n".join(entry_examples)
    if fs:
        return few_shot_template(*fs) + user.format(
            template=template_example, observed_values=entry_examples
        ), gen_system(specs)
    else:
        return user.format(
            template=template_example, observed_values=entry_examples
        ), gen_system(specs)
