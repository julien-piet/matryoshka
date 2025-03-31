import json
import random

system = """You are an expert in databases, specifically schema matching. Your task is to map source attributes to target OCSF fields. For each attribute from the source you always suggest the most relevant attributes in OCSF. You are excellent at this task. If no OCSF field is relevant, you should return an empty list.

The source attributes are part of system log templates. A system log, often referred to as a log file, is a recorded file that contains entries documenting events and activities within a computer system, network, or application. These logs serve as a crucial tool for system administrators and IT professionals for monitoring, troubleshooting, and performing security audits. System logs capture a wide range of events, including user logins, system errors, hardware failures, software operations, and network activities. 

The Open Cybersecurity Schema Framework (OCSF) is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

### Inputs ###
I will provide the following inputs:
* A source attribute, along with its description, and some example values in context

### Outputs ###
Return the list of flattened OCSF attributes that the input attrbute maps to, if any, in a json formated list, for example:
```json
[
    'ssh_activity.src_endpoint.ip',
    'ssh_activity.actor.ip'
]
```
If no OCSF attribute is relevant, return an empty json list:
```json
[]
```"""


user_input = """### Source Attribute ###

Consider the attribute highlighted using three stars in the following templates:
```
{templates}
```

This attribute has been assigned type {type}. We've observed it take the following subset of values:
```
{observed_values}
```

This attribute plays the following role in the log templates:
{description}
"""


def gen_system():
    return system


def gen_prompt(elt, description):
    templates, _, elements, matches = elt
    if not isinstance(elements, list):
        templates = [templates]
        elements = [elements]

    templates = [
        t.highlight(element.id)
        for template_list, element in zip(templates, elements)
        for t in template_list
    ]
    templates = random.sample(templates, min(10, len(templates)))
    values = random.sample(matches, min(20, len(matches)))
    templates = "\n".join(templates)
    values = ", ".join(values)

    prompt = user_input.format(
        templates=templates,
        observed_values=values,
        type=elements[0].type,
        description=description,
    )

    return prompt, [], gen_system()
