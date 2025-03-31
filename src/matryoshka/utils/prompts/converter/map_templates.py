import json
import random

from ...OCSF import OCSFObjectEncoder

system = """You are an expert in databases. You are given a target schema, and an instructed data source. You must fill in attributes in the target that can directly be inferred from the source. You are excellent at this task. 

The source is a system log template. A system log, often referred to as a log file, is a recorded file that contains entries documenting events and activities within a computer system, network, or application. These logs serve as a crucial tool for system administrators and IT professionals for monitoring, troubleshooting, and performing security audits. System logs capture a wide range of events, including user logins, system errors, hardware failures, software operations, and network activities. Templates are a way to group log entries that share the same format. Each template is made up of tokens, which can be constant (fixed parts of the log message) or variable (placeholders for values that change between log entries).

The target is an OCSF event. The Open Cybersecurity Schema Framework (OCSF) is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

### Target Event ###
The target OCSF event is described as: {description}. 

### Inputs ###
I will provide the following inputs:
* A source template (as a format string where variables are replaced with their type), along with its description, and some example values in context
* A subset of target attributes, along with their descriptions, which you must try to fill out.

### Rules ###
I will provide an algorithm that you must follow to fill out the event using information from the source.
Remeber this set of rules:
* Please be consistent with previous completions
* Please only assign values to fields explicitly mentioned in the source. Do not guess values or use external knowledge.

### Algorithm ###
You must adhere to this algorithm to guide your response. 
```
print("### Explanation ###")
for each target attribute T:
  if the source template does not contain information pertaining to this target attribute:
    print the attribute name in quotes, followed by an explanation of why it cannot be inferred from the source
    continue
  elif the source template does contain information pertaining to this target attribute, but part or all of that value is a variable field in the template (denoted by a type in brackets):
    print the attribute name in quotes, followed by an explanation of why it cannot be inferred from the source
    continue
  elif the source template does contain information pertaining to this target attribute, but it is not enough to fully determine the value of the target attribute:
    print the attribute name in quotes, followed by an explanation of why it cannot be inferred from the source
    continue
  else:
    print the attribute name in quotes, followed by an explanation of what value to assign to the target attribute and why.
    mapping[target attribute] = value
    
print("### Mapping ###")
return the mapped target attributes in json format using the following format:
```json
{{
    'field1': 'value1',
    'field2': 'value2',
    ...
}}
```
```"""


user_input = """### Template ###

Consider the following template
```
{template}
```

This template takes, among others, the following values:
```
{observed_values}
```

This template is best described as:
```
{description}
```

### Target Attributes ###

We would like to fill out some of the following attributes:
```     
{fields}
```"""

user_output = """### Explanation ###
{demonstration}

### Mapping ###
```json
{mapping}
```"""


def gen_fewshot(examples):
    fewshot_prompts = []
    for (
        fs_template,
        fs_examples,
        fs_description,
        fs_demonstration,
        fs_candidates,
        fs_mapping,
    ) in examples:
        values = "\n".join(fs_examples)
        prompt = user_input.format(
            template=fs_template,
            observed_values=values,
            description=fs_description,
            fields=json.dumps(
                fs_candidates,
                indent=2,
            ),
        )
        response = user_output.format(
            demonstration=fs_demonstration,
            mapping=json.dumps(fs_mapping, indent=2),
        )
        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append({"role": "assistant", "content": response})

    return fewshot_prompts


def gen_system(event_description):
    return system.format(description=event_description)


def gen_prompt(params, few_shot_examples, event_description):
    template, examples, description, attributes = params

    if few_shot_examples:
        fewshot = gen_fewshot(few_shot_examples)
    else:
        fewshot = []

    prompt = user_input.format(
        template=template,
        observed_values="\n".join(examples),
        description=description,
        fields=json.dumps(attributes, indent=2),
    )

    return (
        prompt,
        fewshot,
        gen_system(event_description),
    )
