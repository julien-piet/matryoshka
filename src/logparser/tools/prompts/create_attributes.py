import json
import random

system = """You are an expert in databases, specifically schema matching. We are mapping elements from source log messages to a target structured format. Your task is to map source attributes to destination attributes. You are excellent at this task. 

The source attributes are part of system log templates. A system log, often referred to as a log file, is a recorded file that contains entries documenting events and activities within a computer system, network, or application. These logs serve as a crucial tool for system administrators and IT professionals for monitoring, troubleshooting, and performing security audits. System logs capture a wide range of events, including user logins, system errors, hardware failures, software operations, and network activities. 

The target attributes are part of the OCSF format. The Open Cybersecurity Schema Framework (OCSF) is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

### Inputs ###
I will provide the following inputs:
* A source attribute, along with its description, and some example values in context
* A list previously created target attributes, along with their descriptions

### Goal ###
You must either select an existing target attribute that is a perfect match for the source attribute, or create a new target attribute that best represents the source attribute.

### Rules ###
I will provide an algorithm that you must follow to match the source and target attributes.
Remeber this set of rules:
* Please be consistent: ff the attribute you are mapping is the same as another you have already mapped, match it to the same target field. Consider attributes to be the same if they share a very similar description and/or play the same exact role as another. 
* Prefer creating a new target attribute if none of the existing target attributes are an exact match.
* Attribute names must be a perfect match for the role of the variable. It must not overgeneralize or misrepresent the variable.
* Names must be snake cased.


### Algorithm ###
Use the following algorithm:
```matching algorithm
print("### Explanation ###")
if there are no target attributes:
    print("Target attribute list is empty, so we will create a new target attribute.")
    target <- new target, with an appropriate short name and description, according to the rules.
else:
    for each target attribute T:
        if the target attribute does not refer to the same entity as the source attribute:
            print the attribute name in quotes, followed by an explanation of why this target attribute is not a good match
            continue
        elif the target attribute does not have the same role as the source attribute:
            print the attribute name in quotes, followed by an explanation of why this target attribute is not a good match
            continue
        elif the target attribute is too broad or too specific:
            print the attribute name in quotes, followed by an explanation of why this target attribute is not a good match
            continue
        else:
            print the attribute name in quotes, followed by an explanation of why this target attribute is a good match
            list.append(target attribute)

    if list is empty:
        print("No target attributes match the source attribute, so we will create a new target attribute.")
        target <- new target, with an appropriate short name and description, according to the rules.
    else:  
        target <- best match from list
        print("A target attribute has been selected.")
    
print("### Mapping ###")
return the target using the following format:
```json
{{
    'name': 'target name',
    'description': 'target description',
}}
```
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

### Target Attributes ###

Here is a list of target attributes:
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
        fs_templates,
        fs_element,
        fs_matches,
        fs_mapping,
        fs_candidates,
        fs_demonstration,
        fs_description,
    ) in examples:
        if isinstance(fs_element, list):
            fs_templates = fs_templates[0]
            fs_element = fs_element[0]

        templates = random.sample(fs_templates, min(10, len(fs_templates)))
        values = random.sample(fs_matches, min(20, len(fs_matches)))
        templates = "\n".join(t.highlight(fs_element.id) for t in templates)
        values = ", ".join(values)

        fs_candidates = [
            {"name": c.name, "description": c.description}
            for c in fs_candidates
        ]
        prompt = user_input.format(
            templates=templates,
            observed_values=values,
            type=fs_element.type,
            description=fs_description,
            fields=json.dumps(fs_candidates, indent=2),
        )
        fs_return = {
            "name": fs_mapping.name,
            "description": fs_mapping.description,
        }
        response = user_output.format(
            demonstration=fs_demonstration,
            mapping=json.dumps(fs_return, indent=2),
        )
        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append({"role": "assistant", "content": response})

    return fewshot_prompts


def gen_system():
    return system


def gen_prompt(elt, few_shot_examples, candidate_attributes, description):
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

    if few_shot_examples:
        fewshot = gen_fewshot(few_shot_examples)
    else:
        fewshot = []

    try:
        mapping = [
            {"name": c.name, "description": c.description}
            for c in candidate_attributes
        ]
    except Exception as e:
        print(e)
        breakpoint()

    prompt = user_input.format(
        templates=templates,
        observed_values=values,
        type=elements[0].type,
        description=description,
        fields=json.dumps(mapping, indent=2),
    )

    return prompt, fewshot, gen_system()
