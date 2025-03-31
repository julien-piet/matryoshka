import json
import random

system = """You are an expert in databases, specifically schema matching. Your task is to create matches between source and target attributes. For each attribute from the source you always suggest the most relevant attributes from the target. You are excellent at this task. If none of the target attributes are relevant, you should return an empty list.

The source attributes are part of system log templates. A system log, often referred to as a log file, is a recorded file that contains entries documenting events and activities within a computer system, network, or application. These logs serve as a crucial tool for system administrators and IT professionals for monitoring, troubleshooting, and performing security audits. System logs capture a wide range of events, including user logins, system errors, hardware failures, software operations, and network activities.

The target attributes are part of the Unified Data Model (UDM) format. The UDM is a Google Security Operations standard data structure that stores information about data received from sources. It is also called the schema. Google SecOps stores the original data it receives in two formats, as the original raw log and as a structured UDM record. The UDM record is a structured representation of the original log. UDM events contain the following high level fields:

* about: Represents entities referenced by the event that are not otherwise described in principal, src, target, intermediary or observer. For example, it could be used to track email file attachments, domains/URLs/IPs embedded within an email body, and DLLs that are loaded during a PROCESS_LAUNCH event.
* extensions: All other first-class, event-specific metadata goes in this message. Don't place protocol metadata in Extensions; put it in Network.
* intermediary: Represents details on one or more intermediate entities processing activity described in the event. This includes device details about a proxy server or SMTP relay server. If an active event (that has a principal and possibly target) passes through any intermediaries, they're added here. Intermediaries can impact the overall action, for example blocking or modifying an ongoing request. A rule of thumb here is that `principal`, `target`, and description of the initial action should be the same regardless of the intermediary or its action. A successful network connection from A->B should look the same in principal/target/intermediary as one blocked by firewall C: principal: A, target B (intermediary: C).
* metadata: Event metadata such as timestamp, source product, etc.
* network: All network details go here, including sub-messages with details on each protocol (for example, DHCP, DNS, or HTTP).
* observer: Represents an observing entity (for example, a packet sniffer or network-based vulnerability scanner), which is not a direct intermediary, but which observes and reports on the event in question.
* principal: Represents the acting entity that generates the activity described in the event. The principal must include at least one machine detail (hostname, MACs, IPs, port, product specific identifiers like an EDR asset ID) or user detail (for example, username), and optionally include process details. It must NOT include any of the following fields: email, files, registry keys, or values.
* security_result: Security related metadata for the event. A security result might be something like “virus detected and quarantined,” “malicious connection blocked,” or “sensitive data included in document foo.doc.”
* src: Represents a source entity being acted upon by the participant along with the device or process context for the source object (the machine where the source object resides). For example, if user U copies file A on machine X to file B on machine Y, both file A and machine X would be specified in the src portion of the UDM event.
* target: Represents a target entity being referenced by the event or an object on the target entity. For example, in a firewall connection from device A to device B, A is described as the principal and B is described as the target. For a process injection by process C into target process D, process C is described as the principal and process D is described as the target.

### Inputs ###
I will provide the following inputs:
* A source attribute, along with its description, and some example values in context
* A shortlist of target attributes, along with their descriptions, which you must chose from

### Rules ###
I will provide an algorithm that you must follow to match the source and target attributes.
Remember this set of rules:
* Please be consistent:
    - If the attribute you are mapping plays the same role in the log as another you have already mapped, match it to the same target fields. Attributes that play similar roles often have similar descriptions.
    - If the attribute you are mapping is a sibling of another you have already mapped, you should map it to sibling UDM attributes. Sibling attributes are fields that refer to the same entity, but describe different aspects (e.g. process name and process id). These have the same UDM prefix, but only the attribute leaf changes (e.g. the last field in the path is different).
* Please only map source attributes to destination attributes if the role and nature of the source attribute is accurately captured by the destination attribute.
* It is OK for the destination attribute to be more generic than the source attribute
* Do not map a source attribute to a destination attribute that is more specific than the source attribute.
* Return all target attributes that are a good match for the source attribute. Make sure these are consistent with the rules, and don't contradict themselves (e.g. you cannot map a source attribute both to the principal and the target fields).
* If none of the target attributes are an exact match, you should return an empty list.

### Algorithm ###
Your job is to match the schemas. This is based on the following algorithm:
```matching algorithm
print("### Explanation ###")
for each target attribute T:
    if the target attribute does not refer to the same entity as the source attribute:
        print the attribute name in quotes, and explain why the target attribute does not refer to the same entity as the source attribute
        continue
    elif the target attribute does not have the same role as the source attribute:
        print the attribute name in quotes, and explain why the target attribute does not have the same role as the source attribute
        continue
    else:
        print the attribute name in quotes, followed by an explanation of why this target attribute is a good match
        list.append(target attribute)

print("### Mapping ###")
return the selected target attributes in the list, in json format, with their descriptions, using the following format:
```json
{
  'field1': 'description1',
  'field2': 'description2',
}
```
```"""

user_input = """### Source Attribute ###

Consider the attribute highlighted using three stars in the following templates:
```
{templates}
```

We've observed it take the following subset of values:
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

### Mapping

```json
{mapping}
```"""


def gen_fewshot(examples):
    fewshot_prompts = []
    for (
        fs_templates,
        fs_element,
        fs_matches,
        fs_description,
        fs_mapping,
        fs_demonstration,
        fs_return,
    ) in examples:
        if isinstance(fs_element, list):
            fs_templates = fs_templates[0]
            fs_element = fs_element[0]

        templates = random.sample(fs_templates, min(10, len(fs_templates)))
        values = random.sample(fs_matches, min(20, len(fs_matches)))
        templates = "\n".join(t.highlight(fs_element.id) for t in templates)
        values = ", ".join(values)
        prompt = user_input.format(
            templates=templates,
            observed_values=values,
            description=fs_description,
            fields=json.dumps(fs_mapping, indent=2),
        )
        response = user_output.format(
            demonstration=fs_demonstration,
            mapping=json.dumps(fs_return, indent=2),
        )
        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append({"role": "assistant", "content": response})
    return fewshot_prompts


def gen_system():
    return system


def gen_prompt(elt, few_shot_examples, mapping, description):
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

    prompt = user_input.format(
        templates=templates,
        observed_values=values,
        description=description,
        fields=json.dumps(mapping, indent=2),
    )

    return prompt, fewshot, gen_system()
