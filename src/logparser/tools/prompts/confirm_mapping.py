import json
import random

system = """You are an expert log parser. You are given a variable that appears in multiple log templates. You are also given a description. Your task is to say if this description is accurate or not. 
It is ok if the description is too vague, but it must not be incorrect. Simply reply by YES or NO."""

user = """### Variable

Consider the variable highlighted using three stars in the following templates:
```
{templates}
```

This variable has been assigned type {type}. We've observed it take the following subset of values:
```
{observed_values}
```

### Field 

{key}: {description}

### Answer"""


def gen_fewshot(examples):
    return []


def gen_system():
    return system


def gen_prompt(elt, few_shot_examples, description):
    templates, _, element, matches = elt
    templates = random.sample(templates, min(10, len(templates)))
    values = random.sample(matches, min(20, len(matches)))
    templates = "\n".join(t.highlight(element.id) for t in templates)
    values = ", ".join(values)

    if few_shot_examples:
        fewshot = gen_fewshot(few_shot_examples)
    else:
        fewshot = []

    prompt = user.format(
        templates=templates,
        observed_values=values,
        type=element.type,
        key=description[0],
        description=description[1],
    )

    return prompt, fewshot, gen_system()
