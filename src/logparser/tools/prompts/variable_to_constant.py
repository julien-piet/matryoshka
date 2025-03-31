variable_confirmation_system_prompt = """
You are a helpful assistant. Your role is to create semi-structured data parsers. The data is a collection of entries, each following a specific template. There can be multiple templates. Each template is made up o constant elements (fixed parts of the template that occur in every instance of the template), and variable elements (parts of the template that change between entries).

More precisely, each entry can be broken down into elements. Elements can be made up of individual characters, single words, or even multiple words. They represent a single semantic unit.

We now define some useful characteristics of elements:

**Variability**

Each element can either be constant, meaning they do not change in entries from the same template, or they can be variable, meaning they can change value in different entries. Elements are variable if they are susceptible to change in other entries, even if rarely. If there is any chance an element might take other values in other instances, we consider it to be variable.

**Type**

Elements can have different types. They can be:
* entities: they represent a distinct concept with independent existence: a resource, device, location, date, time or user. If an element is an entity, we will always consider it to be variable, since it could change in the future. Entities are any element that are not part of the template, but refer to real things: time, location, device, resource, person, etc. We refer to this type as ENTITY
* messages: they describe an action, a message, which do not refer to any independent entity. These include fixed keywords that precede entities. We refer to this type as MSG
* punctuation: non-alphabetical symbols used to structure the entry, and sometime separate elements. These are often used to isolate variables. We refer to this type as PUNCT.


**Instructions**

I will provide a template, some example entries, and the index of one of the template elements.
 
This element was initially parsed as a variable, but in practice, it only takes a single value. Your task is to check if this is indeed a variable.
"""

variable_confirmation_prompt = """Here is a list of templates:

```templates
{}
```


These captured the following lines:

```entries
{}
```

These templates share a common prefix, in which the {} field is labeled as variable, however it only takes value "{}" in practice. Can you tell if this field refers to an entity or a message?



Here are some guidelines to help you in this determination.

* If the content of a field refers to a subject of the template, referring to a real concept such as a location, device, resource..., the field is a variable.

* If the value in the field is not an entity, the field should be constant.

Please first give an answer, then explain your reasoning and critique yourself, and then provide a final answer.

Follow the following output format:

### FIRST TRY

[Provide your first determination]

### CRITIQUE

[Your critique, analysis and explanation of each tag and separation goes here. In particular, if the prompt provided examples of other parsings, you must make sure that you are consistent with the examples, and verify for each field that the annotation is consistent with the annotation of fields with similar contexts in the examples. For each field, indicate with fields in the example are closest to it.]  

### RESULT

[print TRUE if you consider the field should be an entity, and FALSE if not]"""


def generate_variable_confirmation_prompt(
    templates, value, matches, element_idx
):
    template_str = "\n\n".join(
        [
            f"Template #{t_idx}: "
            + " | ".join(
                elt.value if not elt.is_variable() else "VARIABLE"
                for elt in t.elements
            )
            for t_idx, t in enumerate(templates)
        ]
    )

    example_str = "\n\n".join(
        [
            f"Example matches for template #{k}:\n" + "\n".join(v)
            for k, v in enumerate(matches)
        ]
    )

    element_idx = (
        f"{element_idx}th"
        if element_idx >= 4
        else ["first", "second", "third", "fourth"][element_idx]
    )

    return variable_confirmation_prompt.format(
        template_str, example_str, element_idx, value
    )
