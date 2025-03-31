few_shot = """
* Example Log Entries
```
{entries}
```

* Template
{template}
"""


def gen_fewshot(examples, regex=False):
    fewshot_prompts = []
    for example_entries, template in examples:
        entries = "\n".join(example_entries)
        prompt = few_shot.format(
            entries=entries,
            template=template.format_as_example(
                force_match_with_entry=True,
                regex=regex,
                entry=example_entries[0],
            ),
        )
        fewshot_prompts.append(prompt)
    fewshot_prompts = (
        "[ EXAMPLES ]\n\n" + "\n\n".join(fewshot_prompts) + "\n\n[ /EXAMPLES ]"
    )
    return fewshot_prompts
