import json
import random

instructions = """# Description Writing Instructions

## Core Principles
1. **Scope**: Write descriptions that characterize the role of variables in log events, not their values.
2. **Independence**: Descriptions must remain valid regardless of the specific values observed in any instance.
3. **Accuracy**: Ensure descriptions align with the technical reality of the variable's role. Make sure to only describe the variable highlighted in the log lines, not other variables.

## Quality Requirements

### Completeness
- Include all information necessary to understand the variable's role
- Cover the full scope of the variable's function in the context
- Explain any relevant relationships with other fields

### Precision
- Use accurate technical terminology
- Avoid ambiguous or vague language
- Be explicit about the variable's purpose
- Maintain consistent technical depth

### Soundness
- Ensure descriptions align with the technical reality of the variable's role
- Avoid misleading or incorrect characterizations
- Stay within the bounds of what is definitively known about the variable

### Language Standards
- Use clear, professional technical writing
- Write in complete sentences
- Maintain consistent verb tense
- Use active voice

## Consistency
- Use the field name to guide you in writing the description. The field name is the name of the variable that is highlighted in the log lines.

## What to Avoid
- Specific values or examples
- Assumptions about typical or possible values
- Instance-specific observations
- References to implementation details that may change
- Speculative functionality"""

system = """You are an expert log parser. You are given a few log lines that have been generated by the same program. One of the fields in these lines has been highlighted using three stars. Your task is to write a description of the highlighter field.

{instructions}

*** Context-Aware Information ***
All variables you are annotating are part of log templates for a specific log file. This log file is described as:
```
{log_desc}
```
"""


user_input = """Consider the field highlighted in the following lines:
{lines}

Field name: {field_name}
Field description: """


def gen_fewshot():
    fewshot_prompts = []
    for fs_lines, fs_field_name, fs_output in fixed_few_shot_examples:
        lines = "\n".join(fs_lines)
        response = fs_output

        prompt = user_input.format(lines=lines, field_name=fs_field_name)
        fewshot_prompts.append({"role": "user", "content": prompt})
        fewshot_prompts.append({"role": "assistant", "content": response})

    return fewshot_prompts


def gen_system(log_desc):
    return system.format(instructions=instructions, log_desc=log_desc)


def gen_prompt(templates, entries_per_template, node_ids, field_name, log_desc):
    if not isinstance(templates, list):
        templates = [templates]
        entries_per_template = [entries_per_template]
        node_ids = [node_ids]

    lines = []
    for template, entry, node in zip(templates, entries_per_template, node_ids):
        node_position = next(
            i for i, elt in enumerate(template.elements) if elt.id == node
        )
        line = template.format_short(
            highlight=node_position,
            force_match_with_entry=True,
            entry=entry,
            color=False,
            print_types=False,
        )
        lines.append(line)

    lines = "\n".join(lines)

    fewshot = gen_fewshot()

    prompt = user_input.format(
        lines=lines,
        field_name=field_name,
    )

    return prompt, fewshot, gen_system(log_desc)


def gen_fixed_fewshot():
    few_shot_examples = []
    example_lines = [
        'Sep 12 10:17:03 tohuvabohu.balabit audispd[1026]: node=server1.company.com type=ANOM_ABEND msg=audit(1410509823.256:24937): auid=1000 uid=1000 gid=***1000*** ses=13 subj=unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023 pid=3759 comm="gmain" exe="/usr/bin/caja" sig=11',
        'Oct 17 22:12:33 tohuvabohu.balabit audispd[1030]: node=server1.company.com type=ANOM_ABEND msg=audit(1413576753.722:253): auid=1000 uid=1000 gid=***1000*** ses=2 subj=unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023 pid=2755 comm="cinnamon" exe="/usr/bin/cinnamon" sig=11',
        'Sep 15 14:23:45 server1.company.com audispd[2047]: node=db2.company.com type=ANOM_ABEND msg=audit(1410678925.123:35891): auid=2000 uid=2000 gid=***2000*** ses=24 subj=unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023 pid=4891 comm="apache2" exe="/usr/sbin/apache2" sig=6',
        'Sep 18 03:15:22 db2.company.com audispd[3098]: node=tohuvabohu.balabit type=ANOM_ABEND msg=audit(1410891322.445:47129): auid=3000 uid=3000 gid=***3000*** ses=31 subj=unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023 pid=5922 comm="mysqld" exe="/usr/sbin/mysqld" sig=11',
        'Sep 21 19:45:11 cache3.company.com audispd[4156]: node=tohuvabohu.balabit type=ANOM_ABEND msg=audit(1411321511.789:58234): auid=4000 uid=4000 gid=***4000*** ses=42 subj=unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023 pid=7845 comm="redis-server" exe="/usr/bin/redis-server" sig=9',
    ]

    example_field_name = "group_id"

    example_output = (
        "The group ID (gid) of the user who started the process being audited."
    )

    few_shot_examples.append(
        (example_lines, example_field_name, example_output)
    )

    example_lines = [
        "update-alternatives 2022-10-04 22:32:23: run with --install /usr/bin/groovysh groovysh /usr/share/groovy/bin/groovysh 20 --slave ***/usr/bin/groovy*** groovy /usr/share/groovy/bin/groovy --slave /usr/share/man/man1/groovysh.1.gz groovysh.1.gz /usr/share/groovy/man/man1/groovysh.1.gz",
        "update-alternatives 2022-10-04 22:35:45: run with --install /usr/bin/java java /usr/lib/jvm/java-11/bin/java 100 --slave /usr/bin/javac javac /usr/lib/jvm/java-11/bin/javac --slave ***/usr/share/man/man1/java.1.gz*** java.1.gz /usr/lib/jvm/java-11/man/man1/java.1.gz",
        "update-alternatives 2022-10-04 22:37:12: run with --install /usr/bin/python python /usr/bin/python3.8 30 --slave /usr/bin/python-config python-config /usr/bin/python3.8-config --slave ***/usr/share/man/man1/python.1.gz*** python.1.gz /usr/share/man/man1/python3.8.1.gz",
        "update-alternatives 2022-10-04 22:38:01: run with --install /usr/bin/gcc gcc /usr/bin/gcc-9 50 --slave ***/usr/bin/g++*** g++ /usr/bin/g++-9 --slave /usr/share/man/man1/gcc.1.gz gcc.1.gz /usr/share/man/man1/gcc-9.1.gz",
        "update-alternatives 2022-10-04 22:39:15: run with --install /usr/bin/node node /usr/local/node-v14/bin/node 40 --slave ***/usr/bin/npm*** npm /usr/local/node-v14/bin/npm --slave /usr/share/man/m san1/node.1.gz node.1.gz /usr/local/node-v14/share/man/man1/node.1.gz",
    ]

    example_field_name = "slave_destination_path"

    example_output = """The destination path for a slave link that is associated with the main alternative. Slave links are secondary symlinks that are updated along with the master alternative."""

    few_shot_examples.append(
        (example_lines, example_field_name, example_output)
    )

    return few_shot_examples


fixed_few_shot_examples = gen_fixed_fewshot()
