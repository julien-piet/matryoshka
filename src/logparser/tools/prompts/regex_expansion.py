from ..classes import Template
from .few_shot_examples import gen_fewshot

system = """Your role is to create log parsers. Log files are collections of entries, each produced by some program to record state. Log lines can be grouped according to the formatting string used to produce them in the original program. These formatting strings define the template of this group of lines. However, we only observe the log entries, so we must infer the templates.

Templates are made of constant parts (fixed parts of the formatting string) and variable parts (variables in the formatting string).  We can separate templates into tokens. Each token represents a single semantic unit within the log entry, and can be made up of individual characters, single words, or even multiple words. Templates should exactly match the formatting string, but not match anything else. All fixed parts of the formatting string should be constants. We now define some useful characteristics of tokens:

** Variability **

Template are made of two types of tokens:

* Variable tokens: These are the variable parts of the template. Tokens are variable if they were a parameter of the original formatting string that produced them. These are any part of the templates that is susceptible to change in other log lines from the same group, even if rarely. Most often, variable tokens represent a distinct concept with independent existence: a resource, device, location, date, time, user, and others. If the template contains any key-value pairs, the value will always be a variable. 

* Constant tokens: These are the fixed parts of the formatting string. They will always be the same in every line produced by this string. They are often descriptions, punctuations, or delimiters of the template. They can be verbs describing the main action of the log line, or other messages that are not parameters but rather descriptive sentences to contextualize and link variable, independent entities. These include keywords and delimiters in key-value pairs. 

** Tokenization **

* Tokens can either be constants or variable, but not both.
* Variable tokens that contain multiple types of data must be broken down into their individual components: They can only include one parameter from the formatting message. For instance, variables that contain two non-related data types that can vary independently should be in different tokens.
* Punctuation should be kept separate from variables, except if that punctuation is part of the variable (such as punctuation used to separate multiple items in a single variable).
* Fields that are structured data formats (dictionaries, arrays, lists, etc.) must be kept as one single token.
* If you encounter key-value pairs, the key (or keyword) should be in a separate token from the value.
* Units associated with variables should be included in the variable: for instance, time units, size units, or distance units, should be part of the entry they follow. 

** Regular expressions **

Variable tokens are associated with a regular expression. These help capture the expected syntax of the variable, as to avoid overfitting. 
While using wildcards such as .* is permitted when needing to capture variables that can take a very wide range of values, we prefer using more precise ones when possible. 
In particular, we prefer using non-greedy operators in regular expressions when relevant, and fitting as closely as possible to the format of the variable. 
Make sure the regular expressions are properly escaped.

** Representation **

* We represent templates as a list of tokens. 
* Variable tokens have format VAR"value"REGEX"regex", where value is an example value this token can take, and regex is the regular expression associated with this variable.
* Constant tokens have format CST"value", where value is the value of the token.
* For example, log line `update-alternatives 2022-10-04 22:32:23: run with --install git` is produced by template `[CST"update-alternatives", VAR"2022-10-04 22:32:23"REGEX"\\d+-\\d+-\\d+\\s+\\d+:\\d+:\\d+", CST": run with --install", VAR"git"REGEX"\\S+"]`


** Instructions **

I will give you some example log entries and the associated templates and regular expressions to capture variables. 
I will also give you a template that is wrong: one of its variable elements has a regular expression that is too narrow.
Uou must replace the regular expression with a more general one that captures more possible values. """


user = """
* Log entries (all belonging to the same template)
```
{entries}
```

* Close template
{template}

This template only matches the first log entries: it does not match the last one. 
This is because variable elements #{element_index} with regular expression {regexp} is too narrow: it does not capture value {mismatched_value} in the last entry.
You must propose a replacement regular expression.

Copy and fill out the following checklist to guide your answer. 
Remember, you must only replace that regular expression. Do not change the template, only return the regex, making sure the new full template will match all the log entries.
The previous examples are given as guidance. You should be consistent with them, and reuse existing regular expressions as much as possible.

[[ CHECKLIST ]]
* First try
[ Output the replacement regex ]

* Semantics
What do values {list_of_values} and {mismatched_value} represent?
[ Update the regular expression if needed and output it.  ]

* Generalizability
Does the regular expression capture  {list_of_values} and {mismatched_value}?
Is it general enough to capture all possible other values this log could produce for this element?
Is it narrow enough to capture the specific format of entries in this element, if any?
If not, adjust the regular expression to capture all possible values but not overcapture.
[ Update the regular expression if needed and output it.  ]

* Final answer
Return the final regular expression and nothing else.
"""


def gen_prompt(
    examples, entries, template, element_index, list_of_values, mismatched_value
):
    if examples is None or len(examples) == 0:
        raise ValueError("No examples provided")

    fewshot_prompts = gen_fewshot(examples, regex=True)

    if not isinstance(entries, list):
        entries = [entries]

    formatted_entries = "\n".join(entries)

    return system, fewshot_prompts + user.format(
        entries=formatted_entries,
        template=template.format_as_example(
            force_match_with_entry=True, regex=True, entry=entries[0]
        ),
        element_index=element_index,
        list_of_values=", ".join(list_of_values),
        mismatched_value=mismatched_value,
        regexp=Template.escape_string(template.elements[element_index].regexp),
    )
