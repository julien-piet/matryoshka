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
* For example, log line `update-alternatives 2022-10-04 22:32:23: run with --install git` is produced by template `[CST"update-alternatives", VAR"2022-10-04 22:32:23"REGEX"\\\\d+-\\\\d+-\\\\d+\\\\s+\\\\d+:\\\\d+:\\\\d+", CST": run with --install", VAR"git"REGEX"\\S+"]`


** Instructions **

I will give you some example log entries and the associated templates and regular expressions to capture variables. 
I will also give you a template that is wrong: one of its constant elements must be replaced in order to match a set of provided entries. 
Your task is to replace that constant element with an approriate set of elements. """


user = """
* Log entries (all belonging to the same template)
```
{entries}
```

* Close template
{template}

This template only matches the first log entries: it does not match the last one. 
This is because constant elements #{element_index} with value {element} is too constrained: it does not capture value {mismatched_value} in the last entry.
You must replace it with a set of elements that will match all the entries. You have two options:

a/ If {element} and {mismatched_value} refer to the same type of entity but have completely different values, you should replace element #{element_index} with a variable element.
b/ If {element} and {mismatched_value} are made up of multiple parts, some of them entities that changes between both values, others static and the same in both values, replace element #{element_index} with a more granular separation, where the constant parts are isolated from the variable parts. 

Copy and fill out the following checklist to guide your answer. Remember, you must replace the element in the original template so that the new template matches all the log entries. 
The previous examples are given as guidance. You should be consistent with them, and reuse existing regular expressions as much as possible.

[[ CHECKLIST ]]
* First try
[ Output the replacement element(s) for the template ]

* Semantics
What do {element} and {mismatched_value} represent?
What are other possible values you can imagine these taking in other log instances?
[ Answer the questions, then update the replacement element(s) if needed. Output the updated element(s). ]

* Consistency with rules
If the replacement element either a single variable, or a set of constants and variables?
Does it only capture the mismatched value, nothing before, and nothing after?
If not, adjust your answer to follow these rules.
[ Answer the questions, then update the replacement element(s) if needed.Output the updated element(s). ]

* Comparing both values
Do {element} and {mismatched_value} have any parts in common? Do you believe these parts will be the same for all other instances of this template? If so, use constant elements to capture the static parts, and variable elements to capture the variable parts. If not, use a single variable element to capture these values.
[ Answer the questions, then update the replacement element(s) if needed. Output the updated element(s). ]

* Singular and plural values
Does your current answer contain both constant and variable elements? 
If so, are there any plural or singular parts in the constants that depend on the variable parts? 
Make sure these are accounted for in the regular expressions.
[ Answer the questions, then update the replacement element(s) if needed.Output the updated element(s). ]

* Regular expression
Do the new elements capture both {element} and {mismatched_value}?
Can they capture any other possible values this log template could produce?
Are the regular expression properly escaped?
[ Answer the questions, then update the replacement element(s) if needed. Output the updated element(s). ]

* Final answer
Return the final replacement element(s) after taking all these points into consideration, making sure to adhere to all the instructions. Do not return the full template.
"""


def gen_prompt(
    examples, entries, template, element_index, element, mismatched_value
):
    if examples is None or len(examples) == 0:
        raise ValueError("No examples provided")

    fewshot_prompts = gen_fewshot(examples, regex=True)

    if not isinstance(entries, list):
        entries = [entries]

    formatted_entries = "\n".join(entries) + "```"

    return system, fewshot_prompts + user.format(
        entries=formatted_entries,
        template=template.format_as_example(
            force_match_with_entry=True, regex=True, entry=entries[0]
        ),
        element_index=element_index,
        element=element,
        mismatched_value=mismatched_value,
    )
