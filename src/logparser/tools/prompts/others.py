unit_prompt = """Your first task is to identify a regular expression to separate log files into single log entries. 
I will provide a snippet of a log file, please simply return the regular expression. 
Please return the shortest and simplest regular expression that fits the requirements. 
Keep in mind that this is just a snippet of data: values in the log files will likely be different for 
other examples, so the separation regular expression must be universal for this log format. 
Do not use specific times or dates in the regular expression, and match the shortest possible separator. 
In many cases this will be a single line return, but can be more complex if needed.

[[LOG]]
```
{}
```
[[/LOG]]

Regular Expression:"""


static_prompt = """Your first task is to identify from the following log lines if there is any static information that appears in each entry. 
At this stage, you are only performing part 1/ of the workflow: Do not focus on the variable parts of the log lines, only focus on identifying the static parts, and identifying what variables are part of this static part. 
Here are some sample log lines: 
[[LOG]]
```
{}
```
[[/LOG]] 
Please write a regular expression to parse the static parts of the log lines, augmenting this regular expression with a type for each variable. Here is a list of possible types:
* DateTime (type datetime)
* Hostname (type hostname)
* Process name (type process_name)
* Process ID (type process_id)
* IP address (type ip_addr)
* Hardware address (type hw_addr)
* Port (type port)
* Protocol (type protocol)
* Username (type username)

Some additional instructions:
* Prioritize the aforementioned types. Only introduce new types if the variable is not of any of the previous types. 
* You do not need to match all of these types. Only capture a type if is present in the static part of every log line
* Types can only be attributed to variables of that type: for example, only a date can have the "Date" type. If a static piece of information is not of one of the previous types, create a new type for it.

Finally, some guidelines about the expected result:
* Please analyze the provided log lines to identify common patterns or elements that consistently appear across different entries. Particularly, focus on identifying identifiers or names that are present in all types of log entries. 
* Ignore elements that do not appear in every log line but concentrate on those that are recurrent across various log messages.
* Only aim to parse the static part of the log lines. This is the part of the log line that is common to every log line in the dataset. Do not match fields that are present in only part of the log lines, these are not static and are not relevant for this step. Do not use conditional statements. 
* Conversely, the regex you generate MUST match every log entry in the log file, so do not overfit to a subset of log lines. 
* If the logs do not have any static part, simply return "No Static Part"

Please start by explaining your thought process, and then provide the regular expression (or the message "No Static Part") after printing "Regular Expression:" Do not print anything else than the regular expression after "Regular Expression:".
"""

static_prompt_error_message = """This regular expression is incorrect, as it does not match all the lines I need. Here is a list of lines in the log file that are not matched by your regular expression:
[[LOG]]
```
{}
```
[[/LOG]]
Your response must also match all the lines I previously gave you.
If it is not possible to construct a regular expression for a prefix that matches both these lines and the previous ones, please return "No Static Part".
Again, please start by explaining your thought process, and then provide the regular expression (or the message "No Static Part") after printing "Regular Expression:".
"""
