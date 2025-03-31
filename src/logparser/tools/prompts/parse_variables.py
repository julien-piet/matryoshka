from typing import List

system = """Please turn this into the body of a Python function that takes in a string and returns the string as the specified data type.
Do not use any external libraries, or the built-in eval or exec function.
Do not try to handle any exceptions or errors in the function.
Assume that you can use the following libraries as imported below:
```
from datetime import datetime
import time
import re
```

Complete the body of parse and return the entire function as\n```python\n# YOUR CODE HERE\n```.
Respond ONLY with the function body, such that it can be evaluated directly by the eval() function, not the entire function definition.
"""

user = """The specified data type is {data_type}. This data type has the following properties:
{data_type_description}
{examples}

def parse(to_parse: str) -> {data_type}:
    # YOUR CODE HERE
"""


def get_example_str(examples: List[str]):
    ret = ""
    for i, example in enumerate(examples):
        ret += f"Example {i+1}: {example}\n"
    return ret
