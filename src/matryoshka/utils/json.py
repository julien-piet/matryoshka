import json
import re

from ..genai_api.api import Caller, LLMTask


def standard_parse_json(json_str: str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            return json.loads(json_str.strip())
        except json.JSONDecodeError:
            try:
                return json.loads(
                    json_str.strip().replace("\\", "\\\\").strip()
                )
            except json.JSONDecodeError:
                search = re.findall(r"```json(.*?)```", json_str, re.DOTALL)
                if search:
                    value = search[-1].strip()
                    return standard_parse_json(value)
                else:
                    raise e


generation_kwargs_global = {
    "response_mime_type": "application/json",
}


def parse_json(
    json_str: str,
    caller: Caller,
    task: LLMTask = None,
    max_retries: int = 3,
    model: str = "gemini-2.5-flash",
    response_schema=None,
):
    if not model:
        model = "gemini-2.5-flash"
    generation_kwargs = {k: v for k, v in generation_kwargs_global.items()}
    if response_schema:
        generation_kwargs["response_schema"] = response_schema

    retries = 0
    while retries < max_retries:
        try:
            return standard_parse_json(json_str)
        except json.JSONDecodeError as e:
            if task is not None:
                task.update_conversation(
                    json_str,
                    f"We still get an error: {str(e)}. Please fix the original JSON and try again. Return only JSON. Do not return Markdown.",
                )
            else:
                system_prompt = "Return only JSON. Do not change the content or meaning of the JSON: Only fix the syntax."
                user_prompt = f"""
The JSON we are trying to parse is
```json
{json_str}
```

We get the following error: {str(e)}. Please fix the JSON. Return only JSON. Keep the schema the same as the original JSON. Do not change the content of the JSON. Do not change anything that does not need to be changed."
"""
                task = LLMTask(
                    system_prompt=system_prompt,
                    model=model,
                    message=user_prompt,
                    max_tokens=8192,
                    temperature=0,
                )
            orig_json, json_str = (
                json_str,
                caller(task, generation_kwargs=generation_kwargs).candidates[0],
            )
            # gemini tends to return the JSON in Markdown format, so we need to extract the JSON
            search = re.findall(r"```json(.*?)```", json_str, re.DOTALL)
            if search:
                json_str = search[-1].strip()

            # Check if the returned JSON is too different from the original JSON
            def longest_common_substring(str1, str2):
                m = len(str1)
                n = len(str2)
                lcsuff = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
                result = 0
                for i in range(m + 1):
                    for j in range(n + 1):
                        if i == 0 or j == 0:
                            lcsuff[i][j] = 0
                        elif str1[i - 1] == str2[j - 1]:
                            lcsuff[i][j] = lcsuff[i - 1][j - 1] + 1
                            result = max(result, lcsuff[i][j])
                        else:
                            lcsuff[i][j] = 0
                return result

            original_length = len(orig_json)
            lcs_length = longest_common_substring(orig_json, json_str)
            if lcs_length / original_length < 0.1:
                json_str = orig_json

            retries += 1

    raise ValueError(f"Error parsing JSON {json_str}")
