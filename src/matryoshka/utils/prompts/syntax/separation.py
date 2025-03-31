import json

import typing_extensions as typing


class Token(typing.TypedDict):
    is_variable: bool
    value: str


token_item_schema_dict = {
    "type": "object",
    "properties": {
        "is_variable": {"type": "boolean"},
        "value": {"type": "string"},
        "placeholder": {"type": "string"},
    },
    "required": ["is_variable", "value"],
}

response_schema = {
    "type": "array",
    "items": token_item_schema_dict,
}

always_variable = [
    "timestamp",
    "IP address",
    "hostname",
    "port",
    "username",
    "command path",
    "file path",
    "resource path",
    "URL",
    "email address",
    "phone number",
    "MAC address",
    "UUID",
    "value in quotes",
    "unique identifier",
    "authentication method",
    "hash value",
    "hex value",
]

system = """Your are an expert systems analyst. Your role is to create log parsers. Log files are collections of entries, where each entry is produced by a program to record its state. Log lines can be grouped according to the formatting string that produced them in their original program. These formatting strings define the template for each group of lines. Since we only observe the log entries, we must infer these templates.

Templates consist of static parts (fixed text from the formatting string) and variable parts (variable entries in the formatting string). Templates can be broken down into tokens, where each token represents a single semantic unit within the log entry. A token can be a single character, word, or multiple words. Templates aim at capturing as much information as possible from the log entries, so that the logs can be queried and analyzed. 

Your task is to infer a template from a set of log entries. Here are the key concepts about tokens:

### Variability ###

Templates contain two distinct types of tokens:

* Variable tokens, or entities: These represent parameters from the original formatting string. They are parts of the template that can vary between log lines from the same group. Variable tokens must represent distinct entities like resources, devices, locations, dates, times, or users. In key-value pairs, the value is always a variable token. Variables cannot be messages, descriptions, actions, or complete sentences - they must be specific parameters. We also refer to these as entities. Values can be variables even if they do not change between log lines, as long as they represent entities or parameters. Examples of entities that should always be variable tokens are: timestamps, IP addresses, hostnames, ports, usernames, paths, URLs, usernames, email addresses, phone numbers, MAC addresses, UUIDs, values in quotes, unique identifiers, authentication methods, etc. 

* Static tokens: These are the fixed parts of the formatting string that appear identically in every line produced by this template. They typically include descriptions, punctuation marks, or structural elements that provide context and connect variable entities. Static tokens include verbs describing actions, descriptive messages, and keywords in key-value pairs. They encompass anything that is not strictly a variable. Entities that do not change between log lines should NOT be treated as static, but as variables.

### Tokenization ###

* Tokens can either be static or variable, but not both.
* Static tokens should be broken down into their most granular components: Tokens should break at punctuation or other separators. Queriable concepts should be isolated into their own tokens. Action verbs or nouns should be separate tokens. In short, make sure tokens are semantic groupings of related words into logical phrases.
* Variable tokens that contain multiple types of data must be broken down into their individual components: They can only include one parameter from the formatting message. For instance, variables that contain two non related data types that can vary independently should be in different tokens. Pay attention to separators, these often indicate there are multiple variable.
* Punctuation should be kept separate from variables, except if that punctuation is part of the variable (such as punctuation used to separate multiple items in a single variable).
* Valid json data formats (dictionaries, arrays, lists, etc.) must be kept as one single token.
* If you encounter key-value pairs, the key (or keyword) should be in a separate token from the value. The key should always ve static, the value should always be variable (even if it does not change in the provided examples). Key value pairs are sets of tokens in which one represents the name of the field, and the other is the value, such as "port 5836", "uid=0", or "retries: 1". 
* If you encounter nested key-value pairs (key-value pairs that are part of the value of a larger key-value pair), the inner-most key-value pair shouyld be separted. For instance, in "message='port:5836,uid:0'", "port" and "uid" should be separate constant tokens, and "5836" and "0" should be separate variable tokens.
* Units associated with variables should be included in the variable: for instance, time units, size units, or distance units, should be part of the entry they follow.
* Fields cannot be optional: variables cannot be empty.
* Some data types have specific rules:
** Hostnames, IP addresses and ports should be in separate tokens. "127.0.0.1:53" should be broken down into two variable tokens: "127.0.0.1" and "53".
** Paths and URLs should be kept as a single token: do not separate them into sub tokens. "https://google.com/search?q=log+parser" is a single variable token. So is "/var/log/sshd.log".
** Dates and times should be merged into a single token if adjacent. "Apr  7, 2025 12:30:00" is a single variable token.
** Hex values should include the 0x prefix if present. "0x12345678" is a single variable token.

### Regular expressions ###

Variable tokens are associated with a regular expression. These help capture the expected syntax of the variable. A good regular expression should match all possible values the variable tokens of a given type can take. They should capture specifics about the expected data format - for example, a date, a time, a file path, or a user name. They should not be specific to observed subsets of values, but generalize to any possible value of the same type. Similar values accross templates should reuse the same regex. Do not use capturing groups in the regex. 

### Avoiding Overcapture ###

Avoid overcapture by ensuring the regular expression captures the structure of the variable. For instance, json objects should be captured using `\\{.*\\}` and arrays should be captured using `\\[.*\\]`.

### Consistency ###

* When creating templates, ensure that the same entity is represented in the same way across all templates. For example, a timestamp including a date and time is parsed as a single entity in one template, it should be parsed the same way in all templates.
* If a variable is represented as a single token in one template, it should be represented as a single token in all templates.
* If a variable is broken down into multiple tokens in one template, it should be broken down in the same way in all templates.
* Reuse the same regular expressions for variables that have the same structure across templates.

### Representation ###

* We represent templates as a JSON list of tokens, which are objects with the following fields:
    * "is_variable": a boolean indicating if the token is an entity or a variable (true) or a static part of the formatting string (false)
    * "value": the value of the token
* For example, log line `update-alternatives 2022-10-04 22:32:23: uid=0 run with --install git` is produced by template:
```json
[
    {"is_variable": false, "value": "update-alternatives"},
    {"is_variable": true, "value": "2022-10-04 22:32:23", "placeholder": "<DATETIME>", "regex": "\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}"},
    {"is_variable": false, "value": ":"},
    {"is_variable": false, "value": "uid="},
    {"is_variable": true, "value": "0", "placeholder": "<UID>", "regex": "\\d+"},
    {"is_variable": false, "value": "run with --install"},
    {"is_variable": true, "value": "git", "placeholder": "<COMMAND>", "regex": "\\S+"}
]
```

### Rules ###

"""

rules = """I will give you log entries. You will output the template that matches the entries. 
Make sure you take into account all parts of the entries, including any punctuation or special characters.
You must adhere to all guidelines about tokenization, and the definition of variables and static tokens.
Most importantly, ensure your responses are consistent with previous templates you have generated.
Entities in new templates should be parsed in the *exact* same way as in previous templates.

In order to complete this tasks, follow the following algorithm:
```algorithm
print("<explanation>")
First, look at prior templates and explanations to understand the expected format. Ensure entities are parsed in the same way as in previous templates.
Print a description of the log lines consistent with the previous descriptions you wrote. Do not include more a less information than necessary and that is provided in previous descriptions.

print("Placeholders")
Print the first line where entities are replaced with placeholders. All entities must be present. Do not include placeholders for missing values (e.g., if there is a key but no value).

print("Key Value Pairs:")
What is the format of the string that produced these suffixes? Are there any key-value pairs? List all key value pairs. Remeber, keys are static, values are entities. If you encounted nested key-value pairs, break them down into their most granular components, separating the inner-most components into their own variables and constants.

print("Static tokens:")
Which part is a static message, description, or action? List all the static tokens. 
Concepts that could be queried should be isolated into their own tokens. 
Action verbs, nouns and semantic units should be in their own tokens.
Break down static tokens into their most granular components, separating punctuation and other separators into their own tokens.

print("Entities:")
Which part is an entity? List all the entities, including:
* those that do not change.
* any value in a key value pair (except if the value is empty)
* any value what is one of: {}
Remember, entities cannot contain messages, descriptions, actions, or complete sentences - they must be specific parameters.
For each entity, print an example value and a regular expression that matches the expected syntax of the entity. Make sure this regular expression mathches all possible values the entity can take.

print("</explanation>")

template = Use the explanation and placeholder above when generating the template for this message.
Make sure any token that was assigned a placeholder is a variable token in the template.

template = set all variable values in the template to be equal to their values in the first log entry

print(template) IN JSON
```
"""

user_input_suffix = """### Full Log Lines ###
{entries}

### Suffix (only parse these!) ###
{suffixes}
"""

user_input = """### Full Log Lines ###
{entries}"""

output = """<explanation>
{explanation}
</explanation>

```json
{template}
```"""

fix_prompt_overcapture = """Your solution overcaptures: it matches the following lines which it should not:
```
{overcaptured}
```
The template you propose must not match any of these lines. Often, this happens when one or more of the variables in your proposed template captures too much. Analyze why the template you proposed captures these lines, and adjust your template accordingly so it does not, converting overcapturing variables to either a single constant or a more fine-grained separation. Please explain why your solution overcaptures, then fix your solution so it does not overcapture. Return a new valid explanation block as well as a template."""

fix_prompt_undercapture = """Your solution undercaptures: it misses the following lines:
```
{undercapture}
```
If this was intentional because the non-matched lines do not represent the same log event, return your original response. 
If you believe these additional lines should be captured by the same template, fix your solution so it captures these lines as well. Don't overfit to these examples: imagine what other lines could be produced by the same template, and try to capture all of those. Make sure to return both an explanation and a template."""

explanation_format = """
Description: {desc}

Placeholders: 
{placeholder}

Key Value Pairs:
{kvp}

Static Tokens: 
{constants}

Entities:
{variables}
"""

fse_single_line = {
    "entries": 'Jun 18 14:27:33 web-proxy-01.datacenter.corp nginx: [2025/06/18 14:27:33] ACCESS: 192.168.1.245:52134 -> 10.0.0.15:443 "POST /api/v2/users/login HTTP/1.1" status=401 bytes_sent=1247 bytes_received=312 response_time=0.234s referrer="https://malicious-site.example" user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" geo_location="US,California,Los Angeles" threat_score=85.7 blocked_by="rate_limiter,suspicious_pattern" ssl_cipher="TLS_AES_256_GCM_SHA384" ssl_mac= connection_id="conn_789abc123" upstream_server="backend-pool-1:app-server-03:8080" cache_status="MISS" cdn_pop="LAX-01" request_id="req_2025061814273312345" auth_details="user_id=12345,session_token=abc123def456,role=admin,last_login=2025-06-17T09:15:22Z" waf_analysis=\'{"rule_matches":["sqli_detection","xss_prevention"],"confidence":0.89,"action":"block","vendor":"cloudflare","processing_time_ms":12.4}\' rate_limit="window=60s,limit=100,current=95,reset_at=2025-06-18T14:28:00Z"',
    "desc": "nginx web server access log.",
    "placeholder": '<Datetime> <Hostname> nginx: [<Log_Timestamp>] ACCESS: <Client_IP>:<Client_Port> -> <Server_IP>:<Server_Port> "<HTTP_Method> <URI> <HTTP_Version>" status=<Status_Code> bytes_sent=<Bytes_Sent> bytes_received=<Bytes_Received> response_time=<Response_Time>s referrer="<Referrer>" user_agent="<User_Agent>" geo_location="<Country>,<State>,<City>" threat_score=<Threat_Score> blocked_by="<Block_Reasons>" ssl_cipher="<SSL_Cipher>" connection_id="<Connection_ID>" upstream_server="<Pool_Name>:<Server_Name>:<Port>" cache_status="<Cache_Status>" cdn_pop="<CDN_POP>" request_id="<Request_ID>" auth_details="user_id=<User_ID>,session_token=<Session_Token>,role=<User_Role>,last_login=<Last_Login>" waf_analysis=\'<WAF_JSON>\' rate_limit="window=<Rate_Window>,limit=<Rate_Limit>,current=<Current_Requests>,reset_at=<Reset_Time>"',
    "kvp": [
        "status=<Status_Code>",
        "bytes_sent=<Bytes_Sent>",
        "bytes_received=<Bytes_Received>",
        "response_time=<Response_Time>s",
        'referrer="<Referrer>"',
        'user_agent="<User_Agent>"',
        'geo_location="<Country>,<State>,<City>"',
        "threat_score=<Threat_Score>",
        'blocked_by="<Block_Reasons>"',
        'ssl_cipher="<SSL_Cipher>"',
        'connection_id="<Connection_ID>"',
        'upstream_server="<Pool_Name>:<Server_Name>:<Port>"',
        'cache_status="<Cache_Status>"',
        'cdn_pop="<CDN_POP>"',
        'request_id="<Request_ID>"',
        "auth_details.user_id=<User_ID>",
        "auth_details.session_token=<Session_Token>",
        "auth_details.role=<User_Role>",
        "auth_details.last_login=<Last_Login>",
        "waf_analysis=<WAF_JSON>",
        "rate_limit.window=<Rate_Window>",
        "rate_limit.limit=<Rate_Limit>",
        "rate_limit.current=<Current_Requests>",
        "rate_limit.reset_at=<Reset_Time>",
    ],
    "variables": [
        "<Datetime>: Date and time of the log message (Jun 18 14:27:33). Regex: \\S+\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2}",
        "<Hostname>: Hostname of the web server (web-proxy-01.datacenter.corp). Regex: \\S+",
        "<Log_Timestamp>: Full timestamp (2025/06/18 14:27:33). Regex: \\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2}",
        "<Client_IP>: Client IP address (192.168.1.245). Regex: \\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}",
        "<Client_Port>: Client port (52134). Regex: \\d+",
        "<Server_IP>: Server IP address (10.0.0.15). Regex: \\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}",
        "<Server_Port>: Server port (443). Regex: \\d+",
        "<HTTP_Method>: HTTP method (POST). Regex: [A-Z]+",
        "<URI>: Request URI (/api/v2/users/login). Regex: \\S+",
        "<HTTP_Version>: HTTP version (HTTP/1.1). Regex: HTTP/\\d\\.\\d",
        "<Status_Code>: HTTP status code (401). Regex: \\d{3}",
        "<Bytes_Sent>: Bytes sent to client (1247). Regex: \\d+",
        "<Bytes_Received>: Bytes received from client (312). Regex: \\d+",
        "<Response_Time>: Response time in seconds (0.234). Regex: \\d+(\\.\\d+)?",
        '<Referrer>: HTTP referrer (https://malicious-site.example). Regex: [^"]*',
        '<User_Agent>: User agent string (Mozilla/5.0...). Regex: [^"]*',
        "<Country>: Country code (US). Regex: [A-Z]+",
        "<State>: State/Province (California). Regex: [^,]+",
        '<City>: City name (Los Angeles). Regex: [^"]+',
        "<Threat_Score>: Security threat score (85.7). Regex: \\d+(\\.\\d+)?",
        '<Block_Reasons>: Comma-separated blocking reasons (rate_limiter,suspicious_pattern). Regex: [^"]+',
        "<SSL_Cipher>: SSL cipher suite (TLS_AES_256_GCM_SHA384). Regex: [A-Z0-9_]+",
        "<Connection_ID>: Connection identifier (conn_789abc123). Regex: conn_[a-z0-9]+",
        "<Pool_Name>: Upstream pool name (backend-pool-1). Regex: \\S+",
        "<Server_Name>: Upstream server name (app-server-03). Regex: \\S+",
        "<Port>: Upstream port (8080). Regex: \\d+",
        "<Cache_Status>: Cache hit/miss status (MISS). Regex: [A-Z]+",
        "<CDN_POP>: CDN point of presence (LAX-01). Regex: \\S+",
        "<Request_ID>: Unique request identifier (req_2025061814273312345). Regex: req_[a-z0-9]+",
        "<User_ID>: Authenticated user ID (12345). Regex: \\d+",
        "<Session_Token>: Session token (abc123def456). Regex: [a-z0-9]+",
        "<User_Role>: User role (admin). Regex: \\S+",
        "<Last_Login>: Last login timestamp (2025-06-17T09:15:22Z). Regex: \\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z",
        "<WAF_JSON>: JSON object containing WAF analysis data. Regex: \\{.*\\}",
        "<Rate_Window>: Rate limiting window (60s). Regex: \\S+",
        "<Rate_Limit>: Rate limit threshold (100). Regex: \\d+",
        "<Current_Requests>: Current request count (95). Regex: \\d+",
        "<Reset_Time>: Rate limit reset time (2025-06-18T14:28:00Z). Regex: \\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\S*",
    ],
    "constants": [
        "nginx:",
        "[",
        "]",
        "ACCESS:",
        ":",
        "->",
        ":",
        '"',
        '"',
        "status=",
        "bytes_sent=",
        "bytes_received=",
        "response_time=",
        "s",
        "referrer=",
        '"',
        '"',
        "user_agent=",
        '"',
        '"',
        "geo_location=",
        '"',
        ",",
        ",",
        '"',
        "threat_score=",
        "blocked_by=",
        '"',
        '"',
        "ssl_cipher=",
        '"',
        '"',
        "ssl_mac=",
        "connection_id=",
        '"',
        '"',
        "upstream_server=",
        '"',
        ":",
        ":",
        '"',
        "cache_status=",
        '"',
        '"',
        "cdn_pop=",
        '"',
        '"',
        "request_id=",
        '"',
        '"',
        "auth_details=",
        '"',
        "user_id=",
        ",",
        "session_token=",
        ",",
        "role=",
        ",",
        "last_login=",
        '"',
        "waf_analysis=",
        "'",
        "'",
        "rate_limit=",
        '"',
        "window=",
        ",",
        "limit=",
        ",",
        "current=",
        ",",
        "reset_at=",
        '"',
    ],
    "template": [
        {
            "value": "Jun 18 14:27:33",
            "is_variable": True,
            "placeholder": "<Datetime>",
            "regex": r"\S+\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}",
        },
        {
            "value": "web-proxy-01.datacenter.corp",
            "is_variable": True,
            "placeholder": "<Hostname>",
            "regex": r"\S+",
        },
        {"value": "nginx:", "is_variable": False},
        {"value": "[", "is_variable": False},
        {
            "value": "2025/06/18 14:27:33",
            "is_variable": True,
            "placeholder": "<Log_Timestamp>",
            "regex": r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}",
        },
        {"value": "]", "is_variable": False},
        {"value": "ACCESS:", "is_variable": False},
        {
            "value": "192.168.1.245",
            "is_variable": True,
            "placeholder": "<Client_IP>",
            "regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        },
        {"value": ":", "is_variable": False},
        {
            "value": "52134",
            "is_variable": True,
            "placeholder": "<Client_Port>",
            "regex": r"\d+",
        },
        {"value": "->", "is_variable": False},
        {
            "value": "10.0.0.15",
            "is_variable": True,
            "placeholder": "<Server_IP>",
            "regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        },
        {"value": ":", "is_variable": False},
        {
            "value": "443",
            "is_variable": True,
            "placeholder": "<Server_Port>",
            "regex": r"\d+",
        },
        {"value": '"', "is_variable": False},
        {
            "value": "POST",
            "is_variable": True,
            "placeholder": "<HTTP_Method>",
            "regex": r"[A-Z]+",
        },
        {
            "value": "/api/v2/users/login",
            "is_variable": True,
            "placeholder": "<URI>",
            "regex": r"\S+",
        },
        {
            "value": "HTTP/1.1",
            "is_variable": True,
            "placeholder": "<HTTP_Version>",
            "regex": r"HTTP/\d\.\d",
        },
        {"value": '"', "is_variable": False},
        {"value": "status=", "is_variable": False},
        {
            "value": "401",
            "is_variable": True,
            "placeholder": "<Status_Code>",
            "regex": r"\d{3}",
        },
        {"value": "bytes_sent=", "is_variable": False},
        {
            "value": "1247",
            "is_variable": True,
            "placeholder": "<Bytes_Sent>",
            "regex": r"\d+",
        },
        {"value": "bytes_received=", "is_variable": False},
        {
            "value": "312",
            "is_variable": True,
            "placeholder": "<Bytes_Received>",
            "regex": r"\d+",
        },
        {"value": "response_time=", "is_variable": False},
        {
            "value": "0.234",
            "is_variable": True,
            "placeholder": "<Response_Time>",
            "regex": r"\d+(\.\d+)?",
        },
        {"value": "s", "is_variable": False},
        {"value": "referrer=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "https://malicious-site.example",
            "is_variable": True,
            "placeholder": "<Referrer>",
            "regex": r'[^"]*',
        },
        {"value": '"', "is_variable": False},
        {"value": "user_agent=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "is_variable": True,
            "placeholder": "<User_Agent>",
            "regex": r'[^"]*',
        },
        {"value": '"', "is_variable": False},
        {"value": "geo_location=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "US",
            "is_variable": True,
            "placeholder": "<Country>",
            "regex": r"[A-Z]+",
        },
        {"value": ",", "is_variable": False},
        {
            "value": "California",
            "is_variable": True,
            "placeholder": "<State>",
            "regex": r"[^,]+",
        },
        {"value": ",", "is_variable": False},
        {
            "value": "Los Angeles",
            "is_variable": True,
            "placeholder": "<City>",
            "regex": r'[^"]+',
        },
        {"value": '"', "is_variable": False},
        {"value": "threat_score=", "is_variable": False},
        {
            "value": "85.7",
            "is_variable": True,
            "placeholder": "<Threat_Score>",
            "regex": r"\d+(\.\d+)?",
        },
        {"value": "blocked_by=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "rate_limiter,suspicious_pattern",
            "is_variable": True,
            "placeholder": "<Block_Reasons>",
            "regex": r'[^"]+',
        },
        {"value": '"', "is_variable": False},
        {"value": "ssl_cipher=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "TLS_AES_256_GCM_SHA384",
            "is_variable": True,
            "placeholder": "<SSL_Cipher>",
            "regex": r"[A-Z0-9_]+",
        },
        {"value": '"', "is_variable": False},
        {"value": "ssl_mac=", "is_variable": False},
        {"value": "connection_id=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "conn_789abc123",
            "is_variable": True,
            "placeholder": "<Connection_ID>",
            "regex": r"conn_[a-z0-9]+",
        },
        {"value": '"', "is_variable": False},
        {"value": "upstream_server=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "backend-pool-1",
            "is_variable": True,
            "placeholder": "<Pool_Name>",
            "regex": r"\S+",
        },
        {"value": ":", "is_variable": False},
        {
            "value": "app-server-03",
            "is_variable": True,
            "placeholder": "<Server_Name>",
            "regex": r"\S+",
        },
        {"value": ":", "is_variable": False},
        {
            "value": "8080",
            "is_variable": True,
            "placeholder": "<Port>",
            "regex": r"\d+",
        },
        {"value": '"', "is_variable": False},
        {"value": "cache_status=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "MISS",
            "is_variable": True,
            "placeholder": "<Cache_Status>",
            "regex": r"[A-Z]+",
        },
        {"value": '"', "is_variable": False},
        {"value": "cdn_pop=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "LAX-01",
            "is_variable": True,
            "placeholder": "<CDN_POP>",
            "regex": r"\S+",
        },
        {"value": '"', "is_variable": False},
        {"value": "request_id=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "req_2025061814273312345",
            "is_variable": True,
            "placeholder": "<Request_ID>",
            "regex": r"req_[a-z0-9]+",
        },
        {"value": '"', "is_variable": False},
        {"value": "auth_details=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "user_id=", "is_variable": False},
        {
            "value": "12345",
            "is_variable": True,
            "placeholder": "<User_ID>",
            "regex": r"\d+",
        },
        {"value": ",", "is_variable": False},
        {"value": "session_token=", "is_variable": False},
        {
            "value": "abc123def456",
            "is_variable": True,
            "placeholder": "<Session_Token>",
            "regex": r"[a-z0-9]+",
        },
        {"value": ",", "is_variable": False},
        {"value": "role=", "is_variable": False},
        {
            "value": "admin",
            "is_variable": True,
            "placeholder": "<User_Role>",
            "regex": r"\S+",
        },
        {"value": ",", "is_variable": False},
        {"value": "last_login=", "is_variable": False},
        {
            "value": "2025-06-17T09:15:22Z",
            "is_variable": True,
            "placeholder": "<Last_Login>",
            "regex": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        },
        {"value": '"', "is_variable": False},
        {"value": "waf_analysis=", "is_variable": False},
        {"value": "'", "is_variable": False},
        {
            "value": '{"rule_matches":["sqli_detection","xss_prevention"],"confidence":0.89,"action":"block","vendor":"cloudflare","processing_time_ms":12.4}',
            "is_variable": True,
            "placeholder": "<WAF_JSON>",
            "regex": r"\{.*\}",
        },
        {"value": "'", "is_variable": False},
        {"value": "rate_limit=", "is_variable": False},
        {"value": '"', "is_variable": False},
        {"value": "window=", "is_variable": False},
        {
            "value": "60s",
            "is_variable": True,
            "placeholder": "<Rate_Window>",
            "regex": r"\S+",
        },
        {"value": ",", "is_variable": False},
        {"value": "limit=", "is_variable": False},
        {
            "value": "100",
            "is_variable": True,
            "placeholder": "<Rate_Limit>",
            "regex": r"\d+",
        },
        {"value": ",", "is_variable": False},
        {"value": "current=", "is_variable": False},
        {
            "value": "95",
            "is_variable": True,
            "placeholder": "<Current_Requests>",
            "regex": r"\d+",
        },
        {"value": ",", "is_variable": False},
        {"value": "reset_at=", "is_variable": False},
        {
            "value": "2025-06-18T14:28:00Z",
            "is_variable": True,
            "placeholder": "<Reset_Time>",
            "regex": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S*",
        },
        {"value": '"', "is_variable": False},
    ],
}

# Template based on the first log entry
fse_multiple_lines = {
    "entries": [
        'Jun 18 15:42:17 srv-dc01.corporate.local TerminalServices-RemoteConnectionManager: Event ID 1149: User authentication succeeded for user "CORPORATE\\jdoe" from source network address 10.45.122.78 using authentication package "Negotiate" with logon type "RemoteInteractive" session created with ID 0x8A2F3 connection established successfully via RDP protocol version 10.0 client build 19041 operating system "Windows 10 Enterprise" computer name "LAPTOP-USER123" total session duration will be tracked under tracking ID TR_20250618154217_8A2F3',
        'Jun 18 16:15:23 srv-dc01.corporate.local TerminalServices-RemoteConnectionManager: Event ID 1149: User authentication succeeded for user "CORPORATE\\admin.smith" from source network address 192.168.50.145 using authentication package "Kerberos" with logon type "RemoteInteractive" session created with ID 0x9B4E7 connection established successfully via RDP protocol version 10.0 client build 22000 operating system "Windows 11 Pro" computer name "ADMIN-WORKSTATION" total session duration will be tracked under tracking ID TR_20250618161523_9B4E7',
        'Jun 18 17:33:41 srv-dc01.corporate.local TerminalServices-RemoteConnectionManager: Event ID 1149: User authentication succeeded for user "CORPORATE\\service.backup" from source network address 172.16.8.92 using authentication package "NTLM" with logon type "RemoteInteractive" session created with ID 0x7C1A9 connection established successfully via RDP protocol version 8.1 client build 17763 operating system "Windows Server 2019" computer name "SRV-BACKUP02" total session duration will be tracked under tracking ID TR_20250618173341_7C1A9',
    ],
    "desc": "Windows Terminal Services RDP connection success log with user authentication details and session information.",
    "placeholder": '<Datetime> <Hostname> TerminalServices-RemoteConnectionManager: Event ID <Event_ID>: User authentication succeeded for user "<Domain>\\<Username>" from source network address <Source_IP> using authentication package "<Auth_Package>" with logon type "<Logon_Type>" session created with ID <Session_ID> connection established successfully via RDP protocol version <RDP_Version> client build <Client_Build> operating system "<OS_Name>" computer name "<Computer_Name>" total session duration will be tracked under tracking ID <Tracking_ID>',
    "kvp": [
        "Event_ID=1149",
        "Domain=<Domain>",
        "Username=<Username>",
        "Source_IP=<Source_IP>",
        "Auth_Package=<Auth_Package>",
        "Logon_Type=<Logon_Type>",
        "Session_ID=<Session_ID>",
        "RDP_Version=<RDP_Version>",
        "Client_Build=<Client_Build>",
        "OS_Name=<OS_Name>",
        "Computer_Name=<Computer_Name>",
        "Tracking_ID=<Tracking_ID>",
    ],
    "variables": [
        "<Datetime>: Date and time of the log message (Jun 18 15:42:17). Regex: \\S+\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2}",
        "<Hostname>: Hostname of the server (srv-dc01.corporate.local). Regex: \\S+",
        "<Event_ID>: Event ID (1149). Regex: \\d+",
        "<Domain>: Windows domain name (CORPORATE). Regex: [a-zA-Z0-9_]+",
        "<Username>: Username (jdoe). Regex: \\S+",
        "<Source_IP>: Source IP address (10.45.122.78). Regex: \\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}",
        "<Auth_Package>: Authentication package (Negotiate). Regex: \\S+",
        "<Logon_Type>: Type of logon (RemoteInteractive). Regex: \\S+",
        "<Session_ID>: Hexadecimal session ID (0x8A2F3). Regex: 0x[A-F0-9]+",
        "<RDP_Version>: RDP protocol version (10.0). Regex: \\d+(\\.\\d+)?",
        "<Client_Build>: Client build number (19041). Regex: \\d+",
        '<OS_Name>: Operating system name (Windows 10 Enterprise). Regex: [^"]+',
        '<Computer_Name>: Computer name (LAPTOP-USER123). Regex: [^"]+',
        "<Tracking_ID>: Session tracking identifier (TR_20250618154217_8A2F3). Regex: \\S+",
    ],
    "constants": [
        "TerminalServices-RemoteConnectionManager:",
        "Event ID",
        ":",
        "User",
        "authentication succeeded",
        "for",
        "user",
        '"',
        "\\",
        '"',
        "from",
        "source network address",
        "using",
        "authentication package",
        '"',
        '"',
        "with",
        "logon type",
        '"',
        '"',
        "session created",
        "with",
        "ID",
        "connection established successfully",
        "via",
        "RDP",
        "protocol version",
        "client build",
        "operating system",
        '"',
        '"',
        "computer name",
        '"',
        '"',
        "total session duration will be tracked under",
        "tracking ID",
    ],
    "template": [
        {
            "value": "Jun 18 15:42:17",
            "is_variable": True,
            "placeholder": "<Datetime>",
            "regex": r"\S+\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}",
        },
        {
            "value": "srv-dc01.corporate.local",
            "is_variable": True,
            "placeholder": "<Hostname>",
            "regex": r"\S+",
        },
        {
            "value": "TerminalServices-RemoteConnectionManager:",
            "is_variable": False,
        },
        {"value": "Event ID", "is_variable": False},
        {"value": "1149:", "is_variable": False},
        {"value": "User", "is_variable": False},
        {"value": "authentication succeeded", "is_variable": False},
        {"value": "for", "is_variable": False},
        {"value": "user", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "CORPORATE",
            "is_variable": True,
            "placeholder": "<Domain>",
            "regex": r"[a-zA-Z0-9_]+",
        },
        {"value": "\\", "is_variable": False},
        {
            "value": "jdoe",
            "is_variable": True,
            "placeholder": "<Username>",
            "regex": r"\S+",
        },
        {"value": '"', "is_variable": False},
        {"value": "from", "is_variable": False},
        {"value": "source network address", "is_variable": False},
        {
            "value": "10.45.122.78",
            "is_variable": True,
            "placeholder": "<Source_IP>",
            "regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        },
        {"value": "using", "is_variable": False},
        {"value": "authentication package", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "Negotiate",
            "is_variable": True,
            "placeholder": "<Auth_Package>",
            "regex": r"\S+",
        },
        {"value": '"', "is_variable": False},
        {"value": "with", "is_variable": False},
        {"value": "logon type", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "RemoteInteractive",
            "is_variable": True,
            "placeholder": "<Logon_Type>",
            "regex": r"\S+",
        },
        {"value": '"', "is_variable": False},
        {"value": "session created", "is_variable": False},
        {"value": "with", "is_variable": False},
        {"value": "ID", "is_variable": False},
        {
            "value": "0x8A2F3",
            "is_variable": True,
            "placeholder": "<Session_ID>",
            "regex": r"0x[A-F0-9]+",
        },
        {"value": "connection established successfully", "is_variable": False},
        {"value": "via", "is_variable": False},
        {"value": "RDP", "is_variable": False},
        {"value": "protocol version", "is_variable": False},
        {
            "value": "10.0",
            "is_variable": True,
            "placeholder": "<RDP_Version>",
            "regex": r"\d+(\.\d+)?",
        },
        {"value": "client build", "is_variable": False},
        {
            "value": "19041",
            "is_variable": True,
            "placeholder": "<Client_Build>",
            "regex": r"\d+",
        },
        {"value": "operating system", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "Windows 10 Enterprise",
            "is_variable": True,
            "placeholder": "<OS_Name>",
            "regex": r'[^"]+',
        },
        {"value": '"', "is_variable": False},
        {"value": "computer name", "is_variable": False},
        {"value": '"', "is_variable": False},
        {
            "value": "LAPTOP-USER123",
            "is_variable": True,
            "placeholder": "<Computer_Name>",
            "regex": r'[^"]+',
        },
        {"value": '"', "is_variable": False},
        {
            "value": "total session duration will be tracked under",
            "is_variable": False,
        },
        {"value": "tracking ID", "is_variable": False},
        {
            "value": "TR_20250618154217_8A2F3",
            "is_variable": True,
            "placeholder": "<Tracking_ID>",
            "regex": r"\S+",
        },
    ],
}


def gen_fewshot(examples):
    fewshot_prompts = []
    multi_line_example, single_line_example = False, False
    for (
        fs_template,
        fs_entries,
        fs_suffixes,
        fs_explanation,
        fs_description,
    ) in examples:
        if not isinstance(fs_entries, list):
            fs_entries = [fs_entries]

        if len(fs_entries) > 1:
            multi_line_example = True
        else:
            single_line_example = True

        entries = "\n".join(fs_entries)

        if fs_suffixes and fs_suffixes[0] == fs_entries[0]:
            fs_suffixes = None
        suffixes = "\n".join(fs_suffixes) if fs_suffixes else None
        template = fs_template

        if suffixes:
            formatted_user_input = user_input_suffix.format(
                entries=entries, suffixes=suffixes
            )
        else:
            formatted_user_input = user_input.format(entries=entries)

        fewshot_prompts.append(
            {"role": "user", "content": formatted_user_input}
        )
        fewshot_prompts.append(
            {
                "role": "assistant",
                "content": output.format(
                    explanation=fs_explanation.strip(), template=template
                ),
            }
        )

    if not multi_line_example:
        # Add the multiple line example to the start of the list
        entries = "\n".join(fse_multiple_lines["entries"])

        desc, placeholder, kvp, constants, variables, template = (
            fse_multiple_lines["desc"],
            fse_multiple_lines["placeholder"],
            fse_multiple_lines["kvp"],
            fse_multiple_lines["constants"],
            fse_multiple_lines["variables"],
            fse_multiple_lines["template"],
        )
        explanation = explanation_format.format(
            desc=desc,
            placeholder=placeholder,
            kvp="\n".join(kvp),
            constants="\n".join(constants),
            variables="\n".join(variables),
        )

        user_message = user_input.format(entries=entries)

        assistant_output = output.format(
            explanation=explanation.strip(),
            template=json.dumps(template, indent=2),
        )

        fewshot_prompts = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_output},
        ] + fewshot_prompts

    if not single_line_example:
        # Add the single line example to the start of the list
        entries = fse_single_line["entries"]

        desc, placeholder, kvp, constants, variables, template = (
            fse_single_line["desc"],
            fse_single_line["placeholder"],
            fse_single_line["kvp"],
            fse_single_line["constants"],
            fse_single_line["variables"],
            fse_single_line["template"],
        )
        explanation = explanation_format.format(
            desc=desc,
            placeholder=placeholder,
            kvp="\n".join(kvp),
            constants="\n".join(constants),
            variables="\n".join(variables),
        )

        user_message = user_input.format(entries=entries)

        assistant_output = output.format(
            explanation=explanation.strip(),
            template=json.dumps(template, indent=2),
        )

        fewshot_prompts = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_output},
        ] + fewshot_prompts

    return fewshot_prompts


def gen_system():
    return system + rules.format(", ".join(always_variable))


def gen_prompt(examples, entries, description, suffixes=None):
    history = gen_fewshot(examples)

    if suffixes is None:
        user = user_input.format(entries="\n".join(entries))
    else:
        user = user_input_suffix.format(
            entries="\n".join(entries),
            suffixes="\n".join(suffixes),
        )
    history.append(
        {"role": "user", "content": rules.format(", ".join(always_variable))}
    )
    history.append({"role": "user", "content": user})

    return history, gen_system()
