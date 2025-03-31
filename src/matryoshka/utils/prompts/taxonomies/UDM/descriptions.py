import json

system = """You are a log analysis expert. Your role is to write complete yet consice descriptions of fields in the Unified Data Model (UDM).

*** UDM Intro ***

The UDM is a Google Security Operations standard data structure that stores information about data received from sources. It is also called the schema. Google SecOps stores the original data it receives in two formats, as the original raw log and as a structured UDM record. The UDM record is a structured representation of the original log.

*** Goal ***

The attribute tree is a hierarchical structure that defines the fields that can be present in an event. Each node in the tree contains a description of the attribute, some optional constraints, and if the node is not a leaf, a set of nested attributes. Fields are defined as the leaves of the attribute tree.
The issue with the UDM format is that nested fields only have definitions in the scope of their direct parent; however, we want to write descriptions of fields that fully represent their role within a log event.

*** Instructions ***

I will provide you with a field name (given by its path in the attribute tree) as well as a description of each node along the path.
Descriptions of nodes give information of each node in the context of their direct parent. I need you to use this information to write a description of the leaf node (i.e. the field) in the context of the root "event" node.
The rules are the following:
* The description should provide information about the field in the context of the UDM event.
* The description should be complete: it should provide all the information needed to understand what the field refers to.
* The description should be sound: it should be consistent with the descriptions of the parent nodes, and must not mislead into misrepresenting the role of the field.
"""

input_template = """Field: {path}

Description tree:
```
{description_tree}
```

Description:"""

user = """{fs}{input}"""


def gen_fewshot():
    example_1 = """Field: event.principal.user.group_identifiers

Description tree:
```

* #1    ---     principal: "Represents the acting entity that originates the activity described in the event. The principal must include at least one machine detail (hostname, MACS, IPs, port, product-specific identifiers like an EDR asset ID) or user detail (for example, username), and optionally include process details. It must NOT include any of the following fields: email, files, registry keys, or values."
* #2    ---     user: "Information about the user."
* #3    ---     group_identifiers: "Product object identifiers of the group(s) the user belongs to A vendor-specific identifier to uniquely identify the group(s) the user belongs to (a GUID, LDAP OID, or similar)."

```

Description: "Product object identifiers (a GUID, LDAP OID, or similar) to uniquely identify the group(s) of the user that originated the activity described in the event." """

    example_2 = """Field: event.network.http.user_agent

Description tree:
```

* #1    ---     network: "All network details go here, including sub-messages with details on each protocol (for example, DHCP, DNS, or HTTP)."
* #2    ---     http: "HTTP info."
* #3    ---     user_agent: "The User-Agent request header which includes the application type, operating system, software vendor or software version of the requesting software user agent."
```

Description: "The user agent request header in the HTTP connection described in the event." """

    example_header = "### Example Descriptions ###\n\n"
    example_separator = "\n\n################\n\n"
    example_footer = "\n\n### End of Examples ###\n\n"

    example_3 = """Field: event.security_result.detection_fields.value
Description tree:
```

* #1    ---     security_result: "A list of security results."
* #2    ---     detection_fields: "An ordered list of values, that represent fields in detections for a security finding. This list represents mapping of names of requested entities to their values (i.e., the security result matched variables)."
* #3    ---     value: "The value."
```

Description: "The value of a detection field in a security result." """

    example_4 = """Field: event.intermediary.hostname

Description tree:
```

* #1    ---     intermediary: "Represents details on one or more intermediate entities processing activity described in the event. This includes device details about a proxy server or SMTP relay server. If an active event (that has a principal and possibly target) passes through any intermediaries, they are included here. Intermediaries can impact the overall action, for example blocking or modifying an ongoing request. A rule of thumb here is that 'principal', 'target', and description of the initial action should be the same regardless of the intermediary or its action. A successful network connection from A->B should look the same in principal/target/intermediary as one blocked by firewall C: principal: A, target: B (intermediary: C)."
* #2    ---     hostname: "Client hostname or domain name field. Hostname also doubles as the domain for remote entities."
```

Description: "The hostname of the intermediary entity processing the activity described in the event, such as a proxy server, SMTP relay server, or any other network entity that may have impacted the activity but neither the source nor destination of the activity." """

    example_5 = """Field: target.network.dhcp.requested_address

Description tree:
```

* #1    ---     target: "Represents a target entity being referenced by the event or an object on the target entity. For example, in a firewall connection from device A to device B, A is described as the principal and B is described as the target. For a process injection by process C into target process D, process C is described as the principal and process D is described as the target."
* #2    ---     network: "Network details, including sub-messages with details on each protocol (for example, DHCP, DNS, or HTTP)."
* #3    ---     dhcp: "DHCP info."
* #4    ---     requested_address: "Requested IP address. See RFC2132, section 9.1"
```

Description: "The IP address requested by the target network entity referenced by the log event in a DHCP request." """

    fs = (
        example_header
        + example_1
        + example_separator
        + example_2
        + example_separator
        + example_3
        + example_separator
        + example_4
        + example_separator
        + example_5
        + example_separator
        + example_footer
    )

    return fs


def gen_system():
    return system


def gen_prompt(desc):
    desc_tree = []
    for level_idx, level in enumerate(
        desc.name.replace("event.", "").split(".")
    ):
        desc_tree.append(
            [
                f"{1+level_idx}",
                level,
                desc.description_list[level_idx],
            ]
        )

    desc_tree_str = "\n".join(
        [
            f"# {level[0]}\t---\t\t{level[1]}:\t{json.dumps(level[2])}"
            for level in desc_tree
        ]
    )

    ipt = input_template.format(
        path=desc.name,
        description_tree=desc_tree_str,
    )

    return user.format(fs=gen_fewshot(), input=ipt), gen_system()
