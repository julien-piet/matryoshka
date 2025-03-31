import json
import random

system = """You are a log analysis expert. Your role is to write complete yet consice descriptions of fields in the OCSF format.

*** OCSF Format Intro ***

The Open Cybersecurity Schema Framework is an open-source project, delivering an extensible framework for developing schemas, along with a vendor-agnostic core security schema. Vendors and other data producers can adopt and extend the schema for their specific domains. Data engineers can map differing schemas to help security teams simplify data ingestion and normalization, so that data scientists and analysts can work with a common language for threat detection and investigation. The goal is to provide an open standard, adopted in any environment, application, or solution, while complementing existing security standards and processes.

*** OCSF Overview ***

The framework is made up of a set of data types, an attribute tree, and the taxonomy. It is not restricted to the cybersecurity domain nor to events, however the initial focus of the framework has been a schema for cybersecurity events. OCSF is agnostic to storage format, data collection and ETL processes. The core schema for cybersecurity events is intended to be agnostic to implementations. The schema framework definition files and the resulting normative schema are written as JSON.

*** Goal ***

The attribute tree is a hierarchical structure that defines the fields that can be present in an event. Each node in the tree contains a description of the attribute, some optional constraints, and if the node is not a leaf, a set of nested attributes. Fields are defined as the leaves of the attribute tree.
The issue with the OCSF format is that nested fields only have definitions in the scope of their direct parent: however, we want to write descriptions of fields in the context of the root OCSF event they belong to.

*** Instructions ***

I will provide you with a field name (given by its path in the attribute tree) as well as a description of each node along the path.
Descriptions of nodes give information of each node in the context of their direct parent. I need you to use this information to write a description of the leaf node (i.e. the field) in the context of the root node (i.e. the OCSF event).
The rules are the following:
* The description should provide information about the role of the field in the context of the OCSF event it is part of.
* The description should be complete: it should provide all the information needed to understand what the field refers to in the context of the OCSF event.
* The description should be sound: it should be consistent with the descriptions of the parent nodes, and must not mislead into misrepresenting the role of the field.
"""

input_template = """Field: {path}

Description tree:
```
{description_tree}
```

Description:"""

user = """{fs}{input}"""


### TODO: Add both objectve and subjective defnitions for more accuracy


def gen_fewshot():
    example_1 = """Field: dhcp_activity.actor.user.ldap_person.location.continent

Description tree:
```
* Root Event    ---     dhcp_activity:  "DHCP Activity events report MAC to IP assignment via DHCP from a client or server."
* #1    ---     actor:  "The actor object describes details about the user/role/process that was the source of the activity."
* #2    ---     user:   "The user that initiated the activity or the user context from which the activity was initiated."
* #3    ---     ldap_person:    "The additional LDAP attributes that describe a person."
* #4    ---     location:       "The geographical location associated with a user. This is typically the user's usual work location."
* #5    ---     continent:      "The name of the continent."
```

Description: "The continent where the user that generated the DHCP activity (i.e. a MAC to IP assignment) is located." """

    example_2 = """Field: network.connection_info.session.terminal

Description tree:
```
* Root Event    ---     network:        "Network Activity events."
* #1    ---     connection_info:        "The network connection information."
* #2    ---     session:        "The authenticated user or service session."
* #3    ---     terminal:       "The Pseudo Terminal associated with the session. Ex: the tty or pts value."
```

Description: The pseudo terminal (e.g. the tty or pts value) associated with the authenticated user or service session involved in the network activity."""

    example_header = """### Example Descriptions ###\n\n"""
    example_separator = """\n\n##########\n\n"""
    example_footer = """\n\n### End of Examples ###\n\n"""

    example_3 = """Field: finding.device.agent_list.policies.desc

Description tree:
```
* Root Event    ---     finding:        "Findings events report findings, detections, and possible resolutions of malware, anomalies, or other actions performed by security products."
* #1    ---     device: "Describes the affected device/host. It can be used in conjunction with <code>Affected Resource(s)</code>. <p> e.g. Specific details about an AWS EC2 instance, that is affected by the Finding.</p>"
* #2    ---     agent_list:     "A list of <code>agent</code> objects associated with a device, endpoint, or resource."
* #3    ---     policies:       "Describes the various policies that may be applied or enforced by an agent or sensor. E.g., Conditional Access, prevention, auto-update, tamper protection, destination configuration, etc."
* #4    ---     desc:   "The description of the policy."
```

Description: "The description of a policy (e.g. conditional Access, prevention, auto-update, tamper protection, destination configuration, etc.) applied or enforced by an agent associated with a device/host affected by a malware or anomaly." """

    example_4 = """Field: application_lifecycle.device.agent_list.policies.group.domain

Description tree:
```
* Root Event    ---     application_lifecycle:  "Application Lifecycle events report installation, removal, start, stop of an application or service."
* #1    ---     device: "An addressable device, computer system or host."
* #2    ---     agent_list:     "A list of <code>agent</code> objects associated with a device, endpoint, or resource."
* #3    ---     policies:       "Describes the various policies that may be applied or enforced by an agent or sensor. E.g., Conditional Access, prevention, auto-update, tamper protection, destination configuration, etc."
* #4    ---     group:  "The policy group."
* #5    ---     domain: "The domain where the group is defined. For example: the LDAP or Active Directory domain."
```

Description: "The domain (e.g. the LDAP or AD domain) where the policy group is defined for an agent associated with a device in an application lifecycle event (e.g. the installation, removal, start or stop of an application or service)." """

    example_5 = """Field: email_activity.src_endpoint.agent_list.policies.group.privileges

Description tree:
```
* Root Event    ---     email_activity: "Email events report activities of emails."
* #1    ---     src_endpoint:   "The initiator (client) sending the email."
* #2    ---     agent_list:     "A list of <code>agent</code> objects associated with a device, endpoint, or resource."
* #3    ---     policies:       "Describes the various policies that may be applied or enforced by an agent or sensor. E.g., Conditional Access, prevention, auto-update, tamper protection, destination configuration, etc."
* #4    ---     group:  "The policy group."
* #5    ---     privileges:     "The group privileges."
```

Description: "The privileges associated with a policy group for an agent associated with the source network endpoint that sent an email in an email activity event." """

    example_6 = """Field: event_log.src_endpoint.owner.name

Description tree:
```
* Root Event    ---     event_log:      "Event Log Activity events report actions pertaining to the system's event logging service(s), such as disabling logging or clearing the log data."
* #1    ---     src_endpoint:   "The source endpoint for the event log activity."
* #2    ---     owner:  "The identity of the service or user account that owns the endpoint or was last logged into it."
* #3    ---     name:    "The username. For example, janedoe1."
```

Description: "The username of the service or user account that owns the source endpoint responsible for the event log activity, such as disabling logging or clearing the log data." """

    example_7 = """Field: iam.src_endpoint.port 
    
Description tree:
```
* Root Event    ---     iam:    "Identity & Access Management (IAM) events relate to the supervision of the system's authentication and access control model. Examples of such events are the success or failure of authentication, granting of authority, password change, entity change, privileged use etc."
* #1    ---     src_endpoint:   "Details about the source of the IAM activity."
* #2    ---     port:   "The port used for communication within the network connection."
```

Description: "The client network endpoint port used for the IAM (Identity and Access Management) activity. IAM events relate to the supervision of the system's authentication and access control model." """

    example_8 = """Field: ssh_activity.time
    
Description tree:
* Root Event    ---     ssh_activity:   "SSH Activity events report remote client connections to a server using the Secure Shell (SSH) Protocol.
* #1    ---     time:   "The normalized event occurrence time or the finding creation time."
```

Description: "The timestamp of the log entry, indicating when the SSH activity occurred." """

    example_9 = """Field: ssh_activity.client_hassh.fingerprint.value
    
Description tree:
* Root Event    ---     ssh_activity:   "SSH Activity events report remote client connections to a server using the Secure Shell (SSH) Protocol.
* #1    ---     client_hassh:   "	The Client HASSH fingerprinting object."
* #2    ---     fingerprint:   "The Fingerprint object provides detailed information about a digital fingerprint, which is a compact representation of data used to identify a longer piece of information, such as a public key or file content. It contains the algorithm and value of the fingerprint, enabling efficient and reliable identification of the associated data."
* #3    ---     value:   "The digital fingerprint value."
```

Description: "The digital fingerprint of the public key used by the client to authenticate to the server in an SSH connection." """

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
        + example_6
        + example_separator
        + example_7
        + example_separator
        + example_8
        + example_separator
        + example_9
        + example_footer
    )

    return fs


def gen_system():
    return system


def gen_prompt(desc):
    desc_tree = []
    for level_idx, level in enumerate(desc.path.split(".")):
        desc_tree.append(
            [
                "Root Event" if level_idx == 0 else f"#{level_idx}",
                level,
                desc.original_descriptions[level_idx],
            ]
        )
    desc_tree_str = "\n".join(
        [
            f"* {level[0]}\t---\t{level[1]}:\t{json.dumps(level[2])}"
            for level in desc_tree
        ]
    )
    ipt = input_template.format(
        path=desc.path,
        description_tree=desc_tree_str,
    )
    return user.format(fs=gen_fewshot(), input=ipt), gen_system()
