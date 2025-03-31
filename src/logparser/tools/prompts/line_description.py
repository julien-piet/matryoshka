import json
import random

system = """Your role is to annotate log files. Log files are collections of entries produced by programs to record state. Log lines can be grouped according to the formatting string used to produce them in the original program. These formatting strings define the template of this group of lines. We have already parsed the log entries to templates. We now need to write descrpitons for the formatting strings.

Templates are made up of into tokens. Each token represents a single semantic unit within the log entry, and can be made up of individual characters, single words, or even multiple words. Tokens can either be constant (fixed parts of the formatting string) or variable (variables in the formatting string).

*** Instructions ***
I will provide you with a log line. This line was produced by an unknown formatting string to report some event in a program. 
You need to infer the formatting string and write a description of the event this formatting string describes.
You must describe the underlying event without using any specific identifier contained in the log line: this description should apply to any log line produced by the same formatting string. 
The rules are the following:
* The description should provide detailed information about the what event this log line refers to, without refering to specific values, identifiers or examples.
* The description should be complete: it should provide all the information needed to understand what the underlying formatting string refers to.
* The description should be sound: it should be consistent with the role of the log line, and must not mislead into misrepresenting its meaning.
* The description should be precise: it should not be ambiguous, it should use the proper terminology, and should be clear about the meaning of the formatting string.
* The description should be general: it should not reference specific values, identifiers or examples of variable fields in the log line, but instead only reference the abstract event being recorded.

*** Warning ***
Remember, do not include any specific values or identifiers in your description. Values in the log line that are parameters of the event should be described in general terms, without referring to the specific value in the log line. Examples of such fields that should not be included in the description are IP addresses, timestamps, port numbers, process IDs, usernames, etc.

For example, the following log line:
```
03/22 08:52:50 TRACE :......rsvp_event_mapSession: Session=9.67.116.99:1047:6 does not exist
```
can be described as:
```
RSVP session lookup failure: Unable to find an existing RSVP (Resource Reservation Protocol) session matching the specified IP address and port/index identifiers
```
"""


def gen_fewshot(selected):
    ex, desc = [], []
    ex_1 = """2024-12-10T15:23:48.127-0500 [WARN] auth-svc-prod-85cf4d7b9-xk2vp - [correlation_id=8f2e4901-bf3d-4119-8180-a9e21309e628] Failed to validate JWT token: token_expired (issued: 2024-12-09T02:15:33Z) client_ip=172.16.32.45 user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" request_id=req_9f4b23a1"""
    desc_1 = """An authentication message that captures a JWT token validation failure, including the timestamp, log level, service details, correlation ID, error specifics, and request context. The format includes key authentication debugging information such as the token's issue time, client IP, user agent, and request identifier."""
    ex.append(ex_1)
    desc.append(desc_1)

    ex_2 = """Jan 9 6:53:19 laptop-t490 kernel: [87321.342] wlp3s0: disconnect from SSID "Home_WiFi_5G" (00:14:22:01:23:45) reason=4 locally_generated=1"""
    desc_2 = """A wireless network interface disconnection event logged by the system kernel, indicating the network adapter's separation from a specific wireless network with associated identifiers. The log entry includes temporal information, the reason for disconnection, and a flag indicating the event was triggered by the local system rather than the remote access point."""
    ex.append(ex_2)
    desc.append(desc_2)

    ex_3 = """Apr 18 12:24:45 desktop-pc smartd[1234]: Device: /dev/sda [SAT], SMART Usage Attribute: 194 Temperature_Celsius changed from 41 to 43"""
    desc_3 = """A storage device monitoring event that records a change in the drive's temperature as reported by SMART diagnostics functionality. The log captures the specific attribute being monitored, its previous and current values, along with temporal and device identification details."""
    ex.append(ex_3)
    desc.append(desc_3)

    ex_4 = """Jun 23 17:05:58 workstation-ubuntu kernel: [92813.123] python3[23981]: segfault at 7f2c1a3b1000 ip 00007f2c1a3b1000 sp 00007ffd3e2a5830 error 14 in libc.so.6"""
    desc_4 = """A system crash event logged by the kernel, indicating a segmentation fault in a Python process including the associated memory addresses and error code. The log entry provides details on the faulting instruction pointer, stack pointer, and the shared library where the fault occurred."""
    ex.append(ex_4)
    desc.append(desc_4)

    ex_5 = """Mar 2 11:45:07 prod-app-01 audit[31337]: ANOM_ABEND auid=1001 uid=1001 gid=1001 ses=42 pid=31337 comm="unusual_suid_binary" exe="/usr/local/bin/unusual_suid_binary" sig=31 res=1 ARCH=x86_64 SYSCALL=execve AUID="jenkins" UID="jenkins" GID="jenkins" EGID="root" """
    desc_5 = """A security audit log entry capturing the abnormal termination of a privileged binary execution attempt, including user context, process details, and security-relevant information. The event includes the executable's path, signal information, architectural details, and the security context showing privilege escalation details, along with audit session tracking information."""
    ex.append(ex_5)
    desc.append(desc_5)

    fewshot_prompts = []
    for e, d in zip(ex, desc):
        fewshot_prompts.append({"role": "user", "content": e})
        fewshot_prompts.append({"role": "assistant", "content": d})

    for d, o in selected:
        fewshot_prompts.append({"role": "user", "content": d})
        fewshot_prompts.append({"role": "assistant", "content": o})

    return fewshot_prompts


def gen_system():
    return system


def gen_prompt(log_line, few_shot_examples):
    return log_line, gen_fewshot(few_shot_examples), gen_system()
