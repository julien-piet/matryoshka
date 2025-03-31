import re
from abc import ABC, abstractmethod
from typing import List

from ..genai_api.api import Caller


class Module(ABC):

    def __init__(self, name, caller=None, **kwargs) -> None:
        self.name = name
        if caller is None:
            self.caller = Caller()
        else:
            self.caller = caller

    def __call__(self, log_file, **kwargs):
        kwargs["log_file"] = log_file
        return self.process(**kwargs)

    @staticmethod
    def load_log(log_file) -> str:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            log = f.read()
        return log

    @staticmethod
    def load_and_split_log(log_file, split_regex="\n") -> List[str]:
        log = Module.load_log(log_file)
        return re.split(split_regex, log)

    @staticmethod
    def verif_regex(log, regex, strict=False, count=False) -> bool:
        """
        Verifies if the regex matches every log entry in the log file
        """
        if isinstance(log, str):
            log = [log]

        if count:
            return len(re.findall(regex, log[0]))
        elif strict:
            rslt = [re.match(regex, log_entry) for log_entry in log]
            return all(rslt), [
                log_entry for log_entry, r in zip(log, rslt) if not r
            ]

        else:
            rslt = [re.search(regex, log_entry) for log_entry in log]
            return all(rslt), [
                log_entry for log_entry, r in zip(log, rslt) if not r
            ]

    @staticmethod
    def re_compile(s, encoding="utf-8"):
        return re.compile(
            s.encode("latin1")
            .decode("unicode-escape")
            .encode("latin1")
            .decode(encoding)
        )

    @staticmethod
    def update_history(task, response, history=None):
        if history is None and task.system_prompt is not None:
            history = [{"role": "system", "content": task.system_prompt}]
        elif history is None:
            history = []
        history.append({"role": "user", "content": str(task.message)})
        history.append({"role": "assistant", "content": response})
        return history

    @abstractmethod
    def process(self, log_file, **kwargs):
        pass
