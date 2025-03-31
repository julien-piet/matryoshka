import random
import re

from .tools.api import OpenAITask
from .tools.module import Module
from .tools.prompts import (
    static_prompt,
    static_prompt_error_message,
    system_prompt,
)


class StaticTemplates(Module):

    def __init__(
        self,
        caller=None,
        unit_regex=re.compile("\n"),
        parallel_attempts=4,
        line_count=50,
        target_coverage=1,
        **kwargs
    ) -> None:
        super().__init__("statictemplate", caller=caller)
        self.parallel_attempts = parallel_attempts
        self.line_count = line_count
        self.unit_regex = unit_regex
        self.coverage_factor = target_coverage
        self.templates = []

    def process(self, log_file, model="gemini-1.5-flash", **kwargs) -> None:
        lines = [
            line
            for line in self.load_and_split_log(log_file, self.unit_regex)
            if len(line)
        ]
        tasks, filtered_lines = [], lines
        while len(filtered_lines) > len(lines) * (1 - self.coverage_factor):

            for _ in range(self.parallel_attempts):
                sample_length = min(len(filtered_lines), self.line_count)
                random_lines = random.sample(filtered_lines, sample_length)
                excerpt = "\n".join(random_lines)
                task = OpenAITask(
                    static_prompt.format(excerpt),
                    max_tokens=1024,
                    model=model,
                    system_prompt=system_prompt,
                    **kwargs
                )
                tasks.append(task)

            candidates = self.caller(tasks)
            valid_candidates = []
            for c in candidates:

                if "Regular Expression:" not in c:
                    continue

                c = c.split("Regular Expression:")[1].strip()

                if len(c.split("```")) > 1:
                    c = c.split("```")[1].strip()
                else:
                    c = c.replace("```", "").strip()

                if c == "No Static Part":
                    continue

                try:
                    c = self.re_compile(c)
                except re.error:
                    print("Unable to compile regex {}".format(c))
                    continue

                _, errors = self.verif_regex(filtered_lines, c, strict=False)
                valid_candidates.append(
                    (c, len(errors) / len(filtered_lines), errors)
                )

            if not len(valid_candidates):
                break

            best_coverage = min(c[1] for c in valid_candidates)
            best_candidates = [
                c for c in valid_candidates if c[1] == best_coverage
            ]

            shortest_candidate = max(len(c[0].pattern) for c in best_candidates)
            shortest_candidates = [
                c
                for c in best_candidates
                if len(c[0].pattern) == shortest_candidate
            ]

            pattern, _, filtered_lines = random.choice(shortest_candidates)

            self.templates.append(pattern)

        return self.templates
