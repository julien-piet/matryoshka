import random
import re

from .tools.api import OpenAITask
from .tools.module import Module
from .tools.prompts import system_prompt, unit_prompt


class LogUnit(Module):

    def __init__(
        self, caller=None, parallel_attempts=8, line_count=100, **kwargs
    ) -> None:
        super().__init__("logunit", caller=caller)
        self.parallel_attempts = parallel_attempts
        self.line_count = line_count

    def process(self, log_file, model="gemini-1.5-flash", **kwargs) -> None:
        log_contents = self.load_log(log_file)
        lines = log_contents.split("\n")
        tasks, excerpts = [], []

        for _ in range(self.parallel_attempts):
            random_start = random.randint(0, len(lines) - self.line_count)
            excerpt = "\n".join(
                lines[random_start : random_start + self.line_count]
            )
            excerpts.append(excerpt)
            task = OpenAITask(
                unit_prompt.format(excerpt),
                max_tokens=128,
                model=model,
                system_prompt=system_prompt,
                **kwargs
            )
            tasks.append(task)

        candidates = self.caller(tasks)
        valid_candidates = []
        for c_idx, c in enumerate(candidates):
            if len(c.split("```")) > 1:
                c = c.split("```")[1].strip()
            else:
                c = c.replace("```", "").strip()

            try:
                c = self.re_compile(c)
            except re.error:
                print("Unable to compile regex {}".format(c))
                continue

            ratio_in_sample = self.verif_regex(
                excerpts[c_idx], c, count=True
            ) / len(excerpts[c_idx].split("\n"))
            ratio_in_total = self.verif_regex(
                log_contents, c, count=True
            ) / len(lines)
            if (
                1 >= ratio_in_sample > 0.05
                and 1 >= ratio_in_total > 0.05
                and 1.5 > ratio_in_sample / ratio_in_total > 0.75
            ):
                valid_candidates.append(
                    (
                        min(
                            ratio_in_sample / ratio_in_total,
                            ratio_in_total / ratio_in_sample,
                        ),
                        c,
                    )
                )

        if not len(valid_candidates):
            raise ValueError("No valid regex found")

        max_score = max([x[0] for x in valid_candidates])
        filtered_candidates = [
            x[1] for x in valid_candidates if x[0] > max_score - 0.01
        ]

        min_len = min([len(x.pattern) for x in filtered_candidates])
        filtered_candidates = [
            x for x in filtered_candidates if len(x.pattern) == min_len
        ]
        return random.sample(filtered_candidates, 1)[0]
