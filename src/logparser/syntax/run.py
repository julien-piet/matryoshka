import json
import os
import queue
import re
import time
from queue import PriorityQueue

import dill
from tqdm import tqdm

from ..tools.classes import (
    Template,
    TemplateTree,
    Value,
)
from ..tools.embedding import NaiveDistance
from ..tools.logging import get_logger
from ..tools.module import Module
from .heuristics import (
    BuildCluster,
    GenerateRegex,
    GenerateSeparation,
    Greedy,
)


class VariableParser(Module):

    def __init__(
        self,
        caller=None,
        unit_regex=re.compile("\n"),
        parallel_attempts=5,
        few_shot_length=2,
        init_templates=None,
        debug_folder=None,
        model="gemini-1.5-pro",
        buffer_len=10000,
        max_age=200000,
        checkpoint_frequency=1,
        **kwargs,
    ) -> None:

        super().__init__("Variable Parser", caller=caller)
        self.parallel_attempts = parallel_attempts
        self.unit_regex = unit_regex
        self.few_shot_length = few_shot_length
        self.model = model
        self.buffer_len = buffer_len
        self.max_age = max_age
        self.checkpoint_frequency = checkpoint_frequency

        # Init distance
        self.naive_distance = NaiveDistance()

        # Init tree
        self.tree = TemplateTree(distances=[self.naive_distance])

        # Init saved variables
        self.matches = []

        # Init templates
        if init_templates:
            with open(init_templates, "r", encoding="utf-8") as f:
                seeds = json.load(f)
            for seed in seeds:
                self.tree.add_template(
                    Template.load_from_json(
                        seed["template"], seed["examples"][0].strip()
                    ),
                    seed["examples"][0].strip(),
                    fixed=True,
                )

        # Init state
        self.entries_per_template = {
            t_idx: d["examples"] for t_idx, d in enumerate(seeds)
        }
        self.values = {
            v: Value(v)
            for v in range(len(self.tree.nodes))
            if v > 0 and self.tree.nodes[v].is_variable()
        }

        # Debugging info
        self.seen_indices = set()
        self.debug_folder = debug_folder

        # Heuristics
        self.select_lines = BuildCluster(
            self.tree,
            self.caller,
            self.model,
            self.parallel_attempts,
            path=debug_folder,
            few_shot_length=self.few_shot_length,
            values=self.values,
            entries_per_template=self.entries_per_template,
            naive_distance=self.naive_distance,
            max_age=self.max_age,
            **kwargs,
        )

        self.build_template = GenerateSeparation(
            self.tree,
            self.caller,
            self.model,
            self.parallel_attempts,
            path=debug_folder,
            few_shot_length=self.few_shot_length,
            values=self.values,
            entries_per_template=self.entries_per_template,
            naive_distance=self.naive_distance,
            **kwargs,
        )

        self.build_regex = GenerateRegex(
            self.tree,
            self.caller,
            self.model,
            self.parallel_attempts,
            path=debug_folder,
            few_shot_length=self.few_shot_length,
            values=self.values,
            entries_per_template=self.entries_per_template,
            naive_distance=self.naive_distance,
            **kwargs,
        )

        self.change_to_greedy = Greedy(self.tree)

        self.ingestion = []
        self.ingested = set()

        # Init cache in heuristics
        descriptions = self.select_lines.ingest_fewshot(
            [t_idx for t_idx in range(len(seeds))], seeds
        )

        self.build_template.ingest_fewshot(
            [self.tree.gen_template(t_idx) for t_idx in range(len(seeds))],
            seeds,
            descriptions,
        )

        self.build_regex.ingest_fewshot(
            [self.tree.gen_template(t_idx) for t_idx in range(len(seeds))]
        )

        self.unparsed = set()

    def init_caller(self, caller):
        self.caller = caller
        for heuristic in [
            self.select_lines,
            self.build_template,
            self.build_regex,
            self.change_to_greedy,
        ]:
            heuristic.init_caller(caller)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["caller"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caller = None

    def _save_state(self, suffix=None, force=False):
        self.checkpoint_frequency = 50
        if (
            len(self.tree.templates) % self.checkpoint_frequency != 0
            and not force
        ):
            return

        if suffix and not suffix.startswith("_"):
            suffix = "_" + suffix
        else:
            suffix = ""
        if self.debug_folder:
            with open(
                os.path.join(
                    self.debug_folder,
                    f"saved_states_{len(self.tree.templates)}{suffix}.dill",
                ),
                "wb",
            ) as f:
                dill.dump(self, f)
            get_logger().info(
                "Saved state after parsing %s lines in saved_states_%s.dill",
                len(self.seen_indices),
                len(self.tree.templates),
            )

    def _ingest_values(self, matches):
        for elt in matches.elements:
            if elt.is_variable():
                if elt.id not in self.values:
                    self.values[elt.id] = Value(
                        elt.id,
                        values=[elt.value],
                    )
                else:
                    self.values[elt.id].append(elt.value)

        self.naive_distance.update(matches)

    def parse_buffer(self, remaining_lines, bar):
        """Parse buffer, and replenish it using remaining lines

        Does a first parsing pass to get exact matches.
        If a line doesn't match, parses it again once we've seen all the lines in the buffer to check for close matches.
        """

        # Setup priority queue
        inputs = PriorityQueue()
        for line_idx, line in self.ingestion:
            inputs.put((line_idx, line))

        # New ingestion stores the updated buffer.
        new_ingestion = []

        while len(new_ingestion) < self.buffer_len and not (
            inputs.empty() and remaining_lines.empty()
        ):
            if not inputs.empty():
                line_idx, line = inputs.get()
            else:
                line_idx, line = remaining_lines.get()
            line = line.strip()
            if len(line) > 1500:
                self.ingested.add(line_idx)
                get_logger().warning("Line %s is too long, skipping", line_idx)
                continue

            if line.strip() in self.unparsed:
                self.ingested.add(line_idx)
                continue

            match, candidates = self.tree.match(line)

            # If the current line matches a template, add it to the list of matches
            if match:
                t_id, matches = candidates[0].template_id, candidates[0].matches
                self.matches.append((line, t_id, matches))
                self.entries_per_template[t_id].append(line)
                self._ingest_values(matches)
                self.tree.examples[t_id].append(line)
                # self.v2c.run(matches)

                # Add line to seen
                self.ingested.add(line_idx)
                bar.update(1)

                continue

            # Check if the template exists
            if not candidates[0].suffix.strip():
                t_id = self.tree.add_template(
                    Template([self.tree.nodes[i] for i in candidates[0].trail]),
                    line,
                )
                get_logger().info(
                    "Line %s matches an existing path %s that was not a template",
                    line,
                    self.tree.gen_template(t_id),
                )
                self.entries_per_template[t_id] = [line]
                self._ingest_values(candidates[0].matches)
                self.matches.append((line, t_id, candidates[0].matches))
                self.ingested.add(line_idx)
                bar.update(1)

                continue

            new_ingestion.append((line_idx, line))

        self.ingestion = sorted(new_ingestion, key=lambda x: x[0])
        return self.ingestion, None

    def parse(self, log_file, percentage=1, **kwargs) -> None:
        all_lines = [
            re.sub("\s", " ", line)
            for line in self.load_and_split_log(log_file, self.unit_regex)
            if len(line)
        ]

        if percentage < 1:
            all_lines = all_lines[: int(len(all_lines) * percentage)]

        remaining_lines = queue.Queue()

        for line_idx, line in enumerate(all_lines):
            if line_idx not in self.ingested:
                remaining_lines.put((line_idx, line))

        bar = tqdm(
            total=remaining_lines.qsize() + len(self.ingestion),
            desc="Processing log file",
            unit="lines",
            initial=len(self.seen_indices),
        )

        # Parsing existing lines
        self.parse_buffer(remaining_lines, bar)
        while len(self.ingestion):
            lines = [v[1] for v in self.ingestion]

            get_logger().info(
                "Getting cluster at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            # Get cluster
            (
                entries,
                matched_prefix,
                few_shot_ids,
                description,
                cluster_entries,
                cluster_demo,
                cluster_output,
                cluster_save_to_cache,
            ) = self.select_lines.run(lines)

            self.tree.match(entries[0], debug=True)

            blocklist = []
            added_to_tree = False
            reduced = False
            get_logger().info(
                "Entering parsing loop at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            while True:
                new_template, retries = None, 0
                while new_template is None and retries < 3:
                    get_logger().info(
                        "Getting template (attempt #%s) at time %s",
                        1 + retries,
                        time.strftime("%H:%M:%S", time.localtime()),
                    )
                    # Get template
                    (
                        new_template,
                        matched_lines,
                        build_template_demo,
                        build_template_description,
                        suffixes,
                        build_template_result,
                    ) = self.build_template.run(
                        few_shot_ids,
                        entries,
                        matched_prefix,
                        description,
                        model=self.model if retries == 0 else "gemini-1.5-pro",
                        blocklist=blocklist,
                    )
                    if new_template is None or not matched_lines:
                        entries = [entries[0]]
                    retries += 1

                if new_template is None:
                    get_logger().error("Could not parse entry")
                    self.unparsed.add(entries[0].strip())
                    break

                if len(matched_lines) < len(entries):
                    get_logger().warning("Missing some lines (separation)")
                    reduced = True
                    entries = [entries[i] for i in matched_lines]
                    suffixes = [suffixes[i] for i in matched_lines]
                    # self.tree.examples[new_template.id] = [entries[0]]

                # Compute regex suffixes
                undetermined_variables = [
                    i
                    for i, elt in enumerate(new_template.elements)
                    if elt.is_variable() and "_" in str(elt.id)
                ]
                if undetermined_variables:
                    new_template_regex, retries = None, 0
                    while new_template_regex is None and retries < 2:
                        get_logger().info(
                            "Getting regexes (attempt #%s) at time %s",
                            1 + retries,
                            time.strftime("%H:%M:%S", time.localtime()),
                        )
                        (
                            new_template_regex,
                            matched_lines,
                            overlap,
                        ) = self.build_regex.run(
                            few_shot_ids,
                            entries,
                            new_template,
                            model=(
                                self.model if retries == 0 else "gemini-1.5-pro"
                            ),
                            force=retries > 0,
                        )
                        retries += 1

                    if new_template_regex is None or not matched_lines:
                        get_logger().warning(
                            "Automatically generated regex does not match, falling back to naive regex"
                        )
                        for elt in new_template.elements:
                            if elt.is_variable() and "_" in str(elt.id):
                                elt.regexp = (
                                    r"\S+" if " " not in elt.value else r".+"
                                )
                        new_template.generate_regex()
                        matched_lines = []
                        for i, entry in enumerate(entries):
                            if new_template.match(entry)[0]:
                                matched_lines.append(i)

                        new_template_regex = new_template

                    if not matched_lines:
                        get_logger().error("Could not parse entry")
                        for entry in entries:
                            self.unparsed.add(entry.strip())
                        break

                    if len(matched_lines) < len(entries):
                        get_logger().warning("Missing some lines (regex)")
                        reduced = True
                        entries = [entries[i] for i in matched_lines]
                        suffixes = [suffixes[i] for i in matched_lines]

                else:
                    new_template_regex = new_template
                    overlap = []

                if overlap:
                    # If the overlap is with a single template, just give a warning message. If with multiple that do not overlap with each other, ask user what to do.
                    overlapping_templates = {i for i, _ in overlap}
                    get_logger().warning(
                        "Overlap detected for new template %s with template(s) %s",
                        new_template_regex,
                        ", ".join([str(t) for t in overlapping_templates]),
                    )
                    if len(overlapping_templates) > 1:
                        if "overlap_map" not in self.tree.__dict__ or any(
                            j not in self.tree.overlap_map.get(i, [])
                            for i in overlapping_templates
                            for j in overlapping_templates
                            if i != j
                        ):
                            if not self.resolve_overlap(
                                new_template_regex, overlapping_templates
                            ):
                                blocklist = [j for _, j in overlap]
                                entries = [entries[0]]
                                cluster_save_to_cache = False
                                continue

                # Add template to tree
                t_id = self.tree.add_template(new_template_regex, entries[0])
                match, _ = self.change_to_greedy.run(t_id, entries[0])
                if not match:
                    raise ValueError(
                        "Generated template does not match even after adjustment."
                    )

                added_to_tree = True
                break

            if not added_to_tree:
                continue

            # Update caches
            get_logger().info(
                "Saving to cache at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            if cluster_save_to_cache and not reduced:
                self.select_lines.save_to_cache(
                    t_id,
                    cluster_entries,
                    cluster_demo,
                    cluster_output,
                )
            self.build_template.save_to_cache(
                t_id,
                entries,
                suffixes,
                build_template_demo,
                build_template_description,
                build_template_result,
            )

            # Create entries
            self.entries_per_template[t_id] = []
            for elt in new_template_regex.elements:
                elt_id = elt.id
                if elt.is_variable() and elt_id not in self.values:
                    self.values[elt_id] = Value(
                        elt_id,
                    )

            # Save overlaps
            if overlap and "overlap_map" in self.tree.__dict__:
                self.tree.overlap_map[t_id] = []
                for o_id in overlapping_templates:
                    if o_id not in self.tree.overlap_map:
                        self.tree.overlap_map[o_id] = []
                    self.tree.overlap_map[o_id].append(t_id)
                    self.tree.overlap_map[t_id].append(o_id)

            # Parse buffer
            get_logger().info(
                "Parsing buffer at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            self.parse_buffer(remaining_lines, bar)
            self._save_state()

    def resolve_overlap(self, candidate_template, old_template_overlap_ids):
        """Resolve overlap by asking user what to do"""
        print("Candidate template ", str(candidate_template))
        print("Overlapping templates:")
        for t_id in old_template_overlap_ids:
            print(self.tree.gen_template(t_id))

        while True:
            choice = input(
                "Choose an option:\n"
                "1. Keep the new one, discard the old ones.\n"
                "2. Keep the old ones, discard the new one.\n"
                "3. Keep all.\n"
                "Enter 1, 2, or 3: "
            ).strip()
            if choice in {"1", "2", "3"}:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")

        if choice == "1":
            for t_id in old_template_overlap_ids:
                self.tree.remove_template(t_id)
            return True
        elif choice == "2":
            return False
        elif choice == "3":
            return True
        return False

    def process(self, log_file, **kwargs):
        self.parse(log_file, **kwargs)
        self._save_state("final")
