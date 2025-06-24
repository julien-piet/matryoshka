import json
import os
import queue
import re
import time
from collections import defaultdict
from queue import PriorityQueue

import dill
from tqdm import tqdm

from ..classes import Module, Template, TemplateTree, Value
from ..utils.logging import get_logger
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
        model="gemini-2.5-flash",
        buffer_len=10000,
        max_age=200000,
        checkpoint_frequency=1,
        force_overlap=False,
        max_memory=-1,
        use_description_distance=True,
        use_fewshot=False,
        max_line_length=1500,
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
        self.max_memory = max_memory
        self.max_line_length = max_line_length

        # Init tree
        self.tree = TemplateTree(distances=[])

        # Init saved variables
        self.line_to_match = {}

        # Init templates
        seeds = []
        if use_fewshot:
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
        if use_fewshot:
            self.entries_per_template = {
                t_idx: d["examples"] for t_idx, d in enumerate(seeds)
            }
            self.values = {
                v: Value(v)
                for v in range(len(self.tree.nodes))
                if v > 0 and self.tree.nodes[v].is_variable()
            }
        else:
            self.entries_per_template = {}
            self.values = {}

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
            max_age=self.max_age,
            line_to_match=self.line_to_match,
            use_description_distance=use_description_distance,
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
            **kwargs,
        )

        self.change_to_greedy = Greedy(self.tree, debug_folder=debug_folder)

        self.ingestion = []
        self.ingested = set()

        # Init cache in heuristics
        if use_fewshot:
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

        # Overlap decisions
        self.overlap_rejections = []
        self.force_overlap = force_overlap
        self.fixed_template_count = len(seeds) if seeds else 0

    def apply_setting(self, force_overlap, checkpoint_frequency, max_memory):
        self.force_overlap = force_overlap
        self.checkpoint_frequency = checkpoint_frequency
        self.max_memory = max_memory
        if max_memory >= 0:
            self.curtail_memory()

    def curtail_memory(self):
        """Curtail the memory usage by limiting the number of entries per template."""
        if self.max_memory < 0:
            return

        if "matches" in self.__dict__:
            self.__dict__.pop("matches", None)

        for t_id, _ in enumerate(self.tree.templates):
            if (
                self.tree.examples[t_id]
                and len(self.tree.examples[t_id]) > self.max_memory
            ):
                self.tree.examples[t_id] = self.tree.examples[t_id][
                    -self.max_memory :
                ]
            if (
                t_id in self.entries_per_template
                and len(self.entries_per_template[t_id]) > self.max_memory
            ):
                self.entries_per_template[t_id] = self.entries_per_template[
                    t_id
                ][-self.max_memory :]

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
        if self.checkpoint_frequency < 1:
            return
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

    def parse_buffer(
        self,
        remaining_lines,
        bar,
        update_bar_on_failure=False,
    ):
        """
        Parse buffer, and replenish it using remaining lines

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
            if len(line) > self.max_line_length:
                self.ingested.add(line_idx)
                if update_bar_on_failure:
                    bar.update(1)
                get_logger().warning(
                    "Line %s is too long (%d > %d), skipping",
                    line_idx,
                    len(line),
                    self.max_line_length,
                )
                continue

            if line.strip() in self.unparsed:
                self.ingested.add(line_idx)
                continue

            match, candidates = self.tree.match(line)
            self.line_to_match[line] = candidates[0]

            # If the current line matches a template, add it to the list of matches
            if match:
                t_id, matches = candidates[0].template_id, candidates[0].matches
                if (
                    self.max_memory < 0
                    or len(self.entries_per_template[t_id]) < self.max_memory
                ):
                    self.entries_per_template[t_id].append(line)
                    self._ingest_values(matches)
                    self.tree.examples[t_id].append(line)

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
                self.ingested.add(line_idx)
                bar.update(1)

                continue

            new_ingestion.append((line_idx, line))
            if update_bar_on_failure:
                bar.update(1)

        self.ingestion = sorted(new_ingestion, key=lambda x: x[0])
        return self.ingestion, None

    def parse(self, log_file, percentage=1, **kwargs) -> None:
        all_lines = [
            re.sub(r"\s+", " ", line)
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

            get_logger().debug(
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

            blocklist = []
            added_to_tree = False
            reduced = False
            get_logger().debug(
                "Entering parsing loop at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            new_template_regex, overlapping_templates = None, None

            # Main generation loop
            while True:
                (
                    new_template,
                    retries,
                    matched_lines,
                    build_template_demo,
                    build_template_description,
                    suffixes,
                    build_template_result,
                    overlap,
                ) = (None, 0, None, None, None, None, None, None)
                while new_template is None and retries < 3:
                    get_logger().debug(
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
                        model=self.model,
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

                # Compute regex suffixes
                undetermined_variables = [
                    i
                    for i, elt in enumerate(new_template.elements)
                    if elt.is_variable() and "_" in str(elt.id)
                ]
                new_template_regex = new_template
                if undetermined_variables:
                    new_template_regex, retries = None, 0
                    while new_template_regex is None and retries < 2:
                        get_logger().debug(
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
                            model=self.model,
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
                    automated_decision, components = self.overlap_heuristics(
                        new_template_regex,
                        overlapping_templates,
                        entries,
                        force=self.force_overlap,
                    )

                    if automated_decision == 2 or (
                        automated_decision == 0
                        and not self.resolve_overlap(
                            new_template_regex, overlapping_templates
                        )
                    ):
                        blocklist = [j for _, j in overlap]
                        entries = [entries[0]]
                        cluster_save_to_cache = False
                        if automated_decision == 0:
                            self.overlap_rejections.append(
                                [set(l) for l in components]
                            )
                        continue
                    elif automated_decision == 1:
                        for template_id in overlapping_templates:
                            self.tree.remove_template(template_id)
                            del self.entries_per_template[template_id]

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
            get_logger().debug(
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
            get_logger().debug(
                "Parsing buffer at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            self.parse_buffer(remaining_lines, bar)
            self._save_state()

    def overlap_heuristics(
        self, candidate_template, overlap_ids, cluster, force=False
    ):
        """Applies heuristics to resolve overlap
        Inputs:
            - candidate_template: the new template that overlaps with old ones.
            - overlap_ids: set of template IDs that overlap with the candidate.
            - force: if True, will force the decision even if heuristics are inconclusive (no human feedback).
        Returns:
            1 if the candidate template should be kept and old ones discarded.
            2 if the old templates should be kept and the candidate discarded.
            3 if all templates should be kept.
            0 if no decision can be made.
        """
        # First heuristic: Check the overlap map and see if one of the overlaps captures all the others
        overlap_map = self.tree.overlap_map
        overlap_sets = {i: set(overlap_map.get(i, [])) for i in overlap_ids}

        # We want to find the set of overlap partitions, using a union-find approach.
        # An overlap paritition is a group of templates such that for all index i in the group, there exists an index j in the group such that i and j overlap.
        # overlap_sets is a dictionary of sets, where j \in overlap_sets[i] means that template i overlaps with template j. this is symmetric (i.e. if i overlaps with j, then j overlaps with i).

        parent = {}
        rank = {}

        for idx in overlap_ids:
            parent[idx] = idx
            rank[idx] = 0

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                if rank[rootx] < rank[rooty]:
                    parent[rootx] = rooty
                elif rank[rootx] > rank[rooty]:
                    parent[rooty] = rootx
                else:
                    parent[rooty] = rootx
                    rank[rootx] += 1

        for i in overlap_ids:
            for j in overlap_ids:
                if j in overlap_sets[i]:
                    union(i, j)

        comp_dict = defaultdict(list)
        for x in overlap_ids:
            rootx = find(x)
            comp_dict[rootx].append(x)

        # Second heuristic: Check if the overlaps are templates with the same elements but different regexes. We do this by checking that at least one overlap per component has the same elements as the candidate template.
        overlap_templates = {
            tid: self.tree.gen_template(tid) for tid in overlap_ids
        }

        def _is_same_except_for_regex(template1, template2):
            if len(template1.elements) != len(template2.elements):
                return False
            for e1, e2 in zip(template1.elements, template2.elements):
                if e1.is_variable() != e2.is_variable():
                    return False
                if (
                    not e1.is_variable()
                    and e1.value.strip() != e2.value.strip()
                ):
                    return False
            return True

        if all(
            any(
                _is_same_except_for_regex(
                    candidate_template, overlap_templates[overlap_template_id]
                )
                for overlap_template_id in component
            )
            for component in comp_dict.values()
        ):
            get_logger().info(
                "Keeping both the candidate template %s and overlaps due to overlap heuristic #1",
                candidate_template,
            )
            return 3, comp_dict.values()

        # Zeroth heuristic: If any overlap is a fixed template, reject
        if any(i < self.fixed_template_count for i in overlap_ids):
            get_logger().info(
                "Rejecting candidate template %s due to overlap with fixed template",
                candidate_template,
            )
            return 2, overlap_ids

        # Now we have the connected components in comp_dict. If there is only one component, it means all templates are interconnected.
        if len(comp_dict) == 1:
            get_logger().info(
                "Keeping both the candidate template %s and overlaps due to overlap heuristic #2",
                candidate_template,
            )
            return 3, comp_dict.values()

        # Third heuristic: Check if a previous user decision can help
        # More specifically, anytime a user decides a template is too general, we save the set of components that overlapped.
        # If at least two of the components in the current overlap are part of a same previously rejected set, we can conclude that the candidate template is too general.
        for rejected_set_raw in self.overlap_rejections:
            # Remove removed templates from the rejected set
            try:
                rejected_set = [
                    {t for t in component if self.tree.templates[t]}
                    for component in rejected_set_raw
                ]
            except:
                continue
            count = 0

            for component in comp_dict.values():
                if any(
                    old_component in component for old_component in rejected_set
                ):
                    count += 1
            if count >= 2:
                get_logger().info(
                    "Rejecting candidate template %s due to overlap heuristic #3",
                    candidate_template,
                )
                return 2, comp_dict.values()

        # Fourth heuristic: Run the cluster confirmation on one of each of the lines that overlap and the current line
        entries = []
        for component in comp_dict.values():
            # Take the first entry from each component
            if component:
                entries.append(self.tree.examples[component[0]][0])
        entries.extend(cluster)
        clustered_entries, _, _, _ = self.select_lines.query(
            entries, None, verbose=True
        )
        if (
            all(c in entries for c in cluster)
            and len(clustered_entries) - len(cluster) > len(comp_dict) // 2
        ):
            print("Overlap examples: ", "\n".join(entries))
            print("Clustered entries: ", "\n".join(clustered_entries))
            get_logger().info(
                "Accepting candidate template %s due to overlap heuristic #4",
                candidate_template,
            )
            return 1, comp_dict.values()

        # If we reach this point, we cannot make a decision based on heuristics alone.
        # Either we reject by default, or ask a user
        if force:
            get_logger().info(
                "Forcing rejection of candidate template %s due to inconclusive heuristics",
                candidate_template,
            )
            with open(
                os.path.join(
                    self.debug_folder,
                    "user_requests.txt",
                ),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"Candidate template {candidate_template} overlaps with {comp_dict.values()}\n"
                )
            self.overlap_rejections.append([set(l) for l in comp_dict.values()])
            return 2, comp_dict.values()
        else:
            return 0, comp_dict.values()

    def resolve_overlap(self, candidate_template, old_template_overlap_ids):
        """Resolve overlap by applying heuristics, or asking user what to do"""
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
