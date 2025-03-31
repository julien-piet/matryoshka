import json
import os
import queue
import re
import sys
import time
from collections import defaultdict
from queue import PriorityQueue

import dill
from tqdm import tqdm

from matryoshka.validation.syntax import SyntaxValidator

from ..classes import Module, Parser, Template, TemplateTree, Value
from ..utils.logging import get_logger
from .heuristics import (
    BuildCluster,
    GenerateSeparation,
    Greedy,
)
from .utils.generate_explanation import GenerateExplanation

sys.setrecursionlimit(sys.getrecursionlimit() * 10)


class VariableParser(Module):

    def __init__(
        self,
        caller=None,
        unit_regex=re.compile("\n"),
        parallel_attempts=5,
        few_shot_length=5,
        init_templates=None,
        debug_folder=None,
        model="gemini-2.5-flash",
        validation_model="gemini-2.5-pro",
        buffer_len=10000,
        max_age=200000,
        checkpoint_frequency=1,
        force_overlap=False,
        max_memory=-1,
        use_description_distance=True,
        use_fewshot=False,
        max_line_length=1500,
        parser=None,
        validation_size=10,
        run_validation=True,
        skip_cluster=False,
        total_validation_rounds=-1,
        reparse_overlaps=True,
        ablation_fewshot=False,
        ablation_self_correction=False,
        ablation_no_overlap_avoidance=False,
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
        self.existing_parser = parser
        self.skip_cluster = skip_cluster
        self.total_validation_rounds = total_validation_rounds
        self.validation_rounds = 0
        self.reparse_overlaps = reparse_overlaps
        self.ablation_fewshot = ablation_fewshot
        self.ablation_self_correction = ablation_self_correction
        self.ablation_no_overlap_avoidance = ablation_no_overlap_avoidance

        # Init saved variables
        self.line_to_match = {}
        self.all_lines = []
        self.line_to_template = defaultdict(set)
        seeds = None

        # Debugging info
        self.seen_indices = set()
        self.debug_folder = debug_folder

        if not parser:
            # Init tree
            self.tree = TemplateTree(distances=[])

            # Init templates
            if use_fewshot:
                with open(init_templates, "r", encoding="utf-8") as f:
                    seeds = json.load(f)

                self.all_lines = [
                    re.sub(r"\s+", " ", example.strip())
                    for seed in seeds
                    for example in seed["examples"]
                ]
                for seed in seeds:
                    self.tree.add_template(
                        Template.load_from_json(
                            seed["template"], seed["examples"][0].strip()
                        ),
                        seed["examples"][0].strip(),
                        fixed=True,
                    )
        else:
            # Load from parser
            self.tree = parser.tree
            self.tree.reset_regex()

        self.entries_per_template = {
            t_idx: []
            for t_idx in range(len(self.tree.templates))
            if self.tree.templates[t_idx]
        }
        self.values = {
            v: Value(v)
            for v in range(len(self.tree.nodes))
            if v > 0 and self.tree.nodes[v] and self.tree.nodes[v].is_variable()
        }
        self.counts_per_template = {
            t_idx: 0
            for t_idx in range(len(self.tree.templates))
            if self.tree.templates[t_idx]
        }
        self.lines_per_template = {
            t_idx: []
            for t_idx in range(len(self.tree.templates))
            if self.tree.templates[t_idx]
        }

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
            ablation_fewshot=ablation_fewshot,
            ablation_self_correction=ablation_self_correction,
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
            ablation_fewshot=ablation_fewshot,
            ablation_self_correction=ablation_self_correction,
            **kwargs,
        )

        self.change_to_greedy = Greedy(self.tree, debug_folder=debug_folder)

        self.ingestion = []
        self.ingested = set()
        self.unparsed = set()

        # Overlap decisions
        self.overlap_rejections = []
        self.force_overlap = force_overlap
        self.fixed_template_count = len(seeds) if seeds else 0

        # Save flag if we need to generate explanations
        self.generate_explanations = GenerateExplanation(
            self.tree,
            self.caller,
            self.build_template,
            self.select_lines,
            self.model,
            path=debug_folder,
            values=self.values,
            entries_per_template=self.entries_per_template,
        )

        # Init cache in heuristics
        if use_fewshot and not parser:
            template_ids = list(self.entries_per_template.keys())
            parsed_lines = self._reparse_changed_templates(
                template_ids,
                desc="Generating explanations for human labeled few-shot examples.",
            )
            self._create_explanations(template_ids)
            parsed_lines_set = set(parsed_lines) if parsed_lines else set()
            all_lines_indices = set(range(len(self.all_lines)))
            missing_lines = all_lines_indices - parsed_lines_set
            if missing_lines:
                get_logger().error(
                    "Some few-shot examples were not parsed: %s",
                    "\n".join([self.all_lines[i] for i in missing_lines]),
                )
                breakpoint()
                return

        # Validation
        if validation_size == 0:
            run_validation = False

        if run_validation:
            self.validate_heuristic = SyntaxValidator(
                caller,
                tree=self.tree,
                values=self.values,
                entries_per_template=self.entries_per_template,
                save_path=debug_folder,
                model=validation_model,
            )
            self.validated_templates = set()
            self.validation_size = validation_size

            for t_id, template in enumerate(self.tree.templates):
                if not template:
                    continue
                if all(self.tree.nodes[elt_id].fixed for elt_id in template):
                    self.validated_templates.add(t_id)

            self.failed_validation_templates = set()

        else:
            self.validate_heuristic = None
            self.validated_templates = set()
            self.validation_size = 0
            self.failed_validation_templates = set()

        # Lines
        self.all_lines = []

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
                self.lines_per_template[t_id] = self.lines_per_template[t_id][
                    -self.max_memory :
                ]

    def init_caller(self, caller):
        self.caller = caller
        for heuristic in [
            self.select_lines,
            self.build_template,
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
            try:
                with open(
                    os.path.join(
                        self.debug_folder,
                        f"saved_parser_{len(self.tree.templates)}{suffix}.dill",
                    ),
                    "wb",
                ) as f:
                    dill.dump(
                        Parser(
                            tree=self.tree,
                            event_types=(
                                self.existing_parser.event_types
                                if self.existing_parser
                                else None
                            ),
                            var_mapping=(
                                self.existing_parser.var_mapping
                                if self.existing_parser
                                else None
                            ),
                            schema_mapping=(
                                self.existing_parser.schema_mapping
                                if self.existing_parser
                                else None
                            ),
                            schemas=(
                                self.existing_parser.schemas
                                if self.existing_parser
                                else None
                            ),
                            template_mapping=(
                                self.existing_parser.template_mapping
                                if self.existing_parser
                                else None
                            ),
                        ),
                        f,
                    )
            except Exception as e:
                breakpoint()

            get_logger().info(
                "Saved state after parsing %s lines in saved_parser_%s.dill",
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

    def process_line(self, line, line_idx, template_ids=None):
        line = line.strip()
        if len(line) > self.max_line_length:
            get_logger().warning("Line %s is too long, skipping", line_idx)
            return True

        if line.strip() in self.unparsed:
            return True

        match, candidates = self.tree.match(line)

        match_ids = []
        for candidate in candidates:
            match_ids.append(candidate.template_id)

        selected_candidate = None
        if template_ids:
            for candidate in candidates:
                if candidate.template_id in template_ids:
                    selected_candidate = candidate
                    break
        else:
            selected_candidate = candidates[0]

        if match:
            # Update overlap map
            for t_id_1 in match_ids:
                for t_id_2 in match_ids:
                    if t_id_1 != t_id_2:
                        if t_id_1 not in self.tree.overlap_map:
                            self.tree.overlap_map[t_id_1] = set()
                        if t_id_2 not in self.tree.overlap_map:
                            self.tree.overlap_map[t_id_2] = set()
                        self.tree.overlap_map[t_id_1].add(t_id_2)
                        self.tree.overlap_map[t_id_2].add(t_id_1)

            for candidate in candidates:
                template_id, matches = candidate.template_id, candidate.matches
                if (
                    self.max_memory < 0
                    or len(self.entries_per_template[template_id])
                    < self.max_memory
                ):
                    self.entries_per_template[template_id].append(line)
                    self.lines_per_template[template_id].append(line_idx)
                    self._ingest_values(matches)
                    self.tree.examples[template_id].append(line)
                    self.line_to_template[line_idx].add(template_id)

                # Add line to seen
                self.counts_per_template[template_id] = (
                    self.counts_per_template.get(template_id, 0) + 1
                )

            # Update line to match
            if line in self.line_to_match:
                del self.line_to_match[line]
            return True

        if selected_candidate:
            self.line_to_match[line] = selected_candidate

            # Check if the template exists
            if not selected_candidate.suffix.strip():
                t_id = self.tree.add_template(
                    Template(
                        [self.tree.nodes[i] for i in selected_candidate.trail]
                    ),
                    line,
                    debug=True,
                )
                get_logger().info(
                    "Line %s matches an existing path %s that was not a template",
                    line_idx,
                    self.tree.gen_template(t_id),
                )
                self.entries_per_template[t_id] = [line]
                self.lines_per_template[t_id] = [line_idx]
                self.counts_per_template[t_id] = 1
                self._ingest_values(selected_candidate.matches)
                return True

        return False

    def parse_buffer(self, remaining_lines, bar, update_bar_on_failure=False):
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
            ingested = self.process_line(line, line_idx)
            if ingested:
                self.ingested.add(line_idx)
                bar.update(1)
            else:
                new_ingestion.append((line_idx, line))
                if update_bar_on_failure:
                    bar.update(1)

        self.ingestion = sorted(new_ingestion, key=lambda x: x[0])
        return self.ingestion, None

    def parse(self, log_file, percentage=1, **kwargs) -> None:
        all_lines = [
            re.sub(r"\s+", " ", line)
            for line in self.load_and_split_log(log_file)
            if len(line)
        ]

        if percentage < 1:
            all_lines = all_lines[: int(len(all_lines) * percentage)]

        if self.all_lines:
            self.all_lines += all_lines
        else:
            self.all_lines = all_lines
        remaining_lines = queue.Queue()

        if self.existing_parser is None or all(
            [not t for t in self.tree.templates]
        ):
            for line_idx, line in enumerate(all_lines):
                if line_idx not in self.ingested:
                    remaining_lines.put((line_idx, line))
        else:
            templates = [
                t_id for t_id, t in enumerate(self.tree.templates) if t
            ]
            parsed_lines = self._reparse_changed_templates(
                templates,
                desc="Generating explanations for existing templates",
                force_all=True,
            )
            self._create_explanations(templates)
            parsed_lines_set = set(parsed_lines) if parsed_lines else set()
            for line_idx, line in enumerate(all_lines):
                if (
                    line_idx not in parsed_lines_set
                    and line_idx not in self.ingested
                ):
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
            ) = self.select_lines.run(lines, skip_cluster=self.skip_cluster)

            blocklist = []
            added_to_tree = False
            reduced = False
            get_logger().info(
                "Entering parsing loop at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            t_id = -1
            overlapping_templates = None
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
                    get_logger().info(
                        "Getting template (attempt #%s) at time %s",
                        1 + retries,
                        time.strftime("%H:%M:%S", time.localtime()),
                    )
                    model = self.model
                    (
                        new_template,
                        matched_lines,
                        build_template_demo,
                        build_template_description,
                        suffixes,
                        build_template_result,
                        overlap,
                    ) = self.build_template.run(
                        few_shot_ids,
                        entries,
                        matched_prefix,
                        description,
                        model=model,
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
                    get_logger().warning("Missing some lines")
                    reduced = True
                    entries = [entries[i] for i in matched_lines]
                    if suffixes:
                        suffixes = [suffixes[i] for i in matched_lines]

                overlapping_templates = None
                if overlap and not self.ablation_no_overlap_avoidance:
                    # If the overlap is with a single template, just give a warning message. If with multiple that do not overlap with each other, ask user what to do.
                    overlapping_templates = {i for i, _ in overlap}
                    get_logger().warning(
                        "Overlap detected for new template %s with template(s) %s",
                        new_template,
                        ",".join([str(t) for t in overlapping_templates]),
                    )
                    automated_decision, components = self.overlap_heuristics(
                        new_template,
                        overlapping_templates,
                        entries[:],
                        force=self.force_overlap,
                    )
                    if automated_decision == 2 or (
                        automated_decision == 0
                        and not self.resolve_overlap(
                            new_template, overlapping_templates
                        )
                    ):
                        blocklist = [j for _, j in overlap]
                        entries = [entries[0]]
                        cluster_save_to_cache = False
                        if automated_decision == 0:
                            self.overlap_rejections.append(
                                [set([1]) for l in components]
                            )
                        continue
                    elif automated_decision == 1:
                        # Find the overlapped templates that are strictly included in this one
                        redundant_templates = self.filter_redundant_templates(
                            overlapping_templates, new_template
                        )
                        # Remove templates that are redundant
                        for template_id in redundant_templates:
                            get_logger().warning(
                                "Removing template %s due to complete overlap with new template %s",
                                template_id,
                                new_template,
                            )
                            self.tree.remove_template(template_id)
                            del self.entries_per_template[template_id]
                            del self.lines_per_template[template_id]
                            del self.counts_per_template[template_id]
                            if template_id in self.validated_templates:
                                self.validated_templates.remove(template_id)

                        # Mark nodes of other overlapped templates as non fixed, so validation can potentially further fix the overlap.
                        for template_id in overlapping_templates - set(
                            redundant_templates
                        ):
                            if template_id in self.validated_templates:
                                self.validated_templates.remove(template_id)
                            for node_id, node in enumerate(self.tree.nodes):
                                if not node:
                                    continue
                                member_templates = {
                                    t
                                    for t in self.tree.templates_per_node[
                                        node_id
                                    ]
                                    if self.tree.templates[t] and t != 0
                                }
                                if not member_templates.issubset(
                                    self.validated_templates
                                ):
                                    node.fixed = False

                # Add template to tree
                t_id = self.tree.add_template(new_template, entries[0])
                match, _ = self.change_to_greedy.run(t_id, entries[0])
                if not match:
                    breakpoint()
                    raise ValueError(
                        "Generated template does not match even after adjustment."
                    )

                added_to_tree = True

                if overlapping_templates and self.reparse_overlaps:
                    overlapping_templates = list(overlapping_templates) + [t_id]
                    self.reparse(overlapping_templates, run_explanations=False)

                break

            # Check that examples align with templates
            for template_id, template in enumerate(self.tree.templates):
                if not template:
                    continue
                if not self.tree.examples[template_id]:
                    continue
                if not self.tree.gen_template(template_id).match(
                    self.tree.examples[template_id][0]
                )[0]:
                    get_logger().error(
                        "Template %s does not match example %s",
                        template_id,
                        self.tree.examples[template_id][0],
                    )
                    breakpoint()

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
            self.lines_per_template[t_id] = []
            if new_template:
                for elt in new_template.elements:
                    elt_id = elt.id
                    if elt.is_variable() and elt.id not in self.values:
                        self.values[elt_id] = Value(
                            elt_id,
                        )

            # Save overlaps
            if overlap and "overlap_map" in self.tree.__dict__:
                self.tree.overlap_map[t_id] = set()
                if overlapping_templates:
                    for o_id in overlapping_templates:
                        if o_id not in self.tree.overlap_map:
                            self.tree.overlap_map[o_id] = set()
                        self.tree.overlap_map[o_id].add(t_id)
                        self.tree.overlap_map[t_id].add(o_id)

            # Parse buffer
            get_logger().info(
                "Parsing buffer at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            self.parse_buffer(remaining_lines, bar)
            self._save_state()

            # Run validation
            if self.validate_heuristic is not None:
                non_validated_templates = [
                    t_id
                    for t_id, t in enumerate(self.tree.templates)
                    if t
                    and t_id not in self.validated_templates
                    and t_id not in self.failed_validation_templates
                ]
                if len(non_validated_templates) > self.validation_size:
                    self.validate()

                    # Reparse buffer
                    get_logger().info(
                        "Parsing buffer at time %s",
                        time.strftime("%H:%M:%S", time.localtime()),
                    )
                    self.parse_buffer(remaining_lines, bar)
                    self._save_state()

        if self.validate_heuristic is not None and [
            t_id
            for t_id, t in enumerate(self.tree.templates)
            if t
            and t_id not in self.validated_templates
            and t_id not in self.failed_validation_templates
        ]:
            self.validate()
            get_logger().info(
                "Finished validation at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
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
        if len(comp_dict) == 1 and False:
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
            if component and self.tree.examples[component[0]]:
                entries.append(self.tree.examples[component[0]][0])
        entries.extend(cluster)
        clustered_entries, _, _, _ = self.select_lines.query(
            entries, None, normalize=False
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

    def _reparse_changed_templates(
        self,
        changed_templates,
        desc="Reparsing changed templates",
        indices=None,
        empty_cache=True,
        force_all=False,
    ):

        min_line_ids = []

        # Empty state for changed templates
        for template_id in changed_templates:
            min_line_ids += self.lines_per_template.get(template_id, [])
            self.entries_per_template[template_id] = []
            self.lines_per_template[template_id] = []
            self.counts_per_template[template_id] = 0
            if template_id in self.build_template.cache and empty_cache:
                del self.build_template.cache[template_id]
            if template_id in self.select_lines.cache and empty_cache:
                del self.select_lines.cache[template_id]
            self.tree.overlap_map[template_id] = set()

        # Empty state for changed values
        changed_template_set = set(changed_templates)
        for node_id, node in enumerate(self.tree.nodes):
            if not node:
                if node_id in self.values:
                    del self.values[node_id]
                continue

            member_templates = self.tree.templates_per_node[node_id]
            if (
                member_templates.issubset(changed_template_set)
                and node_id in self.values
            ):
                self.values[node_id] = Value(node_id)

        if not indices:
            iterator = list(enumerate(self.all_lines))
            if self.max_memory > 0 and not force_all:
                iterator = iterator[-self.max_memory :]
                selected_lines = set(i[0] for i in iterator)
                for line_id in min_line_ids:
                    if line_id not in selected_lines:
                        iterator.append((line_id, self.all_lines[line_id]))
            total_length = len(iterator)
        else:
            iterator = [(idx, self.all_lines[idx]) for idx in indices]
            if self.max_memory > 0 and not force_all:
                iterator = iterator[-self.max_memory :]
                selected_lines = set(i[0] for i in iterator)
                for line_id in min_line_ids:
                    if line_id not in selected_lines and line_id in indices:
                        iterator.append((line_id, self.all_lines[line_id]))
            total_length = len(indices)

        parsed_lines = []
        for line_idx, line in tqdm(iterator, desc=desc, total=total_length):
            if self.process_line(
                line, line_idx, template_ids=changed_template_set
            ):
                parsed_lines.append(line_idx)

        for template_id in changed_template_set:
            self.tree.examples[template_id] = self.entries_per_template[
                template_id
            ][:5]

        return parsed_lines

    def _create_explanations(self, template_ids):
        if not template_ids:
            return
        self.generate_explanations.run(template_ids)

    def _correct_parsed_lines(self, template_ids):
        """Mark lines that were parsed with changed templates as unparsed"""
        changed_template_set = set(template_ids)
        to_be_reparsed = []
        for line_idx in self.ingested:
            if self.line_to_template[line_idx].issubset(changed_template_set):
                to_be_reparsed.append(line_idx)
            self.line_to_template[line_idx] -= changed_template_set

        return to_be_reparsed

    def filter_redundant_templates(self, template_ids, candidate_template):
        """Filter templates that are strictly included in the candidate template"""
        target_lines, template_ids_set = [], set(template_ids)
        template_to_line_id = {}
        for line_idx in self.ingested:
            if self.line_to_template[line_idx] & template_ids_set:
                target_lines.append(line_idx)
                for t_id in self.line_to_template[line_idx]:
                    if t_id in template_ids_set:
                        if t_id not in template_to_line_id:
                            template_to_line_id[t_id] = set()
                        template_to_line_id[t_id].add(line_idx)

        matched_lines = set()
        for target_line_idx in target_lines:
            if candidate_template.match(self.all_lines[target_line_idx])[0]:
                matched_lines.add(target_line_idx)

        filtered_list = []
        for template_id in template_ids:
            if template_id not in template_to_line_id:
                filtered_list.append(template_id)
            elif template_to_line_id[template_id].issubset(matched_lines):
                filtered_list.append(template_id)
        return filtered_list

    def reparse(self, changed_templates, run_explanations=True):
        to_be_reparsed = self._correct_parsed_lines(changed_templates)
        changed_templates = [
            c for c in changed_templates if self.tree.templates[c]
        ]
        re_ingested = self._reparse_changed_templates(
            changed_templates,
            indices=to_be_reparsed,
            empty_cache=run_explanations,
        )
        all_templates = [
            t_id
            for t_id, t in enumerate(self.tree.templates)
            if t and t_id not in self.validated_templates
        ]
        if run_explanations:
            self._create_explanations(all_templates)

        # Add the non ingested lines to the ingestion queue
        if to_be_reparsed:
            if re_ingested:
                no_longer_parsed = set(to_be_reparsed) - set(re_ingested)
            else:
                no_longer_parsed = to_be_reparsed
            if no_longer_parsed:
                breakpoint()
                get_logger().info(
                    "Adding %s lines to ingestion queue that are no longer parsed",
                    len(no_longer_parsed),
                )
                self.ingestion = [
                    (idx, self.all_lines[idx]) for idx in no_longer_parsed
                ] + self.ingestion
                # Delete cached candidates since they might not be valid anymore
                self.line_to_match.clear()

        if re_ingested:
            self.ingested = self.ingested.union(set(re_ingested))

    def validate(self, validation_size=None):
        """Validates the syntax tree"""
        if self.validate_heuristic is None or (
            self.total_validation_rounds <= self.validation_rounds
            and self.total_validation_rounds >= 0
        ):
            for template_id, template in enumerate(self.tree.templates):
                if not template:
                    continue
                self.validated_templates.add(template_id)
            return

        if not validation_size:
            validation_size = self.validation_size
        elif validation_size <= 0:
            validation_size = len(self.tree.templates)

        # Select most common validated templates
        templates = sorted(
            list(self.validated_templates),
            key=lambda x: self.counts_per_template.get(x, 0),
            reverse=True,
        )[:validation_size]

        # Select all new templates
        templates += [
            t_id
            for t_id, t in enumerate(self.tree.templates)
            if t and t_id not in self.validated_templates
        ]

        # Run validation
        new_tree = None
        for _ in range(3):
            try:
                new_tree = self.validate_heuristic.run(templates)
                break
            except ValueError:
                get_logger().error("Could not validate template - trying again")

        self.validation_rounds += 1

        if not new_tree:
            for template_id in templates:
                self.failed_validation_templates.add(template_id)
            return False

        # Print updates to tree
        get_logger().info("Updating tree")
        changed_templates = self.tree.update_tree(new_tree)
        get_logger().info(
            "Changed templates (%s)", ",".join(map(str, changed_templates))
        )
        self.reparse(changed_templates)

        # Mark all nodes as validated
        for template_id, template in enumerate(self.tree.templates):
            if template_id in self.validated_templates or not template:
                continue
            self.validated_templates.add(template_id)

        for node in self.tree.nodes:
            if node:
                node.fixed = True

        # Clean tree
        self.tree.simplify_tree()

        # Save state
        # self._save_state(suffix="validated", force=True)

        return True

    def process(self, log_file, **kwargs):
        self.parse(log_file, **kwargs)
        self._save_state("final")
