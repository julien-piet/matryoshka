from __future__ import annotations

import json
import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import regex as re

from ..utils.json import parse_json
from ..utils.logging import get_logger
from .element import Element, ElementMatchType
from .match import Match
from .template import Template


@dataclass
class Tree:
    node: int
    parent: Optional[Tree] = None
    branches: dict = field(default_factory=dict)
    terminal: bool = False
    template_id: Optional[int] = None

    def __getitem__(self, item) -> Tree:
        if isinstance(item, int) and item in self.branches:
            return self.branches[item]
        elif isinstance(item, int):
            raise KeyError(f"Key {item} not found in tree")
        else:
            raise ValueError("Invalid key type")

    def __setitem__(self, node: int, value: Tree) -> None:
        if not isinstance(node, int) or not isinstance(value, Tree):
            raise ValueError("Invalid assignment")

        self.branches[node] = value

    def __delitem__(self, item) -> None:
        if isinstance(item, int):
            del self.branches[item]
        else:
            raise ValueError("Invalid key type")

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.branches
        else:
            return False

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["parent"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.parent = None

    def get_element(self, node_to_element: list, element: Element):
        branch_elements = {node_to_element[i]: i for i in self.branches}
        if element in branch_elements:
            return branch_elements[element]
        else:
            return None

    def get_lineage(self):
        lineage = []
        current_node = self
        while current_node.parent is not None:
            lineage.append(current_node.node)
            current_node = current_node.parent
        return lineage[::-1]


@dataclass
class ExplorationNode:
    node: int
    suffix: str
    pending: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    matches: Match | None = None


@dataclass
class TerminalNode:
    suffix: str
    trail: list = field(default_factory=list)
    template_id: Optional[int] = None
    matches: Match | None = None

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TerminalNode):
            return False
        return (
            self.suffix == value.suffix
            and self.template_id == value.template_id
        )


@dataclass
class ExplorationState:
    node: int
    matched_trail: list = field(default_factory=list)
    full_trail: list = field(default_factory=list)
    suffix_index: int = 0
    matches: dict = field(default_factory=dict)


@dataclass
class TemplateTree:

    tree: Tree
    nodes: list = field(default_factory=list)
    templates: list = field(default_factory=list)
    examples: list = field(default_factory=list)

    templates_per_node: dict = field(default_factory=dict)
    node_to_tree: dict = field(default_factory=dict)
    overlap_map: dict = field(default_factory=dict)

    def __init__(self, distances=None):
        self.tree = Tree(0, parent=None, terminal=False)
        self.nodes = [None]
        self.templates_per_node = {0: set()}
        self.node_to_tree = {0: self.tree}

        self.templates = []
        self.examples = []

        self.distances = distances if distances is not None else []
        self.overlap_map = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild parents
        for _, tree in self.node_to_tree.items():
            for branch in tree.branches.values():
                branch.parent = tree

    def add_template(self, elements, example=None, fixed=False):
        if isinstance(elements, Template):
            elements = elements.elements

        added_new_node = False

        current_node = self.tree
        for element_id, element in enumerate(elements):
            next_node_id = current_node.get_element(self.nodes, element)
            if next_node_id is None:
                added_new_node = True
                next_node_id = len(self.nodes)
                current_node[next_node_id] = Tree(
                    next_node_id, parent=current_node, terminal=False
                )
                element.id = next_node_id
                element.fixed = fixed

                # Reset regex
                element.compiled_element_regex = None
                element.compiled_prefix_regex = None
                element.compiled_prefix_regex_terminal = None

                self.nodes.append(element)
                self.templates_per_node[next_node_id] = set()
                self.node_to_tree[next_node_id] = current_node[next_node_id]
            else:
                self.nodes[next_node_id].fixed = (
                    self.nodes[next_node_id].fixed or fixed
                )
                if element_id == len(elements) - 1:
                    # An existing path is being reused
                    added_new_node = True

            current_node = current_node[next_node_id]

        if added_new_node:
            current_node.terminal = True
            new_template_id = len(self.templates)
            current_node.template_id = new_template_id
            self.templates.append(current_node.get_lineage())
            for node in self.templates[-1]:
                self.templates_per_node[node].add(new_template_id)
            self.templates_per_node[0].add(new_template_id)
            self.examples.append([])
            if example:
                self.examples[-1].append(example)

            for distance in self.distances:
                distance.add_template(
                    new_template_id, self.gen_template(new_template_id)
                )

            return new_template_id
        else:
            # Template already exists
            return current_node.template_id

    def remove_template(self, t_id):
        """Remove a template from the tree."""
        elements = self.templates[t_id]

        # Remove terminal flag on last element
        if elements[-1] in self.node_to_tree:
            self.node_to_tree[elements[-1]].terminal = False
            self.node_to_tree[elements[-1]].template_id = None

        # Update templates per element
        for elt in elements[::-1]:
            if elt in self.templates_per_node:
                self.templates_per_node[elt] = set(
                    [t for t in self.templates_per_node[elt] if t != t_id]
                )
                # Remove node if it only was in this template
                if not len(self.templates_per_node[elt]):
                    del self.templates_per_node[elt]
                    parent = self.node_to_tree[elt].parent
                    if parent:
                        del parent[elt]
                    del self.node_to_tree[elt]
                    self.nodes[elt] = None
                continue

            if elt in self.node_to_tree:
                parent = self.node_to_tree[elt].parent
                if parent and elt in parent.branches:
                    del parent.branches[elt]
                    del self.node_to_tree[elt]
                    self.nodes[elt] = None
                    continue

                del self.node_to_tree[elt]
                self.nodes[elt] = None

        self.templates[t_id] = []
        self.examples[t_id] = []

        if t_id in self.overlap_map:
            del self.overlap_map[t_id]

        del_list = []
        for m in self.overlap_map:
            self.overlap_map[m] = [v for v in self.overlap_map[m] if v != t_id]
            if not self.overlap_map[m]:
                del_list.append(m)

        for m in del_list:
            if m in self.overlap_map:
                del self.overlap_map[m]

        # Remove template from root list of templates
        self.templates_per_node[0].remove(t_id)

    def replace_template(self, target_id, elements, example):
        """
        Replace the target template with a new template
        """
        if isinstance(elements, Template):
            elements = elements.elements

        # Unmark elements in original template
        target = self.templates[target_id]
        for elt in target:
            self.templates_per_node[elt] = set(
                [t for t in self.templates_per_node[elt] if t != target_id]
            )
        self.node_to_tree[target[-1]].terminal = False
        self.node_to_tree[target[-1]].template_id = None

        # Add new template
        added_new_node = False
        current_node = self.tree
        for element in elements:
            next_node_id = current_node.get_element(self.nodes, element)
            if next_node_id is None:
                added_new_node = True
                next_node_id = len(self.nodes)
                current_node[next_node_id] = Tree(
                    next_node_id, parent=current_node, terminal=False
                )
                element.id = next_node_id
                element.fixed = False
                self.nodes.append(element)
                self.templates_per_node[next_node_id] = set()
                self.templates_per_node[next_node_id].add(target_id)
                self.node_to_tree[next_node_id] = current_node[next_node_id]
            else:
                self.nodes[next_node_id].fixed = (
                    self.nodes[next_node_id].fixed or False
                )
                self.templates_per_node[next_node_id].add(target_id)

            current_node = current_node[next_node_id]

        if added_new_node:
            current_node.terminal = True
            current_node.template_id = target_id
            self.templates[target_id] = current_node.get_lineage()
            self.templates_per_node[0].add(target_id)
            self.examples[target_id].append(example)
            for distance in self.distances:
                distance.add_template(target_id, self.gen_template(target_id))

        # Remove old nodes
        for elt in target[::-1]:
            if not self.templates_per_node[elt]:
                del self.templates_per_node[elt]
                parent = self.node_to_tree[elt].parent
                del parent[elt]
                del self.node_to_tree[elt]
                self.nodes[elt] = None

        return target_id

    def splice_elements(self, target, new_elements):
        """
        Replace the target element in all templates with new_elements
        """
        new_ids = []
        tree = self.node_to_tree[target].parent
        if target not in tree.branches:
            raise ValueError(f"Target element {target} not found in tree")
        original_tree = tree[target]

        for elt_idx, elt in enumerate(new_elements):
            if elt_idx:
                elt.id = len(self.nodes)
                self.nodes.append(elt)
            else:
                elt.id = target
                self.nodes[target] = elt

            new_ids.append(elt.id)

            if elt_idx < len(new_elements) - 1:
                new_tree = Tree(elt.id, tree, terminal=False)
                tree[elt.id] = new_tree
                tree = new_tree
                self.node_to_tree[elt.id] = tree
            else:
                original_tree.node = elt.id
                original_tree.parent = tree
                tree[elt.id] = original_tree
                self.node_to_tree[elt.id] = original_tree

        # Update templates
        for t_id, trail in enumerate(self.templates):
            if target in trail:
                index = trail.index(target)
                self.templates[t_id][index : index + 1] = new_ids

        # Update templates per node
        for elt in new_ids:
            if elt not in self.templates_per_node:
                self.templates_per_node[elt] = deepcopy(
                    self.templates_per_node[target]
                )

        return new_elements

    def get_deepest_node_elements(self, elements: List[Element]):
        if isinstance(elements, Template):
            elements = elements.elements

        current_node = self.tree
        for element in elements:
            next_node_id = current_node.get_element(self.nodes, element)
            if next_node_id is None:
                return current_node
            else:
                current_node = current_node[next_node_id]

        return current_node

    def get_ordered_templates(self, node: int = 0, compute_percentage=False):
        if node > 0:
            lineage = self.node_to_tree[node].get_lineage()[::-1] + [0]
            seen = set()
            rslt = []
            for idx, node in enumerate(lineage):
                for t_idx in self.templates_per_node[node]:
                    if t_idx not in seen:
                        # Compute the percentage of length shared in terms of number of elements
                        percent = len(
                            [e for e in self.templates[t_idx] if e in lineage]
                        ) / len(self.templates[t_idx])
                        if compute_percentage:
                            rslt.append(
                                (
                                    idx + 1 - percent,
                                    t_idx,
                                )
                            )
                        else:
                            rslt.append(
                                (
                                    idx,
                                    t_idx,
                                )
                            )
                        seen.add(t_idx)
        else:
            rslt = [
                (
                    0,
                    t_idx,
                )
                for t_idx in range(len(self.templates))
            ]
        return rslt

    def degree_of_separation(self, node_a, node_b, norm=False):
        """Return the degree of separation between node_a and node_b.
        The degree of separation is the height of the largest common prefix.
        If both are on the same template, return the depth of the first"""

        lineage_a = self.node_to_tree[node_a].get_lineage()
        lineage_b = self.node_to_tree[node_b].get_lineage()

        if lineage_a[0] != lineage_b[0]:
            return 0

        if node_a in lineage_b or node_b in lineage_a:
            return len(lineage_a)

        i = 0
        while lineage_b[i] == lineage_a[i]:
            i += 1

        if not norm:
            return i
        else:
            return i / min(len(lineage_a), len(lineage_b))

    def gen_template(self, t_id):
        trail = self.templates[t_id]
        elements = [self.nodes[n] for n in trail]
        return Template(
            elements,
            self.examples[t_id][0] if len(self.examples[t_id]) else "",
            id=t_id,
        )

    def gen_matches_from_dict(self, last_node, match_dict):
        """Generate a Template from a match dictionary."""
        elements = []
        for node_id in self.node_to_tree[last_node].get_lineage():
            if node_id:
                elements.append(
                    deepcopy(self.nodes[node_id])
                )  # Deep copy to avoid modifying original

        for elt in elements:
            if elt.is_variable():
                # If the element is a variable, we need to assign matches
                if str(elt.id) in match_dict:
                    elt.value = match_dict[str(elt.id)]
                else:
                    breakpoint()
                    raise ValueError(
                        f"Match dictionary does not contain matches for node {elt.id}"
                    )

        return Match(elements)

    def match(self, entry):
        entry = entry.strip()
        terminal_nodes = {}
        partial_nodes = {}

        # Our queue of states to process. We begin with one state for each branch off the root node.
        queue = deque()
        for branch_id in self.tree.branches:
            initial_state = ExplorationState(node=branch_id)
            queue.append(initial_state)

        while queue:
            expl_state = queue.popleft()

            # Load the current state
            node, full_trail = (
                expl_state.node,
                expl_state.full_trail,
            )
            local_tree, element = self.node_to_tree[node], self.nodes[node]

            # Track if any progress can be made from this node
            dead_end = True

            # Check if this matches the entry
            if node:
                # If this node is the end of a template, check for full matches
                if local_tree.terminal:
                    terminal_match_result = element.match_tree(
                        entry, [self.nodes[p] for p in full_trail], True, True
                    )
                    if (
                        terminal_match_result["result"]
                        == ElementMatchType.MATCH
                    ):
                        # We have a full match, add to terminal nodes
                        dead_end = False
                        if node not in terminal_nodes:
                            terminal_nodes[node] = TerminalNode(
                                suffix="",
                                trail=full_trail + [node],
                                template_id=local_tree.template_id,
                                matches=terminal_match_result["match_dict"],
                            )

                # If this node has children, we still should check if the current prefix matches: if so, we can explore the children
                if local_tree.branches:
                    match_result = element.match_tree(
                        entry, [self.nodes[p] for p in full_trail]
                    )
                    if match_result["result"] != ElementMatchType.NOMATCH:
                        # We have a partial match, we can explore further
                        dead_end = False
                        expl_state.suffix_index = max(
                            match_result["suffix_index"],
                            expl_state.suffix_index,
                        )
                        expl_state.matches.update(match_result["match_dict"])
                        expl_state.full_trail += [node]
                        if match_result["result"] == ElementMatchType.MATCH:
                            expl_state.matched_trail = expl_state.full_trail[:]

                        for branch_id in local_tree.branches:
                            # Create a new exploration state for each branch
                            new_state = ExplorationState(
                                node=branch_id,
                                matched_trail=expl_state.matched_trail[:],
                                full_trail=expl_state.full_trail[:],
                                suffix_index=expl_state.suffix_index,
                                matches=deepcopy(expl_state.matches),
                            )
                            queue.append(new_state)

                # If no progress can be made from this node, we have reached a dead end.
                # In this case, we save the furthest node that matched in partial_nodes
                if dead_end:
                    last_match_node = (
                        expl_state.matched_trail[-1]
                        if expl_state.matched_trail
                        else 0
                    )
                    if last_match_node and last_match_node not in partial_nodes:
                        # We have reached a dead end, save this as a partial match
                        partial_nodes[last_match_node] = TerminalNode(
                            suffix=entry[expl_state.suffix_index :].strip(),
                            trail=expl_state.matched_trail,
                            template_id=None,
                            matches=expl_state.matches,
                        )
                        continue

        # Now we have processed all states, we can finalize the results
        # First, check if there are any complete matches
        if terminal_nodes:
            # Order candidates by length of the match (number of elements)
            terminal_nodes = list(terminal_nodes.values())
            terminal_nodes = sorted(
                terminal_nodes,
                key=lambda x: len(x.trail),
                reverse=True,
            )
            for terminal_state in terminal_nodes:
                terminal_state.matches = self.gen_matches_from_dict(
                    terminal_state.trail[-1], terminal_state.matches
                )
            return True, terminal_nodes

        # If no complete matches, but we have partial matches, we return the largest partial matches
        # We can induce an order over partial matches, where M_a > M_b if the matched trail of M_a contrains the matched trail of M_b.
        # We only want to keep maximal matches according to this ordering
        if partial_nodes:
            non_terminal_partial_nodes = set()
            for state in partial_nodes.values():
                for node in state.trail[:-1]:
                    non_terminal_partial_nodes.add(node)

            partial_nodes = [
                v
                for k, v in partial_nodes.items()
                if k not in non_terminal_partial_nodes
            ]
            partial_nodes = sorted(
                partial_nodes,
                key=lambda x: len(x.trail),
                reverse=True,
            )
            for partial_state in partial_nodes:
                if partial_state.trail:
                    partial_state.matches = self.gen_matches_from_dict(
                        partial_state.trail[-1], partial_state.matches
                    )
            return False, partial_nodes

        # If we neither have complete matches nor partial matches, we can return an empty result.
        return False, [
            TerminalNode(suffix=entry, trail=[], template_id=None, matches=None)
        ]
