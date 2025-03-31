from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx

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

    def add_template(self, elements, example=None, fixed=False, debug=False):
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
                if not self.gen_template(new_template_id).match(example)[0]:
                    breakpoint()
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

        remove_set = set()

        # Update templates per element
        for elt in elements[::-1]:
            is_terminal = self.node_to_tree[elt].terminal
            is_empty = not self.node_to_tree[elt].branches
            if not is_terminal and is_empty:
                remove_set.add(elt)
                if elt in self.templates_per_node:
                    del self.templates_per_node[elt]
                parent = self.node_to_tree[elt].parent
                if parent and elt in parent.branches:
                    del parent[elt]
                del self.node_to_tree[elt]
                self.nodes[elt] = None
            else:
                self.templates_per_node[elt].remove(t_id)

        self.templates[t_id] = []
        self.examples[t_id] = []

        remove_set = set()
        for m in self.overlap_map.get(t_id, []):
            remove_set.add(m)

        for m in remove_set:
            if m != t_id and m in self.overlap_map:
                self.overlap_map[m].remove(t_id)

        if t_id in self.overlap_map:
            del self.overlap_map[t_id]

        if t_id in self.templates_per_node[0]:
            self.templates_per_node[0].remove(t_id)

        return list(remove_set)

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

    def simplify_tree(self):
        """
        Merge identical sibling nodes (same Element) breadth-first from the root.
        For any group of identical direct children under a parent, keep the child
        with the lowest node id and merge the others into it. While merging:
        - All templates referencing the removed node id are updated to the kept id.
        - templates_per_node is updated (union of memberships).
        - node_to_tree mappings, parent pointers, and branches are rewired.
        - terminal/template_id flags are preserved (kept if present; otherwise adopted).
        - SPECIAL-CASE: if both nodes are leaf terminals, delete the template of the
        higher-index node instead of merging templates (self.remove_template).
        The tree is modified in place.
        """

        def _element_key(node_id: int):
            """Create a hashable signature for an Element to detect identical siblings."""
            e = self.nodes[node_id]
            if e is None:
                return 0
            return e.__hash__()

        def _replace_in_templates(old_id: int, new_id: int):
            """Replace occurrences of old_id with new_id across all templates and move memberships."""
            if old_id == new_id:
                return
            tids = list(self.templates_per_node.get(old_id, []))
            if new_id not in self.templates_per_node:
                self.templates_per_node[new_id] = set()
            for t_id in tids:
                trail = self.templates[t_id]
                if trail:
                    self.templates[t_id] = [
                        new_id if x == old_id else x for x in trail
                    ]
                self.templates_per_node[new_id].add(t_id)
            if old_id in self.templates_per_node:
                del self.templates_per_node[old_id]

        def _merge_nodes(keep_id: int, drop_id: int):
            """
            Merge subtree rooted at drop_id into subtree rooted at keep_id.
            Both nodes must represent identical elements.
            """
            if keep_id == drop_id or drop_id not in self.node_to_tree:
                return

            keep_tree = self.node_to_tree[keep_id]
            drop_tree = self.node_to_tree[drop_id]

            # SPECIAL-CASE: if both are leaf terminals, delete the template of the higher-index node
            if (
                keep_tree.terminal
                and not keep_tree.branches
                and drop_tree.terminal
                and not drop_tree.branches
            ):
                # Since we always keep the lowest node id, drop_id is the higher-index node
                if drop_tree.template_id is not None:
                    self.remove_template(drop_tree.template_id)
                return  # Nothing else to merge at this pair

            # Union template memberships at this node and update templates to point to keep_id
            if drop_id in self.templates_per_node:
                if keep_id not in self.templates_per_node:
                    self.templates_per_node[keep_id] = set()
                self.templates_per_node[keep_id].update(
                    self.templates_per_node[drop_id]
                )
            _replace_in_templates(drop_id, keep_id)

            # Terminal handling
            if drop_tree.terminal:
                if not keep_tree.terminal:
                    keep_tree.terminal = True
                    keep_tree.template_id = drop_tree.template_id
                # If both are terminal but not both leaves (i.e., one has children),
                # we keep keep_tree.template_id as-is; templates have been remapped.

            # Merge/move children of drop into keep by element equality
            for child_id in list(drop_tree.branches.keys()):
                # child_id might be removed during recursion
                if child_id not in drop_tree.branches:
                    continue

                child_el = self.nodes[child_id]
                existing_id = keep_tree.get_element(self.nodes, child_el)

                if existing_id is None:
                    # Move this child subtree under keep
                    keep_tree[child_id] = drop_tree.branches[child_id]
                    keep_tree.branches[child_id].parent = keep_tree
                    self.node_to_tree[child_id] = keep_tree.branches[child_id]
                    del drop_tree[child_id]
                else:
                    # Merge identical grandchildren subtrees
                    _merge_nodes(existing_id, child_id)
                    if child_id in drop_tree.branches:
                        del drop_tree[child_id]

            # Remove drop node from its parent and clean mappings
            parent = drop_tree.parent
            if parent and drop_id in parent.branches:
                del parent[drop_id]
            if drop_id in self.node_to_tree:
                del self.node_to_tree[drop_id]
            if 0 <= drop_id < len(self.nodes):
                self.nodes[drop_id] = None

        # BFS from root, merging identical siblings at each level
        queue = deque([self.tree])
        updates = []
        while queue:
            parent_tree = queue.popleft()

            # Group direct children by element signature
            groups = {}
            for cid in list(parent_tree.branches.keys()):
                # Node may have been removed by a previous merge step
                if cid not in parent_tree.branches:
                    continue
                sig = _element_key(cid)
                groups.setdefault(sig, []).append(cid)

            # For each group with duplicates, keep the smallest id and merge others
            for ids in groups.values():
                if len(ids) <= 1:
                    continue
                updates.append(ids)
                ids.sort()
                keep = ids[0]
                for drop in ids[1:]:
                    if drop in parent_tree.branches:  # still present
                        _merge_nodes(keep, drop)

            # Enqueue the (now deduplicated) children
            for cid in list(parent_tree.branches.keys()):
                queue.append(parent_tree.branches[cid])

        return updates

    def splice_elements(self, target, new_elements):
        """
        Replace the target element in all templates with new_elements
        """
        new_ids = []
        tree = self.node_to_tree[target].parent
        if not tree:
            raise ValueError(f"Target element {target} not found in tree")
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

    def reset_regex(self):
        """Reset all regexes in the tree."""
        for node in self.nodes:
            if node is not None:
                node.compiled_element_regex = None
                node.compiled_prefix_regex = None
                node.compiled_prefix_regex_terminal = None

    def create_networkx_graph(
        self,
        template_subset=None,
        add_whitespace=False,
        field_names=None,
        field_descriptions=None,
        include_placeholders=True,
        template_to_lines=None,
    ):
        raw_nodes, raw_edges = {}, set()
        template_endings = {}
        nodes, edges = {}, []

        if field_names:
            field_name_keys = list(field_names.keys())
            for node_id in field_name_keys:
                field_names[str(node_id)] = field_names[node_id]

        if field_descriptions:
            field_description_keys = list(field_descriptions.keys())
            for node_id in field_description_keys:
                field_descriptions[str(node_id)] = field_descriptions[node_id]

        if not template_subset:
            template_subset = list(range(len(self.templates)))

        for template_id in template_subset:
            if not self.templates[template_id]:
                continue
            for node_id in self.templates[template_id]:
                raw_nodes[str(node_id)] = self.nodes[node_id]
                if self.node_to_tree[node_id].parent is not None:
                    if self.node_to_tree[node_id].parent.node > 0:
                        raw_edges.add(
                            (
                                str(self.node_to_tree[node_id].parent.node),
                                str(node_id),
                            )
                        )
            template_endings[str(self.templates[template_id][-1])] = template_id

        raw_edges = list(raw_edges)

        # Add nodes for whitespaces
        whitespace_count = 0
        for e1, e2 in raw_edges:
            if raw_nodes[e1].trailing_whitespace and add_whitespace:
                new_id = f"w{whitespace_count}"
                raw_nodes[new_id] = {
                    "value": " ",
                    "is_variable": False,
                    "id": new_id,
                }
                edges.append((e1, new_id))
                edges.append((new_id, e2))
                whitespace_count += 1
            else:
                edges.append((e1, e2))

        # Convert nodes to json
        for node_id, node in raw_nodes.items():
            node_dict = {}
            node_dict["value"] = node.value
            node_dict["is_variable"] = node.is_variable()
            node_dict["id"] = int(node_id)
            if node_dict["is_variable"]:
                node_dict["regexp"] = node.regexp
            if node.placeholder and include_placeholders:
                node_dict["placeholder"] = node.placeholder
            if node_id in template_endings:
                template_id = template_endings[node_id]
                node_dict["end_of_template_id"] = template_id
                if template_to_lines and template_id in template_to_lines:
                    node_dict["matched_lines"] = template_to_lines[template_id]
            if not add_whitespace:
                node_dict["trailing_whitespace"] = node.trailing_whitespace > 0
            if field_names and node_id in field_names:
                node_dict["field_name"] = field_names[node_id]
            if field_descriptions and node_id in field_descriptions:
                node_dict["field_description"] = field_descriptions[node_id]
            if node.fixed:
                node_dict["frozen"] = True
            nodes[node_id] = node_dict

        # 1. Create a directed graph from the edge list
        graph = nx.DiGraph(edges)

        # 2. Add node attributes from the node attribute dictionary
        nx.set_node_attributes(graph, nodes)

        # 3. Return the populated graph object
        return graph

    def update_tree(self, new_tree, dry_run=False):
        """Update tree using other tree, and return the list of changed templates."""
        # Get list of changed templates
        changed_templates = []
        for template_id, template in enumerate(new_tree.templates):
            # Deleted
            if (
                template_id < len(self.templates)
                and self.templates[template_id]
                and not template
            ):
                changed_templates.append(template_id)
            # Added
            elif template_id >= len(self.templates):
                changed_templates.append(template_id)
            # Edited
            else:
                old_template = self.gen_template(template_id)
                new_template = new_tree.gen_template(template_id)
                old_template.generate_regex()
                new_template.generate_regex()
                if old_template.regex != new_template.regex:
                    changed_templates.append(template_id)

        if dry_run:
            return changed_templates

        self.tree = new_tree.tree
        self.nodes = new_tree.nodes
        self.templates = new_tree.templates
        self.examples = new_tree.examples
        self.templates_per_node = new_tree.templates_per_node
        self.node_to_tree = new_tree.node_to_tree
        self.overlap_map = new_tree.overlap_map

        for template_id in changed_templates:
            remove_list = []
            for node_id in self.overlap_map.get(template_id, []):
                remove_list.append(node_id)
            for n_id in remove_list:
                if n_id in self.overlap_map:
                    self.overlap_map[n_id].remove(template_id)
            self.overlap_map[template_id] = set()
            self.examples[template_id] = []

        self.reset_regex()

        return changed_templates

    @staticmethod
    def load_from_json(json_tree: List[dict]) -> "TemplateTree":
        tree = TemplateTree()
        for template in json_tree:
            template_obj = Template.load_from_json(template)
            tree.add_template(template_obj)
        return tree
