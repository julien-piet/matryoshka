import re

from ...utils.logging import get_logger
from .heuristics import Heuristic


class Greedy(Heuristic):

    def __init__(self, tree, debug_folder=None) -> None:
        super().__init__(
            tree, "Greedy", None, None, None, None, None, save_path=debug_folder
        )

    def run(self, t_id, line, **kwargs):
        _, entries = self.tree.match(line)
        for t in entries:
            if t.template_id == t_id:
                return True, t

        # Change all new regexes to non greedy versions
        get_logger().warning(
            "Generated template does not match. Changing regular expressions to non-greedy versions to avoid overcapture"
        )
        elements = self.tree.templates[t_id]
        for position, e_idx in enumerate(elements):
            self.tree.nodes[e_idx].compiled_element_regex = None
            self.tree.nodes[e_idx].compiled_prefix_regex = None
            self.tree.nodes[e_idx].compiled_prefix_regex_terminal = None
            if (
                self.tree.nodes[e_idx].is_variable()
                and len(self.tree.templates_per_node[e_idx]) == 1
                and position != len(elements) - 1
                and any(
                    self.tree.nodes[e].is_variable()
                    for e in elements[position + 1 :]
                )
            ):
                self.tree.nodes[e_idx].regexp = re.sub(
                    r"(?<!\\)\*(?!\?)",
                    "*?",
                    self.tree.nodes[e_idx].regexp,
                )
                self.tree.nodes[e_idx].regexp = re.sub(
                    r"(?<!\\)\+(?!\?)",
                    "+?",
                    self.tree.nodes[e_idx].regexp,
                )

        # Try again
        _, entries = self.tree.match(line)
        for t in entries:
            if t.template_id == t_id:
                return True, t

        get_logger().error(
            "Generated template does not match even after adjustment."
        )
        return False, None
