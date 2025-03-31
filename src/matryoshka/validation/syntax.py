import copy
import enum
import functools
import json
import random
import re
import time
import trace
import traceback
from collections import defaultdict
from dataclasses import dataclass

from matryoshka.classes.element import Element, ElementType
from matryoshka.classes.template import Template
from matryoshka.classes.tree import TemplateTree, Tree
from matryoshka.genai_api import api
from matryoshka.genai_api.classes import LLMTask, ModelResponses
from matryoshka.utils.logging import get_logger
from matryoshka.utils.prompts.syntax.validate import (
    gen_prompt,
    graph_to_markdown,
)

from .validate import Validator


class TreeAPI:
    def __init__(self, tree, lines, tracked_template_ids):
        self.tree = tree
        self.lines = lines
        self.tracker = []
        self.tracked_template_ids = tracked_template_ids

    def node_exists(self, node_id):
        return node_id < len(self.tree.nodes) and (
            not node_id or self.tree.nodes[node_id]
        )

    def template_exists(self, template_id):
        return (
            template_id < len(self.tree.templates)
            and self.tree.templates[template_id]
        )

    def missing_lines(self):
        missing = []
        for line_idx, line in enumerate(self.lines):
            if not self.tree.match(line)[0]:
                missing.append(line_idx)
        return missing

    def unused_templates(self):
        unused = []
        for template_id in self.tracked_template_ids:
            if template_id < 0 or template_id >= len(self.tree.templates):
                continue
            raw_template = self.tree.templates[template_id]
            if not raw_template:
                continue
            template = self.tree.gen_template(template_id)
            if all(not template.match(line)[0] for line in self.lines):
                unused.append(template_id)
        return unused

    def post_command(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            name = func.__name__.upper()
            self.tree.reset_regex()
            self.validate()
            missing = self.missing_lines()
            unused = self.unused_templates()
            self.tracker.append(
                (name, f"args: {args}, kwargs: {kwargs}", missing, unused)
            )
            return result

        return wrapper

    @post_command
    def create(self, parent_id, value):
        if not isinstance(parent_id, int):
            raise ValueError("Node ids must be numbers")

        if not self.node_exists(parent_id):
            raise ValueError("Parent node does not exist")

        if not isinstance(value, dict):
            raise ValueError("Value must be a dictionary")

        if not value.get("value", ""):
            raise ValueError(
                "Values cannot be empty. You must not include empty nodes in the tree."
            )

        parent_tree_node = self.tree.node_to_tree[parent_id] or self.tree.tree
        new_node_id = len(self.tree.nodes)
        node = Element(
            entity=(
                ElementType.VARIABLE
                if value.get("is_variable", False)
                else ElementType.CONSTANT
            ),
            value=value.get("value", ""),
            regexp=value.get("regexp", ".*?"),
            trailing_whitespace=value.get("trailing_whitespace", 0),
            id=new_node_id,
            placeholder=value.get("placeholder", ""),
        )
        tree_element = Tree(new_node_id, parent_tree_node)
        self.tree.nodes.append(node)
        self.tree.node_to_tree[new_node_id] = tree_element
        self.tree.templates_per_node[new_node_id] = set()
        parent_tree_node.branches[new_node_id] = tree_element

        return new_node_id

    def prune(self):
        """Remove nodes that are not in any template"""
        all_ids = {
            node_id for template in self.tree.templates for node_id in template
        }
        for key in self.tree.node_to_tree:
            all_ids.add(key)
        for node_id in range(len(self.tree.nodes)):
            all_ids.add(node_id)

        id_to_tree, queue = {}, [self.tree.tree]
        while queue:
            node = queue.pop(0)
            id_to_tree[node.node] = node
            for child_tree in node.branches.values():
                queue.append(child_tree)
        for node_id in id_to_tree:
            all_ids.add(node_id)

        for node_id in all_ids:
            if not self.tree.nodes[node_id]:
                if node_id in self.tree.node_to_tree:
                    del self.tree.node_to_tree[node_id]
                if node_id in self.tree.templates_per_node:
                    del self.tree.templates_per_node[node_id]
                if node_id in id_to_tree:
                    parent = id_to_tree[node_id].parent
                    if parent:
                        del parent.branches[node_id]

    def validate(self, strict=False):
        """Make sure the tree is consistent"""
        errors = []

        # Check that the tree is well defined and doesn't contain loops
        id_to_tree, seen, queue = {}, set(), [self.tree.tree]
        while queue:
            node = queue.pop(0)
            if node.node in seen:
                errors.append(f"Loop detected in tree at node {node.node}")
                continue
            if node.node and not node.parent:
                errors.append(f"Node {node.node} has no parent.")
            id_to_tree[node.node] = node
            if strict and not node.branches and not node.terminal:
                errors.append(f"Node {node.node} is a non terminal leaf.")
            for child_id, child_tree in node.branches.items():
                queue.append(child_tree)
                if child_tree.parent != node:
                    errors.append(
                        f"Tree filiation error between nodes {node.node} and {child_id}"
                    )
            seen.add(node.node)

        # Make sure the tree index is the same as the existing one
        for node_id, tree in id_to_tree.items():
            if not node_id:
                continue
            if node_id not in self.tree.node_to_tree:
                errors.append(
                    f"Missing node (in tree, but not in tree index): {node_id}"
                )
            if node_id >= len(self.tree.nodes) or not self.tree.nodes[node_id]:
                errors.append(
                    f"Missing node (in tree, but not in list of nodes): {node_id}"
                )

        # Make sure the tree contains all nodes
        for node_id, node in enumerate(self.tree.nodes):
            if not node_id or not node:
                continue
            if node_id not in id_to_tree:
                errors.append(
                    f"Missing node (in list of nodes, but not in tree): {node_id}"
                )
            if node_id not in self.tree.node_to_tree:
                errors.append(
                    f"Missing node (in list of nodes, but not in tree index): {node_id}"
                )

        # Make sure the index doesn't contain empty nodes
        for node_id, tree in self.tree.node_to_tree.items():
            if not node_id:
                continue
            if node_id >= len(self.tree.nodes) or not self.tree.nodes[node_id]:
                errors.append(
                    f"Missing node (in tree index, but not in list of nodes): {node_id}"
                )
            if node_id not in id_to_tree:
                errors.append(
                    f"Missing node (in tree index, but not in tree): {node_id}"
                )

        # Check if all templates are well defined
        for template_id, template in enumerate(self.tree.templates):
            if not template:
                continue
            for t_idx, t in enumerate(template):
                if t >= len(self.tree.nodes) or not self.tree.nodes[t]:
                    errors.append(
                        f"Missing node (in template, but not in list of nodes): {t}"
                    )
                if t not in id_to_tree:
                    errors.append(
                        f"Missing node (in template, but not in tree): {t}"
                    )
                else:
                    tree_node = id_to_tree[t]
                    if not t_idx and tree_node.parent and tree_node.parent.node:
                        errors.append(
                            f"Root node in template #{template_id} ({t}) has a parent: {tree_node.parent.node}"
                        )
                    elif t_idx and (
                        not tree_node.parent
                        or tree_node.parent.node != template[t_idx - 1]
                    ):
                        errors.append(
                            f"Parent node in template #{template_id} ({t}) <- {template[t_idx - 1]} is not the previous node in tree: {tree_node.parent.node if tree_node.parent else None}"
                        )
                    if (
                        t_idx < len(template) - 1
                        and template[t_idx + 1] not in tree_node.branches
                    ):
                        errors.append(
                            f"Child node in template #{template_id} ({t}) -> {template[t_idx + 1]} is not a child in the tree."
                        )
                    elif t_idx == len(template) - 1 and not tree_node.terminal:
                        errors.append(
                            f"Last node in template #{template_id} ({t}) is not a terminal node."
                        )
                    elif (
                        t_idx == len(template) - 1
                        and tree_node.template_id != template_id
                    ):
                        errors.append(
                            f"Last node in template #{template_id} ({t}) has a different template ID: {tree_node.template_id}"
                        )
            if t not in self.tree.node_to_tree:
                errors.append(
                    f"Missing node (in template, but not in tree index): {t}"
                )
            if t not in self.tree.templates_per_node:
                errors.append(
                    f"Missing node (in template, but not in templates per node): {t}"
                )
            elif template_id not in self.tree.templates_per_node[t]:
                errors.append(
                    f"Missing template #{template_id} (not in templates per node): {t}"
                )

        # Check templates_per_node is consistent with templates
        for node_id, templates in self.tree.templates_per_node.items():
            if node_id != 0:
                for template_id in templates:
                    if node_id not in self.tree.templates[template_id]:
                        errors.append(
                            f"Template #{template_id} in templates per node of node #{node_id} but the node is not in the template"
                        )
            else:
                for template_id in templates:
                    if not self.tree.templates[template_id]:
                        errors.append(
                            f"Template #{template_id} in templates per node of root node but the template is empty"
                        )

        # Return errors
        if errors:
            print(f"{len(errors)} errors detected in tree:")
            for error_idx, error in enumerate(errors):
                print(f" - {error_idx}:\t" + error)
            breakpoint()
            raise ValueError("Tree is not valid")

        return errors

    @post_command
    def edit(self, node_id, new_value):
        if not isinstance(node_id, int):
            raise ValueError("Node ids must be numbers")

        if not self.node_exists(node_id):
            raise ValueError("Node does not exist")

        if not isinstance(new_value, dict):
            raise ValueError("Value must be a dictionary")

        if "value" in new_value and not new_value["value"]:
            raise ValueError(
                "Value cannot be empty. You must not include empty nodes in the tree."
            )

        node = self.tree.nodes[node_id]

        if node.fixed:
            raise ValueError(f"Node #{node_id} is frozen and cannot be edited.")

        node.value = new_value.get("value", node.value)
        node.regexp = new_value.get("regexp", node.regexp)
        node.trailing_whitespace = new_value.get(
            "trailing_whitespace", node.trailing_whitespace
        )
        node.placeholder = new_value.get("placeholder", node.placeholder)
        if "is_variable" in new_value:
            node.entity = (
                ElementType.VARIABLE
                if new_value.get("is_variable", False)
                else ElementType.CONSTANT
            )
        return

    @post_command
    def delete(self, node_id):
        """Delete a node from the tree by its ID.
        Input:
        * node_id: the ID of the node (int)

        Output:
        * None
        """
        if not node_id:
            return
        if not self.node_exists(node_id):
            return

        node = self.tree.nodes[node_id]
        if node.fixed:
            raise ValueError(
                f"Node #{node_id} is frozen and cannot be deleted."
            )

        # First, get the parent node
        tree_node = self.tree.node_to_tree[node_id]
        parent_node = tree_node.parent

        # Then get list of children
        children_nodes = list(self.tree.node_to_tree[node_id].branches.keys())

        # If this node is terminal and its parent is as well, we can use the template deletion command
        if (
            tree_node.terminal
            and parent_node.terminal
            or tree_node.terminal
            and parent_node.node == 0
        ):
            self.delete_template(tree_node.template_id)
            return

        # Add children to the parents
        if parent_node:
            for child in children_nodes:
                parent_node.branches[child] = tree_node.branches[child]

        # Set parent as terminal node if this node was terminal
        if tree_node.terminal and not parent_node.terminal:
            parent_node.template_id = tree_node.template_id
            parent_node.terminal = True

        # Add parent to the children
        for child in children_nodes:
            self.tree.node_to_tree[child].parent = parent_node

        # Delete edge from parent
        if parent_node:
            del parent_node.branches[node_id]

        # Update templates
        for template_id, template in enumerate(self.tree.templates):
            if node_id in template:
                self.tree.templates[template_id] = [
                    i for i in template if i != node_id
                ]

        # Remove node from tree
        del self.tree.node_to_tree[node_id]
        self.tree.nodes[node_id] = None
        del self.tree.templates_per_node[node_id]

        return

    @post_command
    def move(self, node_id, new_parent_id):
        if not isinstance(node_id, int):
            raise ValueError("Node ids must be numbers")

        if not self.node_exists(node_id):
            raise ValueError("Node does not exist")

        if not isinstance(new_parent_id, int):
            raise ValueError("Parent node ids must be numbers")

        if not self.node_exists(new_parent_id):
            raise ValueError("Parent node does not exist")

        if new_parent_id == node_id:
            raise ValueError(
                "Parent node cannot be the same as the node itself"
            )

        node = self.tree.nodes[node_id]
        if node.fixed:
            raise ValueError(f"Node #{node_id} is frozen and cannot be moved.")

        # First, get the parent node and remove the node from its branches
        parent_node = self.tree.node_to_tree[node_id].parent
        del parent_node.branches[node_id]
        old_prefix = parent_node.get_lineage()

        # Then, get the new parent node
        new_parent_node = self.tree.node_to_tree[new_parent_id]
        new_prefix = new_parent_node.get_lineage()
        self.tree.node_to_tree[node_id].parent = new_parent_node
        if new_parent_node:
            new_parent_node.branches[node_id] = self.tree.node_to_tree[node_id]

        # Update templates
        for template_id, template in enumerate(self.tree.templates):
            if node_id in template:
                node_index = template.index(node_id)
                lineage = self.tree.node_to_tree[node_id].get_lineage()[:-1]
                self.tree.templates[template_id] = (
                    lineage + template[node_index:]
                )

        # Update templates_per_node
        templates_to_be_removed = self.tree.templates_per_node[node_id]
        for node in old_prefix:
            for template in templates_to_be_removed:
                if template in self.tree.templates_per_node[node]:
                    self.tree.templates_per_node[node].remove(template)

        for node in new_prefix:
            for template in templates_to_be_removed:
                if template not in self.tree.templates_per_node[node]:
                    self.tree.templates_per_node[node].add(template)

        # Prune nodes with no more templates
        for node in old_prefix:
            if not self.tree.templates_per_node[node]:
                tree_node = self.tree.node_to_tree[node]
                parent_node = tree_node.parent
                for child_id, child_node in tree_node.branches.items():
                    parent_node.branches[child_id] = child_node
                    child_node.parent = parent_node

                del parent_node.branches[node]
                del self.tree.templates_per_node[node]
                del self.tree.node_to_tree[node]
                self.tree.nodes[node] = None

        return

    @post_command
    def replace(self, node_id, values):
        if not isinstance(node_id, int):
            raise ValueError("Node ids must be numbers")

        if not self.node_exists(node_id):
            raise ValueError("Node does not exist")

        if not isinstance(values, list):
            raise ValueError("Values must be a list")

        for value in values:
            if not isinstance(value, dict):
                raise ValueError("Value must be a dictionary")

        node = self.tree.nodes[node_id]
        if node.fixed:
            raise ValueError(
                f"Node #{node_id} is frozen and cannot be replaced."
            )

        # Remove empty nodes from values
        values = [value for value in values if value.get("value", "")]
        if not values:
            return []

        self.edit(node_id, values[0])

        if len(values) == 1:
            return [node_id]

        current_children_ids = list(
            self.tree.node_to_tree[node_id].branches.keys()
        )
        new_node_ids = [node_id]
        for value in values[1:]:
            new_node_ids.append(self.create(new_node_ids[-1], value))

        for child_id in current_children_ids:
            self.move(child_id, new_node_ids[-1])

        # If the node originally at node_id is terminal, we need to unmark it, mark the new last node, and update the template
        if self.tree.node_to_tree[node_id].terminal:
            template_id = self.tree.node_to_tree[node_id].template_id
            self.tree.node_to_tree[node_id].terminal = False
            self.tree.node_to_tree[new_node_ids[-1]].terminal = True
            self.tree.node_to_tree[new_node_ids[-1]].template_id = template_id
            self.tree.templates[template_id] = self.tree.node_to_tree[
                new_node_ids[-1]
            ].get_lineage()
            for node in self.tree.node_to_tree[new_node_ids[-1]].get_lineage():
                self.tree.templates_per_node[node].add(template_id)

        return new_node_ids

    @post_command
    def add_template(self, node_id):
        if not isinstance(node_id, int):
            raise ValueError("Node ids must be numbers")

        if not self.node_exists(node_id):
            raise ValueError("Node does not exist")

        if self.tree.node_to_tree[node_id].terminal:
            raise ValueError("Node is already the end of a template")

        template_id = len(self.tree.templates)
        self.tree.node_to_tree[node_id].terminal = True
        self.tree.node_to_tree[node_id].template_id = template_id

        new_template = self.tree.node_to_tree[node_id].get_lineage()
        self.tree.templates.append(new_template)

        for id in new_template:
            self.tree.templates_per_node[id].add(template_id)

        self.tree.examples.append([])

        return template_id

    @post_command
    def delete_template(self, template_id):
        if not isinstance(template_id, int):
            raise ValueError("Template ids must be numbers")

        if (
            template_id < 0
            or template_id >= len(self.tree.templates)
            or not self.tree.templates[template_id]
        ):
            raise ValueError("Template does not exist")

        return self.tree.remove_template(template_id)


class SyntaxValidator(Validator):
    """Validator for template generation"""

    def __init__(
        self,
        caller,
        parser=None,
        tree=None,
        model="gemini-2.5-pro",
        save_path="./saved_queries/",
        values=None,
        lines_per_template=5,
        entries_per_template=None,
    ):
        if not tree and not parser:
            raise ValueError("Either tree or parser must be provided.")
        super().__init__(
            "syntax_validator",
            caller,
            parser=parser,
            tree=tree,
            model=model,
            save_path=save_path,
            values=values,
            entries_per_template=entries_per_template,
        )
        self.lines_per_template = lines_per_template

    def _extract_answer(self, response, *, schema=None):
        # Identify the code block
        code_block = re.compile(r"```python(.*?)```", re.DOTALL)
        match = code_block.search(response)
        if not match:
            raise ValueError("No code block found in response")
        return match.group(1).strip()

    def _explain_changes(self, old_tree, new_tree, template_ids):
        return_str = ""
        for template_id, template in enumerate(new_tree.templates):
            if not template and (
                template_id >= len(old_tree.templates)
                or not old_tree.templates[template_id]
            ):
                continue
            if (
                not template
                and template_id < len(old_tree.templates)
                and old_tree.templates[template_id]
            ):
                return_str += f"Delete #{template_id}\n"
                continue
            if template and template_id >= len(old_tree.templates):
                return_str += f"Add #{template_id}\n\t{new_tree.gen_template(template_id)}\n"
                continue
            old_template = old_tree.gen_template(template_id)
            new_template = new_tree.gen_template(template_id)
            old_template.generate_regex()
            new_template.generate_regex()
            if old_template.regex != new_template.regex:
                return_str += f"Edit #{template_id}\n\t{old_tree.gen_template(template_id)}\n\t=>\n\t{new_tree.gen_template(template_id)}\n"
                continue

        # Write trees
        old_tree_md = graph_to_markdown(
            self.tree.create_networkx_graph(template_ids)
        )
        new_tree_md = graph_to_markdown(
            new_tree.create_networkx_graph(template_ids)
        )
        return_str += f"#####\nOld tree\n######\n\n\n{old_tree_md}\n\n\n"
        return_str += f"#####\nNew tree\n######\n\n\n{new_tree_md}"

        print(return_str)
        return return_str

    def _parse_answer(
        self, code_block, original_template_ids, lines, force=False
    ):
        editor = TreeAPI(
            copy.deepcopy(self.tree),
            lines,
            tracked_template_ids=original_template_ids,
        )

        # Define the API functions
        api_function_map = {
            "CREATE": editor.create,
            "DELETE": editor.delete,
            "EDIT": editor.edit,
            "ADD_TEMPLATE": editor.add_template,
            "DELETE_TEMPLATE": editor.delete_template,
            "MOVE": editor.move,
            "REPLACE": editor.replace,
        }

        # Run code on tree copy
        try:
            exec(code_block, api_function_map)
        except Exception as e:
            error = traceback.format_exc()
            raise Exception(f"An error occurred during execution: {error}")

        # Make sure the tree is valid
        errors = editor.validate()
        if errors:
            raise ValueError(
                f"The following errors occurred during validation: {errors}"
            )

        # Make sure we still parse all the lines
        editor.tree.reset_regex()
        missing_lines = []
        for line in lines:
            if not editor.tree.match(line)[0]:
                missing_lines.append(line)

        if missing_lines:
            root_cause_index = len(editor.tracker) - 1
            while root_cause_index > 0:
                root_cause_index -= 1
                if not editor.tracker[root_cause_index][2]:
                    root_cause_index += 1
                    break
            root_cause_index += 1
            root_cause_command = editor.tracker[root_cause_index - 1][0].upper()
            root_cause_args = editor.tracker[root_cause_index - 1][1]
            raise ValueError(
                f"The following lines are no longer matched:\n{json.dumps(missing_lines, indent=2)}\n"
                f"It seems like the first change that broke the parsing tree was API call #{root_cause_index}: {root_cause_command}, with {root_cause_args}. "
                f"Please make sure this change is valid by comparing the tree before and after this change to the log lines. Explain your mistake, and propose a new python code."
            )

        # Make sure every template in the tree matches at least one line
        unused_templates = editor.unused_templates()
        if unused_templates and not force:
            root_cause_index = len(editor.tracker) - 1
            while root_cause_index > 0:
                root_cause_index -= 1
                if not editor.tracker[root_cause_index][3]:
                    root_cause_index += 1
                    break
            root_cause_index += 1
            root_cause_command = editor.tracker[root_cause_index - 1][0].upper()
            root_cause_args = editor.tracker[root_cause_index - 1][1]
            raise ValueError(
                f"The following templates do not match any lines:\n{json.dumps(unused_templates)}\n"
                f"If this is intended and the templates are no longer needed, please delete them. It seems like the first change that broke the parsing tree was API call #{root_cause_index}: {root_cause_command}, with {root_cause_args}. "
                f"Please make sure this change is valid by comparing the tree before and after this change to the log lines. Explain your mistake, and propose a new python code."
            )

        # Simplify tree
        changes = editor.tree.simplify_tree()
        print(f"Simplified tree. Changes: {changes}")
        errors = editor.validate()
        if errors:
            breakpoint()

        return editor.tree

    def _apply_changes(self, new_tree, original_template_ids):
        self.tree = new_tree

    def run(self, template_ids, json_tree=False, **kwargs):
        kwargs = self._prepare_kwargs(**kwargs) or {}
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Filter to only keep templates with matches
        template_ids = [
            template_id
            for template_id in template_ids
            if self.entries_per_template.get(template_id, [])
        ]

        # Build prompt
        entries = []
        for template_id in template_ids:
            all_matches = self.entries_per_template.get(template_id, [])
            selected = list(
                random.sample(
                    all_matches, min(self.lines_per_template, len(all_matches))
                )
            )
            selected.append(
                self.tree.gen_template(template_id).print_example_from_values()
            )
            entries.extend(list(set(selected)))

        entries = list(set(entries))

        filtered_entries = []

        # Get list of matched templates for each line
        matched_templates_per_line = []
        for line in entries:
            match, candidates = self.tree.match(line)
            if not match:
                continue
            else:
                filtered_entries.append(line)
                matched_template_ids = [
                    candidate.template_id
                    for candidate in candidates
                    if candidate.template_id in template_ids
                ]
                matched_templates_per_line.append(sorted(matched_template_ids))

        entries = filtered_entries
        # Associate each entry with its matched templates
        paired_entries = []
        for entry, matched_templates in zip(
            entries, matched_templates_per_line
        ):
            if not matched_templates:
                continue
            paired_entries.append((entry, matched_templates))

        # Create map from template to matched line number
        template_to_lines = defaultdict(list)
        for entry_idx, (_, matched_templates) in enumerate(paired_entries):
            for template_id in matched_templates:
                template_to_lines[template_id].append(entry_idx)

        # Generate networkx graph
        tree = self.tree.create_networkx_graph(
            template_ids, template_to_lines=template_to_lines
        )

        # Format entries
        formatted_entries = []
        for entry_idx, (entry, matched_templates) in enumerate(paired_entries):
            formatted_matches = ", ".join(map(str, matched_templates))
            formatted_entries.append(
                f"#{entry_idx} (matched template IDs: {formatted_matches}): {entry}"
            )

        history, system = gen_prompt(
            formatted_entries, tree, json_tree=json_tree
        )
        task = LLMTask(
            system_prompt=system,
            history=history,
            thinking_budget=4096,
            timeout=1600,
            **kwargs,
        )

        get_logger().debug(
            "%s - Running syntax validation.",
            time.strftime("%H:%M:%S", time.localtime()),
        )

        raw_response = self.caller(task)
        if raw_response.failed:
            raise ValueError("Generation failed")
        response = raw_response.candidates[0]
        new_tree, response = self._self_correct(
            response, template_ids, task, entries
        )
        self._write_llm_call(
            task,
            response,
            (
                self._explain_changes(self.tree, new_tree, template_ids)
                if new_tree
                else ""
            ),
        )
        if not new_tree:
            raise ValueError("No tree returned from self-correct")
        return new_tree
