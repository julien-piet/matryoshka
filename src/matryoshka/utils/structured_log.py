import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy

# ----------------- DateParser ----------------------------------------
from datetime import datetime

import dill
import Levenshtein
from dateutil import parser
from tqdm import tqdm

from ..classes import (
    Element,
    ElementType,
    Mapping,
    Parser,
    Template,
    Tree,
    VariableSemantics,
)
from .OCSF import OCSFSchemaClient


def parse_int(value):
    try:
        return int(value)
    except ValueError as e:
        values = re.split(r"\s+", value.strip())
        for val in values:
            fixed_val = re.sub("[^0-9]", "", val)
            if fixed_val:
                return int(fixed_val)
        raise e


def parse_float(value):
    try:
        return float(value)
    except ValueError as e:
        values = re.split(r"\s+", value.strip())
        for val in values:
            fixed_val = re.sub("[^0-9.]", "", val)
            if fixed_val:
                return float(fixed_val)
        raise e


def parse_datetime_to_timestamp(date_string):
    """
    Infer the format of a datetime string and convert it to a Unix timestamp.
    Missing date/time components will be filled with current date values.

    Args:
        date_string (str): A string representing a date and/or time

    Returns:
        int: Unix timestamp (seconds since epoch)

    Raises:
        ValueError: If the date_string cannot be parsed
    """
    # For ISO 8601 with timezone info, use dateutil parser first
    try:
        dt = parser.parse(date_string)
        return int(dt.timestamp())
    except (ValueError, parser.ParserError):
        pass

    # Get current date components to fill in missing parts
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day

    # List of common datetime formats to try
    formats = [
        # ISO formats
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        # Date only formats
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%Y %B %d",
        "%d %b %Y",
        "%b %d, %Y",
        # Date with time formats
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%m-%d-%Y %H:%M",
        # Formats with potentially missing year (like "Mar 9 23:48:56")
        "%b %d %H:%M:%S",
        "%B %d %H:%M:%S",
        "%b  %d %H:%M:%S",
        "%B  %d %H:%M:%S",  # Note the double space
        # Time only formats
        "%H:%M:%S",
        "%H:%M",
        # Other common formats
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    ]

    # Handle Unix timestamps (already in seconds)
    if date_string.isdigit():
        try:
            ts = int(date_string)
            # Verify this looks like a valid timestamp
            # (between 1970 and 2100 approximately)
            if 0 < ts < 4102444800:  # Jan 1, 2100
                return ts
        except:
            pass

    # Remove timezone indicators before trying specific formats
    # This is a fallback in case dateutil parser failed
    timezone_pattern = r"([+-]\d{2}:?\d{2}|\s[A-Z]{3,4})$"
    clean_date_string = re.sub(timezone_pattern, "", date_string)

    # Try each format until one works
    for fmt in formats:
        try:
            dt = datetime.strptime(clean_date_string, fmt)

            # Check for missing year
            if "%Y" not in fmt:
                dt = dt.replace(year=current_year)

            # Check for missing month or day when we just have time
            if "%m" not in fmt and "%b" not in fmt and "%B" not in fmt:
                dt = dt.replace(month=current_month)
            if "%d" not in fmt:
                dt = dt.replace(day=current_day)

            return int(time.mktime(dt.timetuple()))
        except ValueError:
            continue

    raise ValueError(f"Could not parse datetime from: {date_string}")


# ----------------- TreeEditor ----------------------------------------


class TreeEditor:

    def __init__(
        self,
        parser,
        lines,
        output=None,
        lines_per_template=-1,
        client=None,
        caller=None,
        query_file=None,
        run_parse=True,
        cache_dir="~/.cache/",
        save_manager=None,
        **kwargs,
    ):
        self.tree = parser.tree
        self.lines = lines
        self.line_id_per_template = defaultdict(set)
        self.line_id_per_node = defaultdict(set)
        self.matches_per_line_id = defaultdict(list)
        self.nodes_per_line_id = defaultdict(list)

        self.values_per_node_id = defaultdict(Counter)
        self.var_mapping = parser.var_mapping if parser.var_mapping else {}
        self.event_types = parser.event_types if parser.event_types else {}
        self.schema_mapping = (
            parser.schema_mapping if parser.schema_mapping else {}
        )

        self.attribute_to_node_id = defaultdict(set)
        self.node_id_to_match_per_line = defaultdict(lambda: defaultdict(str))
        self.all_field_names = {}
        self.static_templates = defaultdict(str)
        for template_id, template in enumerate(self.tree.templates):
            if template:
                self.static_templates[template_id] = self.tree.gen_template(
                    template_id
                ).convert_to_wildcard_template()

        self.caller = caller
        self.client = (
            client
            if client is not None
            else OCSFSchemaClient(
                caller=self.caller, saved_path=os.path.join(cache_dir, "OCSF")
            )
        )

        self.valid_types = sorted(
            list(self.client.get_basic_types().keys())
            + [
                "composite_t",
                "none_t",
            ]
        )

        self.valid_events = sorted(list(self.client.get_classes().keys()))

        self.output = output

        if isinstance(query_file, list):
            self.query_files = query_file
        elif isinstance(query_file, str):
            self.query_files = [query_file]
        elif self.output:
            self.query_files = [os.path.join(self.output, "queries.json")]
        else:
            self.query_files = None
        self.save_manager = save_manager
        self._validate_tree()
        if run_parse:
            self._parse(keep_lines_per_template=lines_per_template)
            self._validate_tree()

    def tree_stats(self):
        """Get stats about the tree"""
        variable_nodes = [
            n_id
            for n_id, n in enumerate(self.tree.nodes)
            if n and n.is_variable()
        ]
        created_field_names = {
            self.var_mapping[n_id].created_attribute
            for n_id in self.var_mapping
            if self.var_mapping[n_id].created_attribute
        }
        ocsf_mappings = set()
        for mapping in self.var_mapping.values():
            if mapping.mapping:
                continue
            for field in mapping.mapping.field_list:
                ocsf_mappings.add(field)

        stats = {
            "num_nodes": len([n for n in self.tree.nodes if n]),
            "num_variable_nodes": len(variable_nodes),
            "num_templates": len([t for t in self.tree.templates if t]),
            "num_fields": len(created_field_names),
            "num_OCSF_mappings": len(ocsf_mappings),
        }
        return stats

    def _parse(self, keep_lines_per_template=-1):
        self.line_id_per_template = defaultdict(set)
        self.line_id_per_node = defaultdict(set)
        self.matches_per_line_id = defaultdict(list)
        self.nodes_per_line_id = defaultdict(list)
        self.values_per_node_id = defaultdict(Counter)
        self.attribute_to_node_id = defaultdict(set)
        self.node_id_to_match_per_line = defaultdict(lambda: defaultdict(str))
        self.all_field_names = {}

        bar = tqdm(
            total=len(self.lines),
            desc="Parsing log file",
            unit="lines",
        )
        for line_id, line in enumerate(self.lines):
            match, candidates = self.tree.match(line)
            matched_templates = (
                sorted([c.template_id for c in candidates]) if match else []
            )
            self.matches_per_line_id[line_id] = matched_templates
            for template in matched_templates:
                self.line_id_per_template[template].add(line_id)

            nodes = set()
            for candidate in candidates:
                for node_id in candidate.trail:
                    nodes.add(node_id)

            self.nodes_per_line_id[line_id] = sorted(list(nodes))
            for node in nodes:
                self.line_id_per_node[node].add(line_id)

            for candidate in candidates:
                if not candidate.matches:
                    continue
                for match in candidate.matches.elements:
                    created_attr = (
                        self.var_mapping[match.id].created_attribute
                        if self.var_mapping and match.id in self.var_mapping
                        else None
                    )
                    if created_attr and (
                        created_attr == "SYNTAX"
                        or created_attr.endswith("_KEY")
                    ):
                        created_attr = None
                    if match.entity == ElementType.VARIABLE or created_attr:
                        self.values_per_node_id[match.id].update(
                            [match.value.strip()]
                        )

                        attributes = []
                        if created_attr:
                            attributes.append(created_attr)
                            if created_attr not in self.all_field_names:
                                self.all_field_names[created_attr] = (
                                    self.var_mapping[match.id].field_description
                                )

                        ocsf_mappings = (
                            self.var_mapping[match.id].mapping
                            if self.var_mapping
                            and match.id in self.var_mapping
                            and "mapping" in self.var_mapping[match.id].__dict__
                            else None
                        )
                        if ocsf_mappings:
                            for attr in ocsf_mappings.field_list:
                                if attr and attr in self.client.attributes:
                                    attributes.append(attr)
                                    if attr not in self.all_field_names:
                                        self.all_field_names[attr] = (
                                            self.client.attributes[
                                                attr
                                            ].global_description
                                        )

                        for attr in attributes:
                            self.attribute_to_node_id[attr].add(match.id)

                        self.node_id_to_match_per_line[match.id][
                            line_id
                        ] = match.value.strip()

            bar.update(1)

        # Update examples for each template
        for template_id, template in enumerate(self.tree.templates):
            examples = [
                self.lines[i]
                for i in list(self.line_id_per_template[template_id])[:5]
            ]
            self.tree.examples[template_id] = examples

        # Only keep up to N lines per template, to speed things up for future parsings. Include non-captured lines
        if keep_lines_per_template > 0:
            kept_lines = [
                line_id
                for line_id, matches in self.matches_per_line_id.items()
                if not matches
            ]
            for template_id, template in enumerate(self.tree.templates):
                matched_lines = list(self.line_id_per_template[template_id])
                if not matched_lines:
                    continue
                min_overlap = min(
                    len(self.matches_per_line_id[l]) for l in matched_lines
                )
                unique_matched_lines = [
                    l
                    for l in matched_lines
                    if len(self.matches_per_line_id[l]) == min_overlap
                ]
                if len(unique_matched_lines):
                    selected_lines = [unique_matched_lines[0]]
                selected_lines.extend(
                    random.sample(
                        matched_lines,
                        k=min(
                            keep_lines_per_template - len(selected_lines),
                            len(matched_lines),
                        ),
                    )
                )
                selected_lines = list(set(selected_lines))
                kept_lines.extend(selected_lines)

            self.lines = [self.lines[i] for i in kept_lines]
            return self._parse(keep_lines_per_template=-1)

    def _validate_tree(self, blocking=True, debug=True):
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
            if not node.branches and not node.terminal:
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
                        (not tree_node.parent)
                        or tree_node.parent.node != template[t_idx - 1]
                    ):
                        errors.append(
                            f"Parent node in template #{template_id} ({t} <- {template[t_idx - 1]}) is not the previous node in tree: {tree_node.parent.node if tree_node.parent else None}"
                        )
                    if (
                        t_idx < len(template) - 1
                        and template[t_idx + 1] not in tree_node.branches
                    ):
                        errors.append(
                            f"Child node in template #{template_id} ({t} -> {template[t_idx + 1]}) is not a child in the tree."
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
                        f"Missing node (in template #{template_id}, but not in templates per node): {t}"
                    )
                elif template_id not in self.tree.templates_per_node[t]:
                    errors.append(
                        f"Missing template #{template_id} (not in templates per node): {t}"
                    )

        # Check templates_per_node is consistent with templates
        for node_id, templates in self.tree.templates_per_node.items():
            if node_id > 0:
                for template_id in templates:
                    if node_id not in self.tree.templates[template_id]:
                        errors.append(
                            f"Template #{template_id} in templates_per_node of node #{node_id} but the node is not in the template."
                        )
            else:
                for template_id in templates:
                    if not self.tree.templates[template_id]:
                        errors.append(
                            f"Template #{template_id} in templates_per_node of root node but the template is empty."
                        )

        # Return errors
        if errors:
            print(f"#{len(errors)} errors detected in tree:")
            for error_idx, error in enumerate(errors):
                print(f" - {error_idx}:\t" + error)
            if debug:
                breakpoint()
            if blocking:
                raise ValueError("Tree is not valid")

        return errors

    def line_to_dict(self, line_id, OCSF=False, descriptions=True, dedup=True):
        rtn_dict = {}
        if line_id not in self.nodes_per_line_id:
            return rtn_dict
        nodes = self.nodes_per_line_id[line_id]
        if not nodes:
            return rtn_dict

        field_names = defaultdict(set)
        for node_id in nodes:
            if node_id not in self.var_mapping:
                continue
            if not OCSF:
                field_name = self.var_mapping[node_id].created_attribute
                if (
                    field_name
                    and not field_name.endswith("_KEY")
                    and field_name != "SYNTAX"
                ):
                    field_names[field_name].add(node_id)
            else:
                field_added = False
                mapping = self.var_mapping[node_id].mapping
                if mapping and mapping.field_list:
                    for field in mapping.field_list:
                        if field:
                            field_names[field].add(node_id)
                            field_added = True
                if not field_added:
                    field_name = self.var_mapping[node_id].created_attribute
                    if (
                        field_name
                        and not field_name.endswith("_KEY")
                        and field_name != "SYNTAX"
                    ):
                        field_names[field_name].add(node_id)

        field_names = {k: list(v) for k, v in field_names.items()}

        for field_name, node_ids in field_names.items():
            values = []
            for node_id in node_ids:
                val = self.node_id_to_match_per_line[node_id][line_id]
                if val:
                    values.append(val)
                else:
                    values.append(self.tree.nodes[node_id].value)
            if not values:
                continue
            if dedup:
                values = list(set(values))
            if not descriptions:
                rtn_dict[field_name] = values
            else:
                field_descriptions = [
                    self.var_mapping[node_id].field_description
                    for node_id in node_ids
                    if node_id in self.var_mapping
                    and self.var_mapping[node_id].field_description
                ]
                if field_descriptions:
                    rtn_dict[field_name] = {
                        "value": values,
                        "description": field_descriptions[0],
                    }
                else:
                    rtn_dict[field_name] = {"value": values}

        return rtn_dict

    def get_node_events(self, node_id):
        matched_lines = self.line_id_per_node[node_id]
        matched_templates = {
            t for ml in matched_lines for t in self.matches_per_line_id[ml]
        }
        events = {
            event for t in matched_templates for event in self.event_types[t]
        }
        return events

    def check_consistency(self, line_indices):
        """Check if lines match the expected templates.
        Input:
            * line_indices: list of line indices to check
        Output:
            * missed_lines: list of line indices where at least one of the original templates do not match
        """
        missed_lines = []
        for line_idx in line_indices:
            mtch, candidates = self.tree.match(self.lines[line_idx])
            if not mtch:
                missed_lines.append((line_idx, []))
            else:
                templates = set(
                    [candidate.template_id for candidate in candidates]
                )
                if set(self.matches_per_line_id[line_idx]) != templates:
                    missed_lines.append((line_idx, list(sorted(templates))))
        return missed_lines

    def update_state(self, new_matches):
        """Update the internal state of the parser with new matches.
        Input:
            * new_matches: list of tuples (line_index, list of template matches)
        Output:
            * None
        """
        for line_idx, templates in new_matches:
            current_templates = self.matches_per_line_id[line_idx]
            if set(current_templates) != set(templates):
                nodes = set()
                for template in current_templates:
                    self.line_id_per_template[template].discard(line_idx)
                    for node_id in self.tree.templates[template]:
                        self.line_id_per_node[node_id].discard(line_idx)
                for template in templates:
                    self.line_id_per_template[template].add(line_idx)
                    for node_id in self.tree.templates[template]:
                        self.line_id_per_node[node_id].add(line_idx)
                        nodes.add(node_id)

                self.matches_per_line_id[line_idx] = sorted(list(templates))
                self.nodes_per_line_id[line_idx] = sorted(list(nodes))

    def get_node(self, node_id):
        """Get the node from the tree by its ID.
        Input:
            * node_id: the ID of the node (int)

        Output:
            * node: the node object

        Example output:
            {
                "is_variable": true,
                "value": "0",
                "regex": "\\d+",
                "id": 17,
                "trailing_whitespace": 0
            }
        """
        node = self.tree.nodes[node_id]
        seen_values = (
            list(self.values_per_node_id[node_id].most_common(10))
            if node_id in self.values_per_node_id
            else [node.value]
        )
        OCSF_mappings = []
        if node_id in self.var_mapping and self.var_mapping[node_id].mapping:
            OCSF_mappings.extend(
                list(self.var_mapping[node_id].mapping.field_list)
            )
        if OCSF_mappings:
            OCSF_mappings = list(set(OCSF_mappings))

        node_dict = {
            "is_variable": node.is_variable(),
            "value": node.value,
            "regex": node.regexp,
            "id": node.id,
            "trailing_whitespace": node.trailing_whitespace,
            "type": node.type,
            "values_set": seen_values,
            "attribute_name": (
                self.var_mapping[node_id].created_attribute
                if node_id in self.var_mapping
                else None
            ),
            "description": (
                self.var_mapping[node_id].field_description
                if node_id in self.var_mapping
                else None
            ),
            "containing_templates": list(
                self.tree.templates_per_node.get(node_id, [])
            )[:5],
            "OCSF_mappings": OCSF_mappings,
        }
        print("Fetching node: ", node_dict)
        return node_dict

    def change_node_parent(self, node_id, new_parent_id):
        """Change the parent of a node."""
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
                orig_template = template
                node_index = template.index(node_id)
                lineage = self.tree.node_to_tree[node_id].get_lineage()[:-1]
                self.tree.templates[template_id] = (
                    lineage + template[node_index:]
                )
                print(
                    "Changing template {} to {}".format(
                        orig_template, self.tree.templates[template_id]
                    )
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
        to_be_removed = [
            node
            for node in self.tree.templates_per_node.keys()
            if not self.tree.templates_per_node[node]
        ]
        for node in to_be_removed:
            tree_node = self.tree.node_to_tree[node]
            parent_node = tree_node.parent
            for child_id, child_node in tree_node.branches.items():
                parent_node.branches[child_id] = child_node
                child_node.parent = parent_node

            del parent_node.branches[node]
            del self.tree.templates_per_node[node]
            del self.tree.node_to_tree[node]
            self.tree.nodes[node] = None

    def update_var_schema(self, node_id, attribute_name, description):
        if attribute_name is None and description is None:
            return
        if node_id not in self.var_mapping:
            self.var_mapping[node_id] = VariableSemantics(orig_node=node_id)
        if attribute_name:
            self.var_mapping[node_id].created_attribute = attribute_name
        if description:
            self.var_mapping[node_id].field_description = description

    def set_node(
        self,
        node_id,
        node,
        new_parent=None,
        force=False,
        missing_lines=None,
        reparse=True,
    ):
        """Save the node to the tree by its ID.
        Input:
            * node_id: the ID of the node (int)
            * node: the node object (as output by get_node)
            * force: if True, force the update of the node even if it means losing matches

        Output:
            * missed_lines: list of tupples (line_index, list of template matches) where at least one of the original templates do not match
        """
        if node_id != node["id"]:
            raise ValueError("Node ID does not match")

        missing_lines = missing_lines or []

        # Keep copy of original node in case of revert
        old_node = deepcopy(self.get_node(node_id))

        # Update node
        tree_node = self.tree.nodes[node_id]
        tree_node.entity = (
            ElementType.VARIABLE
            if node["is_variable"]
            else ElementType.CONSTANT
        )
        tree_node.value = node["value"]
        tree_node.regexp = node["regex"]
        tree_node.trailing_whitespace = node["trailing_whitespace"]

        if not force:
            # Check if the new node is consistent with the old one
            missed_lines = self.check_consistency(
                self.line_id_per_node[node_id]
            )
            if not missed_lines:
                print(
                    "Changing node to new value: Old value: {} New value: {}".format(
                        old_node, node
                    )
                )
                if new_parent:
                    self.change_node_parent(node_id, new_parent)
                self._validate_tree()
                if reparse:
                    self._parse()
                return []
            else:
                # If not consistent, revert to old node
                tree_node.entity = (
                    ElementType.VARIABLE
                    if old_node["is_variable"]
                    else ElementType.CONSTANT
                )
                tree_node.value = old_node["value"].strip()
                tree_node.regexp = old_node["regex"]
                tree_node.trailing_whitespace = old_node["trailing_whitespace"]
                if old_node["is_variable"] and "type" in old_node:
                    tree_node.type = old_node["type"]
                else:
                    tree_node.type = None

                self._validate_tree()
                return missed_lines
        else:
            # If forced, update the internal state for lines that are no longer matched
            print(
                "Changing node to new value: Old value: {} New value: {}".format(
                    old_node, node
                )
            )
            if new_parent:
                self.change_node_parent(node_id, new_parent)
            self._validate_tree()
            if reparse:
                self._parse()
            return []

    def set_node_semantics(self, node_id, node):
        """
        Change only the node semantics (attribute name, description and type)
        """
        if (
            node_id >= len(self.tree.nodes)
            or not self.tree.nodes[node_id]
            or not self.tree.nodes[node_id].is_variable()
        ):
            return False

        if node_id not in self.var_mapping:
            self.var_mapping[node_id] = VariableSemantics(orig_node=node_id)

        new_type, new_attribute, new_description = (
            node.get("type", None),
            node.get("attribute_name", None),
            node.get("description", None),
        )
        if new_type and new_type not in self.valid_types:
            return False

        if new_type:
            self.tree.nodes[node_id].type = new_type

        self.update_var_schema(node_id, new_attribute, new_description)

    def batch_set_node_semantics(self, node_ids, nodes):
        for node_id, node in zip(node_ids, nodes):
            self.set_node_semantics(node_id, node)

    def add_node(self, new_node, parent, children, reparse=True):
        """Add a node to the tree.
        Input:
            * new_node: the new node object (as output by get_node)
            * parent: the parent node ID (int)
            * children: list of child node IDs (list of int)

        Output:
            * None
        """
        # First, get the parent node
        parent_node = self.tree.node_to_tree[parent]

        # If there are no children, this defines a new template: use the add_template method
        # If this new template does not capture anything, just add the node without adding a template using the logic below
        if not children:
            trail = parent_node.get_lineage()
            template = Template(self.tree.nodes[t] for t in trail if t)
            template_dict = json.loads(
                template.format_as_example(
                    ids=True, regex=True, whitespace=True
                )
            )
            template_dict.append(new_node)
            matched_lines = self.add_template(template_dict)
            if matched_lines:
                return True

        # Add node to tree if non terminal.
        # Create the node
        node = Element(
            entity=(
                ElementType.VARIABLE
                if new_node["is_variable"]
                else ElementType.CONSTANT
            ),
            value=new_node["value"].strip(),
            regexp=new_node.get("regex", ".*?"),
            id=len(self.tree.nodes),
            trailing_whitespace=new_node.get("trailing_whitespace", 0),
        )

        # Create new tree node
        tree_node = Tree(node.id, parent_node)
        self.tree.nodes.append(node)
        self.tree.node_to_tree[node.id] = tree_node
        self.tree.templates_per_node[node.id] = set()
        self.tree.examples.append([])

        # Add children
        for child in children:
            tree_node.branches[child] = self.tree.node_to_tree[child]

        # Add to parent
        parent_node.branches[node.id] = tree_node

        # Remove old connections from parent to child
        for child in children:
            if child in parent_node.branches:
                del parent_node.branches[child]
            self.tree.node_to_tree[child].parent = tree_node

        # Get list of templates
        template_list = set()
        for child in children:
            template_list.update(self.tree.templates_per_node[child])
        self.tree.templates_per_node[node.id] = template_list

        # Add to templates
        insertion_pairs = [(parent, child) for child in children]
        for template_id, template in enumerate(self.tree.templates):
            for p, c in insertion_pairs:
                if p in template and c in template:
                    parent_index = template.index(p)
                    template.insert(parent_index + 1, node.id)
                    self.tree.templates[template_id] = template

        # If the parent node is a template end, make the current node the template end.
        if parent_node.terminal:
            tree_node.terminal = True
            tree_node.template_id = parent_node.template_id
            parent_node.template_id = None
            parent_node.terminal = False

            self.tree.templates_per_node[node.id].add(tree_node.template_id)
            self.tree.templates[tree_node.template_id].append(node.id)

        # Reparse
        self._validate_tree()
        if reparse:
            self._parse()

    def add_nodes(self, new_nodes, parent, children):
        """
        Insert multiple new nodes in a chain, hooking the chain from parent -> first
        and from last -> children.
        """
        pass

    def del_node(self, node_ids, reparse=True):
        """Delete a node from the tree by its ID.
        Input:
            * node_id: the ID of the node (int)

        Output:
            * None
        """
        if not isinstance(node_ids, list):
            node_ids = [node_ids]

        for node_id in node_ids:
            if not node_id:
                continue

            # First, get the parent node
            tree_node = self.tree.node_to_tree[node_id]
            parent_node = tree_node.parent

            # Then get list of children
            children_nodes = list(
                self.tree.node_to_tree[node_id].branches.keys()
            )

            # Add children to the parents
            if parent_node:
                for child in children_nodes:
                    parent_node.branches[child] = tree_node.branches[child]

            # Set parent as terminal node if this node was terminal
            if tree_node.terminal:
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

        # Remove node from var_mapping
        for node_id in node_ids:
            if node_id in self.var_mapping:
                del self.var_mapping[node_id]

        # Reparse
        self._validate_tree()
        if reparse:
            self._parse()

    def get_template(self, template_id):
        """Get the template from the tree by its ID.
        Input:
            * template_id: the ID of the template (int)

        Output:
            * template: the template object

        Example output:
            [
            {
                "is_variable": true,
                "value": "Apr 19 00:25:58",
                "regex": "\\w{3}\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2}",
                "id": 1,
                "trailing_whitespace": 1
            },
            {
                "is_variable": true,
                "value": "tsingtao",
                "regex": "\\S+",
                "id": 2,
                "trailing_whitespace": 1
            },
            {
                "is_variable": true,
                "value": "sshd(pam_unix)",
                "regex": "\\S+",
                "id": 3,
                "trailing_whitespace": 0
            },
            {
                "is_variable": false,
                "value": "[",
                "id": 4,
                "trailing_whitespace": 0
            },
            {
                "is_variable": true,
                "value": "31071",
                "regex": "\\d+",
                "id": 5,
                "trailing_whitespace": 0
            },
            {
                "is_variable": false,
                "value": "]:",
                "id": 6,
                "trailing_whitespace": 1
            },
            {
                "is_variable": false,
                "value": "session opened for user",
                "id": 12,
                "trailing_whitespace": 1
            },
            {
                "is_variable": true,
                "value": "oracle",
                "regex": "\\S+",
                "id": 13,
                "trailing_whitespace": 1
            },
            {
                "is_variable": false,
                "value": "by",
                "id": 14,
                "trailing_whitespace": 0
            },
            {
                "is_variable": false,
                "value": "(",
                "id": 15,
                "trailing_whitespace": 0
            },
            {
                "is_variable": false,
                "value": "uid=",
                "id": 16,
                "trailing_whitespace": 0
            },
            {
                "is_variable": true,
                "value": "0",
                "regex": "\\d+",
                "id": 17,
                "trailing_whitespace": 0
            },
            {
                "is_variable": false,
                "value": ")",
                "id": 18,
                "trailing_whitespace": 0
            }
            ]
        """
        return json.loads(
            self.tree.gen_template(template_id).format_as_example(
                ids=True, regex=True, whitespace=True
            )
        )

    def get_template_events(self, template_id):
        """Get the list of OCSF events associated with this template
        Input:
            * template_id: the id of the template that is being queried
        Output:
            * events: list of OCSF events associated with this template (can be empty if this template is not mapped to any event)
        """
        return (
            list(self.event_types[template_id])
            if template_id in self.event_types
            else []
        )

    def set_template_events(self, template_id, events):
        """Set the list of OCSF events associated with this template
        Input:
            * template_id: the id of the template that is being queried
            * events: list of OCSF events associated with this template
        Output:
            * None
        """
        valid_events = [event for event in events if event in self.valid_events]
        self.event_types[template_id] = tuple(sorted(valid_events))

    def add_template(self, template):
        """Add a template to the tree.
        Input:
            * template: the template object (as output by get_template)

        Output:
            * None
        """
        # Convert template to list of nodes with negative ids, as to not conflict with existing nodes
        new_template = Template(
            [
                Element(
                    entity=(
                        ElementType.VARIABLE
                        if node["is_variable"]
                        else ElementType.CONSTANT
                    ),
                    value=node["value"].strip(),
                    regexp=node.get("regex", ".*?"),
                    trailing_whitespace=node.get("trailing_whitespace", 0),
                    id=len(self.tree.nodes) + i,
                )
                for i, node in enumerate(template)
            ]
        )

        # Get list of matches
        matched_line_ids = [
            line_id
            for line_id, line in enumerate(self.lines)
            if new_template.match(line)[0]
        ]

        if not matched_line_ids:
            return []

        # Add to templates
        template_id = self.tree.add_template(
            new_template, example=self.lines[matched_line_ids[0]]
        )

        # Reparse
        self._validate_tree()
        self._parse()
        return [self.lines[i] for i in self.line_id_per_template[template_id]]

    def set_template(self, template_id, template, reparse=True):
        """Set a template to the tree by its ID.
        Input:
            * template_id: the ID of the template (int)
            * template: the template object (as output by get_template)

        Output:
            * None
        """
        # Convert template to list of nodes with negative ids, as to not conflict with existing nodes
        new_template = Template(
            [
                Element(
                    entity=(
                        ElementType.VARIABLE
                        if node["is_variable"]
                        else ElementType.CONSTANT
                    ),
                    value=node["value"].strip(),
                    regexp=node.get("regex", ".*?"),
                    trailing_whitespace=node.get("trailing_whitespace", 0),
                    id=len(self.tree.nodes) + i,
                )
                for i, node in enumerate(template)
            ]
        )

        # Get list of matches
        matched_line_ids = [
            line_id
            for line_id, line in enumerate(self.lines)
            if new_template.match(line)[0]
        ]

        if not matched_line_ids:
            return []

        self.tree.replace_template(
            template_id, new_template, self.lines[matched_line_ids[0]]
        )

        # Reparse
        self._validate_tree()
        if reparse:
            self._parse()

    def del_template(self, template_id, reparse=True):
        """Delete a template from the tree by its ID.
        Input:
            * template_id: the ID of the template (int)

        Output:
            * None
        """
        self.tree.remove_template(template_id)
        self._validate_tree()
        if reparse:
            self._parse()

    def add_event_to_add(self, events):
        if not isinstance(events, list):
            events = [events]

        for t_id in self.event_types.keys():
            self.event_types[t_id] = tuple(
                sorted(set(self.event_types[t_id] + tuple(events)))
            )

    def remove_event_from_add(self, events):
        if not isinstance(events, list):
            events = [events]

        for t_id in self.event_types.keys():
            self.event_types[t_id] = tuple(
                sorted(set(self.event_types[t_id]) - set(events))
            )

    def get_attributes_missing_embeddings(self, exact=True):
        """
        Get all attributes that are missing embeddings
        Input:
            * exact: if True, only return attributes that are missing all embeddings
        Output:
            * attributes: list of attributes that are missing (all/at least one) embeddings
        """
        attributes_with_emb = defaultdict(list)
        attributes_without_emb = defaultdict(list)
        for node_id, var in self.var_mapping.items():
            if (
                not self.tree.nodes[node_id]
                or not self.tree.nodes[node_id].is_variable()
            ):
                continue
            if var.created_attribute and var.embedding is None:
                attributes_without_emb[var.created_attribute].append(node_id)
            elif var.created_attribute and var.embedding is not None:
                attributes_with_emb[var.created_attribute].append(node_id)

        if not exact:
            return list(attributes_without_emb.keys())

        else:
            missing = {
                k
                for k in attributes_without_emb
                if k not in attributes_with_emb
            }
            return list(missing)

    def get_attributes_missing_map(self, exact=0):
        """
        Get all attributes that are missing mappings
        Input:
            * exact: if 1, only return attributes that are missing all mappings. if 0, return all that are missing at least one. if -1, return only those that have at least one mapping
        Output:
            * attributes: list of attributes that are missing (all/at least one) mappings
        """
        attributes_with_map = defaultdict(list)
        attributes_without_map = defaultdict(list)
        for node_id, var in self.var_mapping.items():
            if not self.tree.nodes[node_id]:
                continue
            if var.mapping:
                fields = set(var.mapping.field_list)
            else:
                fields = set()
            if var.created_attribute and not fields:
                attributes_without_map[var.created_attribute].append(node_id)
            elif var.created_attribute and fields:
                attributes_with_map[var.created_attribute].append(node_id)

        if exact == 0:
            return list(attributes_without_map.keys())
        elif exact == 1:
            missing = {
                k
                for k in attributes_without_map
                if k not in attributes_with_map
            }
            return list(missing)
        else:
            missing = {
                k for k in attributes_without_map if k in attributes_with_map
            }
            return list(missing)

    ### OCSF Mapping Editor ###

    def set_node_mapping(self, node_id, OCSF_fields, description):
        """
        Change only the node mapping (OCSF fields)
        """
        if node_id >= len(self.tree.nodes) or not self.tree.nodes[node_id]:
            return False

        if node_id not in self.var_mapping:
            self.var_mapping[node_id] = VariableSemantics(orig_node=node_id)

        # Reinitialize field lists
        if not self.var_mapping[node_id].mapping:
            self.var_mapping[node_id].mapping = Mapping()
        else:
            self.var_mapping[node_id].mapping.field_list = []

        # Add fields to all events they are part of
        for field in OCSF_fields:
            if field in self.client.attributes:
                self.var_mapping[node_id].mapping.field_list.append(field)

        # Set descriptions
        if description:
            self.var_mapping[node_id].field_description = description

    def batch_set_node_mapping(self, node_ids, OCSF_fields, description):
        for node_id in node_ids:
            self.set_node_mapping(node_id, OCSF_fields, description)

    def set_attribute_mapping(self, attribute_name, OCSF_fields, description):
        """Set the OCSF mapping for all nodes that share the same attribute name"""
        node_ids = [
            node_id
            for node_id, var in self.var_mapping.items()
            if var.created_attribute == attribute_name
        ]

        self.batch_set_node_mapping(node_ids, OCSF_fields, description)

        # Check
        print(OCSF_fields)
        intended_field_set = set(OCSF_fields)
        field_set = set()
        for node_id in node_ids:
            if self.var_mapping[node_id].mapping:
                field_set.update(set(self.var_mapping[node_id].mapping))
        if intended_field_set != set(OCSF_fields):
            breakpoint()

    def get_attribute_mapping(self, attribute_name):
        node_ids = [
            node_id
            for node_id, var in self.var_mapping.items()
            if var.created_attribute == attribute_name
        ]

        # Get the list of OCSF mappings for the attribute
        OCSF_mappings = []
        for node_id in node_ids:
            mapping = self.var_mapping[node_id].mapping
            if mapping:
                OCSF_mappings.extend(list(mapping.field_list))

        if OCSF_mappings:
            OCSF_mappings = list(set(OCSF_mappings))

        # Get a set of template ids that are associated with this attribute
        template_list = list(
            {
                t
                for node_id in node_ids
                for t in self.tree.templates_per_node[node_id]
            }
        )
        template_list = random.sample(template_list, min(5, len(template_list)))

        # Get the description of this attribute
        description = (
            self.var_mapping[node_ids[0]].field_description
            if node_ids
            else None
        )

        return {
            "OCSF_mapping": OCSF_mappings,
            "description": description,
            "templates": template_list,
        }

    def get_all_attribute_mappings(self):
        """Get all attribute mappings"""
        attributes = set()
        for var in self.var_mapping.values():
            if var.created_attribute:
                attributes.add(var.created_attribute)

        attribute_mappings = {
            k: self.get_attribute_mapping(k) for k in attributes
        }
        return attribute_mappings

    def get_all_possible_mappings(self, attribute_name):
        """Get all possible mappings for a node.
        Input:
            * attribute_name: the name of the attribute (str)

        Output:
            * mappings: list of mappings for the node. Format is {
                "exact": {
                    "field_name": "field_description",
                    ...
                },
                "fuzzy": {
                    "field_name": "field_description",
                    ...
                },
                "all": {
                    "field_name": "field_description",
                    ...
                }
            }
            Where "exact" are fields that have the exact same type as the attribute, "fuzzy" are fields that have a similar type to the attribute, "all" are all remaining fields.
        """
        node_ids = [
            node_id
            for node_id, var in self.var_mapping.items()
            if var.created_attribute == attribute_name
        ]
        events = set()
        for node_id in node_ids:
            events.update(self.get_node_events(node_id))

        data_type = self.tree.nodes[node_ids[0]].type
        if data_type in ["user_t", "composite_t"]:
            data_type = None

        descriptions = {"exact": {}, "fuzzy": {}, "all": {}}
        for event in events:
            exact = self.client.get_description(
                event,
                data_type,
                fuzzy=0,
            )
            fuzzy = self.client.get_description(
                event,
                data_type,
                fuzzy=1,
            )
            all_map = self.client.get_description(
                event,
                data_type,
                fuzzy=2,
            )

            descriptions["exact"].update(exact)
            descriptions["fuzzy"].update(fuzzy)
            descriptions["all"].update(all_map)

        # Remove duplicates
        descriptions["fuzzy"] = {
            k: v
            for k, v in descriptions["fuzzy"].items()
            if k not in descriptions["exact"]
        }
        descriptions["all"] = {
            k: v
            for k, v in descriptions["all"].items()
            if k not in descriptions["exact"] and k not in descriptions["fuzzy"]
        }

        return descriptions

    def get_events_per_attribute(self, attribute_name):
        """
        Get the list of events associated with this attribute
        """
        node_ids = [
            node_id
            for node_id, var in self.var_mapping.items()
            if var.created_attribute == attribute_name
        ]
        events = set()
        for node_id in node_ids:
            events.update(self.get_node_events(node_id))
        return list(events)

    def save(self, safe=True):
        if not self.output:
            return

        # Save parser as a dill file
        errors = self._validate_tree(debug=False, blocking=False)
        if not safe or not errors:
            with open(os.path.join(self.output, "parser.dill"), "wb") as f:
                dill.dump(
                    Parser(
                        tree=self.tree,
                        var_mapping=self.var_mapping,
                        schema_mapping=self.schema_mapping,
                        event_types=self.event_types,
                    ),
                    f,
                )
                print("Saved parser to disk")
        else:
            print("Parser not saved to disk: tree is not valid")

    def save_parser_as_json(self, path=None, include_embeddings=False):
        if not path and not self.output:
            return

        if not path:
            path = os.path.join(self.output, "parser.json")

        json_parser = Parser(
            tree=self.tree,
            var_mapping=self.var_mapping,
            event_types=self.event_types,
        ).as_json(include_embeddings=include_embeddings)

        with open(path, "w", encoding="utf-8") as f:
            try:
                json.dump(json_parser, f, indent=2)
            except TypeError:
                breakpoint()
            print(f"Saved parser to {path}")

    def save_as_json(self, output_file=None):
        if not output_file:
            output_file = os.path.join(self.output, "parser.json")

        # Headers = LineID,Content,LogHubContent,EventIDs,EventTemplates,EventMatches
        output_data = []
        for line_id, line in enumerate(self.lines):
            templates = self.matches_per_line_id[line_id]
            template_str = [
                self.tree.gen_template(t).convert_to_wildcard_template()
                for t in templates
            ]
            nodes_per_template = [
                [
                    n
                    for n in self.tree.templates[t]
                    if self.tree.nodes[n] and self.tree.nodes[n].is_variable()
                ]
                for t in templates
            ]
            matches = [
                [self.node_id_to_match_per_line[n][line_id] for n in nodes]
                for nodes in nodes_per_template
            ]
            output_data.append(
                {
                    "LineID": line_id,
                    "Content": line,
                    "LogHubContent": line,
                    "EventIDs": templates,
                    "EventTemplates": template_str,
                    "EventMatches": matches,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

    def export(self, OCSF=False):
        """
        Export the parsed log to JSON format
        Input:
            * OCSF: if True, export the log in OCSF format.
        """

        logs = []
        all_fields = set()
        all_templates = set()
        for line_id, line in enumerate(self.lines):
            matched_templates = self.matches_per_line_id[line_id]
            for t in matched_templates:
                all_templates.add(t)
            matched_nodes = list(
                set(
                    [
                        n
                        for t in matched_templates
                        for n in self.tree.templates[t]
                    ]
                )
            )
            created_attributes = {}
            OCSF_events = list(
                set(
                    [
                        event
                        for t in matched_templates
                        for event in self.event_types.get(t, [])
                    ]
                )
            )
            OCSF_attributes = {}
            for node_id in matched_nodes:
                if node_id not in self.node_id_to_match_per_line:
                    continue
                if line_id not in self.node_id_to_match_per_line[node_id]:
                    continue
                new_value = self.node_id_to_match_per_line[node_id][line_id]
                if self.tree.nodes[node_id].type in [
                    "integer_t",
                    "float_t",
                    "long_t",
                ]:
                    try:
                        new_value = float(new_value)
                    except ValueError:
                        pass
                if self.tree.nodes[node_id].type == "boolean_t":
                    try:
                        new_value = bool(new_value)
                    except ValueError:
                        pass
                if self.tree.nodes[node_id].type == "datetime_t":
                    try:
                        new_value = parse_datetime_to_timestamp(new_value)
                    except ValueError:
                        pass
                if (
                    node_id in self.var_mapping
                    and self.var_mapping[node_id].created_attribute
                ):
                    if (
                        self.var_mapping[node_id].created_attribute
                        not in created_attributes
                    ):
                        created_attributes[
                            self.var_mapping[node_id].created_attribute
                        ] = new_value
                    else:
                        old_value = created_attributes[
                            self.var_mapping[node_id].created_attribute
                        ]
                        if (
                            isinstance(old_value, list)
                            and new_value not in old_value
                        ):
                            created_attributes[
                                self.var_mapping[node_id].created_attribute
                            ].append(new_value)
                        elif (
                            not isinstance(old_value, list)
                            and old_value != new_value
                        ):
                            created_attributes[
                                self.var_mapping[node_id].created_attribute
                            ] = [old_value, new_value]
                    all_fields.add(node_id)
                if (
                    node_id in self.var_mapping
                    and self.var_mapping[node_id].mapping
                ):
                    for field in self.var_mapping[node_id].mapping.field_list:
                        if field not in OCSF_attributes:
                            OCSF_attributes[field] = new_value
                        else:
                            old_value = OCSF_attributes[field]
                            if (
                                isinstance(old_value, list)
                                and new_value not in old_value
                            ):
                                OCSF_attributes[field].append(new_value)
                            elif (
                                not isinstance(old_value, list)
                                and old_value != new_value
                            ):
                                OCSF_attributes[field] = [
                                    old_value,
                                    new_value,
                                ]

            line_dict = {}
            line_dict["templates"] = matched_templates
            line_dict["content"] = line
            if not OCSF:
                line_dict["attributes"] = created_attributes
            else:
                line_dict["OCSF_events"] = OCSF_events
                line_dict["attributes"] = OCSF_attributes
                line_dict["attributes"]["unmapped"] = created_attributes

            logs.append(line_dict)

        templates = {}
        for template_id in all_templates:
            template = self.tree.gen_template(
                template_id
            ).convert_to_wildcard_template(
                name=True, var_mapping=self.var_mapping
            )
            templates[template_id] = {
                "template": template,
                "events": list(self.event_types.get(template_id, [])),
            }

        variables = {}
        for node_id in all_fields:
            field_name = self.var_mapping[node_id].created_attribute
            field_description = self.var_mapping[node_id].field_description
            if field_name not in variables:
                variables[field_name] = {
                    "description": field_description,
                    "type": self.tree.nodes[node_id].type,
                }

        return {
            "entries": logs,
            "templates": templates,
            "variables": variables,
        }

    ### Query Engine ###
    def match_attribute(
        self,
        attribute_name=None,
        attribute_value=None,
        exact=True,
        case_sensitive=True,
        comparision=None,
        negation=False,
        existence=False,
        static_match=False,
        standardized=False,
    ):
        if not attribute_name:
            attribute_name = []
        if not attribute_value:
            attribute_value = []
        if not isinstance(attribute_name, list):
            attribute_name = [attribute_name]
        if not isinstance(attribute_value, list):
            attribute_value = [attribute_value]

        if standardized:
            all_events = self.client.get_classes().keys()
            new_attribute_name = []
            for attr in attribute_name:
                added = False
                for event in all_events:
                    if attr.startswith(event + "."):
                        new_attribute_name.append(attr.replace(event + ".", ""))
                        added = True
                        break
                if not added:
                    new_attribute_name.append(attr)
            attribute_name = new_attribute_name

        if static_match:
            matched_templates = []
            search_regexes = []
            for target_value in attribute_value:
                escaped_value = re.escape(target_value)
                if not case_sensitive:
                    search_regexes.append(
                        re.compile(escaped_value, re.IGNORECASE)
                    )
                else:
                    search_regexes.append(re.compile(escaped_value))
            for template_id, template_string in self.static_templates.items():
                matches = False
                for search_regex in search_regexes:
                    if exact:
                        is_match = search_regex.fullmatch(template_string)
                    else:
                        is_match = search_regex.search(template_string)
                    if is_match:
                        matches = True
                        break

                if matches:
                    matched_templates.append(template_id)

            matching_lines = set()
            for template_id in matched_templates:
                matching_lines.update(self.line_id_per_template[template_id])

            return matching_lines

        node_ids = list(
            {
                i
                for attr in attribute_name
                for i in self.attribute_to_node_id[attr]
            }
        )

        matching_lines = set()
        if not existence:
            for node_id in node_ids:
                for line_id, value in self.node_id_to_match_per_line[
                    node_id
                ].items():
                    for target_value in attribute_value:
                        if not comparision:
                            # Use the re library to match the value
                            if not case_sensitive:
                                target_value = re.escape(target_value)
                                regexp = re.compile(target_value, re.IGNORECASE)
                            else:
                                target_value = re.escape(target_value)
                                regexp = re.compile(target_value)

                            if exact:
                                is_match = regexp.fullmatch(value)
                            else:
                                is_match = regexp.search(value)
                        else:
                            # Parse dates if needed
                            if self.tree.nodes[node_id].type == "datetime_t":
                                try:
                                    target_value = float(target_value)
                                except ValueError:
                                    try:
                                        target_value = (
                                            parse_datetime_to_timestamp(
                                                target_value
                                            )
                                        )
                                    except ValueError:
                                        continue

                                try:
                                    value = float(value)
                                except ValueError:
                                    try:
                                        value = parse_datetime_to_timestamp(
                                            value
                                        )
                                    except ValueError:
                                        continue
                            else:
                                try:
                                    target_value = parse_float(
                                        str(target_value)
                                    )
                                    value = parse_float(value)
                                except ValueError:
                                    continue

                            try:
                                value = float(value)
                                target_value = float(target_value)
                            except ValueError:
                                continue

                            if comparision == ">":
                                is_match = float(value) > float(target_value)
                            elif comparision == "<":
                                is_match = float(value) < float(target_value)
                            elif comparision == "=":
                                is_match = float(value) == float(target_value)
                            elif comparision == ">=":
                                is_match = float(value) >= float(target_value)
                            elif comparision == "<=":
                                is_match = float(value) <= float(target_value)
                            else:
                                is_match = None

                        if negation:
                            is_match = not is_match
                        if is_match:
                            matching_lines.add(line_id)
        else:
            for node_id in node_ids:
                for line_id in self.line_id_per_node[node_id]:
                    matching_lines.add(line_id)
            if negation:
                matching_lines = set(range(len(self.lines))) - matching_lines

        return matching_lines

    def run_query(self, query, standardized=False):
        """
        Run a query on the log file.
        Input:
            * query: a dict, containing two attributes
              - operator: (AND / OR)
              - conditions: list of conditions, where each condition is either a query, or a 6-tuple containing:
                - an attribute name (or list of attibute names)
                - an attribute value (or list of values)
                - a match type boolean (true is exact / false is fuzzy)
                - a case sensitivity boolean (true case_sensitive / false is case_insensitive)
                - a comparison operator (None, >, <, =, >=, <=)
                - a negation boolean (true is negated / false is not negated)
        Output:
            * matching_lines: a set of line ids that match the query
        """

        results = []
        for condition in query["conditions"]:
            if isinstance(condition, dict):
                results.append(self.run_query(condition))
            else:
                results.append(
                    self.match_attribute(*condition, standardized=standardized)
                )

        if query["operator"] == "AND":
            return set.intersection(*results)
        elif query["operator"] == "OR":
            return set.union(*results)

    def save_query(self, query_name, query):
        """
        Saves the given query to a queries.json file in self.output.
        If the file or key doesn't exist, it creates them. If it exists, overwrites.
        """
        queries_path = self.query_files
        if not queries_path:
            raise ValueError(
                "Output or query path are not set, cannot save queries."
            )

        current_data = {}
        for query_path in queries_path:
            if os.path.exists(query_path):
                with open(queries_path, "r", encoding="utf-8") as f:
                    current_data.update(json.load(f))

        current_data[query_name] = query

        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)

    def load_queries(self):
        """
        Loads the queries from a queries.json file in self.output.
        If it doesn't exist or is invalid, returns empty dict.
        """
        queries_path = self.query_files
        if not queries_path:
            raise ValueError(
                "Output or query path are not set, cannot load queries."
            )
        queries = {}
        for query_path in queries_path:
            if not os.path.exists(query_path):
                continue
            with open(query_path, "r", encoding="utf-8") as f:
                try:
                    queries.update(json.load(f))
                except json.JSONDecodeError:
                    pass

        return queries

    def run_queries_from_file(self):
        """
        Loads the queries from queries.json, runs them, and returns
        { queryName: [ listOfLineIds ], ... }
        """
        loaded = self.load_queries()
        results = {}
        for qname, qdef in loaded.items():
            line_ids = self.run_query(qdef, standardized=True)
            results[qname] = sorted(list(line_ids))
        return results

    ### Metrics ###

    def parser_accuracy(self, target_editor, soft=False, is_baseline=True):
        if not isinstance(target_editor, TreeEditor):
            raise ValueError("target_editor must be a TreeEditor object")
        if not len(self.lines) == len(target_editor.lines):
            raise ValueError("Both editors must have parsed the same lines")
        if not is_baseline:
            return target_editor.parser_accuracy(
                self, soft=soft, is_baseline=True
            )

        baseline_templates_per_line = [
            [
                re.sub(
                    r"\s+",
                    " ",
                    self.tree.gen_template(t).convert_to_wildcard_template(),
                ).strip()
                for t in self.matches_per_line_id[line_id]
            ]
            for line_id, _ in enumerate(self.lines)
        ]
        target_templates_per_line = [
            [
                re.sub(
                    r"\s+",
                    " ",
                    target_editor.tree.gen_template(
                        t
                    ).convert_to_wildcard_template(),
                ).strip()
                for t in target_editor.matches_per_line_id[line_id]
            ]
            for line_id, _ in enumerate(target_editor.lines)
        ]

        cached_scores = defaultdict(lambda: defaultdict(lambda: -1))

        score = 0
        for line_id, baseline_templates in enumerate(
            baseline_templates_per_line
        ):
            target_templates = target_templates_per_line[line_id]

            local_scores = []
            for baseline_template in baseline_templates:
                for target_template in target_templates:
                    if cached_scores[baseline_template][target_template] == -1:
                        if soft:
                            cached_scores[baseline_template][
                                target_template
                            ] = 1 - (
                                Levenshtein.distance(
                                    baseline_template, target_template
                                )
                                / max(
                                    len(baseline_template), len(target_template)
                                )
                            )
                        else:
                            cached_scores[baseline_template][
                                target_template
                            ] = (
                                1 if baseline_template == target_template else 0
                            )
                    local_scores.append(
                        cached_scores[baseline_template][target_template]
                    )
            score += max(local_scores) if local_scores else 0

        score /= len(baseline_templates_per_line)
        return score

    def group_accuracy(
        self,
        target_editor,
        soft=False,
        is_baseline=True,
        penalize_overcapture=True,
    ):
        if not isinstance(target_editor, TreeEditor):
            raise ValueError("target_editor must be a TreeEditor object")
        if not is_baseline:
            return target_editor.group_accuracy(
                self, soft=soft, is_baseline=True
            )

        ref_template_to_lines = self.line_id_per_template
        pred_template_to_lines = target_editor.line_id_per_template
        ref_line_to_templates = self.matches_per_line_id
        pred_line_to_templates = target_editor.matches_per_line_id

        cached_scores = defaultdict(lambda: defaultdict(lambda: -1))
        score = 0

        for line_id, ref_templates in ref_line_to_templates.items():
            pred_templates = pred_line_to_templates[line_id]

            local_scores = []
            for ref_template in ref_templates:
                for pred_template in pred_templates:
                    ref_group = ref_template_to_lines[ref_template]
                    pred_group = pred_template_to_lines[pred_template]
                    if not pred_group:
                        continue

                    if cached_scores[ref_template][pred_template] == -1:
                        if soft:
                            if (
                                penalize_overcapture
                                and not pred_group.issubset(ref_group)
                            ):
                                cached_scores[ref_template][pred_template] = 0
                            else:
                                cached_scores[ref_template][pred_template] = (
                                    len(pred_group.intersection(ref_group))
                                    / len(pred_group.union(ref_group))
                                )
                        else:
                            cached_scores[ref_template][pred_template] = (
                                1 if ref_group == pred_group else 0
                            )
                    local_scores.append(
                        cached_scores[ref_template][pred_template]
                    )
            score += max(local_scores) if local_scores else 0

        return score / len(ref_line_to_templates)

    def template_similarity(self, target_editor, is_baseline=True):
        return self.parser_accuracy(
            target_editor, soft=True, is_baseline=is_baseline
        )

    def group_similarity(self, target_editor, is_baseline=True):
        return self.group_accuracy(
            target_editor, soft=True, is_baseline=is_baseline
        )

    def schema_group_accuracy(
        self,
        target_editor,
        soft=False,
        is_baseline=True,
        penalize_overcapture=True,
    ):
        if not isinstance(target_editor, TreeEditor):
            raise ValueError("target_editor must be a TreeEditor object")
        if len(self.tree.nodes) != len(target_editor.tree.nodes) or len(
            self.tree.templates
        ) != len(target_editor.tree.templates):
            raise ValueError("Both editors must have the same tree structure")
        if not is_baseline:
            return target_editor.schema_group_accuracy(self, is_baseline=True)

        # Create a map from each variable node to its field name, and a reverse map from field name to a set of variable node
        baseline_var_to_field = {
            node_id: (
                self.var_mapping[node_id].created_attribute
                if node_id in self.var_mapping
                and self.var_mapping[node_id].created_attribute
                else f"node_{node_id}"
            )
            for node_id, node in enumerate(self.tree.nodes)
            if node_id and node and node.is_variable()
        }
        target_var_to_field = {
            node_id: (
                target_editor.var_mapping[node_id].created_attribute
                if node_id in target_editor.var_mapping
                and target_editor.var_mapping[node_id].created_attribute
                else f"node_{node_id}"
            )
            for node_id, node in enumerate(target_editor.tree.nodes)
            if node_id and node and node.is_variable()
        }

        baseline_field_to_lines = defaultdict(set)
        target_field_to_lines = defaultdict(set)
        for node_id, field_name in baseline_var_to_field.items():
            for line_id in self.line_id_per_node[node_id]:
                baseline_field_to_lines[field_name].add(line_id)
        for node_id, field_name in target_var_to_field.items():
            for line_id in target_editor.line_id_per_node[node_id]:
                target_field_to_lines[field_name].add(line_id)

        weights = {node_id: 0 for node_id in baseline_var_to_field}
        for node_id, field_name in baseline_var_to_field.items():
            weights[node_id] = len(self.line_id_per_node[node_id])

        cached_scores = defaultdict(lambda: defaultdict(lambda: -1))
        score = 0
        for node_id, baseline_field_name in baseline_var_to_field.items():
            target_field_name = target_var_to_field[node_id]

            baseline_group = baseline_field_to_lines[baseline_field_name]
            target_group = target_field_to_lines[target_field_name]

            if cached_scores[baseline_field_name][target_field_name] == -1:
                if soft:
                    if penalize_overcapture and not target_group.issubset(
                        baseline_group
                    ):
                        cached_scores[baseline_field_name][
                            target_field_name
                        ] = 0
                    else:
                        cached_scores[baseline_field_name][
                            target_field_name
                        ] = len(
                            target_group.intersection(baseline_group)
                        ) / len(
                            target_group.union(baseline_group)
                        )
                else:
                    cached_scores[baseline_field_name][target_field_name] = (
                        1 if baseline_group == target_group else 0
                    )

            print(
                f"Score for node {node_id} (baseline name: {baseline_field_name}, target name: {target_field_name}): {cached_scores[baseline_field_name][target_field_name]}"
            )
            score += (
                weights[node_id]
                * cached_scores[baseline_field_name][target_field_name]
            )

        return score / sum(weights.values())

    def schema_group_similarity(self, target_editor, is_baseline=True):
        return self.schema_group_accuracy(
            target_editor, soft=True, is_baseline=is_baseline
        )

    def name_similarity(self, target_editor, is_baseline=True):
        if not isinstance(target_editor, TreeEditor):
            raise ValueError("target_editor must be a TreeEditor object")
        if not is_baseline:
            return target_editor.name_similarity(self, is_baseline=True)

        pass

    def mapping_score(self, target_editor, is_baseline=True):
        if not isinstance(target_editor, TreeEditor):
            raise ValueError("target_editor must be a TreeEditor object")
        if len(self.tree.nodes) != len(target_editor.tree.nodes) or len(
            self.tree.templates
        ) != len(target_editor.tree.templates):
            raise ValueError("Both editors must have the same tree structure")
        if not is_baseline:
            return target_editor.mapping_score(self, is_baseline=True)

        baseline_field_to_ocsf = defaultdict(set)
        target_field_to_ocsf = defaultdict(set)

        def standardize_attribute_name(attr):
            all_events = self.client.get_classes().keys()
            for event in all_events:
                if attr.startswith(event + "."):
                    return attr.replace(event + ".", "")
            return attr

        def populate_field_to_ocsf(target_dict, target_editor):
            for _, var in target_editor.var_mapping.items():
                if not var.created_attribute:
                    continue
                if var.mapping:
                    for ocsf_field in var.mapping.field_list:
                        target_dict[var.created_attribute].add(
                            standardize_attribute_name(ocsf_field)
                        )
                if var.created_attribute not in target_dict:
                    target_dict[var.created_attribute] = set()

        populate_field_to_ocsf(baseline_field_to_ocsf, self)
        populate_field_to_ocsf(target_field_to_ocsf, target_editor)

        weights = {field_name: 0 for field_name in baseline_field_to_ocsf}
        for field_name in weights:
            matched_lines = set()
            for node_id in self.attribute_to_node_id[field_name]:
                matched_lines.update(self.line_id_per_node[node_id])
            weights[field_name] = len(matched_lines)

        score = 0
        for field_name, baseline_ocsf in baseline_field_to_ocsf.items():
            target_ocsf = target_field_to_ocsf[field_name]
            if not target_ocsf and not baseline_ocsf:
                score += weights[field_name]  # both are empty, no contribution
            elif not target_ocsf or not baseline_ocsf:
                continue
            else:
                score += (
                    weights[field_name]
                    * len(target_ocsf.intersection(baseline_ocsf))
                    / len(target_ocsf)
                )
        return score / sum(weights.values())

    def end_to_end_score(self, target_editor, ocsf=False, save_to_file=None):
        if not isinstance(target_editor, TreeEditor):
            raise ValueError("target_editor must be a TreeEditor object")

        results_baseline = self.run_queries_from_file()
        results_target = target_editor.run_queries_from_file()

        common_queries = set(results_baseline.keys()).intersection(
            set(results_target.keys())
        )

        results_per_query = {}
        for query in common_queries:
            baseline_lines = set(results_baseline[query])
            # Save baseline result if needed
            if ocsf and f"{query}_ocsf" in results_target:
                target_lines = set(results_target[f"{query}_ocsf"])
            else:
                target_lines = set(results_target[query])
            precision = (
                len(baseline_lines.intersection(target_lines))
                / len(target_lines)
                if target_lines
                else 0
            )
            recall = (
                len(baseline_lines.intersection(target_lines))
                / len(baseline_lines)
                if baseline_lines
                else 0
            )
            results_per_query[query] = (precision, recall)
            print(
                f"Query: {query}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

        if save_to_file:
            lines_per_query = {
                k: [self.lines[i] for i in v]
                for k, v in results_baseline.items()
            }
            with open(save_to_file, "w", encoding="utf-8") as f:
                json.dump(lines_per_query, f, indent=2)

        # compute averages
        if not results_per_query:
            return 0, 0, {}, {}
        precision_sum = sum(p for p, _ in results_per_query.values())
        recall_sum = sum(r for _, r in results_per_query.values())
        avg_precision = precision_sum / len(results_per_query)
        avg_recall = recall_sum / len(results_per_query)

        baseline_lines = {
            k: [self.lines[i] for i in v] for k, v in results_baseline.items()
        }
        target_lines = {
            k: [target_editor.lines[i] for i in v]
            for k, v in results_target.items()
        }
        return avg_precision, avg_recall, baseline_lines, target_lines
