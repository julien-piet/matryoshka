import re
from collections import defaultdict

import numpy as np
import torch


class ClusterVariables:

    def __init__(
        self,
        caller,
        parser,
        output_dir="output/",
        model="gemini-2.5-flash",
    ):
        self.tree = parser.tree
        self.entries_per_template = parser.entries_per_template
        self.values = parser.values
        self.caller = caller
        self.model = model
        self.output_dir = output_dir

        self.embedding = parser.embedding

        self.tree_cluster = {}
        self.description_cluster = {}

    def build_tree_cluster(self):
        """Check if variables that are branches to the same node represent the same concept"""

        # Helper function to merge constant items
        def merge_constant_items(prefix):
            # Convert constants to their value, and remove whitespaces
            constant_mapping = {}
            for var_idx, var in enumerate(self.tree.nodes):
                if var and not var.is_variable():
                    constant_mapping[var_idx] = re.sub(r"\s", "", var.value)

            # Merge constants that are next to each other
            new_prefixes = {}
            for i, prefix in prefixes.items():
                new_prefix = []
                for j, val in enumerate(prefix):
                    # If this is the first item, add it to the list
                    if j == 0:
                        new_prefix.append(
                            constant_mapping[val]
                            if val in constant_mapping
                            else val
                        )
                        continue

                    # If this is a variable, add it to the list
                    if val not in constant_mapping:
                        new_prefix.append(val)
                        continue

                    # If this is a constant, but the previous item is a variable, add it to the list
                    if prefix[j - 1] not in constant_mapping:
                        new_prefix.append(
                            constant_mapping[val]
                            if val in constant_mapping
                            else val
                        )
                        continue

                    # If you get here, the current value and its predecessor are constants, so merge them
                    new_prefix[-1] += (
                        constant_mapping[val]
                        if val in constant_mapping
                        else val
                    )

                new_prefixes[i] = new_prefix

            return new_prefixes

        # Get list of all variables
        all_variables = [
            var_idx
            for var_idx, var in enumerate(self.tree.nodes)
            if var and var.is_variable()
        ]
        clusters = {i: [i] for i in all_variables}

        # Get prefix of each node
        prefixes = {
            i: self.tree.node_to_tree[i].get_lineage()[:-1]
            for i in all_variables
        }

        # Merge constant items
        prefixes = merge_constant_items(prefixes)

        # Merge variables with equivalent prefixes
        while True:
            equivalent_prefixes = {
                min(clusters[j]): tuple(
                    [min(clusters[i]) if i in clusters else i for i in prefix]
                )
                for j, prefix in prefixes.items()
            }
            equivalence_classes = defaultdict(list)
            for var_idx, prefix in equivalent_prefixes.items():
                equivalence_classes[prefix].append(var_idx)

            if len(equivalence_classes) == len(equivalent_prefixes):
                break

            for values in equivalence_classes.values():
                if len(values) > 1:
                    new_cluster = {i for j in values for i in clusters[j]}
                    for i in new_cluster:
                        clusters[i] = list(new_cluster)

        self.tree_cluster = clusters

        return self.tree_cluster

    def build_descrpition_cluster(self, embeddings, threshold=0.98):
        """Cluster variables based on descriptions

        Merge variables that have descriptions with cosine similarity above a 0.98

        Args:
            embeddings (Dict[int, torch.Tensor]): Embedding of each variable's description
            threshold (float): Threshold for cosine similarity
        """
        # Get list of all variables
        all_variables = sorted(
            [
                var_idx
                for var_idx, var in enumerate(self.tree.nodes)
                if var and var.is_variable()
            ]
        )

        # Compute cosine similarity between all pairs of descriptions
        embedding_list = [embeddings[i].squeeze() for i in all_variables]
        embedding_list = torch.stack(embedding_list).numpy()
        norms = np.linalg.norm(embedding_list, axis=1, keepdims=True)
        normalized_embeddings = embedding_list / norms
        distance_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Cluster variables based on cosine similarity: if two variables have a cosine similarity above the threshold, they are merged
        clusters = {i: {i} for i in all_variables}
        for i, var_idx in enumerate(all_variables):
            for j in range(i + 1, len(all_variables)):
                if distance_matrix[i, j] > threshold:
                    combined_cluster = clusters[var_idx].union(
                        clusters[all_variables[j]]
                    )
                    clusters[var_idx] = combined_cluster
                    clusters[all_variables[j]] = combined_cluster

        self.description_cluster = {k: list(v) for k, v in clusters.items()}

        return self.description_cluster

    @staticmethod
    def merge_clusters(dict1, dict2):
        """
        Merge clusters from two dictionaries where each dictionary maps items to their cluster members.
        Returns a new dictionary with merged clusters.
        """
        # Initialize disjoint set data structure
        parent = {}
        rank = {}

        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1

        # Process clusters from both dictionaries
        for d in [dict1, dict2]:
            for key, cluster in d.items():
                # Union all elements in the cluster
                for item in cluster:
                    union(key, item)

        # Build the final clusters
        clusters = {}
        all_items = set(dict1.keys())  # Both dicts have same keys

        # For each item, find all other items with the same root
        for item in all_items:
            root = find(item)
            cluster = [x for x in all_items if find(x) == root]
            clusters[item] = tuple(sorted(cluster))  # Sort for consistency

        return clusters

    def run(self, var_mapping):
        ### Run both clustering algorithms and combine them
        embeddings = {
            var_idx: var_mapping[var_idx].embedding
            for var_idx in range(len(self.tree.nodes))
            if self.tree.nodes[var_idx]
            and self.tree.nodes[var_idx].is_variable()
        }

        self.build_tree_cluster()
        self.build_descrpition_cluster(embeddings)

        # Combine clusters
        merged_clusters = self.merge_clusters(
            self.tree_cluster, self.description_cluster
        )
        merged_cluster_list = list(
            set([tuple(v) for v in merged_clusters.values()])
        )

        # Break up clusters into clusters of same type
        final_clusters = []
        for cluster in merged_cluster_list:
            types = [self.tree.nodes[i].type for i in cluster]
            sub_clusters = {t: [] for t in types}
            for i, t in zip(cluster, types):
                sub_clusters[t].append(i)

            for cluster in sub_clusters.values():
                final_clusters.append(sorted(cluster))

        final_clusters = {i: tuple(v) for v in final_clusters for i in v}
        return final_clusters
