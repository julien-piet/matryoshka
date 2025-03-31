import json
import os
import random
import time
from collections import Counter, defaultdict

import dill
import numpy as np
import regex as re
import torch

# Calculate pairwise cosine similarities between centroids
from sklearn.metrics.pairwise import cosine_similarity

from ...classes import Template
from ...genai_api.api import LLMTask
from ...utils.json import parse_json
from ...utils.logging import get_logger
from ...utils.prompts.syntax.cluster_confirmation import explanation_format
from ...utils.prompts.syntax.cluster_confirmation import (
    gen_prompt as gen_confirm_prompt,
)
from ...utils.prompts.syntax.cluster_confirmation import (
    response_schema as confirm_response_schema,
)
from ...utils.prompts.syntax.line_description import (
    gen_prompt as gen_description_prompt,
)
from .cluster_helper import FastClusterer, find_k_nearest_logs
from .cluster_helper import cluster as get_naive_representatives
from .heuristics import Heuristic


class BuildCluster(Heuristic):

    def __init__(
        self,
        tree,
        caller,
        model="gemini-2.5-flash",
        parallel_attempts=1,
        temperature=0.5,
        top_p=0.8,
        path="./saved_queries/",
        few_shot_length=4,
        values=None,
        entries_per_template=None,
        N=5,
        max_age=20000,
        line_to_match=None,
        use_description_distance=True,
        ablation_fewshot=False,
        ablation_self_correction=False,
    ) -> None:
        super().__init__(
            tree,
            "cluster",
            caller,
            model,
            parallel_attempts,
            temperature,
            top_p,
            path,
            few_shot_length,
            values,
            entries_per_template,
            line_to_match,
        )

        self.N = N
        self.max_age = max_age

        self.descriptions = {}
        self.embeddings = {}

        self.ingested_count = 0

        self.use_description_distance = use_description_distance
        self.ablation_fewshot = ablation_fewshot
        self.ablation_self_correction = ablation_self_correction

        self.desc_path = os.path.join(self.save_path, "generated_descriptions/")
        if os.path.exists(os.path.join(self.desc_path, "descriptions.tsv")):
            with open(
                os.path.join(self.desc_path, "descriptions.tsv"),
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    try:
                        line, desc = line.split("\t")
                    except ValueError:
                        continue
                    self.descriptions[line] = desc.strip()

        else:
            os.makedirs(self.desc_path, exist_ok=True)
            with open(
                os.path.join(self.desc_path, "descriptions.tsv"),
                "w",
                encoding="utf-8",
            ) as f:
                pass

        if os.path.exists(os.path.join(self.desc_path, "embeddings.pkl")):
            with open(
                os.path.join(self.desc_path, "embeddings.pkl"), "rb"
            ) as f:
                self.embeddings = dill.load(f)

        self.cache = {}  # Cache that links entries to their response
        self.slow_start_samples = 100

        self._fast_clusterer = None

    def compute_cache_embeddings(self):
        """Compute embeddings for all entries in the cache."""
        missing = []
        for idx, (lines, _, _) in self.cache.items():
            if not self.tree.templates[idx]:
                continue
            lines = lines[:5]
            missing.extend([e for e in lines if e not in self.embeddings])
        if missing:
            self.generate_description_slow_start(missing)
            self.generate_embeddings(missing)

    def get_closest_embedding(self, entries):
        """
        Find all targets ordered by cosine similarity to the mean source embedding.

        Args:
            entries: List of source entries to compare against

        Returns:
            List of tuples (index, similarity) ordered by descending similarity
        """
        # Calculate mean embedding for source entries
        source_embedding = torch.mean(
            torch.cat(
                [self.embeddings[e] for e in entries if e in self.embeddings],
                dim=1,
            ),
            dim=1,
        )

        # Collect target embeddings and their indices
        self.compute_cache_embeddings()
        templates = []
        target_embeddings = []
        for idx, (lines, _, _) in self.cache.items():
            if not self.tree.templates[idx]:
                continue
            lines = lines[:5]
            templates.append(idx)
            target_embeddings.append(
                torch.mean(
                    torch.cat([self.embeddings[e] for e in lines], dim=1), dim=1
                )
            )

        if not target_embeddings:
            return []

        # Stack target embeddings
        target_embeddings = torch.stack(target_embeddings, dim=1)

        # Calculate cosine similarity
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        similarities = cos(source_embedding.unsqueeze(1), target_embeddings)

        # Sort similarities in descending order
        sorted_similarities, sorted_indices = similarities.sort(descending=True)

        # Create list of (index, similarity) pairs
        try:
            results = [
                (templates[idx.item()], sim.item())
                for idx, sim in zip(sorted_indices, sorted_similarities)
            ]
        except:
            breakpoint()

        return results

    def get_fewshot_examples(self, entries, prefix=None, debug=False):
        """
        Find few-shot examples

        Few-shot examples are selected based on proximity with the current entries. We always chose the first handlabeled example as a few shot example.

        This is done using an embedding distance: how close is the average embedding of the entries to the average embedding of the templates

        Args:
            entries: List of entries or single entry
            prefix: Prefix object containing matching elements
            debug: Whether to print debug information

        Returns:
            List of selected template indices
        """
        if not isinstance(entries, list):
            entries = [entries]

        last_matching_element = (
            prefix.elements[-1].id if prefix and prefix.elements else 0
        )
        if last_matching_element > 0:
            templates = self.tree.get_ordered_templates(
                last_matching_element, compute_percentage=True
            )
        else:
            templates = self.tree.get_ordered_templates()

        if not templates:
            return []

        closest_embedding = self.get_closest_embedding(entries)

        embedding_sim = dict(closest_embedding)

        # Determine minimum examples per metric
        min_per_metric = self.few_shot_len - 1

        # Sort indices by similarity for each metric
        embedding_indices = sorted(
            embedding_sim.keys(), key=lambda x: embedding_sim[x], reverse=True
        )

        # Initialize selected indices set
        selected = set()

        # Add provided templates
        if "ingested_count" not in self.__dict__:
            self.ingested_count = 1
        for t in range(self.ingested_count):
            selected.add(t)

        # First, select top min_per_metric from each similarity metric
        for indices in [embedding_indices]:
            count = 0
            idx = 0
            while count < min_per_metric and idx < len(indices):
                if indices[idx] is not None and indices[idx] not in selected:
                    selected.add(indices[idx])
                    count += 1
                idx += 1

        # Fill remaining slots with best embedding similarity examples not yet selected
        remaining_slots = self.few_shot_len - len(selected)
        if remaining_slots > 0:
            idx = 0
            while len(selected) < self.few_shot_len and idx < len(
                embedding_indices
            ):
                if (
                    embedding_indices[idx] is not None
                    and embedding_indices[idx] not in selected
                ):
                    selected.add(embedding_indices[idx])
                idx += 1

        # Convert to sorted list based on embedding similarity
        examples = sorted(
            list(selected),
            key=lambda x: (
                1 if x < self.ingested_count else 0,
                (
                    float("inf")
                    if x == 0
                    else embedding_sim.get(x, float("-inf"))
                ),
            ),
            reverse=True,
        )

        return examples

    def generate_description_slow_start(self, entries, **kwargs):
        """Generate descriptions for entries in the list
        Slow start means we first sample SLOW_START_SAMPLES entries using simple metrics to be labeled with no few-shot examples, then we use these as few-shot examples for the rest
        """
        ### Generate zero-shot descriptions of "seed" examples with high quality model
        top_lines = get_naive_representatives(entries, -1)
        self.generate_description(
            [line for line, _ in top_lines],
            model=self.model,
            few_shot_set=(
                list(self.descriptions.keys())
                if len(self.descriptions) > 0
                else None
            ),
        )

        filt_entries = [e for e in entries if e not in self.descriptions]
        if filt_entries:
            self.generate_description(
                filt_entries,
                few_shot_set=(
                    list(self.descriptions.keys())
                    if len(self.descriptions) > 0
                    else None
                ),
            )

    def generate_description(self, entries, few_shot_set=None, **kwargs):
        """Generate descriptions for entries in the list"""
        if not isinstance(entries, list):
            entries = [entries]

        if self.use_description_distance:
            descriptions = []
            tasks = []
            if few_shot_set:
                few_shot_examples = [
                    [(val, self.descriptions[val]) for val, _ in close]
                    for close in find_k_nearest_logs(entries, few_shot_set)
                ]
            else:
                few_shot_examples = [[] for _ in entries]
            for entry, few_shot_example in zip(entries, few_shot_examples):
                user, history, system = gen_description_prompt(
                    entry, few_shot_example
                )
                kwargs["temperature"], kwargs["n"] = 0, 1
                if "model" not in kwargs:
                    kwargs["model"] = self.model
                if self.ablation_fewshot:
                    history = []
                tasks.append(
                    LLMTask(
                        system_prompt=system,
                        history=history,
                        message=user,
                        thinking_budget=128,
                        **kwargs,
                    )
                )

            candidates = self.caller(tasks, use_tqdm=True)
            for candidate in candidates:
                try:
                    description = json.loads(candidate.candidates[0].strip())
                except json.JSONDecodeError:
                    description = re.sub(
                        '(^")|("$)',
                        "",
                        candidate.candidates[0].strip(),
                    )
                description = re.sub(
                    r"\s+",
                    " ",
                    description,
                )
                descriptions.append(description)
        else:
            descriptions = entries[:]

        with open(
            os.path.join(self.desc_path, "descriptions.tsv"),
            "a",
            encoding="utf-8",
        ) as fd:
            for entry, description in zip(entries, descriptions):
                self.descriptions[entry] = description
                fd.write(f"{entry}\t{description}\n")

    def generate_embeddings(self, entries, **kwargs):
        """Generate embeddings for all descriptions"""
        filt_entries = [
            entry for entry in entries if entry in self.descriptions
        ]
        if not filt_entries:
            return
        # Group tasks into chunks of 200 lines
        tasks = []
        chunk_size = 100
        for i in range(0, len(filt_entries), chunk_size):
            chunk = [
                self.descriptions[entry]
                for entry in filt_entries[i : i + chunk_size]
            ]
            tasks.append(
                LLMTask(
                    message=chunk,
                    query_type="embedding",
                    model="text-embedding-005",
                )
            )

        response = self.caller(
            tasks, distribute_parallel_requests=False, use_tqdm=True
        )
        response = [emb for r in response for emb in r]
        for entry, emb in zip(filt_entries, response):
            if emb:
                self.embeddings[entry] = torch.tensor(emb).unsqueeze(1)
            else:
                del self.descriptions[entry]
        with open(os.path.join(self.desc_path, "embeddings.pkl"), "wb") as f:
            dill.dump(self.embeddings, f)

    def merge_similar_clusters(
        self, embeddings_np, labels, similarity_threshold=0.95
    ):
        """
        Merge HDBSCAN clusters whose centroids have cosine similarity above the threshold,
        properly handling transitive relationships between clusters.

        Args:
            embeddings_np: NumPy array of embeddings
            labels: HDBSCAN cluster labels
            similarity_threshold: Threshold for merging clusters (default: 0.9)

        Returns:
            New labels array with merged clusters
        """
        unique_clusters = sorted(set(labels[labels != -1]))
        if len(unique_clusters) <= 1:
            return labels

        # Calculate centroids for each cluster
        centroids = []
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings_np[cluster_mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        similarities = cosine_similarity(centroids)
        np.fill_diagonal(similarities, 0)  # Zero out self-similarities

        # Build connected components graph of similar clusters
        similar_pairs = np.where(similarities > similarity_threshold)
        connected_clusters = {}

        # Initialize each cluster in its own group
        for cluster_id in unique_clusters:
            connected_clusters[cluster_id] = {cluster_id}

        # Merge connected components
        for idx1, idx2 in zip(*similar_pairs):
            cluster1 = unique_clusters[idx1]
            cluster2 = unique_clusters[idx2]

            # Merge the sets of connected clusters
            merged_set = connected_clusters[cluster1].union(
                connected_clusters[cluster2]
            )

            # Update all clusters in the merged set to point to the same set
            for cluster_id in merged_set:
                connected_clusters[cluster_id] = merged_set

        # Create mapping using minimum cluster ID in each connected component
        cluster_mapping = {}
        for cluster_id in unique_clusters:
            cluster_mapping[cluster_id] = min(connected_clusters[cluster_id])

        # Create new labels array
        new_labels = labels.copy()
        for old_cluster, new_cluster in cluster_mapping.items():
            new_labels[labels == old_cluster] = new_cluster

        return new_labels

    def select_diverse_representatives(self, entries, point_indices, n_select):
        """
        Select diverse representatives using MMR-style selection
        considering both density and diversity
        """
        # If we have fewer points than requested representatives, return all points
        if len(point_indices) <= n_select:
            return point_indices

        filtered_entries = [entries[i] for i in point_indices]

        examples = [
            line
            for line, _ in get_naive_representatives(
                filtered_entries, n_select, preprocess_strict=False
            )
        ]

        # Map back to original indices
        return [point_indices[filtered_entries.index(e)] for e in examples]

    def cluster(
        self,
        embeddings,
        prefix_cluster,
        **kwargs,
    ):
        """
        Cluster embeddings using HDBSCAN and return clusters with their densities
        and diverse representative points

        Args:
            embeddings: Tensor of embeddings to cluster
            n_representatives: Number of representative points to select per cluster
            diversity_weight: Weight between 0 and 1 balancing density (0) vs diversity (1)

        Returns:
            list of tuples (cluster_indices, density, representative_indices) ordered by density,
            where:
            - cluster_indices is a list of indices belonging to the cluster
            - density is the cluster density
            - representative_indices is a list of the n most representative point indices
        """
        embeddings = torch.cat(embeddings, dim=1).t()
        embeddings_np = embeddings.detach().cpu().numpy()

        # Create clusterer if not exists
        if self._fast_clusterer is None:
            self._fast_clusterer = FastClusterer(embeddings=embeddings)

        # Get cluster labels
        labels = self._fast_clusterer.fit_predict(embeddings)
        labels = self.merge_similar_clusters(embeddings_np, labels)

        # Get unique cluster labels excluding noise (label -1)
        unique_clusters = sorted(set(labels[labels != -1]))

        # Initialize results list
        clusters_with_density = []

        # Calculate clusters & their densities
        for cluster_id in unique_clusters:
            # Get indices for this cluster
            prefix_agnostic_cluster_indices = np.where(labels == cluster_id)[
                0
            ].tolist()

            # Get list of included prefixes
            local_prefixes = np.unique(
                prefix_cluster[prefix_agnostic_cluster_indices]
            )

            for prefix in local_prefixes:
                # Get indices and values for this cluster
                cluster_indices = np.where(
                    (prefix_cluster == prefix) & (labels == cluster_id)
                )[0].tolist()
                clusters_with_density.append((cluster_indices, 1))

        # Sort clusters by decreasing size
        clusters_with_density.sort(key=lambda x: len(x[0]), reverse=True)

        # Add noise points as final cluster with density 0
        noise_indices = np.where(labels == -1)[0].tolist()
        if noise_indices:
            local_prefixes = np.unique(prefix_cluster[noise_indices])
            for prefix in local_prefixes:
                cluster_indices = np.where(
                    (prefix_cluster == prefix) & (labels == -1)
                )[0].tolist()
            clusters_with_density.append((cluster_indices, 0))

        return clusters_with_density

    def print_cluster(self, entries, cluster):
        print(f"Cluster of length {len(cluster[0])}:")
        samples = random.sample(cluster[0], min(5, len(cluster[0])))
        print(
            "\t"
            + "\n\t".join(
                [
                    "{}\t{}".format(entries[i], self.descriptions[entries[i]])
                    for i in samples
                ]
            )
        )

    def cluster_by_prefix(self, unparsed_entries):
        """
        Separate entries into clusters based on the matched prefix
        """
        prefix_list = self.get_prefix(unparsed_entries)
        prefix_id = defaultdict(lambda: -1)
        for prefix in prefix_list:
            if prefix not in prefix_id:
                prefix_id[prefix] = len(prefix_id)
        prefix_id_array = np.array([prefix_id[p] for p in prefix_list])

        return prefix_id_array

    def get_target_lines(self, unparsed_entries, target_entry=None, **kwargs):
        """
        Return most dense cluster of entries or cluster around target_entry, based on description embeddings.
        """

        # Generate embeddings for all missing descriptions
        get_logger().debug(
            "\t\tGetting descriptions at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        for _ in range(3):
            missing_entries = [
                e for e in unparsed_entries if e not in self.descriptions
            ]
            if missing_entries:
                self.generate_description_slow_start(missing_entries)

            missing_entries = [
                e for e in unparsed_entries if e not in self.embeddings
            ]
            if missing_entries:
                self.generate_embeddings(missing_entries)
            else:
                get_logger().debug("\t\tGot descriptions.")
                break

        # Remove missing
        unparsed_entries = [
            ue for ue in unparsed_entries if ue in self.embeddings
        ]
        # Gather all descriptions and embeddings
        embeddings = [self.embeddings[e] for e in unparsed_entries]

        # Cluster embeddings
        get_logger().debug(
            "\t\tPrefix clustering at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        prefix_clusters = self.cluster_by_prefix(unparsed_entries)

        get_logger().debug(
            "\t\tClustering at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        clusters = self.cluster(embeddings, prefix_clusters)

        get_logger().info(
            "Running confirm cluster heuristic... Found %s clusters, largest of size %s",
            len(clusters),
            max(len(c[0]) for c in clusters),
        )

        if target_entry is None:
            # Return most dense cluster
            get_logger().debug(
                "\t\tGetting representatives at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            representative_indices = self.select_diverse_representatives(
                unparsed_entries, clusters[0][0], self.N
            )
            if len(representative_indices) < self.N and len(
                representative_indices
            ) < len(clusters[0][0]):
                representative_indices += random.sample(
                    [
                        v
                        for v in clusters[0][0]
                        if v not in representative_indices
                    ],
                    min(
                        self.N - len(representative_indices),
                        len(clusters[0][0]),
                    ),
                )

            return (clusters[0][0], representative_indices, unparsed_entries)
        else:
            # Find cluster containing target_entry
            target_index = unparsed_entries.index(target_entry)
            get_logger().debug(
                "\t\tGetting representatives at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            for cluster_indices, _ in clusters:
                if target_index in cluster_indices:
                    representative_indices = (
                        self.select_diverse_representatives(
                            unparsed_entries, cluster_indices, self.N
                        )
                    )
                    return (
                        cluster_indices,
                        representative_indices,
                        unparsed_entries,
                    )

        return ([], [], unparsed_entries)

    def normalize_prefix(self, entries):
        """Replace the prefix of the entries with the same values"""
        # Find the common prefix
        prefix_per_entry = []
        for entry in entries:
            prefix = self.get_prefix(entry)
            try:
                prefix_per_entry.append(
                    [e.id for e in prefix.elements] if prefix else []
                )
            except Exception as e:
                breakpoint()

        common_elements = [
            i
            for i in prefix_per_entry[0]
            if all(i in p for p in prefix_per_entry)
        ]
        largest_element = 0 if not common_elements else max(common_elements)

        if not largest_element:
            return entries

        prefix_ids = self.tree.node_to_tree[largest_element].get_lineage()
        prefix_elements = [self.tree.nodes[i] for i in prefix_ids if i]
        prefix_template = Template(prefix_elements)

        # Get the match for the first entry
        prefix_template.generate_regex()
        prefix_regex = prefix_template.regex
        shared_prefix = re.match(prefix_regex[:-1], entries[0]).group(0)

        # Replace the prefix in all entries
        normalized_entries = []
        for entry in entries:
            normalized_entries.append(
                re.sub(prefix_regex[:-1], shared_prefix, entry)
            )

        return normalized_entries

    def query(self, entries, prefix=None, normalize=True, **kwargs):

        # Get fewshot examples
        few_shot = self.get_fewshot_examples(entries, prefix)

        # Normalize entries
        non_normalized_entries = entries[:]
        if normalize:
            entries = self.normalize_prefix(entries)
        non_shuffled_entries = entries[:]

        local_few_shot = [t for t in few_shot if t < self.ingested_count]
        if len(local_few_shot) < self.few_shot_len:
            local_few_shot += [
                t for t in range(self.ingested_count) if t not in local_few_shot
            ]

        few_shot_data = [
            self.cache[t] for t in local_few_shot
        ]  # Only take the examples that were given by the user.

        # Map element to fields
        kwargs["temperature"] = 0
        kwargs["n"] = 1

        tasks = []
        orig_entries = []
        for _ in range(3):
            random.shuffle(entries)
            user, history, system = gen_confirm_prompt(few_shot_data, entries)
            if self.ablation_fewshot:
                history = []
            tasks.append(
                LLMTask(
                    system_prompt=system,
                    model=self.model,
                    message=user,
                    history=history,
                    thinking_budget=2048,
                    **kwargs,
                )
            )
            orig_entries.append(entries[:])

        candidates = self.caller(tasks)
        valid_candidates = []
        if not isinstance(candidates, list):
            candidates = [candidates]

        result_pattern = re.compile(
            r"###\s+mapping\s+###\s+(.*?)$",
            flags=re.IGNORECASE | re.DOTALL,
        )
        demo_pattern = re.compile(
            r"###\s+explanation\s+###\s+(.*?)\s+### mapping",
            flags=re.IGNORECASE | re.DOTALL,
        )
        response_content = [
            candidate_resp
            for llm_response in candidates
            for candidate_resp in llm_response.candidates
        ]

        if not self.ablation_self_correction:
            # Correction: if any response contains "<MESSAGE>", ask the model to fix itself
            to_be_fixed = []

            for resp_idx, resp in enumerate(response_content):
                if "MESSAGE" in resp.upper():
                    to_be_fixed.append(resp_idx)

            new_tasks = []
            for idx in to_be_fixed:
                orig_task = tasks[idx]
                orig_task.update_conversation(
                    candidates[idx].candidates[0],
                    "Please do not use placeholders for messages, such as <Message>, <ErrorMessage>, <StatusMessage> or others. These placeholders are not allowed in the response. The values they represent are constants. Please fix your response and try again, by outputing your full explanation and response.",
                )
                new_tasks.append(orig_task)

            if new_tasks:
                get_logger().info("Fixing candidates with placeholders...")
                corrected_candidates = self.caller(new_tasks)
                for idx in to_be_fixed:
                    response_content[idx] = corrected_candidates.pop(
                        0
                    ).candidates[0]

        for idx, candidate in enumerate(response_content):
            result_match = result_pattern.search(candidate)
            demo_match = demo_pattern.search(candidate)
            if not result_match or not demo_match:
                continue
            try:
                mapping = list(json.loads(result_match.group(1).strip()))
            except json.JSONDecodeError:
                try:
                    mapping = list(
                        json.loads(
                            result_match.group(1).replace("```", "").strip()
                        )
                    )
                except json.JSONDecodeError:
                    try:
                        mapping = list(
                            json.loads(
                                result_match.group(1)
                                .replace("```json", "")
                                .replace("```", "")
                                .strip()
                            )
                        )
                    except json.JSONDecodeError:
                        try:
                            mapping = parse_json(
                                result_match.group(1),
                                self.caller,
                                response_schema=confirm_response_schema,
                                model=self.model,
                            )
                        except:
                            get_logger().warning(
                                "Cluster confirmation heuristic: Mapping parsing failed."
                            )
                            continue

            if len(mapping) != len(entries):
                get_logger().warning(
                    "Cluster confirmation heuristic: Mapping length mismatch."
                )
                continue

            mapping = [m.strip() for m in mapping]
            counts_per_string = Counter(mapping)
            most_common_mapping = counts_per_string.most_common(1)[0][0]
            subset = [
                i for i, m in enumerate(mapping) if m == most_common_mapping
            ]

            demonstration = demo_match.group(1).strip()
            selected = [orig_entries[idx][i] for i in subset]
            valid_candidates.append(
                (
                    selected,
                    demonstration,
                    mapping,
                )
            )

        if not valid_candidates:
            return [non_normalized_entries[0]], None, None, few_shot

        # Select most common subset
        subset_counts = Counter([tuple(sorted(c[0])) for c in valid_candidates])
        most_common_subset = subset_counts.most_common(1)[0][0]

        # Filter candidates
        most_common_subset, most_common_demo, most_common_mapping = [
            c
            for c in valid_candidates
            if tuple(sorted(c[0])) == most_common_subset
        ][0]
        most_common_subset = tuple(most_common_subset)

        self._write(
            f"{len(self.tree.templates)}_{len(entries)}",
            tasks[-1].system_prompt
            + "\n\n###########\n\n"
            + "".join(
                [
                    f"{c['role']}:\n\n" + c["content"] + "\n\n##########\n\n"
                    for c in tasks[-1].history
                ]
            )
            + "\n\n##########\n\n"
            + tasks[-1].message,
            "".join(
                [
                    f"Response #{c_idx+1}:\n\n"
                    + c.candidates[0]
                    + "\n\n##########\n\n"
                    for c_idx, c in enumerate(candidates)
                ]
            ),
            f"Subset: {most_common_subset}",
            "confirmation",
        )

        most_common_subset = [
            non_normalized_entries[non_shuffled_entries.index(e)]
            for e in most_common_subset
            if e in non_shuffled_entries
        ]
        return (
            most_common_subset,
            most_common_demo,
            most_common_mapping,
            few_shot,
        )

    def run(
        self, unparsed_entries, target_entry=None, skip_cluster=False, **kwargs
    ):
        """
        Run the heuristic to confirm cluster embedding
        """
        get_logger().info(
            "Running confirm cluster heuristic... Generating descriptions and embeddings"
        )
        unparsed_entries = [u.strip() for u in unparsed_entries]
        target_entry = target_entry.strip() if target_entry else None
        get_logger().debug(
            "\tGetting target lines at time %s",
            time.strftime("%H:%M:%S", time.localtime()),
        )
        _, representative_line_indices, unparsed_entries = (
            self.get_target_lines(unparsed_entries, target_entry, **kwargs)
        )
        representative_lines = list(
            set([unparsed_entries[i] for i in representative_line_indices])
        )

        # Dedup representative lines according to shared suffixes
        representative_suffixes = self.get_suffix(representative_lines)
        first_index, seen_suffixes = [], {}
        for idx, suffix in enumerate(representative_suffixes):
            if suffix not in seen_suffixes:
                seen_suffixes[suffix] = idx
                first_index.append(seen_suffixes[suffix])
        representative_lines = list(
            set([representative_lines[i] for i in first_index])
        )

        orig_representative_lines = representative_lines[:]
        prefix = self.get_prefix(representative_lines[0])

        if skip_cluster:
            representative_lines = representative_lines[:1]

        get_logger().info(
            "Running confirm cluster heuristic... Found cluster:\n%s",
            json.dumps(representative_lines, indent=2),
        )
        if len(representative_lines) == 1:
            examples = self.get_fewshot_examples(representative_lines, prefix)
            to_be_saved, explanation, raw_output = False, "", ""
        else:
            get_logger().debug(
                "\tQuerying model to confirm cluster at time %s",
                time.strftime("%H:%M:%S", time.localtime()),
            )
            representative_lines, explanation, raw_output, examples = (
                self.query(representative_lines, prefix, **kwargs)
            )
            to_be_saved = True

        get_logger().info(
            "Running confirm cluster heuristic... Confirmed subset is: \n%s",
            json.dumps(representative_lines, indent=2),
        )

        for lines, _, _ in self.cache.values():
            unmapped = [
                l
                for l in lines[:5]
                if l not in self.descriptions or l not in self.embeddings
            ]
            if unmapped:
                get_logger().debug(
                    "\tGenerating additional descriptions at time %s",
                    time.strftime("%H:%M:%S", time.localtime()),
                )
                self.generate_description_slow_start(lines)
                self.generate_embeddings(lines)

        return (
            representative_lines,
            self.get_prefix(representative_lines[0]),
            examples,
            self.descriptions[representative_lines[0]],
            orig_representative_lines,
            explanation,
            raw_output,
            to_be_saved,
        )

    def save_to_cache(self, t_id, entries, explanation, raw_output):
        entries = list(entries)[:5]
        self.cache[t_id] = (
            entries,
            explanation,
            raw_output,
        )

    def ingest_fewshot(
        self,
        template_ids,
        seeds,
    ):

        entries = [d["examples"] for d in seeds]
        placeholders = [d["format"] for d in seeds]
        constants = [d["constants"] for d in seeds]
        variables = [d["variables"] for d in seeds]
        mappings = [[d["format"] for _ in d["examples"]] for d in seeds]
        self.ingested_count = len(self.tree.templates)

        """Ingest fewshot data into cache"""
        self.generate_description_slow_start(
            [e for entries in entries for e in entries]
        )
        self.generate_embeddings([e for entries in entries for e in entries])
        for (
            template_id,
            local_entries,
            placeholder,
            local_constants,
            local_variables,
            mapping,
        ) in zip(
            template_ids, entries, placeholders, constants, variables, mappings
        ):
            desc = self.descriptions[local_entries[0]]
            explanation = explanation_format.format(
                desc=desc,
                placeholder=placeholder,
                constants="\n".join(local_constants),
                variables="\n".join(local_variables),
                placeholder_list="\n".join(mapping),
            )
            self.cache[template_id] = (local_entries[:], explanation, mapping)

        return [
            self.descriptions[local_entries[0]] for local_entries in entries
        ]
