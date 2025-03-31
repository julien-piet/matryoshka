import re
from collections import defaultdict
from typing import List, Set, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from tqdm import tqdm

from ...utils.logging import get_logger


def preprocess_line(line: str, strict=True) -> Set[str]:
    """
    Preprocess a log line by:
    1. Converting to lowercase
    2. Removing numbers and special characters
    3. Splitting into words
    4. Converting to a set of unique words
    """
    # Remove numbers and special characters, keep only letters and spaces
    if strict:
        cleaned = re.sub(r"[^a-zA-Z\s]", "", line.lower().strip())
    else:
        cleaned = line.lower().strip()
    # Split into words and convert to set
    separators = r"[.\-/_:\\\s]+"
    cleaned = [c for c in re.split(separators, cleaned) if c.strip()]
    return set(cleaned)


def calculate_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets of words.
    Jaccard similarity = size of intersection / size of union
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def dbscan_run(distance_matrix, eps):
    # Use DBSCAN to cluster the log lines
    dbscan = DBSCAN(metric="precomputed", min_samples=2, eps=eps)
    labels = dbscan.fit_predict(distance_matrix)

    # Collect clusters
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(idx)

    return clusters


def cluster(
    log_lines: List[str],
    n_clusters: int = 5,
    bin_search_depth=10,
    preprocess_strict=True,
) -> List[Tuple[str, int]]:
    """
    Cluster log lines using K-medoids clustering and return representative logs with their cluster sizes.

    Args:
        log_lines: List of log lines to cluster
        n_clusters: Number of clusters to create

    Returns:
        List of tuples containing (representative_log, cluster_size)
    """
    if not log_lines:
        return []

    if len(log_lines) <= n_clusters:
        return [(line, 1) for line in log_lines]

    # Preprocess all lines
    preprocessed_lines = [
        preprocess_line(line, strict=preprocess_strict) for line in log_lines
    ]

    # Calculate distance matrix
    n = len(log_lines)
    distances = np.zeros((n, n))
    for i in tqdm(range(n), desc="Naive clustering", total=n, disable=True):
        for j in range(i + 1, n):
            similarity = calculate_similarity(
                preprocessed_lines[i], preprocessed_lines[j]
            )
            distance = 1 - similarity  # Convert similarity to distance
            distances[i][j] = distance
            distances[j][i] = distance

    if n_clusters > 0:
        # Binary Search for the optimal eps
        left, right = 0, 1
        for _ in tqdm(
            range(bin_search_depth),
            total=bin_search_depth,
            desc="Binary search over DBSCAN eps parameter",
        ):
            mid = (left + right) / 2
            clusters = dbscan_run(distances, mid)
            if len(clusters) > n_clusters:
                left = mid
            else:
                right = mid
    else:
        # Use a fixed eps
        clusters = dbscan_run(distances, 0.5)

    # Find representative log for each cluster
    result = []
    for cluster_indices in clusters.values():
        if not cluster_indices:
            continue

        # Choose the log line that is closest to all other lines in the cluster
        min_distance = float("inf")
        representative_idx = cluster_indices[0]
        for idx in cluster_indices:
            distance = sum(
                distances[idx][other_idx] for other_idx in cluster_indices
            )
            if distance < min_distance:
                min_distance = distance
                representative_idx = idx

        result.append((log_lines[representative_idx], len(cluster_indices)))

    result.sort(key=lambda x: x[1], reverse=True)

    return result[:n_clusters]


def find_k_nearest_logs(
    source_lines: List[str], target_lines: List[str], k: int = 3
) -> List[List[Tuple[str, float]]]:
    """
    For each source log line, find the k most similar lines from the target lines.

    Args:
        source_lines: List of log lines to find matches for
        target_lines: List of log lines to search within
        k: Number of nearest neighbors to return for each source line

    Returns:
        List of lists, where each inner list contains k tuples of (target_line, similarity_score)
        for the corresponding source line
    """
    # Preprocess all lines
    processed_sources = [
        (i, preprocess_line(line)) for i, line in enumerate(source_lines)
    ]
    processed_targets = [
        (i, preprocess_line(line)) for i, line in enumerate(target_lines)
    ]

    results = []

    # For each source line
    for _, source_words in processed_sources:
        # Calculate similarity with all target lines
        similarities = []
        for target_idx, target_words in processed_targets:
            similarity = calculate_similarity(source_words, target_words)
            similarities.append((target_lines[target_idx], similarity))

        # Sort by similarity in descending order and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results.append(similarities[:k])

    return results


class FastClusterer:
    """
    Fast clustering implementation using DBSCAN with BallTree for efficient
    nearest neighbor calculations. Much faster than HDBSCAN while still
    being density-based.
    """

    def __init__(
        self,
        eps: float = 0.25,
        min_samples: int = 2,
        embeddings: torch.Tensor = None,
    ):
        if embeddings is not None:
            # eps = self.estimate_eps(embeddings)
            eps = 0.05
            get_logger().info(f"Estimated eps: {eps}")

        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="ball_tree",
        )

    def fit_predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Cluster embeddings using DBSCAN.
        Returns cluster labels.
        """
        # Convert to numpy
        embeddings_np = embeddings.detach().cpu().numpy()

        # Run DBSCAN clustering
        labels = self.dbscan.fit_predict(embeddings_np)

        return labels

    def estimate_eps(
        self,
        embeddings: torch.Tensor,
        n_neighbors: int = 15,
        dedup: bool = False,
    ) -> float:
        """
        Estimate eps parameter for DBSCAN using nearest neighbor distances.
        Uses the standard method of finding the knee/elbow in the k-nearest neighbor distance curve.
        Removes duplicate embeddings that are closer than 1e-3 to previous embeddings.

        The knee point typically indicates the distance threshold that separates clusters.
        """
        # First deduplicate embeddings
        embeddings_np = embeddings.detach().cpu().numpy()
        if len(embeddings_np) > 1 and dedup:
            # Build tree for initial set
            tree = BallTree(embeddings_np[:1].reshape(1, -1))
            keep_indices = [0]  # Always keep first embedding

            # Check each subsequent embedding
            for i in range(1, len(embeddings_np)):
                # Get distance to nearest neighbor
                dist, _ = tree.query(embeddings_np[i].reshape(1, -1), k=1)
                if dist[0][0] >= 1e-4:  # Not a duplicate
                    keep_indices.append(i)
                    # Update tree with the new unique embedding
                    tree = BallTree(embeddings_np[keep_indices])

            # Keep only unique embeddings
            embeddings_np = embeddings_np[keep_indices]

        abs_min = 0.05

        # Need enough neighbors to get reliable density estimate
        n_neighbors = min(n_neighbors, len(embeddings_np) - 1)
        if n_neighbors < 2:
            # Not enough points for meaningful clustering
            return 0.1  # Conservative default

        # Build ball tree
        tree = BallTree(embeddings_np)

        # Get distances to k neighbors for each point
        distances, _ = tree.query(embeddings_np, k=n_neighbors + 1)

        # Use the k-th nearest neighbor distance for each point
        k_distances = np.sort(distances[:, -1])

        # If too few points or all distances very small, use conservative estimate
        if len(k_distances) < 4 or np.mean(k_distances) < 1e-6:
            return max(float(np.percentile(distances.flatten(), 75)), abs_min)

        # Calculate moving average to smooth curve
        window = max(2, len(k_distances) // 10)
        smoothed = np.convolve(
            k_distances, np.ones(window) / window, mode="valid"
        )

        # Find the point of maximum curvature in the smoothed curve
        # This is typically where the distance starts growing rapidly
        diffs = np.diff(smoothed)
        acceleration = np.diff(diffs)
        knee_idx = np.argmax(acceleration) + 1

        # Use the corresponding distance as eps
        if knee_idx >= len(k_distances):
            knee_idx = len(k_distances) - 1
        eps = float(k_distances[knee_idx])

        # Ensure reasonable bounds
        min_eps = min(
            np.percentile(distances.flatten(), 10), abs_min
        )  # Conservative lower bound
        max_eps = max(
            np.percentile(distances.flatten(), 90), 0.9
        )  # Conservative upper bound
        eps = min(max(eps, min_eps), max_eps)

        return eps
