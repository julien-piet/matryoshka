from __future__ import annotations

import math
from typing import Any, List, Optional, Union


class Value:
    def __init__(self, element_id, values=None):
        self.element_id = element_id

        self.values = []
        self.value_counts = {}
        self.running_sum = None

        if values is not None:
            for v in values:
                self.append(v)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def append(self, value: str):
        self.values.append(value)
        if value not in self.value_counts:
            self.value_counts[value] = 1
        else:
            self.value_counts[value] += 1

    def __len__(self):
        return len(self.values)

    def get_closest(
        self,
        values: List[Value],
        metrics: Optional[Union[List[str], str]] = None,
        k=5,
        tree=None,
        embedding=None,
    ):
        if not values:
            return []

        overall = False
        if metrics is None:
            metrics = [
                "intersection",
                "cousins",
                "template_embedding",
            ]
            overall = True

        if not isinstance(metrics, list):
            metrics = [metrics]

        # Compute all the metrics
        distances = [[0] * len(metrics) for _ in values]
        for m_idx, metric in enumerate(metrics):
            if metric not in [
                "intersection",
                "cousins",
                "template_embedding",
            ]:
                raise ValueError(f"Unknown value closeness metric: {metric}")
            elif metric == "intersection":
                dist = self.get_most_common(values)
            elif metric == "cousins":
                dist = self.get_cousins(values, tree)
            else:
                dist = self.get_closest_template_embedding(
                    values, tree, embedding
                )

            for i, (_, d) in enumerate(dist):
                distances[i][m_idx] = d

        # Augment values
        values = [(v, d) for v, d in zip(values, distances)]

        # Rank and return
        if not overall:
            values = sorted(values, key=lambda x: x[1])
            return [v[0] for v in (values[:k] if k > 0 else values)]
        else:
            # Custom ranking
            selections = [
                (0,),
                (2,),
                (1, 2),
            ]
            selected, selected_idx = [], []

            for s_idx, s in enumerate(selections):
                ct = (
                    k // len(selections)
                    if s_idx >= k % len(selections)
                    else k // len(selections) + 1
                )
                ranked = sorted(
                    values,
                    key=lambda x: tuple(x[1][i] for i in s),  # noqa: E741
                )
                if 1 in s:
                    ranked = [r for r in ranked if r[1][1] != math.inf]
                ranked = [
                    r[0] for r in ranked if r[0].element_id not in selected_idx
                ][:ct]
                for r in ranked:
                    selected.append(r)
                    selected_idx.append(r.element_id)

            return selected

    def get_most_common(self, values: List[Value]):
        augmented_values = []
        for v in values:
            intersection_keys = self.value_counts.keys() & v.value_counts.keys()
            if len(intersection_keys) > 0:
                weight = sum(
                    self.value_counts[k] for k in intersection_keys
                ) + sum(v.value_counts[k] for k in intersection_keys)
                weight /= len(self) + len(v)
            else:
                weight = -math.inf
            augmented_values.append((v, -weight))

        return augmented_values

    def get_cousins(self, values: List[Value], tree: Any):
        """Order other variables by degree of separation on tree"""
        return [
            (v, -tree.degree_of_separation(self.element_id, v.element_id))
            for v in values
        ]

    def get_closest_template_embedding(
        self, values: List[Value], tree: Any, embedding: Any
    ):
        """Order other variables by template embedding distance"""

        relevant_templates = [
            t for v in values for t in tree.templates_per_node[v.element_id]
        ]

        distances = {t: 0 for t in relevant_templates}
        for t in tree.templates_per_node[self.element_id]:
            for t_id, dist in embedding.template_distance(
                t, templates=relevant_templates
            ):
                distances[t_id] += dist

        distance_per_value = {v.element_id: [0, 0] for v in values}

        for v in values:
            for t in tree.templates_per_node[v.element_id]:
                distance_per_value[v.element_id][0] += distances[t]
                distance_per_value[v.element_id][1] += 1

        distance_per_value = {
            v: d[0] / d[1] for v, d in distance_per_value.items() if d[1] > 0
        }

        return [
            (
                v,
                (
                    -distance_per_value[v.element_id]
                    if v.element_id in distance_per_value
                    else 0
                ),
            )
            for v in values
        ]
