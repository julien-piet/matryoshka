import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..classes import Template


class Embedding(ABC):
    """
    Abstract class for embedding models
    """

    def __init__(self, filt) -> None:
        self.filt = filt

    @abstractmethod
    def _embed(self, sentences: Union[list[str], str]) -> np.array:
        pass

    def _tokenize(self, text):
        rtn = []
        for sentence in text:
            sentence = re.sub(r"['\"\`]", " ", sentence)
            tokens = re.split(r"[\[\]\(\)\{\}.:|,;\s]", sentence)
            filt_tokens = []
            for tok in tokens:
                if re.match(r"^[0-9\$]+$", tok):
                    continue
                filt_tokens.append(tok)
            rtn.append(" ".join([f for f in filt_tokens if f]))

        return rtn

    def __call__(self, sentences: Union[list[str], str]) -> np.array:
        return self._embed(sentences)


class Distance(ABC):
    """
    Abstract class for distance models
    """

    def __init__(self) -> None:
        self.templates = {}

    @abstractmethod
    def _embed(self, sentences: Union[list[str], str]):
        pass

    def add_template(self, index, template):
        self.templates[index] = self._embed(template.example_entry)

    def get_template(self, entry, k=-1, templates=None, debug=False):
        embedding = self._embed(entry)
        distances = {
            k: np.linalg.norm(v - embedding) for k, v in self.templates.items()
        }
        ordered_templates = sorted(distances.items(), key=lambda x: x[1])

        if templates:
            ordered_templates = [
                v for v in ordered_templates if v[0] in templates
            ]
        if k > 0:
            return ordered_templates[:k]
        else:
            return ordered_templates

    def init_embedding(self, model):
        pass

    def __call__(self, sentence: str, k: int = -1, templates=None, debug=False):
        return self.get_template(sentence, k, templates=templates, debug=debug)


@dataclass
class ElementTokenCounter:
    full_tokens: dict = field(default_factory=dict)
    partial_tokens: dict = field(default_factory=dict)
    observed_count: int = 0

    def get(self):
        return {
            t: v / self.observed_count for t, v in self.full_tokens.items()
        }, {t: v / self.observed_count for t, v in self.partial_tokens.items()}


@dataclass
class TemplateElementCounter:
    full_tokens: dict = field(default_factory=dict)
    full_token_sum: list = field(default_factory=list)
    partial_tokens: dict = field(default_factory=dict)
    partial_token_sum: list = field(default_factory=list)
    count_per_token: list = field(default_factory=list)
    element_ids: list = field(default_factory=list)

    def simplify(self, thsd=0.2):
        if len(self.full_tokens) < 100:
            return
        self.full_tokens = {
            k: v
            for k, v in self.full_tokens.items()
            if sum(
                [
                    w / self.count_per_token[w_idx] if w else 0
                    for w_idx, w in enumerate(v)
                ]
            )
            > thsd
        }
        self.partial_tokens = {
            k: v
            for k, v in self.partial_tokens.items()
            if sum(
                [
                    w / self.count_per_token[w_idx] if w else 0
                    for w_idx, w in enumerate(v)
                ]
            )
            > thsd
        }

    def reset(
        self,
        element_ids: list,
        all_elements: dict,
    ):
        self.element_ids = element_ids
        self.full_tokens = {}
        self.full_token_sum = [0 for _ in range(len(element_ids))]
        self.partial_tokens = {}
        self.partial_token_sum = [0 for _ in range(len(element_ids))]
        self.count_per_token = [0 for _ in range(len(element_ids))]

        for elt_idx, elt_id in enumerate(element_ids):
            elt_full, elt_part, ct = (
                all_elements[elt_id].full_tokens,
                all_elements[elt_id].partial_tokens,
                all_elements[elt_id].observed_count,
            )
            for key, value in elt_full.items():
                if key not in self.full_tokens:
                    self.full_tokens[key] = [0 for _ in range(len(element_ids))]
                self.full_tokens[key][elt_idx] += value
                self.full_token_sum[elt_idx] += value

            for key, value in elt_part.items():
                if key not in self.partial_tokens:
                    self.partial_tokens[key] = [
                        0 for _ in range(len(element_ids))
                    ]
                self.partial_tokens[key][elt_idx] += value
                self.partial_token_sum[elt_idx] += value

            self.count_per_token[elt_idx] = ct

    def update(self, elt_id, full_value, partial_values):
        if elt_id not in self.element_ids:
            return
        index = self.element_ids.index(elt_id)
        self.count_per_token[index] += 1

        if full_value not in self.full_tokens:
            self.full_tokens[full_value] = [
                0 for _ in range(len(self.element_ids))
            ]
        self.full_tokens[full_value][index] += 1
        self.full_token_sum[index] += 1

        for kw in partial_values:
            if kw not in self.partial_tokens:
                self.partial_tokens[kw] = [
                    0 for _ in range(len(self.element_ids))
                ]
            self.partial_tokens[kw][index] += 1
            self.partial_token_sum[index] += 1


class NaiveDistance(Distance):

    def __init__(self) -> None:
        super().__init__()
        self.elements = {}
        self.template_embeddings = {}
        self.replace_regex = re.compile(r"[^a-zA-Z0-9]+")

    def add_template(self, index: int, template: Template):
        self.templates[index] = [t.id for t in template.elements]
        for t in template.elements:
            if t.id not in self.elements:
                self.elements[t.id] = ElementTokenCounter()
        self.template_embeddings[index] = TemplateElementCounter()
        self.template_embeddings[index].reset(
            self.templates[index], self.elements
        )

    def update(self, match):
        for elt in match.elements:
            if elt.id not in self.elements:
                self.elements[elt.id] = ElementTokenCounter()

            self.elements[elt.id].observed_count += 1

            # Compute values
            full_value = elt.value.strip()
            partial_values = [
                kw.strip()
                for kw in re.split(self.replace_regex, elt.value)
                if kw.strip()
            ]

            # Save elements
            self.elements[elt.id].full_tokens[full_value] = (
                self.elements[elt.id].full_tokens.get(full_value, 0) + 1
            )
            for kw in partial_values:
                self.elements[elt.id].partial_tokens[kw] = (
                    self.elements[elt.id].partial_tokens.get(kw, 0) + 1
                )

            # Update template elements
            for t_emb in self.template_embeddings.values():
                t_emb.update(elt.id, full_value, partial_values)

    def get_template(self, entries, k=-1, templates=None, debug=False):
        score = {}

        if not isinstance(entries, list):
            entries = [entries]

        if templates:
            selected_templates = [k for k in self.templates if k in templates]
        else:
            selected_templates = [k for k in self.templates]

        template_embeddings = {}
        for t_idx in selected_templates:
            t_emb = self.template_embeddings[t_idx]
            t_emb.simplify()
            template_embeddings[t_idx] = (
                t_emb.full_tokens,
                t_emb.full_token_sum,
                t_emb.partial_tokens,
                t_emb.partial_token_sum,
                t_emb.count_per_token,
            )

        for key, (
            full,
            full_sum,
            partial,
            partial_sum,
            counts,
        ) in template_embeddings.items():

            def find_occurrences(entry, word):
                indices, start = [], 0
                while start < len(entry):
                    index = entry.find(word, start)
                    if index != -1:
                        indices.append(index)
                        start = index + 1
                    else:
                        break
                return indices

            def get_similarity(entry, words, word_sum, counts):
                mask = [0 for _ in range(len(entry))]
                accounted_weight = 0
                for word, weights in words.items():
                    if word in entry:
                        weight = sum(
                            [
                                w / counts[w_idx] if w else 0
                                for w_idx, w in enumerate(weights)
                            ]
                        )
                        accounted_weight += weight
                        for i in find_occurrences(entry, word):
                            for j in range(len(word)):
                                mask[i + j] = max(mask[i + j], weight)

                score = sum(mask) / len(entry) if len(entry) > 0 else 0
                total_weight = sum(
                    w / c if w else 0 for w, c in zip(word_sum, counts)
                )
                if total_weight > 0:
                    score *= accounted_weight / total_weight
                return score

            full_ct = sum(
                get_similarity(entry, full, full_sum, counts)
                for entry in entries
            ) / len(entries)
            part_ct = sum(
                get_similarity(entry, partial, partial_sum, counts)
                for entry in entries
            ) / len(entries)

            if debug:
                breakpoint()

            score[key] = 0.5 * (full_ct + part_ct)

        ordered_templates = sorted(
            score.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        if k > 0:
            return ordered_templates[:k]
        else:
            return ordered_templates

    def template_distance(self, template_id, k=-1, templates=None, debug=False):
        score = {}

        if templates:
            selected_templates = [k for k in self.templates if k in templates]
        else:
            selected_templates = [k for k in self.templates]

        template_embeddings = {}
        for t_idx in selected_templates:
            t_emb = self.template_embeddings[t_idx]
            t_emb.simplify()
            template_embeddings[t_idx] = (
                t_emb.full_tokens,
                t_emb.full_token_sum,
                t_emb.partial_tokens,
                t_emb.partial_token_sum,
                t_emb.count_per_token,
            )

        target_template_emb = self.template_embeddings[template_id]

        for key, (
            full,
            full_sum,
            partial,
            partial_sum,
            counts,
        ) in template_embeddings.items():

            def get_similarity(
                words_a, words_b, word_sum_a, word_sum_b, counts_a, counts_b
            ):
                accounted_weight = 0
                for word, weights_a in words_a.items():
                    if word in words_b:
                        weights_b = words_b[word]
                        weight = sum(
                            [
                                (w_a + w_b) / (c_a + c_b) if (c_a + c_b) else 0
                                for w_a, w_b, c_a, c_b in zip(
                                    weights_a, weights_b, counts_a, counts_b
                                )
                            ]
                        )
                        accounted_weight += weight

                total_weight_a = sum(
                    w / c if w else 0 for w, c in zip(word_sum_a, counts_a)
                )
                total_weight_b = sum(
                    w / c if w else 0 for w, c in zip(word_sum_b, counts_b)
                )
                tw = total_weight_a + total_weight_b
                if tw:
                    return accounted_weight / tw
                else:
                    return 0

            full_ct = get_similarity(
                target_template_emb.full_tokens,
                full,
                target_template_emb.full_token_sum,
                full_sum,
                target_template_emb.count_per_token,
                counts,
            )
            part_ct = get_similarity(
                target_template_emb.partial_tokens,
                partial,
                target_template_emb.partial_token_sum,
                partial_sum,
                target_template_emb.count_per_token,
                counts,
            )

            score[key] = 0.5 * (full_ct + part_ct)

        ordered_templates = sorted(
            score.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        if k > 0:
            return ordered_templates[:k]
        else:
            return ordered_templates

    def _embed(self, sentences: Union[list[str], str]):
        raise NotImplementedError


class DescriptionDistance:

    def __init__(self, clusters, embeddings):
        self.clusters = list(set([tuple(v) for v in clusters.values()]))
        self.raw_embeddings = embeddings
        self.embeddings = [
            torch.stack([self.raw_embeddings[i] for i in cluster]).mean(dim=0)
            for cluster in self.clusters
        ]
        self.embeddings_vector = torch.stack(self.embeddings)

    def rank(self, source_embedding, k=-1):
        sim = torch.nn.CosineSimilarity()(
            source_embedding, self.embeddings_vector
        )

        if k == -1:
            sorted_clusters = sorted(
                [
                    (self.clusters[i], sim[i].item())
                    for i in range(len(self.clusters))
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_clusters

        else:
            closest = sim.topk(
                k=min(self.embeddings_vector.shape[0], k), largest=True
            ).indices
            return [(self.clusters[i], sim[i].item()) for i in closest]
