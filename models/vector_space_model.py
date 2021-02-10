import math
from collections import Counter

from utils import TermDocumentDict
import re
import numpy as np
from enum import Enum, auto


class SimilarityFunctions(Enum):
    DOT = auto()
    DICE = auto()
    COSINUS = auto()
    JACCARD = auto()


class VectorSpaceModel:
    def __init__(self, dictionary):
        self._dict = TermDocumentDict(dictionary)
        self._dict.weight_inplace()
        self._norm_documents = {doc: np.linalg.norm(list(self._dict.terms(doc).values())) for doc in
                                self._dict.all_documents}
        self._sim = {
            SimilarityFunctions.DOT: self._eval_dot_product,
            SimilarityFunctions.COSINUS: self._eval_cosinus,
            SimilarityFunctions.DICE: self._eval_dice,
            SimilarityFunctions.JACCARD: self._eval_jaccard
        }

    def _eval_dot_product(self, l):
        document_scores = {}
        for d in self._dict.all_documents:
            document_scores[d] = 0
            for t, w in l.items():
                document_scores[d] += w*self._dict[t,d]
        return document_scores

    def _eval_dice(self, l):
        l_norm_square = (np.array(list(l.values()))**2).sum()
        document_scores = self._eval_dot_product(l)
        document_scores = {k: 2 * v / (self._norm_documents[k] ** 2 + l_norm_square) for k, v in document_scores.items()}
        return document_scores

    def _eval_cosinus(self, l):
        l_norm_square = (np.array(list(l.values())) ** 2).sum()
        document_scores = self._eval_dot_product(l)
        document_scores = {k: v / (self._norm_documents[k] * math.sqrt(l_norm_square)) for k, v in document_scores.items()}
        return document_scores

    def _eval_jaccard(self, l):
        l_norm_square = (np.array(list(l.values())) ** 2).sum()
        document_scores = self._eval_dot_product(l)
        document_scores = {k: v / (self._norm_documents[k] ** 2 + l_norm_square - v) for k, v in document_scores.items()}
        return document_scores

    def eval(self, query, similarity_function=SimilarityFunctions.DOT):
        query_terms = re.findall(r'\w+', query)
        query_terms = Counter(query_terms)
        max_freq = max(query_terms.values())
        query_terms = {k:v/max_freq for k,v in query_terms.items()}
        return self._sim[similarity_function](query_terms)
