import math
from collections import Counter

import nltk

from utils import TermDocumentDict
import re
import numpy as np
from enum import Enum, auto
import scipy.sparse


class SimilarityFunctions(Enum):
    DOT = auto()
    DICE = auto()
    COSINUS = auto()
    JACCARD = auto()


class VectorSpaceModel:
    def __init__(self, dictionary, sparse=False):
        self._dict = TermDocumentDict(dictionary)
        self._dict.weight_inplace()
        self._norm_documents = {doc: np.linalg.norm(list(self._dict.terms(doc).values())) for doc in
                                self._dict.all_documents}
        if sparse:
            self.tokens = {x: i for i,x in enumerate(list(self._dict.dict.keys()))}
            self.docs = {x:i for i,x in enumerate(list(self._dict.all_documents))}
            self.rdocs = {x: i for i, x in self.docs.items()}
            row, column, data = [], [], []
            for t, docs in self._dict.dict.items():
                for d, v in docs.items():
                    row.append(self.tokens[t])
                    column.append(self.docs[d])
                    data.append(v)
            self._dict = scipy.sparse.csr_matrix((data, (row, column)),
                                                 shape=(len(self._dict.dict), len(self._norm_documents)),
                                                 dtype=np.double)
            self._eval_dot_product = self._eval_dot_product_sparse

        self._sim = {
            SimilarityFunctions.DOT: (self._eval_dot_product, 10),
            SimilarityFunctions.COSINUS: (self._eval_cosinus, 9),
            SimilarityFunctions.DICE: (self._eval_dice, 7),
            SimilarityFunctions.JACCARD: (self._eval_jaccard, 7)
        }

        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.sparse = sparse

    def _eval_dot_product_sparse(self, l):
        l = list(l.items())
        l = [(x,y) for x,y in l if x in self.tokens]
        row = (0,) * len(l)
        column = [self.tokens[x] for x, _ in l]
        data = [x for _, x in l]
        v = scipy.sparse.csr_matrix((data, (row, column)), shape=(1, len(self.tokens)), dtype=np.double)
        v = v @ self._dict
        data = v.data.tolist()
        coord = v.nonzero()[1]
        v = {self.rdocs[coord[i]] : data[i] for i in range(len(coord))}
        return v

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

    def eval(self, query, similarity_function=SimilarityFunctions.DOT, k=None):
        query = query.lower()
        query_terms = re.findall(r'\w+', query)
        query_terms = [x for x in query_terms if x not in self.stopwords]
        query_terms = Counter(query_terms)
        max_freq = max(query_terms.values())
        query_terms = {k:v/max_freq for k,v in query_terms.items()}
        sim, k_pred = self._sim[similarity_function]
        res = sim(query_terms)
        res = sorted([(k,v) for k,v in res.items()], reverse=True, key=lambda x : x[1])
        if k is None:
            return res[:k_pred]
        else:
            return res[:k]

