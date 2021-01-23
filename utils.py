import math
import copy

class TermDocumentDict:
    def __init__(self, dictionary: dict = None):
        self._dict = {}
        self.N = len(dictionary)
        for k in dictionary:
            for term, value in dictionary[k].items():
                if term not in self._dict:
                    self._dict[term] = {}
                self._dict[term][k] = value

    def documents(self, term):
        """Get all the docuemnts that contains the term with their value"""
        if term not in self._dict:
            return {}
        return self._dict[term]

    def terms(self, document):
        """Get all the terms that are in the document with their value"""
        terms = {}
        for term, d in self._dict.items():
            if document in d:
                terms[term] = d[document]
        return terms

    def add(self, term, document, value):
        if term not in self._dict:
            self._dict[term] = {}
        self._dict[term][document] = value

    def __getitem__(self, key):
        term, document = key
        if term not in self._dict:
            return 0
        d = self._dict[term]
        if document not in d:
            return 0
        return d[document]

    def weight_inplace(self):
        for k in self._dict:
            term = self._dict[k]
            ni = len(term)
            sup = max(term.values())
            for document in term:
                term[document] = term[document] / sup * math.log10(self.N / ni + 1)

    def weight(self):
        c = copy.deepcopy(self)
        c.weight_inplace()
        return c

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return str(self)
