import math
import copy
from collections import Counter

import nltk
import re

def read_cacm(path):
    """Reads CACM File and extracts the ID (I), Title (T), Authors (A) and Summary (W) (if present) of all the
    documents in a dictionary"""
    with open(path, 'r') as f:
        data = f.read()
    l = re.findall(r'(\.I(.|\n)+?(?=(\n\.I|$)))', data)
    l = [x[0] for x in l]
    r1 = r'\.(I) (\d+)'
    r2 = r'\.(T)\n((.|\n)+?)(?=(\n\.|$))'
    r3 = r'\.(A)\n((.|\n)+?)(?=(\n\.|$))'
    r4 = r'\.(W)\n((.|\n)+?)(?=(\n\.|$))'
    r = r'{}|{}|{}|{}'.format(r1,r2,r3,r4)

    dictionary = {}
    for doc in l:
        x = re.findall(r, doc)
        i = 0
        id = None
        while i < len(x):
            x[i] = tuple(filter(len, x[i]))[:2]
            if x[i][0] == 'I':
                id = int(x[i][1])
                x.pop(i)
                i -= 1
            i += 1
        dictionary[id] = dict(x)
    return dictionary

def preprocess_cacm(dictionary : dict):
    """Preprocess CACM dictionary inplace : lower + remove stopwords + Count frequencies"""
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for k in dictionary:
        s = ' '.join(dictionary[k].values()).lower()
        s = re.findall(r'\w+', s)
        s = [x for x in s if x not in stop_words]
        s = dict(Counter(s))
        dictionary[k] = s
    return dictionary

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

