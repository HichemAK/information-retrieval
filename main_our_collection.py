import os
import re
from collections import Counter

from utils import TermDocumentDict

with open('stopwords_fr.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read()

stop_words = set(stop_words.split("\n"))

collection_path = 'collection-RI'
documents = {}
for s in os.listdir(collection_path):
    with open(os.path.join(collection_path, s), 'rb') as f:
        text = f.read()
        try:
            text = text.decode()
        except UnicodeDecodeError:
            text = text.decode('ansi')
        text = text.lower()
        spl = re.split(r'\W+', text)
        if len(spl[0]) == 0 : spl.pop(0)
        if len(spl[-1]) == 0 : spl.pop(-1)
        spl = [x for x in spl if x not in stop_words]
        documents[s] = spl

for k in documents:
    c = Counter(documents[k])
    documents[k] = dict(c)

struct = TermDocumentDict(documents)
weight = struct.weight()