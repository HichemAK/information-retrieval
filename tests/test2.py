from models.vector_space_model import VectorSpaceModel, SimilarityFunctions
from utils import read_cacm_query, read_cacm, preprocess_cacm

dictionary = read_cacm('../CACM/cacm.all')
dictionary = preprocess_cacm(dictionary)
query_dict, qrels_dict = read_cacm_query('../CACM/query.text', '../CACM/qrels.text')

vm = VectorSpaceModel(dictionary)
i = 1
print(query_dict[i])
res = vm.eval(query_dict[i])
rels = qrels_dict[i]
print(rels, '\n')
for i, (k, v) in enumerate(res, start=1):
    if k in rels or i < 10:
        print(i , (k,v))
