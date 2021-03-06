from pprint import pprint

from models.boolean_model import BooleanModel
from models.vector_space_model import VectorSpaceModel, SimilarityFunctions
from utils import read_cacm, preprocess_cacm

# Charger les données et construction des modèles
dictionary = read_cacm('CACM/cacm.all')
dictionary = preprocess_cacm(dictionary)

vm = VectorSpaceModel(dictionary, sparse=True)
bm = BooleanModel(dictionary)

# Tester le modèle booléen
query = " ('science' or 'compiler') and not 'algebra' and 'code'"
pprint(bm.eval(query))

# Tester le modèle vectoriel
# k = -1 veut dire retourner tous les documents qui ont une similarité supèrieur à 0.
# k = 10 veut dire retourner les 10 meilleurs documents.
query = "I want to consult a document about code optimization and compilers"
pprint(vm.eval(query, k=-1, similarity_function=SimilarityFunctions.DOT))
