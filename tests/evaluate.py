from models.evaluator import Evaluator
from models.vector_space_model import VectorSpaceModel, SimilarityFunctions
from utils import read_cacm, preprocess_cacm, read_cacm_query
import numpy as np

dictionary = read_cacm('../CACM/cacm.all')
dictionary = preprocess_cacm(dictionary)
query_dict, qrels_dict = read_cacm_query('../CACM/query.text', '../CACM/qrels.text')

vm = VectorSpaceModel(dictionary, sparse=True)

evaluator = Evaluator(vm, query_dict, qrels_dict)
performances = evaluator.precision_recall_range('f', np.arange(0.01, 1.001, 0.02),
                                                sims=[SimilarityFunctions.COSINUS, SimilarityFunctions.DICE, SimilarityFunctions.JACCARD])