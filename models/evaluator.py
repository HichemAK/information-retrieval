from models.boolean_model import BooleanModel
from models.vector_space_model import VectorSpaceModel, SimilarityFunctions
import numpy as np


class Evaluator:
    def __init__(self, model : VectorSpaceModel, query_dict : dict, qrels_dict : dict, k=None):
        self.model = model
        self.query_dict = query_dict
        self.qrels_dict = qrels_dict
        self.k = k

    def precision_recall(self):
        model_perform = {}
        for sim in SimilarityFunctions:
            results = {k: {x for x,_ in self.model.eval(query, sim, self.k)} for k, query in self.query_dict.items()}
            precision = [len(v.intersection(self.qrels_dict[k]))/len(v) for k, v in results.items()]
            recall = [len(v.intersection(self.qrels_dict[k]))/len(self.qrels_dict[k]) for k, v in results.items()]
            mean_precision = np.array(precision).mean()
            mean_recall = np.array(recall).mean()
            model_perform[sim] = {
                "mean_precision" : mean_precision,
                "mean_recall" : mean_recall,
                "f1_score" : 2*mean_recall*mean_precision/(mean_recall+mean_precision)
            }
        return model_perform

    def precision_recall_k(self, values):
        c = self.k
        model_perfs = {sim:{
            "mean_precision" : [],
            "mean_recall" : [],
            "f1_score" : []
        } for sim in SimilarityFunctions}
        for k in values:
            self.k = k
            perf = self.precision_recall()
            for sim, p in perf.items():
                m = model_perfs[sim]
                for k, v in p.items():
                    m[k].append(v)
        self.k = c
        return model_perfs



