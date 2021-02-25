import math

import numpy as np

from models.vector_space_model import VectorSpaceModel, SimilarityFunctions


class Evaluator:
    def __init__(self, model: VectorSpaceModel, query_dict: dict, qrels_dict: dict, k=None):
        self.model = model
        self.query_dict = query_dict
        self.qrels_dict = qrels_dict
        self.k = k

    def precision_recall(self, reduce_mean=True, sims=SimilarityFunctions, **kwargs):
        model_perform = {}
        for sim in sims:
            results = {k: {x for x, _ in self.model.eval(query, sim, **kwargs)} for k, query in self.query_dict.items()}
            f = lambda k, v: len(v.intersection(self.qrels_dict[k])) / len(v) if len(v) else 0
            precision = [f(k, v) for k, v in results.items()]
            recall = [len(v.intersection(self.qrels_dict[k])) / len(self.qrels_dict[k]) for k, v in results.items()]
            precision = np.array(precision)
            recall = np.array(recall)
            if reduce_mean:
                precision = precision.mean()
                recall = recall.mean()
            model_perform[sim] = {
                "precision": precision,
                "recall": recall,
                "f1_score": 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            }
        return model_perform

    def precision_recall_range(self, var, values, interpolate=False, sims=SimilarityFunctions):
        model_perfs = {sim: {
            "precision": [],
            "recall": [],
            "f1_score": []
        } for sim in sims}
        for k in values:
            for sim in sims:
                precision, recall = self.precision_recall_query_sim_all(sim=sim, interpolate=interpolate, **{var: k})
                if interpolate:
                    precision = precision.mean()
                    recall = precision.mean()
                f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
                m = model_perfs[sim]
                m['precision'].append(precision)
                m['recall'].append(recall)
                m['f1_score'].append(f1_score)
        return model_perfs

    def precision_recall_query(self, query_id, sim=SimilarityFunctions.DOT, interpolate=False, **kwargs):
        precision, recall = [], []
        query = self.query_dict[query_id]
        query_rel = set(self.qrels_dict[query_id])
        e = self.model.eval(query, sim, **kwargs)
        e = [x for x, _ in e]
        for i in range(1, len(e) + 1):
            if e[i-1] not in query_rel:
                continue
            t = e[:i]
            intersect = query_rel.intersection(t)
            p = len(intersect) / len(t)
            r = len(intersect) / len(query_rel)
            precision.append(p)
            recall.append(r)
        if interpolate:
            m = np.array([recall, precision])

            def f(x):
                try:
                    return m[1][m[0] >= x].max()
                except ValueError:
                    return 0

            recall = np.arange(0, 1.01, 0.1)
            precision = [f(x) for x in recall]
        return precision, recall

    def precision_recall_query_all(self, sims=list(SimilarityFunctions), interpolate=False, **kwargs):
        d = dict()
        for sim in sims:
            d[sim] = self.precision_recall_query_sim_all(sim, interpolate, **kwargs)
        return d

    def precision_recall_query_sim_all(self, sim, interpolate, **kwargs):
        precision = []
        recall = []
        for query_id in self.query_dict:
            p, r = self.precision_recall_query(query_id, sim, interpolate=interpolate, **kwargs)
            precision.append(p), recall.append(r)
        if interpolate:
            precision = np.row_stack(precision).mean(axis=0)
            recall = np.row_stack(recall).mean(axis=0)
        else:
            precision = sum(np.array(x).mean() if len(x) else 0 for x in precision)/len(precision)
            recall = sum(np.array(x).mean() if len(x) else 0 for x in recall)/len(recall)
        return precision, recall
