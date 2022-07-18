"""
    Statistic scores & losses, during the training
"""

import torch

class ScoreCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {}
        self.totals = {}
        self.descriptions = {}

    def update(self, score_dict:dict):
        for k in score_dict.keys():
            if len(score_dict[k]) == 2:
                score, count = score_dict[k]
            else:
                score, count, desc = score_dict[k]
                self.descriptions[k] = desc
            self.counts[k] = self.counts.get(k, 0) + count
            self.totals[k] = self.totals.get(k, 0) + score*count
    
    def average(self, keys:list=None, group_by_description=False):
        if keys is None:
            keys = self.totals.keys()
        res = {}
        for k in keys:
            if k in self.totals.keys():
                avg = self.totals[k] / self.counts[k]
                if group_by_description:
                    desc = self.descriptions[k]
                    if desc not in res.keys():
                        res[desc] = {}
                    res[desc][k] = avg
                else:
                    res[k] = avg
            else:
                # print(f"[Warning] {k} not counted before.")
                res[k] = None
        return res

class HistoCounter:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.samples = None

    def __len__(self):
        return len(self.samples)

    def update(self, samples):
        if self.samples is None:
            self.samples = samples
        else:
            self.samples = torch.cat([self.samples, samples], dim=0)

    def get_values(self):
        return self.samples