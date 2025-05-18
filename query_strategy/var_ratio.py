import numpy as np
import torch
from .strategy import Strategy
from scipy.stats import mode

class VarRatio(Strategy):
    def __init__(self, dataset, net, args_input, args_task,loader):
        super(VarRatio, self).__init__(dataset, net, args_input, args_task,loader)

    def query(self, n):
        #unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_idxs=np.arange(len(self.loader.dataset))
        probs = self.predict_prob1(self.loader)
        preds = torch.max(probs, 1)[0]
        uncertainties = 1.0 - preds
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
