import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task,loader):
        super(EntropySampling, self).__init__(dataset, net, args_input, args_task,loader)

    def query(self, n):
        #unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_idxs=np.arange(len(self.loader.dataset))
        probs = self.predict_prob1(self.loader)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
