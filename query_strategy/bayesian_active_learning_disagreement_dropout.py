import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset, net, args_input, args_task, loader, n_drop=10):
        super(BALDDropout, self).__init__(dataset, net, args_input, args_task,loader)
        self.n_drop = n_drop

    def query(self, n):
        #unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_idxs=np.arange(len(self.loader.dataset))
        probs = self.predict_prob_dropout_split1(self.loader, n_drop=self.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
