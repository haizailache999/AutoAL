import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task,loader):
        super(MarginSampling, self).__init__(dataset, net, args_input, args_task,loader)

    def query(self, n):
        #unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_idxs=np.arange(len(self.loader.dataset))
        probs = self.predict_prob1(self.loader)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
