import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset, net, args_input, args_task,loader):
        super(LeastConfidence, self).__init__(dataset, net, args_input, args_task,loader)

    def query(self, n):
        #unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_idxs=np.arange(len(self.loader.dataset))
        probs = self.predict_prob1(self.loader)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
