import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net, args_input, args_task,loader):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task
        self.loader=loader
    def query(self, n):
        pass
    
    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)
    
    def get_model(self):
        return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def predict_prob1(self, data):
        probs = self.net.predict_prob1(data)
        return probs


    def predict_prob_dropout_split1(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split1(data, n_drop=n_drop)
        return probs
    

    def get_embeddings1(self, data):
        embeddings = self.net.get_embeddings1(data)
        return embeddings
    

    def get_grad_embeddings1(self, data):
        embeddings = self.net.get_grad_embeddings1(data)
        return embeddings

