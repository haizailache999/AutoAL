import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task,loader):
        super(KMeansSampling, self).__init__(dataset, net, args_input, args_task,loader)

    def query(self, n):
        #print(n)
        #unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_idxs=np.arange(len(self.loader.dataset))
        embeddings = self.get_embeddings1(self.loader)
        embeddings = embeddings.numpy()
        #print(embeddings.shape)
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
        
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        #print(dis[cluster_idxs==i] for i in range(n))
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

        return unlabeled_idxs[q_idxs]
