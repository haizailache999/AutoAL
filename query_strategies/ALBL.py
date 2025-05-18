import numpy as np
from .strategy import Strategy
from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .kmeans_sampling import KMeansSampling
from .bayesian_active_learning_disagreement_dropout import BALDDropout
from .var_ratio import VarRatio
from .mean_std import MeanSTD

class ActiveLearningByLearning(Strategy):
	def __init__(self,dataset, net, args_input, args_task, strategy_list=[], delta = 0.1):
		super(ActiveLearningByLearning, self).__init__(dataset, net, args_input, args_task,)
		self.strategy_list = strategy_list
		self.strategy_list.append(LeastConfidence(dataset, net, args_input, args_task))
		self.strategy_list.append(MarginSampling(dataset, net, args_input, args_task))
		self.strategy_list.append(EntropySampling(dataset, net, args_input, args_task))
		self.strategy_list.append(KMeansSampling(dataset, net, args_input, args_task))
		self.strategy_list.append(BALDDropout(dataset, net, args_input, args_task))
		self.strategy_list.append(VarRatio(dataset, net, args_input, args_task))
		self.strategy_list.append(MeanSTD(dataset, net, args_input, args_task))
		self.n_strategy = len(self.strategy_list)
		self.delta = delta
		self.w = np.ones((self.n_strategy, ))
		self.pmin = 1.0 / (self.n_strategy * 10.0)
		self.start = True
		self.Y = self.dataset.get_alldata().Y
		self.aw = np.zeros((len(self.Y)))
		self.idxs_lb=self.dataset.labeled_idxs
		self.aw[self.idxs_lb] = 1.0
		self.dataset=dataset
		self.clf=net

	def query(self, n):
		unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
		if not self.start:
			idxs_labeled,labeled_data=self.dataset.get_labeled_data()
			#print(idxs_labeled)
			P = self.predict(labeled_data)
			fn = (P.numpy() == self.Y[idxs_labeled].numpy()).astype(float)
			#print(sum(fn))
			for i in idxs_labeled:
				if self.aw[i]==0:
					self.aw[i]=1
			#print("aw",self.aw[idxs_labeled])
			reward = (fn / self.aw[idxs_labeled]).mean()

			self.w[self.s_idx] *= np.exp(self.pmin/2.0 * (reward + 1.0 / self.last_p * np.sqrt(np.log(self.n_strategy / self.delta) / self.n_strategy)))

		self.start = False
		W = self.w.sum()
		p = (1.0 - self.n_strategy * self.pmin) * self.w / W + self.pmin

		for i, stgy in enumerate(self.strategy_list):
			print('  {} {}'.format(p[i], type(stgy).__name__))

		self.s_idx = np.random.choice(np.arange(self.n_strategy), p=p)
		print('  select {}'.format(type(self.strategy_list[self.s_idx]).__name__))
		self.strategy_list[self.s_idx].clf = self.clf
		q_idxs = self.strategy_list[self.s_idx].query(n)
		self.aw[q_idxs] = p[self.s_idx]
		self.last_p = p[self.s_idx]

		return q_idxs
