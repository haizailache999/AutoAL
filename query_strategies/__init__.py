from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .least_confidence_dropout import LeastConfidenceDropout
from .margin_sampling_dropout import MarginSamplingDropout
from .entropy_sampling_dropout import EntropySamplingDropout
from .kmeans_sampling import KMeansSampling
from .kcenter_greedy import KCenterGreedy
from .bayesian_active_learning_disagreement_dropout import BALDDropout
from .adversarial_bim import AdversarialBIM
from .adversarial_deepfool import AdversarialDeepFool

from .kcenter_greedy_pca import KCenterGreedyPCA
from .kmeans_sampling_gpu import KMeansSamplingGPU
from .var_ratio import VarRatio
from .mean_std import MeanSTD
from .badge_sampling import BadgeSampling
from .ceal import CEALSampling
from .loss_prediction import LossPredictionLoss
from .vaal import VAAL
from .waal import WAAL

from .ALBL import ActiveLearningByLearning
