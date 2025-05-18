from query_strategy import LeastConfidence, MarginSampling, EntropySampling,KMeansSampling,KCenterGreedyPCA, BALDDropout,VarRatio, MeanSTD, BadgeSampling
import numpy as np
import math

def get_idxs(dataset, net, args_input, args_task,NUM_QUERY,unlabeled_idxs,loader):
    #print("dataset",len(dataset))
    net1=net
    #st1=RandomSampling(dataset, net, args_input, args_task)
    st2=LeastConfidence(dataset=dataset, net=net, args_input=args_input, args_task=args_task,loader=loader)
    st3=MarginSampling(dataset, net, args_input, args_task,loader)
    st4=EntropySampling(dataset, net, args_input, args_task,loader)
    st8=KMeansSampling(dataset, net, args_input, args_task,loader)
    st12=BALDDropout(dataset, net, args_input, args_task,loader)
    st13=VarRatio(dataset, net, args_input, args_task,loader)
    st14=MeanSTD(dataset, net, args_input, args_task,loader)
    st_list=[st2,st3,st4,st8,st12,st13,st14]
    result_list=np.zeros((len(loader.dataset), len(st_list)))
    for i,strategy in enumerate(st_list):
        if len(loader.dataset)>args_input.quota+args_input.initseed:
            q_idx=strategy.query(args_input.batch)
        else:
            if math.ceil(len(loader.dataset)*args_input.ratio)==1:
                num_q=2            
            else:
                num_q=math.ceil(len(loader.dataset)*args_input.ratio)
            q_idx=strategy.query(num_q)
        for t in q_idx:
            result_list[t][i]=1
    return result_list
