import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import math
import torch.nn.init as init
	
class Architect(object):
    def __init__(self, model,loss_module,ratio):
        self.model = model
        self.loss_module=loss_module
        self.ratio=ratio
        self.optimizer1 = torch.optim.SGD([{'params':[ param for name, param in model.named_parameters() if '_1' in name]}],
            lr=0.05, momentum=0.9, weight_decay=5e-4)
        self.optimizer2 = torch.optim.Adam([{'params':[ param for name, param in model.named_parameters() if '_1' not in name]}],
            lr=0.05, betas=(0.9, 0.999), weight_decay=5e-4)
        self.loss_module_optimizer=optim.SGD(self.loss_module.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        self.scheduler_loss_module=torch.optim.lr_scheduler.MultiStepLR(optimizer = self.loss_module_optimizer, milestones=[120])
    
    def step(self, input_valid, target_valid,device,score,length,optim,init_loss,output_feature):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.loss_module_optimizer.zero_grad()
        if optim%3==1:
            if optim>=50:
                for param_group in self.optimizer1.param_groups:
                    param_group['lr'] = param_group['lr']*(1-0.0005*optim)
            loss,loss1,loss2,loss_max,loss1_test,overlap,n=self._backward_step(input_valid.to(device), target_valid.to(device),score,device,length,init_loss,output_feature)
            self.optimizer1.step()
        elif optim%3==2:
            if optim>=50:
                for param_group in self.loss_module_optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*(1-0.0005*optim)
            loss,loss1,loss2,loss_max,loss1_test,overlap,n=self._backward_step_module(input_valid.to(device), target_valid.to(device),score,device,length,init_loss,output_feature)
            self.loss_module_optimizer.step()
            self.scheduler_loss_module.step()
        else:
            loss,loss1,loss2,loss_max,loss1_test,overlap,n=self._backward_step1(input_valid.to(device), target_valid.to(device),score,device,length)
            self.optimizer2.step()
        return loss,loss1,loss2,loss_max,loss1_test,overlap,n

    def LossPredLoss(self,input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape
        input = (input - input.flip(0))[:len(input)] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()
        
        return loss

    def _backward_step(self, input_valid, target_valid,score,device,length,init_loss,output_feature):
        pred_loss = self.loss_module(output_feature)
        pred_loss = pred_loss.view(pred_loss.size(0))
        loss = F.cross_entropy(input_valid, target_valid,reduction='none')
        loss=loss.detach()
        loss_test=F.cross_entropy(input_valid, target_valid,reduction='mean')
        loss_test=loss_test.detach()  #add
        init_loss=init_loss.detach()  #add
        m_module_loss = torch.clamp(self.LossPredLoss(pred_loss, loss, margin=1),min=0,max=10)
        n=torch.sum(score > 0.5).item()
        if n>int(length*self.ratio):
            add_loss=(1/(1+math.exp(0.5*(int(length*self.ratio)-n)))-0.5)*2
        else:
            add_loss=(1/(1+math.exp(0.5*(n-int(length*self.ratio))))-0.5)*2
        top_n_values, top_n_indices = torch.topk(loss, int(length*self.ratio))
        indices = torch.nonzero(score > 0.5).squeeze()
        if n==1:
            set1=set([indices.item()])
        else:
            set1 = set(indices.tolist())
        set2 = set(top_n_indices.tolist())
        overlap=len(set1.intersection(set2))
        loss_max=torch.sum(top_n_values)
        loss1_test=(loss*score).sum()
        loss1=(loss*(score)).mean()     #newloss1
        loss2=add_loss
        loss=-loss1-2*add_loss
        loss.backward()
        return loss,loss1,loss2,loss_max,loss1_test,overlap,n

    def _backward_step1(self, input_valid, target_valid,score,device,length):
        loss = F.cross_entropy(input_valid, target_valid,reduction='none')
        loss_test=F.cross_entropy(input_valid, target_valid,reduction='mean')
        n=torch.sum(score > 0.5).item()
        if n>int(length*self.ratio):
            add_loss=(1/(1+math.exp(0.5*(int(length*self.ratio)-n)))-0.5)*2
        else:
            add_loss=(1/(1+math.exp(0.5*(n-int(length*self.ratio))))-0.5)*2
        score=score.detach()
        top_n_values, _ = torch.topk(loss, n)
        loss_max=torch.sum(top_n_values)
        loss1_test=(loss*score).sum()
        loss1=(loss*score).mean()
        loss2=add_loss
        loss=loss1+0.5*loss2
        loss.backward()
        overlap=-1
        return loss,loss1,loss2,loss_max,loss1_test,overlap,n

    def _backward_step_module(self, input_valid, target_valid,score,device,length,init_loss,output_feature):
        pred_loss = self.loss_module(output_feature)
        pred_loss = pred_loss.view(pred_loss.size(0))
        loss = F.cross_entropy(input_valid, target_valid,reduction='none')
        loss=loss.detach()
        loss_test=F.cross_entropy(input_valid, target_valid,reduction='mean')
        loss_test=loss_test.detach()  #add
        init_loss=init_loss.detach()  #add
        m_module_loss = torch.clamp(self.LossPredLoss(pred_loss, loss, margin=1),min=0,max=10)
        n=torch.sum(score > 0.5).item()
        if n>int(length*self.ratio):
            add_loss=(1/(1+math.exp(0.5*(int(length*self.ratio)-n)))-0.5)*2
        else:
            add_loss=(1/(1+math.exp(0.5*(n-int(length*self.ratio))))-0.5)*2
        top_n_values, top_n_indices = torch.topk(loss, int(length*self.ratio))
        indices = torch.nonzero(score > 0.5).squeeze()
        if n==1:
            set1=set([indices.item()])
        else:
            set1 = set(indices.tolist())
        set2 = set(top_n_indices.tolist())
        overlap=len(set1.intersection(set2))
        loss_max=torch.sum(top_n_values)
        loss1_test=torch.zeros([1])
        loss1=torch.zeros([1])   #newloss1
        loss2=add_loss
        loss=m_module_loss
        loss.backward()
        return loss,loss1,loss2,loss_max,loss1_test,overlap,n