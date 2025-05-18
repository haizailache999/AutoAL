from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np

class myDataset(Dataset): #继承Dataset
    def __init__(self, idx_list,unlabeled_data): #__init__是初始化该类的一些基础参数
        self.idx_list = idx_list   #文件目录
        self.unlabeled_data=unlabeled_data
    
    def __len__(self):#返回整个数据集的大小
        return len(self.idx_list)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        #print(self.idx_list)
        image_index = self.idx_list[index]#根据索引index获取该图片
        #print(image_index)
        img = self.unlabeled_data[image_index][0]
        #print(self.unlabeled_data[image_index][1])
        '''label_numpy=np.zeros(1,dtype=np.int64)
        label_tensor = self.unlabeled_data[image_index][1].item()
        label_numpy[0]=label_tensor
        label=torch.from_numpy(label_numpy)'''
        label=self.unlabeled_data[image_index][1]
        return img,label
