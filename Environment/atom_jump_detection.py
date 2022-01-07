import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, f1_score


class conv_dataset(Dataset):
    def __init__(self, currents, atom_moved, length=2048):
        
        self.currents = currents
        self.atom_moved = atom_moved
        self.length = length
        
    def __len__(self):
        return len(self.currents)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        currents_same_len, atom_moved = [], []
        current = self.currents[idx]
        
        current_ = current
        while len(current)<self.length:
            current = np.hstack((current,current_))
        new_current = current[:self.length]
        new_current = (new_current - np.mean(new_current))/np.std(new_current)
        currents_same_len.append(new_current)
        atom_moved.append(self.atom_moved[idx])
        
        sample = {'current': np.vstack(currents_same_len), 'atom_moved': np.array(atom_moved)}
        return sample


class CONV(nn.Module):
    def __init__(self,input_dim, kernel_size, max_pool_kernel_size, stride, max_pool_stride, output_dim):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size = kernel_size, stride=stride)
        lout1 = self.get_size(input_dim, kernel_size, stride=stride)
        self.max_pool1 = nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride)
        lout1_1 = self.get_size(lout1, max_pool_kernel_size, stride=max_pool_stride)
        self.conv2 = nn.Conv1d(1, 1, kernel_size = kernel_size, stride=stride)
        lout2 = self.get_size(lout1_1, kernel_size, stride=stride)
        self.max_pool2 = nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride)
        lout2_1 = int(self.get_size(lout2, max_pool_kernel_size, stride=max_pool_stride))
        
        self.fc3 = nn.Linear(lout2_1, output_dim)
        self.dropout= nn.Dropout(0.1)
        self.float()
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.dropout(x)
        x= torch.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
    def get_size(self, Lin, kernel_size, stride = 1, padding = 0, dilation = 1):
        Lout = (Lin + 2*padding - dilation*(kernel_size-1)-1)/stride + 1
        return Lout

class AtomJumpDetector_conv:
    def __init__(self, data_len, load_weight = None, load_folder = None):
        self.data_len = data_len
        self.conv_net = CONV(data_len, 64, 4, 4, 2, 1)
        if load_folder is not None:
            currents, atom_moved = self.load_data(load_folder)
        else:
            currents, atom_moved = [], []
        
        if load_weight is not None:
            self.load_weight(load_weight)
        self.dataset = conv_dataset(currents, atom_moved, length=data_len)
        self.dataloader = DataLoader(self.dataset, batch_size=64,
                        shuffle=True, num_workers=0)

            
            
        self.optim = Adam(self.conv_net.parameters(),lr=1e-3)
        self.criterion = nn.BCELoss()
        
    def push(self, currents, atom_moved):
        new_dataset = conv_dataset(currents, atom_moved, length=self.data_len)
        self.dataset.__add__(new_dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=64,
                        shuffle=True, num_workers=0)
    
    def train(self):
        for _, sample_batched in enumerate(self.dataloader):
            current = sample_batched['current']
            r = sample_batched['atom_moved']
            prediction = self.conv(current.float())
            loss = self.criterion(torch.squeeze(prediction,-1), r.type(torch.float32))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
    def predict(self, current):
        while len(current)<self.data_len:
            current = np.hstack((current,current))
        current = current[:self.data_len]
        current = torch.FloatTensor(current.reshape((1,1,-1)))
        pred = self.conv_net(current)
        return pred.detach().item()>0.5
        
    def load_data(self, folder_name):
        currents, atom_moved = [], []

        for i, np_name in enumerate(glob.glob(folder_name+'/*.npy')):
            try:
                data = np.load(np_name,allow_pickle=True).item()
                atom_old = data['episode_start_info']['info']['start_absolute_nm']
                for info in data['transitions']['info']:
                    atom_new = info['atom_absolute_nm']
                    current = info['current_series']
                    if current is not None:
                        currents.append(current)
                        atom_moved.append(np.linalg.norm(atom_new - atom_old)<0.16)
                    atom_old = atom_new
            except:
                pass
        return currents, atom_moved
    
    def load_weight(self, load_weight):
        self.conv_net.load_state_dict(torch.load(load_weight))

        
        

        