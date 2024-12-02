import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', k='', edge_index='', mode='default'):
        super(TrafficDataset, self).__init__()
        self.mode = mode
        if mode == 'default':
            self.x = inputs[split + '_x']  # [T, Len, N]
            self.y = inputs[split + '_y']  # [T, Len, N]
        else:
            self.x = x
            self.y = y
            self.k = k

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.mode == 'default':
            x = torch.Tensor(self.x[index].T)
            y = torch.Tensor(self.y[index].T)
            return x,y
        else:
            x = torch.Tensor(self.x[index].T)
            y = torch.Tensor(self.y[index].T)
            k = torch.Tensor(self.k[index].T)
            return x, y, k


class SplitedTrafficDataset(Dataset):
    def __init__(self, inputs, split, start_idx, end_idx, x='', y='', k='', edge_index='', mode='default'):
        super(SplitedTrafficDataset, self).__init__()
        self.mode = mode
        if mode == 'default':
            self.x = inputs[split + '_x'][..., start_idx:end_idx]  # [T, Len, N]
            self.y = inputs[split + '_y'][..., start_idx:end_idx]  # [T, Len, N]
        else:
            self.x = x
            self.y = y
            self.k = k

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.mode == 'default':
            x = torch.Tensor(self.x[index].T)
            y = torch.Tensor(self.y[index].T)
            return x, y
        else:
            x = torch.Tensor(self.x[index].T)
            y = torch.Tensor(self.y[index].T)
            k = torch.Tensor(self.k[index].T)
            return x, y, k


class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs  # [T, Len, N]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return x


class continue_learning_CrossDataset(Dataset):
    def __init__(self, inputs_x, inputs_y):
        super(continue_learning_CrossDataset, self).__init__()
        self.x = inputs_x  # [T, Len, N]
        self.y = inputs_y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        return x, y


# from torch_geometric.data import Data, Dataset
#
#
# class TrafficDataset(Dataset):
#     def __init__(self, inputs, split, x='', y='', k='', edge_index='', mode='default'):
#         super(TrafficDataset, self).__init__()
#         self.mode = mode
#         if mode == 'default':
#             self.x = inputs[split+'_x'] # [T, Len, N]
#             self.y = inputs[split+'_y'] # [T, Len, N]
#         else:
#             self.x = x
#             self.y = y
#             self.k = k
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, index):
#         if self.mode == 'default':
#             x = torch.Tensor(self.x[index].T)
#             y = torch.Tensor(self.y[index].T)
#             return Data(x=x, y=y)
#         else:
#             x = torch.Tensor(self.x[index].T)
#             y = torch.Tensor(self.y[index].T)
#             k = torch.Tensor(self.k[index].T)
#             return Data(x=x, y=y, k=k)
#
# class SplitedTrafficDataset(Dataset):
#     def __init__(self, inputs, split, start_idx, end_idx, x='', y='', k='', edge_index='', mode='default'):
#         super(SplitedTrafficDataset, self).__init__()
#         self.mode = mode
#         if mode == 'default':
#             self.x = inputs[split + '_x'][..., start_idx:end_idx]  # [T, Len, N]
#             self.y = inputs[split + '_y'][..., start_idx:end_idx]  # [T, Len, N]
#         else:
#             self.x = x
#             self.y = y
#             self.k = k
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, index):
#         if self.mode == 'default':
#             x = torch.Tensor(self.x[index].T)
#             y = torch.Tensor(self.y[index].T)
#             return Data(x=x, y=y)
#         else:
#             x = torch.Tensor(self.x[index].T)
#             y = torch.Tensor(self.y[index].T)
#             k = torch.Tensor(self.k[index].T)
#             return Data(x=x, y=y, k=k)
#
# class continue_learning_Dataset(Dataset):
#     def __init__(self, inputs):
#         self.x = inputs # [T, Len, N]
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, index):
#         x = torch.Tensor(self.x[index].T)
#         return Data(x=x)
#
#
# class continue_learning_CrossDataset(Dataset):
#     def __init__(self, inputs_x, inputs_y):
#         super(continue_learning_CrossDataset, self).__init__()
#         self.x = inputs_x  # [T, Len, N]
#         self.y = inputs_y
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, index):
#         x = torch.Tensor(self.x[index].T)
#         y = torch.Tensor(self.y[index].T)
#         return Data(x=x, y=y)