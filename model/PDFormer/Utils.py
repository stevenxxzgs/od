import os
import scipy.sparse as ss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from functools import partial
from logging import getLogger
import logging


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def masked_huber_loss(preds, labels, delta=1.0, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
# 损失函数实现
def masked_mae_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mape_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


# 抽象模型基类

class AbstractModel(nn.Module):

    def __init__(self, config, data_feature):
        nn.Module.__init__(self)

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """

class AbstractTrafficStateModel(AbstractModel):

    def __init__(self, config, data_feature):
        self.data_feature = data_feature
        super().__init__(config, data_feature)

    def predict(self, batch):
        """

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """

    def get_data_feature(self):
        return self.data_feature


class StandardScaler_:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataInput(object):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self):
        # 加载OD数据
        ODPATH = self.data_dir + '/od_day20180101_20210228.npz'
        OD_DAYS = [date.strftime('%Y-%m-%d') for date in pd.date_range(start='2020-01-01', end='2021-02-28', freq='1D')]
        prov_day_data = ss.load_npz(ODPATH)
        prov_day_data_dense = np.array(prov_day_data.todense()).reshape((-1, 47, 47))
        data = prov_day_data_dense[-len(OD_DAYS):, :, :, np.newaxis]
        ODdata = np.log(data + 1.0)  # log transformation

        # 加载邻接矩阵
        adj = np.load(self.data_dir + '/adjacency_matrix.npy')

        # 计算DTW矩阵
        dtw_matrix = self.calculate_dtw_matrix(ODdata)
        
        # 计算距离矩阵和跳数矩阵
        sd_mx = self.calculate_distance_matrix(adj)
        sh_mx = self.calculate_hop_matrix(adj)

        # 计算交通模式键
        pattern_keys = self.calculate_pattern_keys(ODdata)

        # 构建数据字典
        dataset = {
            'OD': ODdata,
            'adj': adj,
            'dtw_matrix': dtw_matrix,
            'sd_mx': sd_mx,
            'sh_mx': sh_mx,
            'pattern_keys': pattern_keys,
            'scaler': self._get_scaler(ODdata)  # 添加标准化器
        }
        return dataset

    def _get_scaler(self, data):
        """创建数据标准化器"""
        data_flat = data.reshape(-1, data.shape[-1])
        mean = np.mean(data_flat, axis=0)
        std = np.std(data_flat, axis=0)
        return StandardScaler_(mean, std)

    def calculate_dtw_matrix(self, data):
        """计算DTW距离矩阵"""
        N = data.shape[1]  # 节点数量
        dtw_matrix = np.zeros((N, N))
        
        # 提取每个节点的时间序列
        node_sequences = []
        for i in range(N):
            # 将每个节点的所有出发和到达OD流量序列求和，并确保是一维数组
            out_flow = data[:, i, :, 0].sum(axis=1)  # 出发流量
            in_flow = data[:, :, i, 0].sum(axis=1)   # 到达流量
            seq = out_flow + in_flow  # 合并流量
            # 确保序列是一维的并且是连续的内存布局
            seq = np.ascontiguousarray(seq.ravel(), dtype=np.float64)
            node_sequences.append(seq)
        
        # 计算DTW距离
        for i in range(N):
            for j in range(i+1, N):  # 只计算上三角矩阵
                try:
                    # 确保序列是连续的内存布局的一维数组
                    seq1 = np.ascontiguousarray(node_sequences[i])
                    seq2 = np.ascontiguousarray(node_sequences[j])
                    
                    # 将序列重塑为二维数组，每个时间点一行
                    seq1_2d = seq1.reshape(-1, 1)
                    seq2_2d = seq2.reshape(-1, 1)
                    
                    distance, _ = fastdtw(seq1_2d, seq2_2d, dist=euclidean)
                    dtw_matrix[i, j] = distance
                    dtw_matrix[j, i] = distance  # 矩阵是对称的
                except Exception as e:
                    print(f"Error calculating DTW for nodes {i} and {j}")
                    print(f"Sequence 1 shape: {seq1.shape}, dtype: {seq1.dtype}")
                    print(f"Sequence 2 shape: {seq2.shape}, dtype: {seq2.dtype}")
                    print(f"Error message: {str(e)}")
                    raise
                
        return dtw_matrix

    def calculate_distance_matrix(self, adj):
        """计算节点间的距离矩阵"""
        N = adj.shape[0]
        sd_mx = np.zeros((N, N)) + np.inf
        
        # Floyd-Warshall算法
        for i in range(N):
            sd_mx[i, i] = 0
            for j in range(N):
                if adj[i, j] > 0:
                    sd_mx[i, j] = 1

        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if sd_mx[i, k] + sd_mx[k, j] < sd_mx[i, j]:
                        sd_mx[i, j] = sd_mx[i, k] + sd_mx[k, j]
                        
        return sd_mx

    def calculate_hop_matrix(self, adj):
        """计算节点间的跳数矩阵"""
        N = adj.shape[0]
        sh_mx = np.zeros((N, N))
        
        # 广度优先搜索计算最短路径跳数
        for i in range(N):
            visited = set()
            queue = [(i, 0)]
            visited.add(i)
            
            while queue:
                node, hops = queue.pop(0)
                sh_mx[i, node] = hops
                
                for j in range(N):
                    if adj[node, j] > 0 and j not in visited:
                        queue.append((j, hops + 1))
                        visited.add(j)
                        
        return sh_mx

    def calculate_pattern_keys(self, data):
        """计算交通模式键"""
        B, T, N, C = data.shape
        s_attn_size = 3  # 默认空间注意力大小
        
        # 使用滑动窗口提取模式
        pattern_keys = np.zeros((N, N, s_attn_size, C))
        for i in range(N):
            for j in range(N):
                # 计算每个OD对的时间序列
                od_series = data[:, i, j, :]  # (B, C)
                
                # 使用滑动窗口计算模式
                for k in range(s_attn_size):
                    if k + s_attn_size <= T:
                        # 计算窗口内的平均值
                        window_mean = np.mean(od_series[k:k+s_attn_size], axis=0)
                        pattern_keys[i, j, k] = window_mean
                        
        return pattern_keys


class DataGenerator:
    def __init__(self, obs_len, pred_len, data_split_ratio):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def get_data_loader(self, data, params):
        # 准备数据
        x_seq = {}
        y_true = {}
        
        # 数据标准化
        scaler = data['scaler']
        od_data = scaler.transform(data['OD'])
        
        # 生成序列
        num_samples = len(od_data) - self.obs_len - self.pred_len + 1
        for i in range(num_samples):
            if i == 0:
                x = od_data[i:i+self.obs_len]
                y = od_data[i+self.obs_len:i+self.obs_len+self.pred_len]
                x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                y = y.reshape(1, y.shape[0], y.shape[1], y.shape[2], y.shape[3])
            else:
                tmp_x = od_data[i:i+self.obs_len]
                tmp_y = od_data[i+self.obs_len:i+self.obs_len+self.pred_len]
                tmp_x = tmp_x.reshape(1, tmp_x.shape[0], tmp_x.shape[1], tmp_x.shape[2], tmp_x.shape[3])
                tmp_y = tmp_y.reshape(1, tmp_y.shape[0], tmp_y.shape[1], tmp_y.shape[2], tmp_y.shape[3])
                x = np.concatenate([x, tmp_x], axis=0)
                y = np.concatenate([y, tmp_y], axis=0)
        
        # 数据分割
        total_samples = x.shape[0]
        train_size = int(total_samples * self.data_split_ratio[0] / sum(self.data_split_ratio))
        val_size = int(total_samples * self.data_split_ratio[1] / sum(self.data_split_ratio))
        
        # 确保pattern_keys的维度正确
        pattern_keys = data['pattern_keys']
        if pattern_keys.shape[-1] != 1:
            pattern_keys = pattern_keys[..., np.newaxis]  # 添加最后一个维度
        
        # 转换为张量
        x_seq['train'] = torch.FloatTensor(x[:train_size])
        y_true['train'] = torch.FloatTensor(y[:train_size])
        pattern_keys_tensor = torch.FloatTensor(pattern_keys)
        
        x_seq['validate'] = torch.FloatTensor(x[train_size:train_size+val_size])
        y_true['validate'] = torch.FloatTensor(y[train_size:train_size+val_size])
        
        x_seq['test'] = torch.FloatTensor(x[train_size+val_size:])
        y_true['test'] = torch.FloatTensor(y[train_size+val_size:])
        
        # 创建数据加载器
        data_loader = {
            mode: DataLoader(
                dataset=torch.utils.data.TensorDataset(x_seq[mode], y_true[mode]),
                batch_size=params['batch_size'],
                shuffle=(mode == 'train'),
                pin_memory=False,  # 禁用pin_memory
                num_workers=0  # 设置为0以避免多进程问题
            ) for mode in ['train', 'validate', 'test']
        }
        
        return data_loader


class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: torch.Tensor, mode: str, mode_len: dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.output[item]

    def prepare_xy(self, inputs: dict, output: torch.Tensor):
        """准备数据集的输入和输出"""
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:  # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y 