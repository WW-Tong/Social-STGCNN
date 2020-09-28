import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import pickle
from constants import *
def get_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)

def get_norm(p):
    return math.sqrt(p[0]**2+ p[1]**2)

def get_cosine(p1,p2,p3):
    '''
    use three pos to get the cosine between two vector
    '''
    p4=np.zeros(2)
    p4[0]=p2[0]-p1[0]
    p4[1]=p2[1]-p1[1]
    m1=math.sqrt(p4[0]**2+ p4[1]**2)
    m2=math.sqrt(p3[0]**2+ p3[1]**2)
    m=p4[0]*p3[0]+p4[1]*p3[1]
    return m/(m1*m2)

def anorm(p1,p2,p3,p4): 
    
    cosine_ij=get_cosine(p1,p2,p3)
    vi_norm=get_norm(p3)
    cosine_ji=get_cosine(p2,p1,p4)
    vj_norm=get_norm(p4)
    dis=get_distance(p1,p2)
    norm=(vi_norm*cosine_ij+vj_norm*cosine_ji)/dis
    if torch.isnan(norm):
        norm = 0
    #     return 0
    if norm <0:
        norm=0
    return norm*100.0
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel =seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_[h],step_[k],step_rel[h],step_rel[k])        # 距离差的l2  求倒数
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])          # 生成图
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()      # 返回图的拉普拉斯规范化形式
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)
def relative_to_abs(rel_traj, start_pos):       # T V C   V C
    rel_traj = rel_traj.permute(1, 0, 2)    # V*T*2
    displacement = torch.cumsum(rel_traj, dim=1)    # 沿第二个维度求和
    start_pos = torch.unsqueeze(start_pos, dim=1)   # 变为3*1*2
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)        # T V C

def bce_loss(input, target):
    neg_abs = -input.abs()          # 取负值
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)    # 真实数据在1左右
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)     # 预测数据在0左右
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def l2_loss(pred_traj, pred_traj_gt, mode='average'):
    pred_traj_gt=torch.squeeze(pred_traj_gt)    # VCT
    seq_len, batch, _ = pred_traj.size()        # T V C
    loss = (pred_traj_gt.permute(0, 2, 1) - pred_traj.permute(1, 0, 2))**2  #        V T C
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / (seq_len * batch)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)   # V

def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    '''
    计算位移误差总和
    :param pred_traj: 预测结果 12*n*2
    :param pred_traj_gt: 实际路径 12*n*2
    :param mode:
    :return:
    '''
    seq_len, _, _ = pred_traj.size()    # 12
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)   # 3*12*2
    loss = loss**2      # 平方差
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)   # n*1 平方根
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(
    pred_pos, pred_pos_gt, mode='sum'
):
    '''
    计算FDE
    :param pred_pos: 终点
    :param pred_pos_gt: 真实终点
    :param mode:
    :return:
    '''
    loss = pred_pos_gt - pred_pos
    loss = loss**2 # 平方差
    loss = torch.sqrt(loss.sum(dim=1))  # 求和取平方根
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        # all_files = os.listdir(self.data_dir)
        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".txt")]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        fet_map = {}
        fet_list = []
        
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            hkl_path = os.path.splitext(path)[0] + ".pkl"   # 获取数据文件的文件名和路径，读取对应的pkl文件
            with open(hkl_path, 'rb') as handle:
                new_fet = pickle.load(handle)
            fet_map[hkl_path] = torch.from_numpy(new_fet)
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))  # 向上取整

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))      # 序列中的人数
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,  # n*2*20/16
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))   # n*2*20/16
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),   # n*20/16
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])    # 第二帧以后的所有帧转置
                    curr_ped_seq = curr_ped_seq             # ？？？？
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq  #
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    fet_list.append(hkl_path)     # 每个序列对应一个hkl_path

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.fet_map = fet_map
        self.fet_list = fet_list

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.fet_map[self.fet_list[index]]
        ]
        return out
#test loader
das=TrajectoryDataset(data_dir="./datasets/eth/test")
loader=DataLoader(das,1,shuffle=False)
for step,(obs,pre,obs_re,pred_re,non,los,vo,ao,vp,ap,vgg) in enumerate(loader):
    print(obs.shape)
    print(pre.shape)
    print(vo.shape)
    print(vgg.shape)
    print("ok")
