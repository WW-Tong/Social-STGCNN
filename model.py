import os
import math
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from constants import *
import torch.optim as optim
from utils import seq_to_graph



class ConvTemporalGraphical(nn.Module):         # 时空图
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    """The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),   # 空洞率
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size    # K==Kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)


        assert len(kernel_size) == 2        # 
        assert kernel_size[0] % 2 == 1      # 卷积核尺寸为奇数
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

def make_mlp(dim_list):
    '''
    批量生成全连接层
    :param dim_list: 维度列表
    :return: 一系列线性层的感知机
    '''
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def get_noise(shape):
    '''
    生成高斯噪声
    :param shape: 噪声的shape
    :return:
    '''
    return torch.randn(*shape).cuda()

class Encoder(nn.Module):
    def __init__(self,out_channels=5):
        super(Encoder, self).__init__()

        self.h_dim = H_DIM  # 64 
        self.embedding_dim = out_channels   # 节点通道

        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)        # 输入的维度=out_chanels，隐状态维度64，层数1，LSTM编码
        # self.spatial_embedding = nn.Linear(2, self.embedding_dim)           # 输入2维，输出embedding_dim

    def init_hidden(self, batch):
        '''
        隐层状态初始化，细胞状态初始化
        :param batch:
        :return:
        '''
        h = torch.zeros(1, batch, self.h_dim).cuda()        # 1*bat*64
        c = torch.zeros(1, batch, self.h_dim).cuda()
        return (h, c)

    def forward(self, V):       # v={N C T V}

        # padded = len(obs_trajV.shape) == 4    # 观测轨迹的维度数目4
        npeds = V.size(3)             # 节点数
        total = npeds * V.size(1)        # Bantch=npeds*bantch

        # obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))   # 展成n*2输入线性嵌入层,输出16 8*3*64*16
        V = V.view(-1, total, self.embedding_dim)  # 调整维度（-1，4096，16）
        state = self.init_hidden(total)        # 以total初始化隐藏状态 1*4096*64
        output, state = self.encoder(V, state)         # 输入编码器输出到state中64
        final_h = state[0]      # 取隐藏状态
        
        final_h = final_h.view(npeds, self.h_dim)
        return final_h      # 最后一步的输出

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.seq_len = PRED_LEN         # 解码长度 12
        self.h_dim = H_DIM              # 64
        self.embedding_dim = EMBEDDING_DIM      # 16

        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)   # 输入为16 ，输出为64 ，1层隐藏层
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)   # 输入2输出16
        self.hidden2pos = nn.Linear(self.h_dim, 2)                  # 转换为坐标序列 输入64，输出2

    def forward(self, last_pos, last_pos_rel, state_tuple):
        npeds = last_pos.size(0)        # 行人编号
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)        # 输入64*2，输出维度64*16
        decoder_input = decoder_input.view(1, npeds, self.embedding_dim)        # 调整维度1*64*16

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)      # 输出1*64*64的隐状态和细胞状态
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))              # 将out变为64*64再输入坐标转换64*2
            curr_pos = rel_pos + last_pos   # 当前坐标为坐标差和最后坐标之和
            embedding_input = rel_pos       # 64*2

            decoder_input = self.spatial_embedding(embedding_input)     # 输入64*2输出64*16
            decoder_input = decoder_input.view(1, npeds, self.embedding_dim)   # 变为 1*64*16
            pred_traj_fake_rel.append(rel_pos.view(npeds, -1))   # 预测轨迹加入rel_pos  12*64*2
            last_pos = curr_pos  # 当前坐标作为最后坐标

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)     # 按0维拼接
        return pred_traj_fake_rel       # 12*64*2
class EndpointGenerate(nn.Module):
    def __init__(self):
        super(EndpointGenerate, self).__init__()
        self.mlp_dim = MLP_DIM  # 64
        self.h_dim = H_DIM  # 64
        self.embedding_dim = EMBEDDING_DIM  # 16
        self.bottleneck_dim = BOTTLENECK_DIM  # 32
        self.noise_dim = NOISE_DIM  # 8
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)  
        # self.pattn = PhysicalAttention()
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, 1)
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.hidden2pos = nn.Linear(self.h_dim, 2)
        input_dim = self.h_dim + self.bottleneck_dim  # 96
        mlp_decoder_context_dims = [input_dim, self.mlp_dim, self.h_dim - self.noise_dim]  # 96,64,56
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)  # 96--56
                   # 64*56 +64*8 =64*64
    def forward(self,obs_traj,obs_traj_rel):
        '''
        参数暂时包括轨迹和坐标差
        :param obs_traj:
        :param obs_traj_rel:
        :return:
        '''
        total=obs_traj.size(0)*obs_traj.size(1)
        obs_traj=obs_traj.permute(3,0,1,2)      # 转置成 L N P C
        obs_traj_rel=obs_traj_rel.permute(3,0,1,2)      # 转置成 L N P C
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))   # 展成n*2输入线性嵌入层,输出16 LNP*16
        obs_traj_embedding = obs_traj_embedding.view(-1, total, self.embedding_dim)  # 调整维度（L，N*P，16）
        state = self.init_hidden(total)        # 以total初始化隐藏状态 1*4096*64
        output, state = self.encoder(obs_traj_embedding, state)         # 输入编码器输出到state中64
        final_h = state[0]
        h_c=torch.zeros_like(final_h)
        state_tuple = (final_h, h_c)
        last_pos = obs_traj[-1, :, :, :]
        # obs_speed = obs_traj_rel.view(-1, total, self.embedding_dim)  # 8*n*2
        obs_speed = torch.sum(obs_traj_rel, dim=0)/OBS_LEN      #  N P C
        move = obs_speed*PRED_LEN
        npeds = last_pos.size(2)        # 行人编号
        decoder_input = self.spatial_embedding(move)  
        # decoder_input = decoder_input.view(1, npeds, self.embedding_dim)
        output, state_tuple = self.decoder(decoder_input, state_tuple)      # 输出1*64*64  根据速度和注意力解码
        rel_pos = self.hidden2pos(output.view(-1, self.h_dim))              # 解码输出变为位移差  64*2
        pre_end_pos = rel_pos + last_pos   # 终点坐标为坐标差和最后坐标之和
        return pre_end_pos 
class PhysicalAttention(nn.Module):
    def __init__(self):
        super(PhysicalAttention, self).__init__()

        self.L = ATTN_L   # 1000
        self.D = ATTN_D   # 500
        self.D_down = ATTN_D_DOWN  # 16
        self.bottleneck_dim = BOTTLENECK_DIM  # 32
        self.embedding_dim = EMBEDDING_DIM    # 16

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)     # 2-16
        self.pre_att_proj = nn.Linear(self.D, self.D_down)        # 512-16

        mlp_pre_dim = self.embedding_dim + self.D_down    # 32
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)   # 32-512-32

        self.attn = nn.Linear(self.L*self.bottleneck_dim, self.L)     # 1000*32--1000

    def forward(self, vgg, end_pos):

        npeds = end_pos.size(0)
        end_pos = end_pos[:, 0, :]        # n*2
        curr_rel_embedding = self.spatial_embedding(end_pos)  # n*16
        curr_rel_embedding = curr_rel_embedding.view(-1, 1, self.embedding_dim).repeat(1, self.L, 1)  # n*900*16

        vgg = vgg.view(-1, self.D)    # x*512
        features_proj = self.pre_att_proj(vgg)        # x*16   x=n*900
        features_proj = features_proj.view(-1, self.L, self.D_down)   # y*900*16

        mlp_h_input = torch.cat([features_proj, curr_rel_embedding], dim=2)   # n*900*32
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.D_down))  # -1，32--32
        attn_h = attn_h.view(npeds, self.L, self.bottleneck_dim)  # n*900*32

        attn_w = F.softmax(self.attn(attn_h.view(npeds, -1)), dim=1)  # n*28800--n*900
        attn_w = attn_w.view(npeds, self.L, 1)      # n*900*1

        attn_h = torch.sum(attn_h * attn_w, dim=1)      # n*1000*32**
        return attn_h
class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = OBS_LEN      # 8
        self.pred_len = PRED_LEN    # 12
        self.mlp_dim = MLP_DIM      # 64
        self.h_dim = H_DIM          # 64
        self.embedding_dim = EMBEDDING_DIM  # 16
        self.bottleneck_dim = BOTTLENECK_DIM    # 32
        self.noise_dim = NOISE_DIM      # 8

        self.encoder = Encoder()
        # self.sattn = SocialAttention()
        # self.pattn = PhysicalAttention()
        self.decoder = Decoder()

        # input_dim = self.h_dim + 2*self.bottleneck_dim      # 128
        input_dim = self.h_dim + self.bottleneck_dim  # 96
        mlp_decoder_context_dims = [input_dim, self.mlp_dim, self.h_dim - self.noise_dim]       # 96,64,56
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)       # 96--56

    def add_noise(self, _input):
        npeds = _input.size(0)      # 64
        noise_shape = (self.noise_dim,)     # 8,
        z_decoder = get_noise(noise_shape)  # 8
        vec = z_decoder.view(1, -1).repeat(npeds, 1)        # 64*8
        return torch.cat((_input, vec), dim=1)              # 64*56 +64*8 =64*64

    def forward(self, obs_traj, obs_traj_rel):

        npeds = obs_traj_rel.size(1)    # 64
        final_encoder_h = self.encoder(obs_traj_rel)    # 64*64*64

        end_pos = obs_traj[-1, :, :, :]  # 3*64*2
        attn_s = self.sattn(final_encoder_h, end_pos)       # 64*32
        # attn_p = self.pattn(vgg_list, end_pos)
        # mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s, attn_p], dim=1)
        mlp_decoder_context_input = torch.cat([final_encoder_h[:, 0, :], attn_s], dim=1)        # 64*96
        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)       # 64*56
        decoder_h = self.add_noise(noise_input)   # 64*64
        decoder_h = torch.unsqueeze(decoder_h, 0)  # 1*64*64

        decoder_c = torch.zeros(1, npeds, self.h_dim).cuda()    # 1*64*64 细胞状态为空，隐藏状态为编码器隐藏状态加上注意力
        state_tuple = (decoder_h, decoder_c)

        last_pos = obs_traj[-1, :, 0, :]        # 64*2
        last_pos_rel = obs_traj_rel[-1, :, 0, :]    # 64*2
        pred_traj_fake_rel = self.decoder(last_pos, last_pos_rel, state_tuple)
        return pred_traj_fake_rel
