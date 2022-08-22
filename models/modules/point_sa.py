"""
    Apply self-attention across the points
"""
import torch
from torch import nn
import math
import numpy as np
from utils.point_ops import get_knn_values

def cal_sinusoid_encoding(dims, coords):
    # coords: (B, N)
    position_embedding_channel = dims
    device = coords.device
    hidden = torch.arange(position_embedding_channel, device=device)
    hidden = torch.div(hidden, 2, rounding_mode='floor') * 2 / position_embedding_channel
    hidden = torch.pow(10000, hidden)
    coords = coords.unsqueeze(-1) / hidden.view(1, 1, -1)    # (B, N, D)
    coords[:, :, 0::2] = torch.sin(coords[:, :, 0::2])
    coords[:, :, 1::2] = torch.cos(coords[:, :, 1::2])
    return coords

class PointSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head_size = cfg.hidden_size // cfg.num_heads

        # self.key = nn.Linear(cfg.position_embedding_channel+cfg.input_pts_channel + cfg.input_img_channel, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        
        # self.value = nn.Linear(cfg.position_embedding_channel+cfg.input_pts_channel + cfg.input_img_channel, cfg.hidden_size)
        
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.layerNorm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, query_states, key_states, attn_mask):
        """
            Args: 
                query_states: (B, N+M, Ci) float
                key_states: (B, N, Ci) float
                attn_mask: (B, N)
        """
        values = self.value(key_states)
        query = self.query(query_states)
        key = self.key(key_states)

        query, key, values = self.split_heads(query), self.split_heads(key), self.split_heads(values)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))       # (B, num_heads, N+M, N)
        if attn_mask is not None:
            attention_scores = attention_scores - ((~attn_mask).float()*1e10).unsqueeze(1).unsqueeze(2)
        attention_scores = nn.Softmax(dim=-1)(attention_scores / math.sqrt(self.head_size))
        attention_scores = self.dropout(attention_scores)
        values = torch.matmul(attention_scores, values)                 # (B, num_heads, N, head_size)
        values = self.merge_heads(values)                               # (B, N, hidden_size)

        values = self.dropout(self.dense(values))
        values = self.layerNorm(values + query_states)
        return values        

    def split_heads(self, values):
        # values: (B, N, C)
        num_heads = self.cfg.num_heads
        B, N, C = values.size()
        values = values.view(B, N, num_heads, -1).permute(0, 2, 1, 3)         # (B, num_heads, N, head_size)
        return values

    def merge_heads(self, values):
        # values: (B, num_heads, N, head_size)
        B, _, N, _ = values.size()
        values = values.permute(0, 2, 1, 3).reshape(B, N, -1)
        return values

class PointAttentionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = PointSelfAttention(cfg)
        self.dense1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.act = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.layernorm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, query_states, key_states, attn_mask):
        new_feature_3d = self.attention(query_states, key_states, attn_mask)
        intermediate = self.act(self.dense1(new_feature_3d))
        intermediate = self.dropout(self.dense2(intermediate))
        output = self.layernorm(intermediate + new_feature_3d)  # (B, N, Cp)
        return output

class AttentionPointEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_layers = cfg.num_layers
        layers = [PointAttentionLayer(cfg) for i in range(num_layers)]
        self.layers = nn.ModuleList(layers)

        # fusion method
        if cfg.fuse_method=='ADD':
            self.ln1 = nn.LayerNorm(cfg.input_pts_channel)
            self.down_channel = nn.Sequential(
                nn.Linear(cfg.input_pts_channel, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size)
            )
        elif cfg.fuse_method=='CAT':
            self.down_channel = nn.Sequential(
                nn.Linear(cfg.input_pts_channel*3, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size)
            )
        elif cfg.fuse_method=='GATE':
            self.gating = nn.Sequential(
                nn.Linear(cfg.input_pts_channel*3, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Softmax(dim=-1)
            )
            self.down_channel = nn.Linear(cfg.input_pts_channel, cfg.hidden_size)

        if cfg.fore_attn:
            self.fore_attn = nn.Sequential(
                nn.Linear(cfg.hidden_size, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
                )

        if cfg.use_cls_token:
            self.cls_f3d = nn.Parameter(torch.zeros(cfg.hidden_size))
            self.cls_f3d = nn.init.normal_(self.cls_f3d)    
    
    def forward(self, query_c2d, query_f2d, query_f3d, key_c2d, key_f2d, key_f3d, attn_mask):
        """
            Args: 
                query_c2d: (B, M, 2) float
                query_f2d: (B, M, Ci) float
                query_f3d: (B, M, Cp) float or None
                key_c2d:    (B, N, 2) 
                key_f2d:    (B, N, Ci)
                key_f3d:    (B, N, Cp)
                attn_mask:  (B, N)
        """
        B, N, _ = key_c2d.size()
        query_c2d = torch.cat([cal_sinusoid_encoding(query_f2d.size(-1)//2, query_c2d[:, :, 0]),
                               cal_sinusoid_encoding(query_f2d.size(-1)//2, query_c2d[:, :, 1]),], dim=-1)
        key_c2d = torch.cat([cal_sinusoid_encoding(key_f2d.size(-1)//2, key_c2d[:, :, 0]),
                             cal_sinusoid_encoding(key_f2d.size(-1)//2, key_c2d[:, :, 1]),], dim=-1)

        if self.cfg.fuse_method=='ADD':
            query_states = self.down_channel(self.ln1(query_f2d + query_f3d + query_c2d))
            key_states = self.down_channel(self.ln1(key_f2d + key_f3d + key_c2d))
        elif self.cfg.fuse_method=='CAT':
            query_states = self.down_channel(torch.cat([query_f2d, query_f3d, query_c2d], dim=-1))
            key_states = self.down_channel(torch.cat([key_f2d, key_f3d, key_c2d], dim=-1))
        elif self.cfg.fuse_method=='GATE':
            query_weights = self.gating(torch.cat([query_f2d, query_f3d, query_c2d], dim=-1))
            key_weights = self.gating(torch.cat([key_f2d, key_f3d, key_c2d], dim=-1))
            query_states = torch.matmul(query_weights.unsqueeze(-2), torch.stack([query_f2d, query_f3d, query_c2d], dim=2))
            key_states = torch.matmul(key_weights.unsqueeze(-2), torch.stack([key_f2d, key_f3d, key_c2d], dim=2))
            query_states = self.down_channel(query_states.squeeze(-2))
            key_states = self.down_channel(key_states.squeeze(-2))

        cls_f3d = self.cls_f3d.view(1, 1, -1).repeat(B, 1, 1)

        for l in self.layers:
            res = l(torch.cat([key_states, query_states], dim=1), key_states, attn_mask)
            key_states, query_states = res[:, :N, :], res[:, N:, :]
            if self.cfg.fore_attn:
                res = self.fore_attn(res) * res
            
            cls_f3d = l(cls_f3d, torch.cat([cls_f3d, res], dim=1), None)

        return query_states, key_states, cls_f3d
        