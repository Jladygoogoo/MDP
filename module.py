import numpy as np
import torch
import torch.nn as nn

from activity import ActivityModule


class MLP(torch.nn.Module):
    def __init__(self, dims, activation="relu", dropout=0.1):
        super().__init__()
        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "tanh":
            activation_layer = nn.Tanh
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation_layer())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.module = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.module(x)
    



class DACT(nn.Module):
    def __init__(self, act_module: ActivityModule, device, n_feat=None, dropout=0.1) -> None:
        super().__init__()
        act_dim = act_module.feat_dim
        feat_dim = act_dim
        self.node_raw_embed = None
        if n_feat is not None:
            self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
            self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
            feat_dim += n_feat.shape[1]
        self.mlp = MLP(dims=[feat_dim*2, feat_dim, act_dim, 1], dropout=dropout)
        self.act_module = act_module
        self.device = device
        # self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        # self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)

    
    def contrast(self, batch_src_idx, batch_dst_idx, batch_neg_idx, batch_ts_l):
        batch_src_act_embed = self.get_embed(batch_src_idx, batch_ts_l, node_type="source")
        batch_dst_act_embed = self.get_embed(batch_dst_idx, batch_ts_l, node_type="target")
        batch_neg_act_embed = self.get_embed(batch_neg_idx, batch_ts_l, node_type="target")

        pos_score = self.mlp(torch.cat([batch_src_act_embed, batch_dst_act_embed], dim=1)).squeeze(dim=-1)
        neg_score = self.mlp(torch.cat([batch_src_act_embed, batch_neg_act_embed], dim=1)).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()


    def get_embed(self, batch_node_idx, batch_ts_l, node_type):
        act_embed = self.act_module.get_features(batch_node_idx, batch_ts_l, node_type)
        if self.node_raw_embed is not None:
            batch_node_idx_th = torch.from_numpy(batch_node_idx).long().to(self.device)
            semantic_embed = self.node_raw_embed(batch_node_idx_th)
            embed = torch.cat((act_embed, semantic_embed), dim=1)
            return embed
        return act_embed
    

    