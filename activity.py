import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
from tqdm import tqdm


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MyLSTM(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = nn.LSTM(input_size=self.input_dim, 
                                hidden_size=self.hidden_dim, 
                                num_layers=1, 
                                batch_first=True)
        self.merger = MergeLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_ = x.reshape(batch_size, self.seq_len, -1)
        _, (hn, _) = self.model(x_)
        hn = hn[-1, :, :] #hn.squeeze(dim=0)
        out = self.merger.forward(hn, x)
        return out




class ActivityModule:
    def __init__(self, data_name, device, n_windows=None, window_size_l=None, feat_dim=None, 
                 active_direction="both", method="degree", encoder="mlp", lstm_seq_len=None) -> None:
        print("Initiating activity Module...")
        self.data_name = data_name
        self.active_direction = active_direction
        self.method = method
        self.device = device
        self.encoder_type = encoder
        self.lstm_seq_len = lstm_seq_len

        df, adj_list = self.init_data()
        self.init_window_parameters(n_windows, window_size_l, df, adj_list)
        self.init_activity_by_intvs_nodes(df, adj_list)

        self.act_seq_dim = self.n_windows
        # self.act_seq_dim = self.n_windows*len(self.window_size_l)
        self.feat_dim = feat_dim if feat_dim else self.act_seq_dim

        self.init_encoder()

        

    def init_encoder(self):
        activation = nn.Tanh if self.method=="degree" else nn.ReLU
        self.encoders = []
        for s in self.window_size_l:
            self.encoders.append(nn.Sequential(
                nn.Linear(self.n_windows, self.feat_dim),
                activation(),
                nn.Linear(self.feat_dim, self.feat_dim),
                activation(),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.Dropout(p=0.1)
            ))
        self.merger = nn.Sequential(
                nn.Linear(self.feat_dim*len(self.window_size_l), self.feat_dim),
                activation(),
                nn.Linear(self.feat_dim, self.feat_dim))
        for encoder in self.encoders:
            encoder.to(self.device)
        self.merger.to(self.device)


    def reset_encoder(self):
        self.init_encoder()
    # def reset_encoder(self):
    #     self.encoder = nn.Sequential(
    #             nn.Linear(self.n_windows, self.n_windows),
    #             self.activation(),
    #             nn.Linear(self.n_windows, self.n_windows),
    #             self.activation(),
    #             nn.Linear(self.n_windows, self.feat_dim),
    #             nn.Dropout(p=0.1)
    #         )
    #     self.encoder.to(self.device) 
    
    def get_parameters(self):
        params = []
        for encoder in self.encoders:
            params += list(encoder.parameters())
        params += list(self.merger.parameters())
        return params
      
        
    def get_features(self, batch_node_idx, batch_ts, node_type="source"):
        # if node_type == "source":
        #     encoder = self.source_encoder
        # elif node_type == "target":
        #     encoder = self.target_encoder
        batch_activity_seqs = self.get_sequences(batch_node_idx, batch_ts)
        batch_activity_feats = []
        for i in range(len(self.window_size_l)):
            batch_activity_seq = torch.from_numpy(batch_activity_seqs[i]).float().to(self.device)
            batch_activity_feat = self.encoders[i](batch_activity_seq)
            batch_activity_feats.append(batch_activity_feat)
        batch_activity_feats = torch.cat(batch_activity_feats, dim=1)
        batch_activity_feat = self.merger(batch_activity_feats)
        return batch_activity_feat
    
    def get_sequences(self, batch_node_idx, batch_ts):
        batch_size = len(batch_node_idx)
        res = []
        for j in range(len(self.window_size_l)):
            batch_activity_seq = np.zeros((batch_size, self.act_seq_dim)).astype(np.int32)
            for i in range(batch_size):
                batch_activity_seq[i, :] = self._get_sequence(batch_node_idx[i], batch_ts[i], j)
            # print(batch_activity_seq[:5]) 
            res.append(batch_activity_seq)
        return res
        # return batch_activity_seq


    def _get_sequence(self, node_idx, ts, window_size_index):
        window_size = self.window_size_l[window_size_index]
        intv = int(ts // window_size)
        # res = self.activity_by_intvs_nodes[node_idx][intv-self.n_windows+1: intv+1]
        act_seq = self.activity_by_intvs_nodes_s[window_size_index][node_idx][intv-self.n_windows: intv]
        # act_seq = self.activity_by_intvs_nodes_s[window_size_index][node_idx][intv-(self.n_windows+1): intv]
        act_seq = np.concatenate((np.zeros(self.n_windows-len(act_seq)), act_seq)) 
        # res = act_seq[1:] - act_seq[:-1]
        # return res
        return act_seq

    # def _get_sequence(self, node_idx, ts):
    #     res = []
    #     for i, s in enumerate(self.window_size_l):
    #         intv = int(ts // s)
    #         # res = self.activity_by_intvs_nodes[node_idx][intv-self.n_windows+1: intv+1]
    #         act_seq = self.activity_by_intvs_nodes_s[i][node_idx][intv-self.n_windows: intv]
    #         act_seq = np.concatenate((np.zeros(self.n_windows-len(act_seq)), act_seq)) 
    #         res.append(act_seq)
    #     return np.concatenate(res, axis=0)
    

    def init_data(self):
        df = pd.read_csv("./processed/{}/{}_new.csv".format(self.data_name, self.data_name))

        src_l = df.src.values
        dst_l = df.dst.values
        e_idx_l = df.idx.values
        label_l = df.label.values
        ts_l = df.ts.values
        max_idx = max(src_l.max(), dst_l.max())

        src_active = 1 if self.active_direction in ["src","both"] else 0
        dst_active = 1 if self.active_direction in ["dst","both"] else 0

        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            adj_list[src].append((dst, eidx, ts, src_active))
            adj_list[dst].append((src, eidx, ts, dst_active))

        return df, adj_list


    def init_window_parameters(self, n_windows, window_size_l, df, adj_list):
        if self.method == "degree":
            if not n_windows:
                n_windows = 10
            if not window_size_l:
                window_size_l = choose_best_window_size_degree(df, adj_list)
        
        elif self.method == "pagerank" or self.method == "eigv":
            if not n_windows:
                n_windows = 10
            if not window_size_l:
                window_size_l = choose_best_window_size_pagerank(df, adj_list)

        elif self.method == "closeness":
            if not n_windows:
                n_windows = 10
            if not window_size_l:
                window_size = choose_best_window_size_closeness(df, adj_list)

        self.n_windows = n_windows    
        self.window_size_l = [int(x) for x in window_size_l]
        print("n_windows: {}, window_size: {}".format(self.n_windows, self.window_size_l))


    def init_activity_by_intvs_nodes(self, df, adj_list):
        self.activity_by_intvs_nodes_s = []

        for window_size in self.window_size_l:
            
            save_path = "./processed/{}/{}_activity_{}_s{}_{}.pkl".format(self.data_name, self.data_name, self.method, window_size, self.active_direction)
            if os.path.exists(save_path):
                print("Loading activity from: {}".format(save_path))
                with open(save_path, "rb") as f:
                    self.activity_by_intvs_nodes_s.append(pickle.load(f))            
            else:
                print("Calculating activity for all using method-{}, window size-{}...".format(self.method, window_size))
                if self.method == "degree":
                    degree_by_intvs_nodes = get_degree_by_intvs_nodes(adj_list, window_size)
                    self.activity_by_intvs_nodes_s.append(degree_by_intvs_nodes)
                
                # elif self.method in ("pagerank", "closeness", "eigv"):
                #     centrality_by_intvs_nodes = get_other_centrality_by_intvs_nodes(df, len(adj_list), self.window_size, self.method)
                #     centrality_by_intvs_nodes = standardize(centrality_by_intvs_nodes)
                #     self.activity_by_intvs_nodes = centrality_by_intvs_nodes
                with open(save_path, "wb") as f:
                    pickle.dump(degree_by_intvs_nodes, f)


def standardize(data):
    data = data / (np.sum(data,axis=0) / np.sum((data>0), axis=0))
    return data

def get_degree_by_intvs_nodes(adj_list, window_size):
    degree_by_intvs_nodes = [] 
    for i in range(len(adj_list)):
        curr = adj_list[i]
        curr = sorted(curr, key=lambda x: x[1]) # sorted by time
        new_edge_intvs = [int(x[2]//window_size) for x in curr if x[3]==1] # intervals of all edges for the current node
        if len(new_edge_intvs):
            degree_by_intvs = np.zeros(new_edge_intvs[-1]+1) # 从 intv:0 开始一直到 intv:last_ngh
            for intv in new_edge_intvs:
                degree_by_intvs[intv:] = degree_by_intvs[intv:] + 1      
            degree_by_intvs_nodes.append(degree_by_intvs)  
        else:
            degree_by_intvs_nodes.append([])
    return degree_by_intvs_nodes


def get_other_centrality_by_intvs_nodes(df, n_nodes, window_size, method):
    n_windows_total = int(np.ceil((df.ts.max() - df.ts.min()) / window_size))
    graph = nx.Graph()
    if method == "pagerank":
        centrality = nx.pagerank
    elif method == "closeness":
        centrality = nx.closeness_centrality
    elif method == "eigv":
        centrality = nx.eigenvector_centrality

    centrality_by_nodes_intvs = np.zeros((n_windows_total, n_nodes))
    for i in tqdm(range(n_windows_total)):
        tmp_df = df[(df.ts >= i*window_size) & (df.ts <= (i+1)*window_size)]
        graph.add_edges_from(zip(tmp_df.src.values, tmp_df.dst.values))
        res = centrality(graph) # {node:centrality}
        for nid, pr in res.items():
            centrality_by_nodes_intvs[i][nid] = pr
    
    centrality_by_intvs_nodes = centrality_by_nodes_intvs.T
    print(centrality_by_intvs_nodes.shape)
    print(centrality_by_intvs_nodes)

    return centrality_by_intvs_nodes








def choose_best_window_size_degree(df, adj_list):
    print("Choosing best window size...")

    src_l = df.src.values
    dst_l = df.dst.values
    ts_l = df.ts.values    
    e_idx_l = df.idx.values

    timestamps = np.array(df.ts.values)
    timestamps_lag = np.zeros(timestamps.shape[0])
    timestamps_lag[1:] = timestamps[:-1]
    timestamps_gap = timestamps-timestamps_lag

    avg_gap = timestamps_gap.mean()
    print("average time gap for all edges: {}".format(avg_gap))
    baseline = np.ceil(avg_gap/10) * 10
    print("using {} as baseline".format(baseline))

    best_entropy = 0
    best_window_size = 0
    for factor in range(1, 21):
        window_size = factor * baseline
        degree_by_intvs = get_degree_by_intvs_nodes(adj_list, window_size)

        n_neighbors_in_last_intv_l = []
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            cur_intv = int(ts // window_size)
            pre_intv = 0 if cur_intv == 0 else cur_intv - 1
            val = degree_by_intvs[src][cur_intv] - degree_by_intvs[src][pre_intv]
            n_neighbors_in_last_intv_l.append(val)
        
        hist, bin_edges = np.histogram(n_neighbors_in_last_intv_l, bins=10, density=True)
        hist = hist + 1e-5
        entropy = -np.sum(hist * np.log(hist))
        print("entropy for window size {}={}*{}: {}".format(window_size, factor, baseline, entropy))

        if entropy > best_entropy:
            best_entropy = entropy
            best_window_size = window_size

    print("best window size: {}, best entropy: {}\n".format(best_window_size, best_entropy))
    return best_window_size



def choose_best_window_size_pagerank(df, adj_list):
    ts_min, ts_max = df.ts.min(), df.ts.max()
    n_intvs = 1000
    window_size = np.ceil((ts_max - ts_min) / n_intvs)
    return window_size


def choose_best_window_size_closeness(df, adj_list):
    ts_min, ts_max = df.ts.min(), df.ts.max()
    n_intvs = 100
    window_size = np.ceil((ts_max - ts_min) / n_intvs)
    return window_size