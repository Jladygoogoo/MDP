import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd

from graph import NeighborFinder, RandEdgeSampler

random.seed(2023)
torch.manual_seed(2023)


class MyDataset(Dataset):
    def __init__(self, src_l, dst_l, ts_l, e_idx_l, label_l) -> None:
        super().__init__()
        self.src_l = src_l
        self.dst_l = dst_l
        self.ts_l = ts_l
        self.e_idx_l = e_idx_l
        self.label_l = label_l
    
    def __getitem__(self, index):
        return self.src_l[index], self.dst_l[index], self.ts_l[index], self.e_idx_l[index], self.label_l[index]

    def __len__(self):
        return len(self.src_l)


def train_val_test_split(df, mask_train=True):
    src_l = df.src.values
    dst_l = df.dst.values
    ts_l = df.ts.values

    # train, val, test split
    # 按照时间粗划分
    val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

    total_node_set = set(np.unique(np.hstack([src_l, dst_l])))
    num_total_unique_nodes = len(total_node_set)

    # 在训练集中 mask 10% 来自测试集的点
    if mask_train:
        mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    else:
        mask_node_set = set()
    mask_src_flag = df.src.map(lambda x: x in mask_node_set).values
    mask_dst_flag = df.dst.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    
    train_node_set = set(src_l[valid_train_flag]).union(dst_l[valid_train_flag]) # 所有训练集中出现过的结点
    assert(len(train_node_set - mask_node_set) == len(train_node_set))
    new_node_set = total_node_set - train_node_set # 所有在训练集中没有出现过的点

    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

    val_node_set = set(src_l[valid_val_flag]).union(dst_l[valid_val_flag])
    test_node_set = set(src_l[valid_test_flag]).union(dst_l[valid_test_flag])

    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    print("Total: num of edges: {}, num of nodes: {}".format(len(src_l), num_total_unique_nodes))
    print("Train: num of edges: {}, num of nodes: {}".format(len(src_l[valid_train_flag]), len(train_node_set)))
    print("Val: num of edges: {}(new:{}), num of nodes: {}(new:{})".format(
        len(src_l[valid_val_flag]), len(src_l[nn_val_flag]), len(val_node_set), len(val_node_set.intersection(new_node_set))))
    print("Test: num of edges: {}(new:{}), num of nodes: {}(new:{})".format(
        len(src_l[valid_test_flag]), len(src_l[nn_test_flag]), len(test_node_set), len(test_node_set.intersection(new_node_set))))
    print()

    return valid_train_flag, valid_val_flag, valid_test_flag, nn_val_flag, nn_test_flag, new_node_set


def get_dataloaders(df, args, valid_train_flag, valid_val_flag, valid_test_flag, nn_val_flag, nn_test_flag):
    batch_size = args.bs

    src_l = df.src.values
    dst_l = df.dst.values
    e_idx_l = df.idx.values
    label_l = df.label.values
    ts_l = df.ts.values

    # train
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    # val
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]
    
    nn_val_src_l = src_l[nn_val_flag] # 至少一个结点没有在训练集出现过
    nn_val_dst_l = dst_l[nn_val_flag]
    nn_val_ts_l = ts_l[nn_val_flag]
    nn_val_e_idx_l = e_idx_l[nn_val_flag]
    nn_val_label_l = label_l[nn_val_flag]

    # test
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]

    nn_test_src_l = src_l[nn_test_flag] # 至少一个结点没有在训练集出现过
    nn_test_dst_l = dst_l[nn_test_flag]
    nn_test_ts_l = ts_l[nn_test_flag]
    nn_test_e_idx_l = e_idx_l[nn_test_flag]
    nn_test_label_l = label_l[nn_test_flag]    


    train_dataset = MyDataset(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l)    
    val_dataset = MyDataset(val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l)    
    nn_val_dataset = MyDataset(nn_val_src_l, nn_val_dst_l, nn_val_ts_l, nn_val_e_idx_l, nn_val_label_l)    
    test_dataset = MyDataset(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l)    
    nn_test_dataset = MyDataset(nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_e_idx_l, nn_test_label_l)    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    nn_val_dataloader = DataLoader(nn_val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    nn_test_dataloader = DataLoader(nn_test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, nn_val_dataloader, test_dataloader, nn_test_dataloader



def get_dataloaders_ngh_finders_cls(df, args):
    batch_size = args.bs

    src_l = df.src.values
    dst_l = df.dst.values
    e_idx_l = df.idx.values
    label_l = df.label.values
    ts_l = df.ts.values

    use_val = False

    # train, val, test split
    # 按照时间粗划分
    val_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

    train_flag = ts_l <= val_time if use_val else ts_l <= test_time
    test_flag = ts_l > test_time
    val_flag = np.logical_and(ts_l <= test_time, ts_l > val_time) if use_val else test_flag

    # train
    train_src_l = src_l[train_flag]
    train_dst_l = dst_l[train_flag]
    train_ts_l = ts_l[train_flag]
    train_e_idx_l = e_idx_l[train_flag]
    train_label_l = label_l[train_flag]

    # val
    val_src_l = src_l[val_flag]
    val_dst_l = dst_l[val_flag]
    val_ts_l = ts_l[val_flag]
    val_e_idx_l = e_idx_l[val_flag]
    val_label_l = label_l[val_flag]

    # test
    test_src_l = src_l[test_flag]
    test_dst_l = dst_l[test_flag]
    test_ts_l = ts_l[test_flag]
    test_e_idx_l = e_idx_l[test_flag]
    test_label_l = label_l[test_flag]

    train_dataset = MyDataset(train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l)    
    val_dataset = MyDataset(val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l)    
    test_dataset = MyDataset(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l)    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    max_idx = max(src_l.max(), dst_l.max())

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=args.uniform)

    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=args.uniform)    

    return train_dataloader, val_dataloader, test_dataloader, train_ngh_finder, full_ngh_finder



def get_neighbor_finders(df, args, valid_train_flag):
    src_l = df.src.values
    dst_l = df.dst.values
    e_idx_l = df.idx.values
    ts_l = df.ts.values

    max_idx = max(src_l.max(), dst_l.max())

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=args.uniform)

    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=args.uniform)

    return train_ngh_finder, full_ngh_finder


def get_rand_samplers(df, args, valid_train_flag, nn_val_flag, nn_test_flag):
    src_l = df.src.values
    dst_l = df.dst.values

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    nn_val_src_l = src_l[nn_val_flag]
    nn_val_dst_l = dst_l[nn_val_flag]
    nn_test_src_l = src_l[nn_test_flag]
    nn_test_dst_l = dst_l[nn_test_flag]


    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
    test_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)    

    return train_rand_sampler, val_rand_sampler, nn_val_rand_sampler, test_rand_sampler, nn_test_rand_sampler
