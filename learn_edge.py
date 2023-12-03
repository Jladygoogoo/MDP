"""Unified interface to all dynamic graph model experiments"""
import os
import time
import pickle
import logging
import sys
import argparse
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from utils import EarlyStopMonitor
from dataloader import *
from activity import ActivityModule
from module import DACT

import warnings
warnings.filterwarnings('ignore')

### Argument and global variables
parser = argparse.ArgumentParser("Interface for TGAT experiments on link predictions")
parser.add_argument("-d", "--data", type=str, help="data sources to use, try wikipedia or reddit", default="wikipedia")
parser.add_argument("--bs", type=int, default=200, help="batch_size")
parser.add_argument("--n_neighbors", type=int, default=20, help="number of neighbors to sample")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--n_runs", type=int, default=1, help="")
parser.add_argument("--n_layers", type=int, default=2, help="number of network layers")
parser.add_argument("--n_patience", type=int, default=5, help="")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
parser.add_argument("--gpu", type=int, default=0, help="idx for the gpu to use")

# added
parser.add_argument("--wandb", action="store_true", help="use wandb to record")
parser.add_argument("--window_size", type=int, help="", nargs='+')
parser.add_argument("--n_windows", type=int, help="")
parser.add_argument("--feat_dim", type=int, help="")
parser.add_argument("--active_direction", type=str, default="both", choices=["both", "src", "dst"], help="active direction when caculating node importance")
parser.add_argument("--method", type=str, default="degree", choices=["degree", "pagerank", "closeness", "eigv"], help="method for caculating node importance")
parser.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "lstm"], help="")
parser.add_argument("--use_semantic", action="store_true")




try:
    args = parser.parse_args()
    print(args)
except:
    parser.print_help()
    sys.exit(0)
args.uniform = True



def eval_one_epoch(hint, model, sampler, dataloader):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        model = model.eval()
        for i, data in enumerate(dataloader):
            batch_src_idx, batch_dst_idx, batch_ts, batch_edge_idx, batch_label = data
            batch_src_idx = batch_src_idx.numpy()
            batch_dst_idx = batch_dst_idx.numpy()
            batch_ts = batch_ts.numpy()
            batch_label = batch_label.numpy()            
            if len(batch_src_idx) <= 1:
                break

            size = len(batch_src_idx)
            src_l_fake, batch_neg_idx = sampler.sample(size)

            pos_prob, neg_prob = model.contrast(batch_src_idx, batch_dst_idx, batch_neg_idx, batch_ts)
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)


# device
device = torch.device("cuda:{}".format(args.gpu) if args.gpu>=0 else "cpu")

act_module = ActivityModule(
    data_name=args.data, device=device, 
    n_windows=args.n_windows, window_size_l=args.window_size, 
    feat_dim=args.feat_dim, active_direction=args.active_direction, 
    method=args.method, encoder=args.encoder, lstm_seq_len=3)

project_name = "pure-act"
project_name += "_s{}_n{}_d{}_{}_{}".format(
    '-'.join(map(str,act_module.window_size_l)), act_module.n_windows, act_module.feat_dim, args.method, args.active_direction)
project_name += "_{}".format(args.data)

if args.wandb:
    wandb.login()
    wandb.init(
        project="activity-dynamic",
        name=project_name,
        config=args,
        tags=[args.data])

MODEL_SAVE_PATH = f"./saved_models/{project_name}.pth"
get_checkpoint_path = lambda epoch: f"./saved_checkpoints/{project_name}-{epoch}.pth"

# load data
df = pd.read_csv("./processed/{}/{}_new.csv".format(args.data, args.data))
if args.use_semantic:
    raw_node_feat = np.load("./processed/{}/{}_node.npy".format(args.data, args.data))
else:
    raw_node_feat = None
# raw_edge_feat = np.load("./processed/{}/{}_edge.npy".format(args.data, args.data))

# train, validation, test 
# mask_train = False if args.data == "dblp" else True
valid_train_flag, valid_val_flag, valid_test_flag, nn_val_flag, nn_test_flag, new_node_set = \
    train_val_test_split(df)

# dataloaders
train_dataloader, val_dataloader, nn_val_dataloader, test_dataloader, nn_test_dataloader = \
    get_dataloaders(df, args, valid_train_flag, valid_val_flag, valid_test_flag, nn_val_flag, nn_test_flag)

# neighbor finders
train_ngh_finder, full_ngh_finder = get_neighbor_finders(df, args, valid_train_flag)

# rand edge samplers
train_rand_sampler, val_rand_sampler, nn_val_rand_sampler, test_rand_sampler, nn_test_rand_sampler = \
    get_rand_samplers(df, args, valid_train_flag, nn_val_flag, nn_test_flag)



# for data in train_dataloader:
#     print(data)
#     break
# sys.exit(0)

# train
ts = []
for k in range(args.n_runs): # runs
    print("Run {} start training...".format(k))
    print("train batch size: {}, train num of batched per epoch: {}".format(args.bs, len(train_dataloader)))

    # model initialize
    # if act_module:
    #     act_module.reset_encoder()

    model = DACT(act_module, device, raw_node_feat, args.dropout)
    parameters = list(model.parameters())
    parameters += act_module.get_parameters()
    num_parameters =  sum(param.numel() for param in parameters)
    print("Parameters number:", num_parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    criterion = torch.nn.BCELoss()
    model = model.to(device)

    early_stopper = EarlyStopMonitor(max_round=args.n_patience)

    for epoch in range(args.n_epochs):

        model.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        print("Start {} epoch".format(epoch))

        t = time.time()
        for i, data in tqdm(enumerate(train_dataloader)):

            batch_src_idx, batch_dst_idx, batch_ts, batch_edge_idx, batch_label = data
            batch_src_idx = batch_src_idx.numpy()
            batch_dst_idx = batch_dst_idx.numpy()
            batch_ts = batch_ts.numpy()
            batch_label = batch_label.numpy()

            size = len(batch_src_idx)
            _, batch_neg_idx = train_rand_sampler.sample(size) # 1j: 但是随机采样得到的不一定是严格的负样本
            
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            model = model.train()
            pos_prob, neg_prob = model.contrast(batch_src_idx, batch_dst_idx, batch_neg_idx, batch_ts)
            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            if args.wandb:
                wandb.log({"loss(batch)": loss.item()})

            # train results
            with torch.no_grad():
                model = model.eval()
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                auc.append(roc_auc_score(true_label, pred_score))
                # f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
        print("Epoch train time:", time.time()-t)
        ts.append(time.time()-t)

        # validation results
        model.ngh_finder = full_ngh_finder
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch("val for old nodes", model, val_rand_sampler, test_dataloader)
        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch("val for new nodes", model, val_rand_sampler, nn_test_dataloader)
            
        print("epoch: {}:".format(epoch))
        print("Epoch mean loss: {}".format(np.mean(m_loss)))
        print("train acc: {}, val acc: {}, new node val acc: {}".format(np.mean(acc), val_acc, nn_val_acc))
        print("train auc: {}, val auc: {}, new node val auc: {}".format(np.mean(auc), val_auc, nn_val_auc))
        print("train ap: {}, val ap: {}, new node val ap: {}".format(np.mean(ap), val_ap, nn_val_ap))
        # print("train f1: {}, val f1: {}, new node val f1: {}".format(np.mean(f1), val_f1, nn_val_f1))

        if args.wandb:
            wandb.log({
                "loss(epoch)": np.mean(m_loss),
                "acc(epoch)/train": np.mean(acc),
                "acc(epoch)/val": val_acc,
                "acc(epoch)/val(new node)": nn_val_acc,
                "auc(epoch)/train": np.mean(auc),
                "auc(epoch)/val": val_auc,
                "auc(epoch)/val(new node)": nn_val_auc,
                "ap(epoch)/train": np.mean(ap),
                "ap(epoch)/val": val_ap,
                "ap(epoch)/val(new node)": nn_val_ap
            })    

        if early_stopper.early_stop_check(val_ap):
            print("No improvment over {} epochs, stop training".format(early_stopper.max_round))
            print(f"Loading the best model at epoch {early_stopper.best_epoch}")
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded the best model at epoch {early_stopper.best_epoch} for inference")
            model.eval()
            break
        else:
            torch.save(model.state_dict(), get_checkpoint_path(epoch))


    # test results
    model.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch("test for old nodes", model, test_rand_sampler, test_dataloader)
    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch("test for new nodes", model, nn_test_rand_sampler, nn_test_dataloader)

    print("Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}".format(test_acc, test_auc, test_ap))
    print("Test statistics: New nodes -- acc: {}, auc: {}, ap: {}".format(nn_test_acc, nn_test_auc, nn_test_ap))

    print("Avg epoch train time:", np.mean(ts))
    print("Parameters number:", num_parameters)


    if args.wandb:
        wandb.log({
            "acc(epoch)/test": test_acc,
            "acc(epoch)/test(new node)": nn_test_acc,
            "auc(epoch)/test": test_auc,
            "auc(epoch)/test(new node)": nn_test_auc,
            "ap(epoch)/test": test_ap,
            "ap(epoch)/test(new node)": nn_test_ap
        })           

    print("Saving model model")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("model models saved")

 



