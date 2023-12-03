import os
import json
import torch
import gensim
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch


def preprocess(data_name):
    dst_list, src_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        # print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            
            src = int(e[0])
            dst = int(e[1])

            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])
            
            dst_list.append(dst)
            src_list.append(src)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'src': dst_list, 
                         'dst':src_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)



def reindex(df):
    assert(df.dst.max() - df.dst.min() + 1 == len(df.dst.unique()))
    assert(df.src.max() - df.src.min() + 1 == len(df.src.unique()))
    
    upper_u = df.dst.max() + 1
    new_src = df.src + upper_u
    
    new_df = df.copy()
    
    new_df.src = new_src
    new_df.dst += 1
    new_df.src += 1
    new_df.idx += 1
    
    return new_df



def preprocess_reindex_dblp(filepath):
    '''
    处理来自 M2DNE 的数据，每行为一条：src_node dst_node timestamp
    dblp.txt: src_node 于 timestamp 引用 dts_node
    '''
    data = open(filepath).read().splitlines()
    src_list = []
    dst_list = []
    ts_list = []
    label_list = []
    idx_list = []

    node_set = set()

    for item in data:
        node_set.add(int(item.split()[0]))
        node_set.add(int(item.split()[1]))
    
    print("total nodes num:", len(node_set))
    print("node index from {} to {}".format(1, len(node_set)))
    
    # 从1开始顺序编码
    node_set = sorted(list(node_set))
    new_node_idx = dict([(n,i) for i,n in enumerate(node_set, start=1)])
    for i, item in enumerate(data, start=1):
        item_split = item.split()
        src_list.append(new_node_idx[int(item_split[0])])
        dst_list.append(new_node_idx[int(item_split[1])])
        ts_list.append(float(item_split[2]))
        label_list.append(0)
        idx_list.append(i)

    df = pd.DataFrame({
        "src": src_list, 
        "dst":dst_list, 
        "ts":ts_list, 
        "label":label_list, 
        "idx":idx_list
    })

    df.sort_values(by="ts", inplace=True)

    return df



def preprocess_reindex_hepth_trend(filepath):
    '''
    处理来自 TREND 的数据，每行为一条：src_node dst_node timestamp
    dblp.txt: src_node 于 timestamp 引用 dts_node    
    '''
    data = torch.load(filepath)
    df = pd.DataFrame({
        "src": data[:,0].type(torch.int).numpy(),
        "dst": data[:,1].type(torch.int).numpy(),
        "ts": data[:,2].type(torch.float).numpy(),
    })
    df["src"] += 1
    df["dst"] += 1
    df["label"] = [0] * len(df)
    df["idx"] = list(range(1, len(df)+1))

    nid_set = set(list(df.dst.values) + list(df.src.values))
    # print(np.max(np.array(list(nid_set))), len(nid_set))
    assert(np.max(np.array(list(nid_set))) == len(nid_set))

    return df


def preprocess_reindex_hepth(rel_filepath, date_filepath, abs_dirpath):
    rels = [line.split('\t') for line in open(rel_filepath).read().splitlines() if '#' not in line]
    src_l = np.array([int(p[0]) for p in rels])
    dst_l = np.array([int(p[1]) for p in rels])

    dates = [line.split('\t') for line in open(date_filepath).read().splitlines() if '#' not in line]
    # print(date_to_special_ts(dates[0][1]))
    ts_dict = dict([(int(p[0]), date_to_special_ts(p[1])) for p in dates])

    has_ts_node_set = set(ts_dict.keys())

    total_node_set = set(src_l).union(dst_l)
    print(np.max(np.array(len(total_node_set))), len(total_node_set)) # 9912293 27770 需要重新编码
    print(len(has_ts_node_set)) # 37621

    valid_node_flag = np.array(list(map(lambda x: x in has_ts_node_set, src_l)))
    # print(valid_node_flag)
    valid_src_l = src_l[valid_node_flag]
    valid_dst_l = dst_l[valid_node_flag]
    valid_ts_l = np.array(list(map(lambda n:ts_dict[n], valid_src_l)))

    df = pd.DataFrame({
        "src": valid_src_l, 
        "dst": valid_dst_l, 
        "ts": valid_ts_l,
        "label": [0] * len(valid_dst_l)
    })

    df.sort_values(by="ts", inplace=True, ignore_index=True)
    df["idx"] =  list(range(1, len(valid_dst_l)+1))

    valid_node_set = list(set(df.src.values).union(df.dst.values))
    print("num of valid nodes: {}".format(len(valid_node_set))) # 18477
    # 得到的结点数量大于 TREND 的原因：这里只要求 src 有时间戳，所以最后的结点集并不是都有对应的发表时间

    # reindex
    valid_node_reindex = dict(zip(valid_node_set, range(1, len(valid_node_set)+1)))
    df.src = df.src.map(lambda n: valid_node_reindex[n])
    df.dst = df.dst.map(lambda n: valid_node_reindex[n])

    # generate node features
    reindex_node_absts = [''] * (len(valid_node_set) + 1)
    count_has_abst = 0
    for root, dir, files in os.walk(abs_dirpath):
        for file in files:
            if ".abs" not in file: continue
            filepath = os.path.join(root, file)
            node_id = int(file.split('.')[0])
            if node_id not in valid_node_set:
                continue
            file_content = open(filepath).read().replace('\n', ' ').strip()
            splitted_content = list(filter(None, file_content.split("\\")))
            abst = ''.join(splitted_content[2:])
            if len(abst) < 10:
                print(filepath)
                print(splitted_content)
                continue
            reindex_node_absts[valid_node_reindex[node_id]] = abst
            count_has_abst += 1

    print("num of nodes that have abstracts: {}".format(count_has_abst))


    train_data = []
    for i, abst in enumerate(reindex_node_absts):
        tokens = gensim.utils.simple_preprocess(abst)
        train_data.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))

    print("Start training Doc2Vec model...")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=172, min_count=2, epochs=10)                
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./doc2vec/model.pkl")
    # model = gensim.models.doc2vec.Doc2Vec.load("./doc2vec/model.pkl")

    reindex_node_embeds = []
    for doc_id in range(len(train_data)):
        reindex_node_embeds.append(model.infer_vector(train_data[doc_id].words))
    reindex_node_embeds = np.stack(reindex_node_embeds)


    # bert = BertVectorizer()
    # batch_size = 10
    # reindex_node_embeds = []
    # for i in range(0, len(reindex_node_absts), batch_size):
    #     batch_embeds = bert.vectorize(reindex_node_absts[i:i+batch_size])
    #     reindex_node_embeds.append(batch_embeds)
    # reindex_node_embeds = np.concatenate(reindex_node_embeds, axis=0)

    # return df, None, len(valid_node_set)
    return df, reindex_node_embeds, len(valid_node_set)



def date_to_special_ts(s):
    ts = datetime.strptime(s, "%Y-%M-%d").timestamp()
    return int(ts // 10000) # 大概处理到天的精度



class BertVectorizer:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def vectorize(self, batch_text, layers=[-3,-2,-1], mode="mean", return_tensor=False):
        '''
        获取BERT词向量
        param: layers: 指定参与计算词向量的隐藏层，-1表示最后一层
        param: mode: 隐藏层合并策略
        return: torch.Tensor, size=[768]
        '''
        input_ = self.tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)
        input_ = input_.to(self.device)
        output = self.model(**input_)
        # specified_hidden_states = [output.hidden_states[i] for i in layers]
        # specified_embeddings = torch.stack(specified_hidden_states, dim=0)
        # # layers to one strategy
        # if mode == "sum":
        #     token_embeddings = torch.squeeze(torch.sum(specified_embeddings, dim=0))
        # elif mode == "mean":
        #     token_embeddings = torch.squeeze(torch.mean(specified_embeddings, dim=0))        
        # # tokens to one strategy
        # word_embedding = torch.mean(token_embeddings, dim=0)
        last_hidden_state = output.last_hidden_state # shape:(batch_size, 512, 768)
        batch_embeds = last_hidden_state.mean(dim=1) # shape:(batch_size, 768)
        if not return_tensor:
            batch_embeds = batch_embeds.detach().cpu().numpy()
        return batch_embeds
    
    def has_word(self, word):
        return True
    




def run(args):
    data_name = args.data
    print("Preprocessing datasets: {}".format(data_name))

    OUT_DF = './processed/{}/{}_new.csv'.format(data_name, data_name)
    OUT_FEAT = './processed/{}/{}_edge.npy'.format(data_name, data_name)
    OUT_NODE_FEAT = './processed/{}/{}_node.npy'.format(data_name, data_name)
    
    if data_name in ["wikipedia", "reddit"]:
        PATH = './processed/{}.csv'.format(data_name)
        df, edge_feat = preprocess(PATH)
        empty = np.zeros(edge_feat.shape[1])[np.newaxis, :]
        edge_feat = np.vstack([empty, edge_feat])
        new_df = reindex(df)
        max_idx = max(new_df.dst.max(), new_df.src.max())
        node_feat = np.zeros((max_idx + 1, edge_feat.shape[1]))        

    elif data_name == "dblp":
        PATH = './processed/{}.txt'.format(data_name)
        feat_dim=150
        new_df = preprocess_reindex_dblp(PATH)
        edge_feat = np.zeros((len(new_df)+1, feat_dim))
        max_idx = max(new_df.dst.max(), new_df.src.max())
        node_feat = np.zeros((max_idx + 1, edge_feat.shape[1]))        

    elif data_name == "hepth":
        feat_dim = 172
        rel_filepath = "./processed/hepth/cit-HepTh.txt"
        date_filepath = "processed/hepth/cit-HepTh-dates.txt"
        abs_dirpath = "processed/hepth/cit-HepTh-abstracts/"
        new_df, node_feat, num_nodes = preprocess_reindex_hepth(rel_filepath, date_filepath, abs_dirpath)
        edge_feat = np.zeros((len(new_df)+1, feat_dim))
        max_idx = max(new_df.dst.max(), new_df.src.max())

    
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, edge_feat)
    np.save(OUT_NODE_FEAT, node_feat)

    print("number of unique src nodes: {}".format(len(new_df.src.unique())))
    print("number of unique dst nodes: {}".format(len(new_df.dst.unique())))
    print("number of edges: {}".format(len(new_df)))
    print("node features shape: {}".format(node_feat.shape))
    print("edge features shape: {}".format(edge_feat.shape))
    

def check_label(data):
    filepath = "./processed/ml_{}.csv".format(data)
    df = pd.read_csv(filepath)
    print(df.label.value_counts())
    '''
    wiki:
    0    157257
    1       217
    reddit:
    0    672081
    1       366    
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default="wikipedia")
    parser.add_argument("--rand_node_feat_dim", type=int, default=150, help="dimension of random node feature")
    parser.add_argument("--rand_edge_feat_dim", type=int, default=150, help="dimension of random node feature")
    parser.add_argument("--random_feat_mode", type=str, choices=["zero", "random"], default="zero")

    args = parser.parse_args()
    print(args)
    run(args)


# bert = BertVectorizer()
# texts = ["  We introduce a new 1-matrix model with arbitrary potential and the matrix-valued background field. Its partition function is a $\tau$-function of KP-hierarchy, subjected to a kind of ${\cal L}_{-1}$-constraint. Moreover, partition function behaves smoothly in the limit of infinitely large matrices. If the potential is equal to $X^{K+1}$, this partition function becomes a $\tau$-function of $K$-reduced KP-hierarchy, obeying a set of ${\cal W} _K$-algebra constraints identical to those conjectured in \cite{FKN91} for double-scaling continuum limit of $(K-1)$-matrix model. In the case of $K=2$ the statement reduces to the early established \cite{MMM91b} relation between Kontsevich model and the ordinary $2d$ quantum gravity . Kontsevich model with generic potential may be considered as interpolation between all the models of $2d$ quantum gravity with $c<1$ preserving the property of integrability and the analogue of string equation."]
# texts.append("  The Ward identities in Kontsevich-like 1-matrix models are used to prove at the level of discrete matrix models the suggestion of Gava and Narain, which relates the degree of potential in asymmetric 2-matrix model to the form of $\cal W$-constraints imposed on its partition function.")
# bert.vectorize(texts)


