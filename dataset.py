import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, filename):
        # 数据集初始化
        """
        dataset,labels,len_pad
        mr     ['negtive', 'positive'] 76
        R8     ['ship', 'money-fx', 'grain', 'acq', 'trade', 'earn', 'crude', 'interest'] 400
        SST2   ['0', '1'] 65
        TREC   ['ABBR', 'DESC','ENTY','HUM','LOC','NUM'] 50
        sst1   ['0', '1','2','3','4'] 64
        """
        self.labels = ['0', '1','2','3','4']
        self.labels_id = list(range(len(self.labels)))
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        self.len_pad = 64
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.adj = []
        self.adj_cos = []
        self.load_data(filename)

    def change_adj(self, x,  lp):
        # Adjust the shape of the adjacency matrix.
        if lp > x.size()[1]:
            return torch.nn.functional.pad(x, (0, lp-x.size()[1], 0, lp-x.size()[1]), mode='constant', value=0)
        elif lp < x.size()[1]:
            x_0 = x[:lp, :lp]
            x_1 = x_0.clone()
            del x
            del x_0
            return x_1
        else:
            return x
    def load_data(self, filename):
        # load data.
        print('loading data from:', filename)
        with open(f'{filename}'+'.texts_clean.txt', 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        adj = np.load(f"{filename}.cos_adj_q2_pad.npy")

        for line, a in tqdm(list(zip(lines, adj)), ncols=100):

            token = self.tokenizer(line, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(
                np.array(token['token_type_ids']))
            self.attention_mask.append(
                np.array(token['attention_mask']))

            a = self.change_adj(torch.from_numpy(a), self.len_pad)
            adj_cos = torch.tensor(a, dtype=torch.float32)
            self.adj.append(adj_cos)

        self.label_id = np.load(f"{filename}"+".targets.npy")  # 得到所有文本的标签id
        self.label_id = self.label_id.astype(float)
        self.y = torch.tensor(self.label_id, dtype=torch.long)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.y[index],\
            self.adj[index]


    def __len__(self):
        return len(self.input_ids)

