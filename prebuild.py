from tqdm import tqdm
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

"""
Text Graph Construction.
"""

model_name = './bert-base-uncased'
pad_len = 300
dataset0 = 'data/sst1'
dataset1 = 'train'
dataset2 ='q3'

# # normalize 归一化邻接矩阵
# def normalize_adj(adj):
#     row_sum = np.array(adj.sum(1))
#     with np.errstate(divide='ignore'):
#         d_inv_sqrt = np.power(row_sum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = np.diag(d_inv_sqrt)
#     adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
#     return adj_normalized
#
# # Normalization-euclidean similarity in l2 normal form.(l2范式归一化-欧几里得相似度)
# def normalize_vector(vec):
#     norm = np.linalg.norm(vec)
#     if norm == 0:
#         return vec
#     return vec / norm

# 余弦相似度
def cosine_similarity(vector1, vector2):
    # 计算向量的范数（模）
    norm_vector1 = norm(np.array(vector1))
    norm_vector2 = norm(np.array(vector2))
    # 计算向量的内积
    dot_product = np.dot(vector1, vector2)
    # 计算余弦相似度
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def cut(input,l):
    # Adjust the shape of the input.
    if len(input) > l:
        input = input[:l]
        return input
    else: return input

def pad_adj(adj, pl):
    # Adjust the shape of the adjacency matrix.
    if len(adj) > pl:
        adj = adj[:pl, :pl]
        return torch.from_numpy(adj)
    else:
        adj = torch.from_numpy(adj)
        return torch.nn.functional.pad(adj, (0, pl-adj.size()[0], 0, pl-adj.size()[0]), mode='constant', value=0)

if __name__ =='__main__':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    with open(f"{dataset0}/" + f"{dataset1}.texts_clean.txt", 'r', encoding='utf-8') as rf:
        texts = rf.readlines()
    inputs, inputs_pad = [], []
    adj_cos, adj_cos_pad = [], []
    for text in tqdm(texts):
        bert_model = BertModel.from_pretrained(model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(model_name)

        tokens1 = bert_tokenizer.encode(text, add_special_tokens=True)
        input_ids1 = cut(torch.tensor(tokens1),300)

        with torch.no_grad():
            outputs1 = bert_model(input_ids1.unsqueeze(0))
        embeddings1 = outputs1[0].squeeze(0)
        embedding_np1 = embeddings1.detach().numpy()

        """Calculate the cosine similarity matrix (计算余弦相似度矩阵)"""
        similarity_c = []
        for x in range(len(embedding_np1)):
            for y in range(len(embedding_np1)):
                similarity_c.append(cosine_similarity(embedding_np1[x], embedding_np1[y]))

        sorted_data = sorted(similarity_c)  # Sort the data in ascending order .(对数据进行升序排序)
        n = len(similarity_c)
        q1_index = (n + 1) / 4  # Position of the 1st quartile.(第1四分位数的位置)
        q2_index = (n + 1) / 2  # Position of the 2st quartile(第2四分位数的位置)
        q3_index = (n + 1) * 3 / 4  # Position of the 3st quartile(第3四分位数的位置)
        q = sorted_data[int(q3_index) - 1]
        similarity_c = torch.tensor(similarity_c, dtype=torch.float32)
        adj_c = torch.ones_like(similarity_c)
        zero = torch.zeros_like(adj_c)
        adj = torch.where(similarity_c > q,  adj_c, zero)
        # Generating matrix (生成矩阵)
        cos_adj = np.array(adj).reshape(len(embedding_np1), len(embedding_np1))

        inputs.append(input_ids1)
        adj_cos.append(cos_adj)

    len_inputs = [len(e) for e in inputs]
    adj_cos_pad = [np.array(pad_adj(adj, pad_len)) for adj in adj_cos]
    # save
    np.save(f"{dataset0}/" + f"{dataset1}.cos_adj"+f"_{dataset2}_pad.npy", adj_cos_pad)