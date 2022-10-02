import spacy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def tokens_to_vecs(tokens, sent_len, embed_func, emb_size=300):
    glove_matrix = np.zeros((sent_len, emb_size), dtype=float)

    sent_len = min(sent_len, len(tokens))

    for i in range(sent_len):
        glove_matrix[i, :] = embed_func(tokens[i]).vector

    return glove_matrix


class Tweets(Dataset):

    NUM_CLASSES = 3

    CLASS_NAMES = ['negative', 'neutral', 'positive']

    CLASS_TABLE = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __init__(self, sent_len, data_path='data/Tweets.csv', split='train'):
        self.sent_len = sent_len
        self.data = pd.read_csv(data_path)
        self.embedding_func = spacy.load('en_vectors_web_lg')
        split_cur = int(len(self.data) * 0.8)
        if split == 'train':
            self.data = self.data[:split_cur]
        else:
            self.data = self.data[split_cur:]
        self.dict_data = self.data.to_dict()
        for k, v in self.dict_data.items():
            self.dict_data[k] = list(v.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.dict_data['selected_text'][index]).strip()
        class_name = self.dict_data['sentiment'][index]
        tokens = text.split()
        embedding = tokens_to_vecs(tokens, self.sent_len, self.embedding_func)
        label = self.CLASS_TABLE[class_name]

        embedding = torch.as_tensor(embedding, dtype=torch.float32)
        label = torch.as_tensor(np.ascontiguousarray(label, dtype=np.int64))

        dataset_dict = {'txt': embedding, 'txt_gt': label, 'sent_len': tokens}

        return dataset_dict
