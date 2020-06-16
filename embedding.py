import embed_preprocess
import pandas as pd
# import sif_embedding
import string
import torch
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def workflow(df):
    pytorch_data_dir = './data/pytorch/'

    train_iter, val_iter, test_iter, TEXT = process(pytorch_data_dir)

    vocab_size = len(TEXT.vocab)


    

class Embd(nn.Module):
    def __init__(self, vocab_size, emb_size=128):
        super(Embd, self).__init__()
        self.emb_size = emb_size

        self.em = nn.Embedding(vocab_size, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, 16, bidirectional=True, num_layers=1)
        self.last = nn.Linear(32, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.em(x)
        # print(x.shape)
        x, _ = self.lstm(x)
        # print(x.shape)
        x = x[-1,:,:]
        # print(x.shape)
        x = self.last(x)
        # print(x.shape)
        return x


def process(pytorch_data_dir):
    TEXT, LABEL, train, val, test = embed_preprocess.create_pytorch_dataset(pytorch_data_dir)

    TEXT.build_vocab(train)
    train_iter, val_iter, test_iter =  build_iterator(train, val, test)
    return train_iter, val_iter, test_iter, TEXT



def build_iterator(train, val, test, batch_size=64):
    train_iter, val_iter = BucketIterator.splits((train, val)
                                        , batch_size=batch_size
                                        , device='cpu'
                                        , sort_key=lambda x: len(x.title)
                                        , sort_within_batch=False
                                        , repeat=True)
    test_iter = Iterator(test, batch_size=batch_size, device='cpu', sort=False, sort_within_batch=False, repeat=True)

    return train_iter, val_iter, test_iter




def predict_cat():
    '''Train embedding by just predicting if the word is title or author, then fine tune on actual labels'''
    return None






if __name__ == '__main__':
    df = pd.read_csv('./data/processed/matches.csv')
    workflow(df)