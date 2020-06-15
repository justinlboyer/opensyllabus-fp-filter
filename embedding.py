import embed_preprocess
import pandas as pd
# import sif_embedding
import string
import torch
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn



def workflow(df):
    pytorch_data_dir = './data/pytorch/'

    train_iter, val_iter, test_iter = preprocess(pytorch_data_dir)
    

def preprocess(pytorch_data_dir):
    TEXT, LABEL, train, val, test = embed_preprocess.create_pytorch_dataset(pytorch_data_dir)

    TEXT.build_vocab(train)
    train_iter, val_iter, test_iter =  build_iterator(train, val, test)
    return train_iter, val_iter, test_iter



def build_iterator(train, val, test, batch_size=64):
    train_iter, val_iter = BucketIterator((train, val)
                                        , batch_size=(batch_size, batch_size)
                                        , device=-1
                                        , sort_key=lambda x: len(x.title)
                                        , sort_within_batch=False
                                        , repeat=True)
    test_iter = Iterator(test, batch_size, device=-1, sort=False, sort_within_batch=False, repeat=True)

    return train_iter, val_iter, test_iter


class Embd(nn.Module):
    def __init__(self, n_letters=57, emb_size=512):
        super(Embd, self).__init__()
        self.emb_size = emb_size

        self.em = nn.Embedding(n_letters, emb_size)
        self.sif = None # tbd
        self.last = nn.Linear(512, 1)

    def forward(self, x):
        x = self.em(x)
        x = self.sif(x)
        return nn.LogSoftmax(self.last(x))

def predict_cat():
    '''Train embedding by just predicting if the word is title or author, then fine tune on actual labels'''
    return None


def preprocess(df):
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)





def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor




if __name__ == '__main__':
    df = pd.read_csv('./data/processed/matches.csv')
    workflow(df)