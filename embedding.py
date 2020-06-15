import pandas as pd
import string
import torch
import torch.nn as nn


def workflow(df):
    train_df = df[df['split'] == 'train']
    val_df = df[df['split']=='dev']
    test_df = df[df['split']=='test']


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



def preprocess():
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