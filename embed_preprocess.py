import pandas as pd
import torch
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

def process(df, rewrite=False):
    pytorch_data_dir = './data/pytorch'
    if rewrite:
        split_n_write(df)

    TEXT, LABEL, train, val, test = create_pytorch_dataset(pytorch_data_dir)



def filter_write(df, keep_cols, col, dir):
    out = df.loc[df['split']== col, keep_cols]
    out.to_csv(f"{dir}{col}.csv", index=False)

def split_n_write(df, keep_cols = ['author', 'middle', 'title', 'bin_label'], pt_data_dir = './data/pytorch/'):
    # need to mkdir if not exists
    cols = ['train', 'dev', 'test']

    df['bin_label'] = df['label'].astype(int) # terrible place to do this, but just putting it here for now
    for col in cols:
        filter_write(df, keep_cols, col, pt_data_dir)



def create_pytorch_dataset(pt_data_dir):
    tokenize = lambda x: list(x)
    TEXT = Field(sequential=True, tokenize=tokenize)
    LABEL = Field(sequential=False, use_vocab=False)

    matches_fields = [('author', TEXT)
                        , ('middle', TEXT)
                        , ('title', TEXT)
                        , ('bin_label', LABEL)]

    trn, vld, tst = TabularDataset.splits(pt_data_dir
                                    , format='csv'
                                    , train='train.csv'
                                    , validation='dev.csv'
                                    , test='test.csv'
                                    , fields=matches_fields
                                    , skip_header=True)


    return TEXT, LABEL, trn, vld, tst


if __name__ == '__main__':
    df = pd.read_csv('./data/processed/matches.json')
    train_df = df[df['split']=='train']
    process(train_df)