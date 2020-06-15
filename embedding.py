import pandas as pd
import torch


def workflow(df):
    train_df = df[df['split'] == 'train']
    val_df = df[df['split']=='dev']
    test_df = df[df['split']=='test']






if __name__ == '__main__':
    df = pd.read_csv('./data/processed/matches.csv')
    workflow(df)