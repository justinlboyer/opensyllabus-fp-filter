import click
import embed_preprocess
import mlflow
import numpy as np
import pandas as pd
# import sif_embedding
import time
import torch
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils


@click.command()
@click.argument('epochs', type=click.INT)
@click.argument('learning_rate', type=click.FLOAT)
@click.argument('make_data', type=click.BOOL)
def workflow(epochs, learning_rate, make_data=False):
    pytorch_data_dir = './data/pytorch/'
    utils.make_path(pytorch_data_dir)
    print(f'Training model with {epochs} epochs, learning rate={learning_rate}'.center(90,'~'))
    train_iter, val_iter, test_iter, TEXT = process(pytorch_data_dir, make_data)

    vocab_size = len(TEXT.vocab)

    model = Emb3in(vocab_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    with mlflow.start_run(run_name='deep_learning_test'):
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('learning_rate', learning_rate)
        for ep in range(epochs):
            trainer(model, train_iter, optimizer, criterion, ep)
            evaluate(model, val_iter, criterion, ep, 'val_')
        evaluate(model, test_iter, criterion, ep, 'test_', timed=True)
            


class Emb3in(nn.Module):
    def __init__(self, vocab_size, emb_size=128):
        super(Emb3in, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        self.embd_author = Embd(self.vocab_size)
        self.embd_middle = Embd(self.vocab_size)
        self.embd_title = Embd(self.vocab_size)

        self.last = nn.Linear(3,1)

    def forward(self, author, middle, title):
        out1 = self.embd_author(author)
        out2 = self.embd_middle(middle)
        out3 = self.embd_title(title)
        # print(out3.shape)
        out = torch.cat((out1, out2, out3), 1)
        # print(out.shape)
        out = self.last(out)
        return out
    

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

def trainer(model, train_iter, optimizer, criterion, epoch):
    model.train()
    all_true = []
    all_preds = []
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        out = model(batch.author, batch.middle, batch.title)
        y = batch.bin_label.unsqueeze(1).float()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        mlflow.log_metric('loss', loss.item(), step = i)

        preds = np.round(torch.sigmoid(out).tolist())
        all_preds.extend(preds)
        all_true.extend(y.tolist())

        if i > len(train_iter):
            break
    
    utils.get_and_log_metrics(all_true, all_preds, name='train_', step=epoch)

def evaluate(model, iterator, criterion, epoch, name, timed=False):
    model.eval()
    
    epoch_loss = 0
    if timed:
        timer = 0

    all_preds = []
    all_true = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if timed: 
                start = time.time()
            
            out = model(batch.author, batch.middle, batch.title)
            
            if timed:
                end = time.time()
                timer += end-start

            y = batch.bin_label.unsqueeze(1).float()

            loss = criterion(out, y)

            epoch_loss += loss.item()
            preds = np.round(torch.sigmoid(out).tolist())
            all_preds.extend(preds)
            all_true.extend(y.tolist())

            if i > len(iterator):
                break
        mlflow.log_metric(f'{name}epoch_loss', epoch_loss, step=epoch)
        utils.get_and_log_metrics(all_true, all_preds, name=name, step=epoch)
        if timed:
            mlflow.log_metric('prediction_time', timer)


def process(pytorch_data_dir, make_data):
    if make_data:
        df = pd.read_csv('./data/processed/matches.csv')
        embed_preprocess.split_n_write(df)

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





if __name__ == '__main__':
    workflow()