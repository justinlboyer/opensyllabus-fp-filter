import click
import mlflow
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import time
import utils

@click.command()
@click.argument('pth')
@click.argument('vec_pth')
def workflow(pth, vec_pth):
    pth = os.path.join(pth, 'processed/matches.csv')
    df = pd.read_csv(pth)
    
    gaussain_nb_word_len(df)

    df, vec_cols = utils.preprocess(df, vec_pth, 'middle')
    multinominal_nb_char_ngram(df, vec_cols)

    df, vec_cols = utils.preprocess(df, vec_pth, 'middle', tfidf=True)
    gaussian_nb_char_ngram(df, vec_cols)


def gaussain_nb_word_len(df):
    utils.reset_mlflow_run()
    with mlflow.start_run(run_name='gaussian_nb_word_len'):
        cols = ['middle_len', 'title_len', 'author_len']
        
        train_df = df[df['split']=='train']
        gnb = GaussianNB()
        gnb.fit(train_df[cols], train_df['label'])

        test_df = df[df['split']=='test']
        start = time.time()
        preds = gnb.predict(test_df[cols])
        stop = time.time()
        f1 = f1_score(test_df['label'], preds)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('prediction_time', stop-start)



def multinominal_nb_char_ngram(df, vec_cols):
    with mlflow.start_run(run_name='multinomial_nb_bow'):
        train_df = df[df['split']=='train']
        
        mnb = GaussianNB()
        mnb.fit(train_df[vec_cols], train_df['label'])

        test_df = df[df['split']=='test']
        start = time.time()
        preds = mnb.predict(test_df[vec_cols])
        stop = time.time()
        f1 = f1_score(test_df['label'], preds)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('prediction_time', stop-start)

def gaussian_nb_char_ngram(df, vec_cols):
    with mlflow.start_run(run_name='gaussian_nb_bow'):
        train_df = df[df['split']=='train']
        
        mnb = MultinomialNB()
        mnb.fit(train_df[vec_cols], train_df['label'])

        test_df = df[df['split']=='test']
        start = time.time()
        preds = mnb.predict(test_df[vec_cols])
        stop = time.time()
        f1 = f1_score(test_df['label'], preds)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('prediction_time', stop-start)


if __name__ == '__main__':
    workflow()