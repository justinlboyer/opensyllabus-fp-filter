import click
import mlflow
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import utils


@click.command(help="Path to raw data")
@click.argument("pth") 
def workflow(pth):
    with mlflow.start_run(run_name='process_data'):
        csv_path = './data/processed/'
        utils.make_path(csv_path)

        df = load_raw_data(pth)
        df = calculate_lengths(df)
        train_df = df[df['split']=='train']
        mid_cnt_path = char_grams(train_df, 'middle', 2, 2)
        mlflow.log_artifact(mid_cnt_path, 'vec-dir')
        df.to_csv(csv_path + 'matches.csv', index=False)

        mlflow.log_artifact(csv_path, 'processed-data-dir')


def load_raw_data(pth):
    return pd.read_json(pth, lines=True)

def char_grams(train_df, col, n_start, n_stop):
    cvec = CountVectorizer(analyzer='char_wb', ngram_range= (n_start,n_stop))
    count = cvec.fit(train_df[col])
    cnt_path = utils.serialize_feature_extractor(count, './models/', f'count_{col}.joblib')
    return cnt_path



def calculate_lengths(df):
    df['author_len'] = df.author.str.len()
    df['title_len'] = df.title.str.len()
    df['middle_len'] = df.middle.str.len()
    return df

if __name__ == '__main__':
    workflow()