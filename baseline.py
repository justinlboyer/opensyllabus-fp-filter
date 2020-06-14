import click
import mlflow
import os
import pandas as pd
from sklearn.metrics import f1_score
import time
import utils

@click.command()
@click.argument('pth')
@click.argument('middle_len_threshold')
@click.argument('title_len_threshold')
def workflow(pth, middle_len_threshold, title_len_threshold):
    pth = os.path.join(pth, 'processed/matches.csv')
    utils.reset_mlflow_run()
    with mlflow.start_run(run_name='baseline'):
        middle_len_threshold = int(middle_len_threshold)
        title_len_threshold = int(title_len_threshold)
        mlflow.log_param('middle_len_threshold', middle_len_threshold)
        mlflow.log_param('title_len_threshold', title_len_threshold)
        df = pd.read_csv(pth)
        start = time.time()
        df['preds'] = df.apply(model
                                , middle_len_threshold=middle_len_threshold
                                , title_len_threshold=title_len_threshold
                                , axis=1)
        stop = time.time()
        mlflow.log_metric('prediction_time', stop-start)
        f1 = get_f1(df[df['split']=='test'])
        mlflow.log_metric('f1_score', f1)

def get_f1(test_df):
    return f1_score(test_df['label'], test_df['preds'])


def model(x, **kwargs):
    # think of a better way to set kwargs
    if 'middle_len_threshold' not in kwargs.keys():
        kwargs['middle_len_threshold'] = 13
    if 'title_len_threshold' not in kwargs.keys():
        kwargs['title_len_threshold'] = 0

    if x['middle_len'] >= kwargs['middle_len_threshold']:
        return False
    elif x['title_len'] <= kwargs['title_len_threshold']:
        return False
    else:
        return True



if __name__ == '__main__':
    workflow()