import click
import mlflow
import pandas as pd
from sklearn.metrics import f1_score
import time
# from sklearn.metrics import confusion_matrix

@click.command()
@click.argument('pth') #/home/j/projects/opensyllabus/mlruns/0/24ca970cae9641c686dd5056e7a66871/artifacts/processed-data-dir/matches.csv
@click.argument('mlt')
@click.argument('tlt')
def workflow(pth, mlt, tlt):
    with mlflow.start_run(run_name='baseline'):
        mlt = int(mlt)
        tlt = int(tlt)
        mlflow.log_param('middle_len_threshold', mlt)
        mlflow.log_param('title_len_threshold', tlt)
        df = pd.read_csv(pth)
        start = time.time()
        df['preds'] = df.apply(model
                                , middle_len_threshold=mlt
                                , title_len_threshold=tlt
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