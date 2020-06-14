from joblib import dump, load
import os
import mlflow
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score

def reset_mlflow_run():
    ''' Hack to get around lack of run_name labelling '''
    mlflow.start_run()
    mlflow.end_run()

def get_and_log_metrics(true, preds):
    f1 = f1_score(true, preds)
    precision = precision_score(true, preds)
    recall = recall_score(true, preds)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('precision_score', precision)
    mlflow.log_metric('recall_score', recall)

def load_vectorizers(pth):
    return load(pth)

def serialize_feature_extractor(feature_extractor, file_path, extractor_name):
    make_path(file_path)
    extractor_path = os.path.join(file_path, extractor_name)
    dump(feature_extractor, extractor_path)
    return extractor_path


def make_path(pth):
    if not os.path.isdir(pth):
        os.mkdir(pth)

def vectorizer_transform(vec, df, col, tfidf):
    cols = vec.get_feature_names()
    vecs = vec.transform(df[col])
    if tfidf:
        tf = TfidfTransformer()
        vecs = tf.fit_transform(vecs)
    vec_df = pd.DataFrame(vecs.todense(), columns=cols)
    return pd.concat([df, vec_df], axis =1), cols

def preprocess(df, vectorizer_paths, col, tfidf=False):
    cnt_mid_vec = load_vectorizers(os.path.join(vectorizer_paths, f'count_{col}.joblib'))
    df, cols = vectorizer_transform(cnt_mid_vec, df, col, tfidf)
    return df, cols