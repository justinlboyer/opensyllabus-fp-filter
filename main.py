import click
import os
import mlflow
import utils

@click.command()
@click.argument('pth')
def workflow(pth):
    load_data_run = mlflow.run('.', 'data_clean'
                                , parameters={'pth': pth})
    
    load_data_run_info = mlflow.tracking.MlflowClient().get_run(load_data_run.run_id)
    processed_data_uri = os.path.join(load_data_run_info.info.artifact_uri, 'processed-data-dir')
    vec_dir_uri = os.path.join(load_data_run_info.info.artifact_uri, 'vec-dir')
    
    baseline_run = mlflow.run('.', 'baseline'
                                , parameters={'pth': processed_data_uri
                                                , 'middle_len_threshold': 12
                                                , 'title_len_threshold': 0})
    nb_run = mlflow.run('.', 'nb'
                            , parameters={'pth': processed_data_uri
                                            , 'vec_pth': vec_dir_uri})

    dl_run = mlflow.run('.', 'emb'
                        , parameters={'epochs': 5
                                        , 'learning_rate': 0.01
                                        , 'make_data': True})


if __name__ == '__main__':
    workflow()