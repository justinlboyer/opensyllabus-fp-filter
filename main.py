import click
import os

import mlflow

@click.command()
@click.argument('pth')
def workflow(pth):
    with mlflow.start_run():
        load_data_run = mlflow.run('.', 'data_clean'
                                    , parameters={'pth': pth})
        load_data_run_info = mlflow.tracking.MlflowClient().get_run(load_data_run.run_id)
        processed_data_uri = os.path.join(load_data_run_info.info.artifact_uri, 'processed-data-dir')
        baseline_run = mlflow.run('.', 'baseline'
                                    , parameters={'pth': processed_data_uri
                                                    , 'mlt': 13
                                                    , 'tlt': 0})
        train_run = mlflow.run('.', 'train'
                                , parameters={'pth': processed_data_uri})


if __name__ == '__main__':
    workflow()