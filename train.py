import click
import logging
import mlflow
import os
import pandas as pd


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

@click.command()
@click.argument('pth')
def workflow(pth):
    try:
        df = load_processed_data(pth)
    except Exception as e:
        logger.exception(f"Unable to load data\n{e}")
    print(df.columns)

def load_processed_data(pth):
    file_pth = os.path.join(pth, 'matches.csv')
    if os.path.isfile(file_pth):
        return pd.read_csv(file_pth)
    else:
        print(f'Path {file_pth} does not exist') # need to handle

if __name__ == '__main__':
    workflow()
