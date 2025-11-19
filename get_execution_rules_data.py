import os
import logging
from datetime import date, timedelta

import yaml

from snowflake_utils import execute_query_and_return_formatted_data
import helper_functions as hf

# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


def get_unpadded_dea(
        prefix,
        query_path,
        output_path,
        s,
        e
        ):
    """
    Query to get Unpadded DEA data from Snowflake. 

    Args:
        output_path: Main folder of Unpadded DEA output in Local.
        s: Start date.
        e: End date.
    """
    logger.info('pulling unpadded DEA...')

    if not hf.path_exists(output_path):
        hf.ensure_directory_exists(output_path)

    if prefix.startswith('s3://'):
        query_path = '.'

    df = execute_query_and_return_formatted_data(
        query_path=query_path,
        query_name='unpadded_edd_dea',
        start_date=s,
        end_date=e,
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path),
        partition_cols='unpadded_edd',
        existing_data_behavior='delete_matching'
        )

    logger.info('Unpadded DEA table created and saved.')


def get_backlog(
        prefix,
        query_path,
        output_path,
        s,
        e
        ):
    """
    Query to get Backlog data from Snowflake. 

    Args:
        output_path: Main folder of Backlog output in Local.
        s: Start date.
        e: End date.
    """
    logger.info('pulling backlog...')

    if not hf.path_exists(output_path):
        hf.ensure_directory_exists(output_path)

    if prefix.startswith('s3://'):
        query_path = '.'

    df = execute_query_and_return_formatted_data(
        query_path=query_path,
        query_name='backlog',
        start_date=s,
        end_date=e,
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path),
        partition_cols='date',
        existing_data_behavior='delete_matching'
        )

    logger.info('backlog table created and saved.')


if __name__ == '__main__':

    with open("./configs.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    PREFIX = config['ENVIRONMENT']['prefix']
    LOOKBACK_DAY_COUNT = config['EXECUTION_DATA']['lookback_day_count']
    START_DATE = config['EXECUTION_DATA']['start_date']
    END_DATE = config['EXECUTION_DATA']['end_date']

    if START_DATE and END_DATE:
        pass
    else:
        END_DATE = date.today()
        START_DATE = END_DATE - timedelta(days=LOOKBACK_DAY_COUNT)
        END_DATE = END_DATE.strftime('%Y-%m-%d')
        START_DATE = START_DATE.strftime('%Y-%m-%d')

    logger.info('Running for %s - %s ...', START_DATE, END_DATE)

    get_unpadded_dea(
        prefix=PREFIX,
        query_path='./sql',
        output_path=f'{PREFIX}/data/execution_data/unpadded_dea',
        s=START_DATE,
        e=END_DATE
        )

    get_backlog(
        prefix=PREFIX,
        query_path='./sql',
        output_path=f'{PREFIX}/data/execution_data/backlog',
        s=START_DATE,
        e=END_DATE
        )

