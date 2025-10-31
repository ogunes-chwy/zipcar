import pandas as pd
from snowflake_utils import execute_query_and_return_formatted_data, execute_query
import os
from datetime import date, timedelta
import logging
import yaml


# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


def get_unpadded_dea(
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = execute_query_and_return_formatted_data(
        query_path='./data',
        query_name='unpadded_edd_dea',
        start_date=s,
        end_date=e,
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path),
        partition_cols='delivery_date',
        existing_data_behavior='delete_matching'
        )

    logger.info('Unpadded DEA table created and saved.')


def get_backlog(
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = execute_query_and_return_formatted_data(
        query_path='./data',
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

    with open("./configs.yaml") as f:
        config = yaml.safe_load(f)

    lookback_day_count = config['EXECUTION_DATA']['lookback_day_count']
    start_date = config['EXECUTION_DATA']['start_date']
    end_date = config['EXECUTION_DATA']['end_date']

    if start_date and end_date:
        pass
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_day_count)
        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')

    logger.info('Running for %s - %s ...', start_date, end_date)

    get_unpadded_dea(
        './data/execution_data/unpadded_dea',
        s=start_date,
        e=end_date
        )

    get_backlog(
        './data/execution_data/backlog',
        s=start_date,
        e=end_date
        )

