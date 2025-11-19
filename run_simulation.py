import os
import logging
from datetime import date, timedelta

import pandas as pd
import yaml

from snowflake_utils import execute_multiple_query_and_return_formatted_data
import helper_functions as hf


# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


def generate_ship_map_file(
        prefix,
        query_path,
        output_path,
        scenario,
        s,
        e):
    """
        Generate Ship Map File data and save to Local. 

    Args:
        output_path: Main folder of Ship Map File output in Local. Must exist.
        scenario: Scenario name for simulation. 
            - Must be one of 'baseline', 'remediation', 'expansion'.
        s: Start date of simulation. Must be in 'YYYY-MM-DD' format.
        e: End date of simulation. Must be in 'YYYY-MM-DD' format.
    """

    logger.info('creating SMF table...')

    if not hf.path_exists(output_path + f'/{scenario}'):
        hf.ensure_directory_exists(output_path + f'/{scenario}')

    if prefix.startswith('s3://'):
        query_path = '.'
    else:
        query_path = f'{query_path}/create_smf_tables'

    # read SMF table creation SQL
    if scenario == 'baseline':
        with open(
            f"{query_path}/smf_baseline.sql", 
            encoding='utf-8'
            ) as f:
            sql_text = f.read()

    elif scenario == 'remediation':
        with open(
            f"{query_path}/smf_remediation.sql", 
            encoding='utf-8'
            ) as f:
            sql_text = f.read()

    elif scenario == 'expansion':
        with open(
            f"{query_path}/smf_expansion.sql", 
            encoding='utf-8'
            ) as f:
            sql_text = f.read()

    else:
        raise ValueError(f"Invalid scenario: {scenario}")

    # add select statement to get the SMF table
    sql_text = sql_text + 'select * from edldb_dev.sc_promise_sandbox.simulation_smf;'

    df = execute_multiple_query_and_return_formatted_data(
        query=sql_text,
        start_date=s,
        end_date=e,
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path, scenario),
        partition_cols='shipdate',
        existing_data_behavior='delete_matching'
        )

    logger.info('SMF table created and saved.')


def simulation(
        prefix,
        query_path,
        output_path,
        scenario,
        s,
        e):
    """
        Run simulation for Best FC-Route selection and save to Local. 

    Args:
        output_path: Main folder of Simulations output in Local. Must exist.
        scenario: Scenario name for simulation. 
            - Must be one of 'baseline', 'remediation', 'expansion'.
        s: Start date of simulation. Must be in 'YYYY-MM-DD' format.
        e: End date of simulation. Must be in 'YYYY-MM-DD' format.
    """

    logger.info('running simulation...')

    if not hf.path_exists(output_path + f'/{scenario}'):
        hf.ensure_directory_exists(output_path + f'/{scenario}')

    if prefix.startswith('s3://'):
        query_path = '.'
        query_path_smf = '.'
    else:
        query_path_smf = f'{query_path}/create_smf_tables'

    # read SMF table creation SQL
    if scenario == 'baseline':
        with open(
            f"{query_path_smf}/smf_baseline.sql", 
            encoding='utf-8'
            ) as f:
            sql_text_smf = f.read()

    elif scenario == 'remediation':
        with open(
            f"{query_path_smf}/smf_remediation.sql", 
            encoding='utf-8'
            ) as f:
            sql_text_smf = f.read()

    elif scenario == 'expansion':
        with open(
            f"{query_path_smf}/smf_expansion.sql", 
            encoding='utf-8'
            ) as f:
            sql_text_smf = f.read()

    else:
        raise ValueError(f"Invalid scenario: {scenario}")

    # read simulation SQL
    with open(f"{query_path}/run_simulation.sql", encoding='utf-8') as f:
        sql_text_sim = f.read()

    # combine SMF table creation and simulation SQL
    sql_text = sql_text_smf + sql_text_sim

    df = execute_multiple_query_and_return_formatted_data(
        query=sql_text,
        start_date=s,
        end_date=e,
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path, scenario),
        partition_cols='order_placed_date',
        existing_data_behavior='delete_matching'
        )

    logger.info('simulation run finished...')



if __name__ == '__main__':

    # read configs
    with open("./configs.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    PREFIX = config['ENVIRONMENT']['prefix']
    SIMULATION_SCENARIO = config['SIMULATION']['simulation_scenario']
    LOOKBACK_DAY_COUNT = config['SIMULATION']['lookback_day_count']
    START_DATE = config['SIMULATION']['start_date']
    END_DATE = config['SIMULATION']['end_date']

    # if start date and end date are not provided,
    # use lookback day count to calculate start and end date from today
    if START_DATE and END_DATE:
        pass
    else:
        END_DATE = date.today()
        START_DATE = END_DATE - timedelta(days=LOOKBACK_DAY_COUNT)
        END_DATE = END_DATE.strftime('%Y-%m-%d')
        START_DATE = START_DATE.strftime('%Y-%m-%d')

    # run simulation for each scenario
    for scenario in SIMULATION_SCENARIO:

        logger.info('Running %s for %s - %s ...', scenario, START_DATE, END_DATE)


        generate_ship_map_file(
            prefix=PREFIX,
            query_path='./sql',
            output_path=f'{PREFIX}/data/smf/',
            scenario=scenario,
            s=START_DATE,
            e=END_DATE
            )

        simulation(
            prefix=PREFIX,
            query_path='./sql',
            output_path=f'{PREFIX}/data/simulations/',
            scenario=scenario,
            s=START_DATE,
            e=END_DATE
            )
