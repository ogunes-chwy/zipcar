import os
import logging
from datetime import date, timedelta

import pandas as pd
import yaml

from snowflake_utils import execute_multiple_query_and_return_formatted_data


# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


def generate_ship_map_file(
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

    if not os.path.exists(output_path + f'/{scenario}'):
        os.makedirs(output_path + f'/{scenario}')

    # read SMF table creation SQL
    if scenario == 'baseline':
        with open(
            "data/create_smf_tables/smf_baseline.sql", 
            encoding='utf-8'
            ) as f:
            sql_text = f.read()

    elif scenario == 'remediation':
        with open(
            "data/create_smf_tables/smf_remediation.sql", 
            encoding='utf-8'
            ) as f:
            sql_text = f.read()

    elif scenario == 'expansion':
        with open(
            "data/create_smf_tables/smf_expansion.sql", 
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

    if not os.path.exists(output_path + f'/{scenario}'):
        os.makedirs(output_path + f'/{scenario}')

    # read SMF table creation SQL
    if scenario == 'baseline':
        with open(
            "data/create_smf_tables/smf_baseline.sql", 
            encoding='utf-8'
            ) as f:
            sql_text_smf = f.read()

    elif scenario == 'remediation':
        with open(
            "data/create_smf_tables/smf_remediation.sql", 
            encoding='utf-8'
            ) as f:
            sql_text_smf = f.read()

    elif scenario == 'expansion':
        with open(
            "data/create_smf_tables/smf_expansion.sql", 
            encoding='utf-8'
            ) as f:
            sql_text_smf = f.read()

    else:
        raise ValueError(f"Invalid scenario: {scenario}")

    # read simulation SQL
    with open("data/run_simulation.sql", encoding='utf-8') as f:
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

    simulation_scenario = config['SIMULATION']['simulation_scenario']
    lookback_day_count = config['SIMULATION']['lookback_day_count']
    start_date = config['SIMULATION']['start_date']
    end_date = config['SIMULATION']['end_date']

    # if start date and end date are not provided,
    # use lookback day count to calculate start and end date from today
    if start_date and end_date:
        pass
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_day_count)
        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')

    end_date_smf = (pd.to_datetime(end_date) + timedelta(days=3)).strftime('%Y-%m-%d')

    # run simulation for each scenario
    for scenario in simulation_scenario:

        logger.info('Running %s for %s - %s ...', scenario, start_date, end_date)


        generate_ship_map_file(
            output_path='./data/smf/',
            scenario=scenario,
            s=start_date,
            e=end_date_smf
            )

        simulation(
            output_path='./data/simulations/',
            scenario=scenario,
            s=start_date,
            e=end_date
            )
