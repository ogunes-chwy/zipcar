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


def generate_inventory(
        s,
        e):
    """
    Query and insert inventory data to Snowflake.

    Args:
        s: Start date of simulation.
        e: End date of simulation.
    """
    logger.info('creating inventory table...')

    execute_query(query="drop table if exists edldb_dev.sc_promise_sandbox.simulation_inventory;")
    execute_query(query_name='create_inventory_table',
                  query_path='./data/create_tables')
    execute_query(query_name='insert_inventory_table',
                  query_path='./data/insert_tables',
                  start_date=s,
                  end_date=e)
    execute_query(query="commit;")

    logger.info('inventory table created.')


def generate_ship_map_file(
        output_path,
        scenario,
        s,
        e):
    """
    Query and insert Ship Map File data to Snowflake and Local. 

    Args:
        output_path: Main folder of Ship Map File output in Local.
        scenario: Scenario name for simulation.
        s: Start date of simulation.
        e: End date of simulation.
    """
    logger.info('creating SMF table...')

    if not os.path.exists(output_path + f'/{scenario}'):
        os.makedirs(output_path + f'/{scenario}')

    execute_query(query="drop table if exists edldb_dev.sc_promise_sandbox.simulation_smf;")
    execute_query(query_name='create_smf_table',
                  query_path='./data/create_tables')
    execute_query(query_name='insert_smf_table',
                  query_path='./data/insert_tables',
                  start_date=s,
                  end_date=e)
    execute_query(query="commit;")

    df = execute_query_and_return_formatted_data(
        query='select * from edldb_dev.sc_promise_sandbox.simulation_smf;',
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path, scenario),
        partition_cols='shipdate',
        existing_data_behavior='delete_matching'
        )

    logger.info('SMF table created and saved.')


def generate_cost_estimation(
        s,
        e):
    """
    Query and insert Cost estimation data to Snowflake.
    Dependent on Inventory and Ship Map File data availability. 

    Args:
        s: Start date of simulation.
        e: End date of simulation.
    """
    logger.info('creating cost estimation table...')

    execute_query(query="drop table if exists edldb_dev.sc_promise_sandbox.simulation_cost_estimation;")
    execute_query(query_name='create_cost_table',
                  query_path='./data/create_tables')
    execute_query(query_name='insert_cost_table',
                  query_path='./data/insert_tables',
                  start_date=s,
                  end_date=e)
    execute_query(query="commit;")

    logger.info('cost estimation table created.')


def simulation(
        output_path,
        scenario,
        dt):
    """
        Query simulation for Best FC-Route selection and saves to Local. 

    Args:
        output_path: Main folder of Simulations output in Local.
        scenario: Scenario name for simulation.
        s: Start date of simulation.
        e: End date of simulation.
    """
    logger.info('running simulation...')

    if not os.path.exists(output_path + f'/{scenario}'):
        os.makedirs(output_path + f'/{scenario}')

    df = execute_query_and_return_formatted_data(
        query_name='run_simulation',
        query_path='./data',
        convert_to_lowercase=True)

    df.to_parquet(
        os.path.join(output_path, scenario),
        partition_cols='order_placed_date',
        existing_data_behavior='delete_matching'
        )

    logger.info('simulation run finished...')



if __name__ == '__main__':

    with open("./configs.yaml") as f:
        config = yaml.safe_load(f)

    simulation_scenario = config['SIMULATION']['simulation_scenario']
    lookback_day_count = config['SIMULATION']['lookback_day_count']
    start_date = config['SIMULATION']['start_date']
    end_date = config['SIMULATION']['end_date']

    if start_date and end_date:
        pass
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_day_count)
        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')

    end_date_smf = (pd.to_datetime(end_date) + timedelta(days=3)).strftime('%Y-%m-%d')

    logger.info('Running for %s - %s ...', start_date, end_date)

    generate_inventory(
        s=start_date,
        e=end_date
        )
    generate_ship_map_file(
        output_path='./data/smf/',
        scenario=simulation_scenario,
        s=start_date,
        e=end_date_smf
        )
    generate_cost_estimation(
        s=start_date,
        e=end_date
        )
    simulation(
        output_path='./data/simulations/',
        scenario=simulation_scenario,
        dt=start_date
        )
