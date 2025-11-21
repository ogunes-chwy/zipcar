import os
from pathlib import Path
import logging
import json
from datetime import date, timedelta, datetime
import pandas as pd
import yaml
from snowflake_utils import insert_data_to_snowflake,execute_query_and_return_formatted_data

import helper_functions as hf

# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    # Load configuration parameters
    with open("./configs.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    CUSTOM_SCENARIO = 'remediation'

    PREFIX = config['ENVIRONMENT']['prefix']
    BASELINE_SCENARIO = config['EXECUTION']['baseline_scenario']

    LOOKBACK_DAY_COUNT = config['EXECUTION']['lookback_day_count']
    START_DATE = config['EXECUTION']['start_date']
    END_DATE = config['EXECUTION']['end_date']

    # set start and end date for simulation data
    if START_DATE and END_DATE:
        pass
    else:
        END_DATE = date.today()
        START_DATE = END_DATE - timedelta(days=LOOKBACK_DAY_COUNT)
        END_DATE = END_DATE.strftime('%Y-%m-%d')
        START_DATE = START_DATE.strftime('%Y-%m-%d')


    baseline_sim_df = hf.read_helper(
        os.path.join(f'{PREFIX}/data/simulations', BASELINE_SCENARIO),
        cols=[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'units',
            'zip5',
            'sim_fc_name',
            'sim_carrier_code',
            'sim_route',
            'sim_tnt',
            'sim_transit_cost',
            'act_fc_name',
            'act_carrier_code',
            'act_route',
            'act_transit_cost',
            'std',
            'dea_flag'],
        start_date=START_DATE,
        end_date=END_DATE,
        col_names=[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'units',
            'zip5',
            'base_fc_name',
            'base_carrier_code',
            'base_route',
            'base_tnt',
            'base_transit_cost',
            'act_fc_name',
            'act_carrier_code',
            'act_route',
            'act_transit_cost',
            'std',
            'dea_flag']
    )
    custom_sim_df = hf.read_helper(
        os.path.join(f'{PREFIX}/data/simulations', CUSTOM_SCENARIO),
        cols=[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'zip5',
            'sim_fc_name',
            'sim_carrier_code',
            'sim_route',
            'sim_tnt',
            'sim_transit_cost'],
        start_date=START_DATE,
        end_date=END_DATE
    )

    # CALCULATE METRICS

    # Estimate network level ONTRGD % before WMS proxy
    carrier_change = hf.calculate_package_distribution_change_by_groups(
        baseline_sim_df.rename(
            columns={'base_carrier_code': 'sim_carrier_code'},
            inplace=False
        ),
        custom_sim_df,
        ['sim_carrier_code'],
        'shipment_tracking_number',
        'nunique'
    )
    logger.info('NETWORK LEVEL ONTRGD %% - BEFORE WMS PROXY\n%s', carrier_change.to_string())

    # Estimate network level ONTRGD % after WMS proxy
    baseline_sim_df_proxy = hf.apply_wms_proxy(
        baseline_sim_df.rename(
            columns={
                'base_carrier_code': 'sim_carrier_code',
                'base_fc_name': 'sim_fc_name'
            },
            inplace=False
        ),
        'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1026_1108;'
    )
    custom_sim_df_proxy = hf.apply_wms_proxy(
        custom_sim_df,
        'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1026_1108;'
    )

    carrier_change_proxy = hf.calculate_package_distribution_change_by_groups(
        baseline_sim_df_proxy,
        custom_sim_df_proxy,
        ['carrier'],
        'package_count',
        'sum'
    )
    logger.info('NETWORK LEVEL ONTRGD %% - AFTER WMS PROXY\n%s',
    carrier_change_proxy.to_string())

    # Estimate cost saving
    cost_saving = hf.calculate_cost_change(
        baseline_sim_df,
        custom_sim_df,
    )
    cost_saving = pd.DataFrame({'cost_saving_iter-base': [cost_saving]})
    logger.info('COST SAVING (ITER - BASE)\n%s', cost_saving)

    # Estimate FC charge changes
    fc_charge_change = hf.calculate_package_distribution_change_by_groups(
        baseline_sim_df.rename(
            columns={'base_fc_name': 'sim_fc_name'},
            inplace=False
        ),
        custom_sim_df,
        ['sim_fc_name'],
        'shipment_tracking_number',
        'nunique'
    )
    logger.info('FC CHARGE CHANGES\n%s', fc_charge_change.to_string())