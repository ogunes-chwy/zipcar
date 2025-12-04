import os
import logging
from datetime import date, timedelta
import pandas as pd
import yaml

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

    PREFIX = config['ENVIRONMENT']['prefix']
    BASELINE_SCENARIO = 'baseline'
    REMEDIATION_SCENARIO = 'remediation'
    EXPANSION_SCENARIO = 'expansion'

    REMEDIATION = True
    EXPANSION = False
    REMEDIATION_ZIP_PATH = './analysis/adhoc-zip-remediation/2025-12-01/OnTrac-first-pass-removal-20251201.csv'
    EXPANSION_ZIP_PATH = ''

    START_DATE = '2025-11-16' # config['EXECUTION']['start_date']
    END_DATE = '2025-11-22' # config['EXECUTION']['end_date']

    baseline_sim_df = hf.read_helper(
        os.path.join(f'{PREFIX}/data/simulations', BASELINE_SCENARIO),
        cols=[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'units',
            'zip5',
            'sim_fc_name',
            'cost_neutral_fc',
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
            'base_cost_neutral_fc',
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
    remediation_sim_df = hf.read_helper(
        os.path.join(f'{PREFIX}/data/simulations', REMEDIATION_SCENARIO),
        cols=[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'units',
            'zip5',
            'sim_fc_name',
            'cost_neutral_fc',
            'sim_carrier_code',
            'sim_route',
            'sim_tnt',
            'sim_transit_cost'],
        start_date=START_DATE,
        end_date=END_DATE
    )
    expansion_sim_df = hf.read_helper(
        os.path.join(f'{PREFIX}/data/simulations', EXPANSION_SCENARIO),
        cols=[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'units',
            'zip5',
            'sim_fc_name',
            'cost_neutral_fc',
            'sim_carrier_code',
            'sim_route',
            'sim_tnt',
            'sim_transit_cost'],
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Cherry-pick Zips for Custom Scenarios
    baseline_sim_df_temp = baseline_sim_df.copy()
    custom_sim_df = pd.DataFrame()

    if REMEDIATION:

        # remediation zips
        remediation_sim_zips = pd.read_csv(
            REMEDIATION_ZIP_PATH,
        )
        remediation_sim_zips['zip5'] = remediation_sim_zips['zip5'].astype(int)
        remediation_sim_zips['zip5'] = remediation_sim_zips['zip5'].astype(str)
        remediation_sim_zips['zip5'] = remediation_sim_zips['zip5'].str.pad(5, fillchar='0')

        remediation_sim_df_temp = remediation_sim_df.copy()
        remediation_sim_df_temp = remediation_sim_df_temp.merge(remediation_sim_zips, on='zip5')
        remediation_sim_df_temp = remediation_sim_df_temp[[
            'order_id', 'order_placed_date', 'shipment_tracking_number','units', 'zip5', 'sim_fc_name', 'cost_neutral_fc', 'sim_carrier_code','sim_tnt', 'sim_transit_cost']
            ]
        custom_sim_df = pd.concat([custom_sim_df, remediation_sim_df_temp])
        baseline_sim_df_temp = baseline_sim_df_temp.loc[
            (~baseline_sim_df_temp['zip5'].isin(remediation_sim_zips['zip5']))
            ]

    if EXPANSION:
        # expansion zips
        expansion_sim_zips = pd.read_csv(
            EXPANSION_ZIP_PATH,
        )
        expansion_sim_zips['zip5'] = expansion_sim_zips['zip5'].astype(int)
        expansion_sim_zips['zip5'] = expansion_sim_zips['zip5'].astype(str)
        expansion_sim_zips['zip5'] = expansion_sim_zips['zip5'].str.pad(5, fillchar='0')

        expansion_sim_df_temp = expansion_sim_df.copy()
        expansion_sim_df_temp = expansion_sim_df_temp.merge(expansion_sim_zips, on='zip5')
        expansion_sim_df_temp = expansion_sim_df_temp[[
            'order_id', 'order_placed_date', 'shipment_tracking_number','units', 'zip5', 'sim_fc_name', 'cost_neutral_fc', 'sim_carrier_code','sim_tnt', 'sim_transit_cost']
            ]
        custom_sim_df = pd.concat([custom_sim_df, expansion_sim_df_temp])
        baseline_sim_df_temp = baseline_sim_df_temp.loc[
            (~baseline_sim_df_temp['zip5'].isin(expansion_sim_zips['zip5']))
            ]

    # union all zips with different scenarios
    baseline_sim_df_temp = baseline_sim_df_temp[
        ['order_id', 'order_placed_date', 'shipment_tracking_number','units', 'zip5', 'base_fc_name', 'base_cost_neutral_fc', 'base_carrier_code','base_tnt', 'base_transit_cost']]
    baseline_sim_df_temp.columns = [
        'order_id', 'order_placed_date', 'shipment_tracking_number','units', 'zip5', 'sim_fc_name', 'cost_neutral_fc', 'sim_carrier_code','sim_tnt', 'sim_transit_cost']
    custom_sim_df = pd.concat([custom_sim_df, baseline_sim_df_temp])

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
        'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1109_1122;'
    )
    custom_sim_df_proxy = hf.apply_wms_proxy(
        custom_sim_df,
        'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1109_1122;'
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
        'units',
        'sum'
    )
    fc_charge_change['change_to_baseline'] = (
        fc_charge_change['iter_value'] - fc_charge_change['base_value'])
    fc_charge_change['percent_change_to_baseline'] = (
        fc_charge_change['iter_value'] - fc_charge_change['base_value']) / fc_charge_change['base_value']
    logger.info('FC CHARGE CHANGES\n%s', fc_charge_change.to_string())

    # Estimate FC Flow & FC-Carrier FLow
    fc_carrier_flow = baseline_sim_df.merge(
        custom_sim_df,
        on=['order_id','shipment_tracking_number','order_placed_date','zip5','units']
        )
    fc_flow = hf.calculate_package_distribution_by_groups(
        fc_carrier_flow,
        ['base_fc_name', 'sim_fc_name'],
        'units',
        'sum',
        ['base_fc_name']
    )
    fc_flow = fc_flow.sort_values(['base_fc_name', 'value'], ascending=[True, False])
    fc_flow['id'] = fc_flow.groupby('base_fc_name').cumcount()
    fc_flow = fc_flow[fc_flow['id'] < 4]
    fc_flow = fc_flow[['base_fc_name', 'sim_fc_name', 'value','percent']]
    logger.info('FC FLOW\n%s', fc_flow.to_string())

    carrier_flow = hf.calculate_package_distribution_by_groups(
        fc_carrier_flow,
        ['base_fc_name','base_carrier_code','sim_fc_name','sim_carrier_code'],
        'units',
        'sum',
        ['base_fc_name', 'base_carrier_code']
    )
    carrier_flow = carrier_flow.sort_values(
        ['base_fc_name', 'base_carrier_code', 'value'], 
        ascending=[True, True, False])
    carrier_flow['id'] = carrier_flow.groupby(['base_fc_name', 'base_carrier_code']).cumcount()
    carrier_flow = carrier_flow[carrier_flow['id'] < 4]
    carrier_flow = carrier_flow[
        ['base_fc_name', 'base_carrier_code', 'sim_fc_name', 'sim_carrier_code', 'value','percent']]
    logger.info('FC CARRIER FLOW\n%s', carrier_flow.to_string())

    # Estimate TnT changes
    tnt_change = hf.calculate_package_distribution_change_by_groups(
        baseline_sim_df.rename(
            columns={'base_tnt': 'sim_tnt'},
            inplace=False
        ),
        custom_sim_df,
        ['sim_tnt'],
        'shipment_tracking_number',
        'nunique'
    )
    logger.info('TNT DIST CHANGES\n%s', tnt_change.to_string())
