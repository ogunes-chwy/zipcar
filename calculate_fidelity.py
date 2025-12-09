"""
Fidelity Analysis Script

This script compares simulation results with actual ORS (Order Routing System) data
to assess fidelity across carrier coverages and FC charges.
"""
import os
import pandas as pd
import psycopg2
from datetime import date, timedelta, datetime
import helper_functions as hf
from snowflake_utils import insert_data_to_snowflake, execute_query
import yaml
import logging
import boto3
from botocore.exceptions import ClientError
import json

# pylint: disable=abstract-class-instantiated

# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)

# Configuration
WMS_PROXY_QUERY = 'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1109_1122;'

# Date configuration
LOOKBACK_DAY_COUNT = 7
END_DATE = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAY_COUNT+1)

# FC names
FC_NAMES = ['AVP1', 'AVP2', 'BNA1', 'CFC1', 'CLT1', 'DAY1', 'HOU1',
            'DFW1', 'MCI1', 'MCO1', 'MDT1', 'PHX1', 'RNO1']


def get_secret():
    """
    Get the Snowflake connection secret from the Secrets Manager.
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId="zipcar-dev-postgres-connection-ogunes"
        )
    except ClientError as e:
        raise e
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret

POSTGRES_INFO = {
    'host': 'db-writer-dev-global-ors-simulations-db.scff.dev.chewy.com',
    'port': 5432,
    'database': 'orssimsdb',
    'options': '-c idle_in_transaction_session_timeout=360000000',
    'user': get_secret()["user"],
    'password': get_secret()["password"],
    'keepalives': 1,
    'keepalives_idle': 30,
    'keepalives_interval': 10,
    'keepalives_count': 5
}

def assign_fc_region(df, fc_name, fc_region_name):
    """
    Assign FC region based on FC name.
    
    Args:
        df: DataFrame to modify
        fc_name: Column name containing FC name
        fc_region_name: Column name to store FC region
        
    Returns:
        DataFrame with FC region assigned
    """
    df = df.copy()
    df[fc_region_name] = 'None'
    df.loc[df[fc_name].isin(['AVP1', 'AVP2', 'MDT1']), fc_region_name] = 'NorthEast'
    df.loc[df[fc_name].isin(['MCI1', 'DAY1', 'DFW1', 'CFC1', 'HOU1']), fc_region_name] = 'Central'
    df.loc[df[fc_name].isin(['BNA1', 'CLT1', 'MCO1']), fc_region_name] = 'SouthEast'
    df.loc[df[fc_name].isin(['PHX1', 'RNO1']), fc_region_name] = 'West'
    return df


def add_week_start_date(df, date_col):
    """
    Add week_start_date column based on date column.
    
    Args:
        df: DataFrame
        date_col: Name of date column
        
    Returns:
        DataFrame with week_start_date column
    """
    df = df.copy()
    df['week_start_date'] = pd.to_datetime(df[date_col]).dt.to_period('W-SUN').dt.start_time - timedelta(days=1)
    df['week_start_date'] = df['week_start_date'].dt.strftime('%Y-%m-%d')
    return df


def load_ors_data(start_date, end_date):
    """
    Load ORS data from PostgreSQL.
    
    Args:
        start_date: Start date for query
        end_date: End date for query
        
    Returns:
        DataFrame with ORS data
    """
    sql = f"""
    SELECT
        DATE(batch_dttm - INTERVAL '4 hours') AS date,
        fc_name AS ors_fc_name,
        ship_method AS ors_carrier_code,
        COUNT(DISTINCT CAST(order_id AS VARCHAR)||'-'||CAST(cartonization_id::INT AS VARCHAR)||'-'||fc_name||'-'||ship_method) AS packages
    FROM
        ors2_athena_final_routed_packages
    WHERE
        itemtype = 'NORMAL'
        AND fc_name IN ({','.join([f"'{fc}'" for fc in FC_NAMES])})
        AND batch_dttm BETWEEN '{str(start_date)}' AND '{str(end_date)}'
    GROUP BY
        1, 2, 3
    ORDER BY
        1, 2, 3
    """

    with psycopg2.connect(**POSTGRES_INFO) as conn:
        df_ors = pd.read_sql(sql, conn)

    return df_ors


def calculate_carrier_coverage(df, group_cols, value_col, func, denom_cols,
                                carrier_col, prefix=''):
    """
    Calculate carrier coverage distribution.
    
    Args:
        df: DataFrame
        group_cols: Columns to group by
        value_col: Column to aggregate
        func: Aggregation function ('sum' or 'nunique')
        denom_cols: Columns for denominator calculation
        carrier_col: Name of carrier column
        prefix: Prefix for output column names
        
    Returns:
        DataFrame with carrier coverage metrics
    """
    coverage = hf.calculate_package_distribution_by_groups(
        df, group_cols=group_cols, value_col=value_col,
        f=func, denom_cols=denom_cols
    )

    coverage = coverage[group_cols + ['value', 'percent']].copy()
    coverage.columns = group_cols + [f'{prefix}_package_count', f'{prefix}_package_percent']

    # Rename carrier column if needed
    if carrier_col in coverage.columns:
        coverage = coverage.rename(columns={carrier_col: 'carrier'})

    if 'carrier' in coverage.columns:
        coverage['carrier'] = coverage['carrier'].str.strip()

    return coverage


def calculate_fc_charges(df, group_cols, value_col, func, denom_cols,
                          fc_col, prefix='', target_col='fc_name'):
    """
    Calculate FC charge distribution.
    
    Args:
        df: DataFrame
        group_cols: Columns to group by
        value_col: Column to aggregate
        func: Aggregation function ('sum' or 'nunique')
        denom_cols: Columns for denominator calculation
        fc_col: Name of FC column
        prefix: Prefix for output column names
        target_col: Target column name after renaming (default: 'fc_name')
        
    Returns:
        DataFrame with FC charge metrics
    """
    charges = hf.calculate_package_distribution_by_groups(
        df, group_cols=group_cols, value_col=value_col, 
        f=func, denom_cols=denom_cols
    )

    charges = charges[group_cols + ['value', 'percent']].copy()
    charges.columns = group_cols + [f'{prefix}_package_count', f'{prefix}_package_percent']

    # Rename FC column if needed
    if fc_col in charges.columns:
        charges = charges.rename(columns={fc_col: target_col})

    if target_col in charges.columns:
        charges[target_col] = charges[target_col].str.strip()

    return charges


def main(PREFIX):
    """Main execution function."""

    print("Loading ORS data...")
    df_ors = load_ors_data(START_DATE, END_DATE)
    df_ors = add_week_start_date(df_ors, 'date')
    df_ors = df_ors.loc[df_ors['week_start_date'] == df_ors['week_start_date'].max()]

    print("Loading simulation data...")
    df_sim = hf.read_helper(
        f'{PREFIX}/data/simulations/baseline/',
        start_date=START_DATE.strftime('%Y-%m-%d'),
        end_date=END_DATE.strftime('%Y-%m-%d')
    )
    df_sim = add_week_start_date(df_sim, 'order_placed_date')
    df_sim = df_sim.loc[df_sim['week_start_date'] == df_sim['week_start_date'].max()]

    print("Applying WMS proxy...")
    df_sim_wms_proxy = hf.apply_wms_proxy(df_sim, WMS_PROXY_QUERY)
    df_sim_wms_proxy = add_week_start_date(df_sim_wms_proxy, 'order_placed_date')

    # Carrier Coverage Analysis
    print("\n=== Carrier Coverage Analysis ===")

    # Before WMS Proxy
    print("\n--- Before WMS Proxy ---")
    sim_carrier_coverage = calculate_carrier_coverage(
        df_sim,
        group_cols=['week_start_date', 'sim_carrier_code'],
        value_col='shipment_tracking_number',
        func='nunique',
        denom_cols='week_start_date',
        carrier_col='sim_carrier_code',
        prefix='sim'
    )

    ors_carrier_coverage = calculate_carrier_coverage(
        df_ors,
        group_cols=['week_start_date', 'ors_carrier_code'],
        value_col='packages',
        func='sum',
        denom_cols='week_start_date',
        carrier_col='ors_carrier_code',
        prefix='ors'
    )

    carrier_coverage_comp_before = sim_carrier_coverage.merge(
        ors_carrier_coverage,
        on=['week_start_date', 'carrier'],
        how='left'
    )
    carrier_coverage_comp_before['abs_per_diff'] = (
        carrier_coverage_comp_before['sim_package_percent'] -
        carrier_coverage_comp_before['ors_package_percent']
    )
    print(carrier_coverage_comp_before)

    # After WMS Proxy
    print("\n--- After WMS Proxy ---")
    sim_carrier_coverage_after = calculate_carrier_coverage(
        df_sim_wms_proxy,
        group_cols=['week_start_date', 'carrier'],
        value_col='package_count',
        func='sum',
        denom_cols='week_start_date',
        carrier_col='carrier',
        prefix='sim'
    )

    act_carrier_coverage = calculate_carrier_coverage(
        df_sim,
        group_cols=['week_start_date', 'act_carrier_code'],
        value_col='shipment_tracking_number',
        func='nunique',
        denom_cols='week_start_date',
        carrier_col='act_carrier_code',
        prefix='act'
    )

    carrier_coverage_comp_after = sim_carrier_coverage_after.merge(
        act_carrier_coverage,
        on=['week_start_date', 'carrier'],
        how='left'
    )
    carrier_coverage_comp_after['abs_per_diff'] = (
        carrier_coverage_comp_after['sim_package_percent'] -
        carrier_coverage_comp_after['act_package_percent']
    )
    print(carrier_coverage_comp_after)

    # FC Charges Analysis
    print("\n=== FC Charges Analysis ===")

    # By Region
    print("\n--- By Region ---")
    df_sim_region = assign_fc_region(df_sim, 'sim_fc_name', 'sim_fc_region')
    df_sim_region = assign_fc_region(df_sim_region, 'act_fc_name', 'act_fc_region')

    sim_fc_charges_region = calculate_fc_charges(
        df_sim_region,
        group_cols=['week_start_date', 'sim_fc_region'],
        value_col='shipment_tracking_number',
        func='nunique',
        denom_cols='week_start_date',
        fc_col='sim_fc_region',
        prefix='sim',
        target_col='fc_region'
    )

    act_fc_charges_region = calculate_fc_charges(
        df_sim_region,
        group_cols=['week_start_date', 'act_fc_region'],
        value_col='shipment_tracking_number',
        func='nunique',
        denom_cols='week_start_date',
        fc_col='act_fc_region',
        prefix='act',
        target_col='fc_region'
    )

    fc_region_charge_comp = sim_fc_charges_region.merge(
        act_fc_charges_region,
        on=['week_start_date', 'fc_region'],
        how='left'
    )
    fc_region_charge_comp['abs_per_diff'] = (
        fc_region_charge_comp['sim_package_percent'] - 
        fc_region_charge_comp['act_package_percent']
    )
    print(fc_region_charge_comp)

    # By FC
    print("\n--- By FC ---")
    sim_fc_charges = calculate_fc_charges(
        df_sim,
        group_cols=['week_start_date', 'sim_fc_name'],
        value_col='shipment_tracking_number',
        func='nunique',
        denom_cols='week_start_date',
        fc_col='sim_fc_name',
        prefix='sim'
    )

    act_fc_charges = calculate_fc_charges(
        df_sim,
        group_cols=['week_start_date', 'act_fc_name'],
        value_col='shipment_tracking_number',
        func='nunique',
        denom_cols='week_start_date',
        fc_col='act_fc_name',
        prefix='act'
    )

    fc_charge_comp = sim_fc_charges.merge(
        act_fc_charges,
        on=['week_start_date', 'fc_name'],
        how='left'
    )
    fc_charge_comp['abs_per_diff'] = (
        fc_charge_comp['sim_package_percent'] - 
        fc_charge_comp['act_package_percent']
    )
    print(fc_charge_comp)

    return (
        carrier_coverage_comp_before,
        carrier_coverage_comp_after,
        fc_region_charge_comp,
        fc_charge_comp
            )


if __name__ == '__main__':

    # Load configuration parameters
    with open("./configs.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    PREFIX = config['ENVIRONMENT']['prefix']
    RUN_DATE = date.today().strftime('%Y-%m-%d')
    RUN_DTTM = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    OUTPUT_PATH = os.path.join(f'{PREFIX}/baseline_fidelity', RUN_DATE)

    carrier_coverage_before_wms_proxy, carrier_coverage_after_wms_proxy, fc_region_charge_comp, fc_charge_comp = main(PREFIX)
    carrier_coverage_before_wms_proxy['run_dttm'] = RUN_DTTM
    carrier_coverage_after_wms_proxy['run_dttm'] = RUN_DTTM
    fc_region_charge_comp['run_dttm'] = RUN_DTTM
    fc_charge_comp['run_dttm'] = RUN_DTTM


    # SAVING OUTPUTS

    week_start_date_list = list(carrier_coverage_before_wms_proxy['week_start_date'].drop_duplicates())
    SQL_DATE_LIST = ", ".join([f"'{date}'" for date in week_start_date_list])

    execute_query(query=f'delete from edldb.sc_promise_sandbox.ZIPCAR_BASELINE_FIDELITY_CARRIER_COVERAGE_BEFORE_WMS_PROXY where week_start_date in ({SQL_DATE_LIST})')
    execute_query(query=f'delete from edldb.sc_promise_sandbox.ZIPCAR_BASELINE_FIDELITY_CARRIER_COVERAGE_AFTER_WMS_PROXY where week_start_date in ({SQL_DATE_LIST})')
    execute_query(query=f'delete from edldb.sc_promise_sandbox.ZIPCAR_BASELINE_FIDELITY_FC_REGION_CHARGE where week_start_date in ({SQL_DATE_LIST})')
    execute_query(query=f'delete from edldb.sc_promise_sandbox.ZIPCAR_BASELINE_FIDELITY_FC_CHARGE where week_start_date in ({SQL_DATE_LIST})')
    execute_query(query='commit;')

    logger.info('Saving outputs...')

    success, nchunks, nrows, _ = insert_data_to_snowflake(
        df=carrier_coverage_before_wms_proxy,
        table_name='ZIPCAR_BASELINE_FIDELITY_CARRIER_COVERAGE_BEFORE_WMS_PROXY',
        database='EDLDB',
        schema='SC_PROMISE_SANDBOX')
    success, nchunks, nrows, _ = insert_data_to_snowflake(
        df=carrier_coverage_after_wms_proxy,
        table_name='ZIPCAR_BASELINE_FIDELITY_CARRIER_COVERAGE_AFTER_WMS_PROXY',
        database='EDLDB',
        schema='SC_PROMISE_SANDBOX')
    success, nchunks, nrows, _ = insert_data_to_snowflake(
        df=fc_region_charge_comp,
        table_name='ZIPCAR_BASELINE_FIDELITY_FC_REGION_CHARGE',
        database='EDLDB',
        schema='SC_PROMISE_SANDBOX')
    success, nchunks, nrows, _ = insert_data_to_snowflake(
        df=fc_charge_comp,
        table_name='ZIPCAR_BASELINE_FIDELITY_FC_CHARGE',
        database='EDLDB',
        schema='SC_PROMISE_SANDBOX')

    if not hf.path_exists(OUTPUT_PATH):
        hf.ensure_directory_exists(OUTPUT_PATH)

    excel_file_path = os.path.join(OUTPUT_PATH, 'baseline_fidelity.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        carrier_coverage_before_wms_proxy.to_excel(
            writer,
            sheet_name='carrier_coverage_before_wms_proxy',
            index=False,
            na_rep=''
        )
        carrier_coverage_after_wms_proxy.to_excel(
            writer,
            sheet_name='carrier_coverage_after_wms_proxy',
            index=False,
            na_rep=''
        )
        fc_region_charge_comp.to_excel(
            writer,
            sheet_name='fc_region_charge_comp',
            index=False,
            na_rep=''
        )
        fc_charge_comp.to_excel(
            writer,
            sheet_name='fc_charge_comp',
            index=False,
            na_rep=''
        )

