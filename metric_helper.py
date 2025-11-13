import pandas as pd
import numpy as np
from snowflake_utils import execute_query_and_return_formatted_data
import os
import logging


# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


# read helper
def read_helper(
        path,
        start_date,
        end_date,
        date_col_name = 'order_placed_date',
        cols = None,
        col_names = None):
    """
    Args:
        path: Path to data.
        cols: Cols to select.
        start_date: Data start date. 
        end_date: Data end date .
        date_col_name: Date col name used to save by partition.
        col_names: New col names.

    Returns:
        pd.DataFrame: DataFrame in path.
    """
    df = pd.DataFrame()

    date_list = pd.date_range(start=start_date, end=end_date, freq='D')

    for dt in date_list:

        dt = dt.strftime("%Y-%m-%d")
        path_temp = os.path.join(path, f'{date_col_name}={dt}')

        if os.path.exists(path_temp):

            df_temp = pd.read_parquet(path_temp)
            df_temp[date_col_name] = dt

            if cols:
                df_temp = df_temp[cols]

            if col_names:
                df_temp.columns = col_names

            df = pd.concat([df,df_temp])

        else:
            logger.info('%s does not exist', path_temp)

    return df


# cost change
def calculate_cost_change(
        baseline_df,
        iteration_df):
    """
        Args:
            baseline_df: baseline simulation DataFrame.
            iteration_df: iteration simulation DataFrame.

        Returns:
            float: cost change = iteration - baseline.
    """
    baseline_cost = baseline_df['base_transit_cost'].sum()
    iteration_cost = iteration_df['sim_transit_cost'].sum()
    return iteration_cost - baseline_cost


# package count change
def calculate_package_distribution_change_by_groups(
        baseline_df,
        iteration_df,
        group_cols,
        value_col,
        f,
        denom_cols=None
        ):
    """
        Args:
            baseline_df (df): baseline simulation DataFrame.
            iteration_df (df): iteration simulation DataFrame.
            group_cols (list): group by cols for aggregation.
            value_col (str): name of field to apply function on.
            f (str): function to apply e.g. sum, nunique.
            denom_cols (list): cols to calculate percent denominator

        Returns:
            DataFrame: change values & percents.
    """

    baseline_agg = calculate_package_distribution_by_groups(
        baseline_df,
        group_cols,
        value_col,
        f,
        denom_cols
    )
    baseline_agg = baseline_agg.rename(
        columns={'value': 'base_value', 'percent': 'base_percent'}
    )
    baseline_agg = baseline_agg.drop('total',axis=1)

    iteration_agg = calculate_package_distribution_by_groups(
        iteration_df,
        group_cols,
        value_col,
        f,
        denom_cols
    )
    iteration_agg = iteration_agg.rename(
        columns={'value': 'iter_value', 'percent': 'iter_percent'}
    )
    iteration_agg = iteration_agg.drop('total',axis=1)

    change = baseline_agg.merge(iteration_agg,on=group_cols)
    change['abs_percent_change'] = change['iter_percent'] - change['base_percent']
    change = change.sort_values(group_cols)

    return change


# package count distribution
def calculate_package_distribution_by_groups(
        df,
        group_cols,
        value_col,
        f,
        denom_cols=None
        ):
    """
        Args:
            df (df): simulation DataFrame.
            group_cols (list): group by cols for aggregation.
            value_col (str): name of field to apply function on.
            f (str): function to apply e.g. sum, nunique.
            denom_cols (list): cols to calculate percent denominator

        Returns:
            DataFrame: values & percents.
    """
    df_agg = df.groupby(group_cols)[value_col].agg(f).reset_index()
    df_agg.rename(columns={value_col: 'value'}, inplace=True)

    if denom_cols:
        df_agg['total'] = df_agg.groupby(denom_cols)['value'].transform('sum')
    else:
        df_agg['total'] = df_agg['value'].sum()

    df_agg['percent'] = np.round((df_agg['value'] / df_agg['total'])*100,4)
    df_agg = df_agg.sort_values(group_cols)

    return df_agg


# apply wms proxy at aggregate level
def apply_wms_proxy(
        df,
        wms_query
    ):
    """
        Args:
            df: simulation DataFrame.
            wms_query: query to pull WMS proxies from Snowflake.

        Returns:
            DataFrame: date,fc,carrier level WMS proxy package count.
    """
    wms_proxy = execute_query_and_return_formatted_data(
        query=wms_query,
        convert_to_lowercase=True)

    df['dayofweek'] = pd.to_datetime(df['order_placed_date']).dt.dayofweek
    df['dayofweek'] = df['dayofweek'] + 1
    df.loc[df['dayofweek'] == 7,
           'dayofweek'] = 0
    df_agg = df\
        .groupby(['order_placed_date',
                  'dayofweek',
                  'sim_fc_name',
                  'sim_carrier_code'])['shipment_tracking_number']\
        .nunique()\
        .reset_index()
    df_agg.columns = ['order_placed_date','dayofweek','fc','carrier','package_count']

    df_agg = df_agg.merge(wms_proxy,on=['dayofweek','fc'])
    df_agg = df_agg.loc[df_agg['carrier'] == df_agg['proxy_carrier']]
    df_agg['FDXHD'] = round(df_agg['package_count']*df_agg['fdxhd_proxy'])
    df_agg['ONTRGD'] = round(df_agg['package_count']*df_agg['ontrgd_proxy'])

    df_agg = df_agg\
        .groupby(['order_placed_date','fc'])[['FDXHD','ONTRGD']]\
        .sum()\
        .reset_index()

    df_agg = pd.melt(df_agg, id_vars=['order_placed_date','fc'], value_vars=['FDXHD', 'ONTRGD'])
    df_agg.columns = ['order_placed_date','fc','carrier','package_count']

    return df_agg


# execution metrics - unpadded dea
def calculate_execution_metrics():
    return 0


if __name__ == '__main__':

    df_b = read_helper(
        os.path.join('./data/simulations', 'baseline'),
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
        start_date='2025-10-05',
        end_date='2025-10-11'
    )

    o1 = calculate_package_distribution_by_groups(df_b,
                                                ['sim_carrier_code'],
                                                'shipment_tracking_number',
                                                'nunique'
                                                )

    df_b = apply_wms_proxy(df_b,'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1030;')

    o2 = calculate_package_distribution_by_groups(df_b,
                                                ['sim_carrier_code'],
                                                'shipment_tracking_number',
                                                'nunique'
                                                )

    print(o1,o2)

