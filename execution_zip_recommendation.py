import os
from pathlib import Path
import logging
import json
from datetime import date, timedelta, datetime
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from snowflake_utils import insert_data_to_snowflake,execute_query_and_return_formatted_data

import helper_functions as hf

from helper_functions import (
    read_helper,
    calculate_package_distribution_change_by_groups,
    apply_wms_proxy,
    calculate_cost_change,
    fillna_with_mean)

# pylint: disable=abstract-class-instantiated

# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


def get_zip_status(  # pylint: disable=redefined-outer-name
        smf_baseline_df,
        smf_expansion_df,
        check_date
    ):
    """
    Determine zip code status (active/inactive) based on SMF data.

    Args:
        smf_baseline_df: DataFrame containing baseline SMF.
        smf_expansion_df: DataFrame containing expansion SMF.
        check_date: Date to check zip status.

    Returns:
        pd.DataFrame: DataFrame containing zip status with 'active' column (1=active, 0=inactive).
    """
    # Get active zips from baseline (ONTRGD mode on check_date)
    smf_baseline_df = smf_baseline_df.loc[
        (smf_baseline_df['mode'] == 'ONTRGD')
        & (smf_baseline_df['shipdate'] == check_date),
        ['zip5']
    ].drop_duplicates()
    smf_baseline_df['active'] = 1

    # Get all eligible zips from expansion (ONTRGD mode on check_date)
    smf_expansion_df = smf_expansion_df.loc[
        (smf_expansion_df['mode'] == 'ONTRGD')
        & (smf_expansion_df['shipdate'] == check_date),
        ['zip5']
    ].drop_duplicates()

    # Merge to determine active status
    eligible_ontrgd = smf_expansion_df.merge(
        smf_baseline_df,
        on='zip5',
        how='left'
    )
    eligible_ontrgd['active'] = eligible_ontrgd['active'].fillna(0)

    return eligible_ontrgd


def get_last_recommendation(  # pylint: disable=redefined-outer-name
        prefix,
        run_date,
        run_name='default'
    ):
    """
    Retrieve previous zip recommendations from remediation and expansion directories.

    Args:
        run_date: Date of the run.
        run_name: Name of the run.
    
    Returns:
        pd.DataFrame: DataFrame containing previous zips to remediate and expand.
    """
    run_date = pd.to_datetime(run_date)
    start_date = (run_date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = run_date.strftime('%Y-%m-%d')

    input_cols = ['zip5', 'final_recommendation', 'run_dttm',
                'final_reason_dea', 'final_reason_shutdown', 'final_reason_backlog']
    output_cols = ['zip5', 'last_recommendation', 'recommendation_date',
                'last_reason_dea', 'last_reason_shutdown', 'last_reason_backlog']

    if prefix.startswith('s3://'):
        query_path = '.'
    else:
        query_path = './sql'

    try:
        # remediation
        remediation_df = execute_query_and_return_formatted_data(
            query_path=query_path,
            query_name='get_last_remediation',
            start_date=start_date,
            end_date=end_date,
            convert_to_lowercase=True)

        remediation_df = remediation_df.loc[remediation_df['run_name'] == run_name]

        remediation_df = remediation_df[input_cols]
        remediation_df.columns = output_cols

    except Exception as e:
        logger.error('Error getting last remediation: %s', e)
        remediation_df = pd.DataFrame(columns=['zip5', 'last_recommendation', 'recommendation_date',
                    'last_reason_dea', 'last_reason_shutdown', 'last_reason_backlog'])

    try:
        # expansion
        expansion_df = execute_query_and_return_formatted_data(
            query_path=query_path,
            query_name='get_last_expansion',
            start_date=start_date,
            end_date=end_date,
            convert_to_lowercase=True)

        expansion_df = expansion_df.loc[remediation_df['run_name'] == run_name]

        expansion_df = expansion_df[input_cols]
        expansion_df.columns = output_cols

    except Exception as e:
        logger.error('Error getting last expansion: %s', e)
        expansion_df = pd.DataFrame(columns=['zip5', 'last_recommendation', 'recommendation_date',
                    'last_reason_dea', 'last_reason_shutdown', 'last_reason_backlog'])


    df = pd.concat([remediation_df, expansion_df])
    df['last_recommendation_date'] = df.groupby('zip5')['recommendation_date'].transform('max')
    df = df.loc[df['recommendation_date'] == df['last_recommendation_date']]
    df = df.drop('last_recommendation_date', axis=1)
    df['recommendation_date'] = pd.to_datetime(df['recommendation_date']).dt.strftime('%Y-%m-%d')

    return df


def calculate_zip_volume_daily(baseline_sim_df):  # pylint: disable=redefined-outer-name
    """
    Calculate average daily package volume per zip code.

    Args:
        baseline_sim_df: Baseline simulation that run with prd Ship Map File.

    Returns:
        pd.DataFrame: DataFrame containing zip average daily volume in selected time period.
    """
    demand_by_zip = baseline_sim_df.groupby(['zip5'])['shipment_tracking_number'].nunique().reset_index()
    demand_by_zip.columns = ['zip5', 'package_count']

    date_count = len(baseline_sim_df['order_placed_date'].drop_duplicates())
    demand_by_zip['daily_package_count_avg'] = demand_by_zip['package_count'] / date_count
    demand_by_zip = demand_by_zip[['zip5', 'daily_package_count_avg']]

    return demand_by_zip


def calculate_fc_carrier_switch(baseline_sim_df, iteration_sim_df):  # pylint: disable=redefined-outer-name
    """
    Calculate FC and carrier switch metrics between baseline and iteration simulations.

    Args:
        baseline_sim_df: Baseline simulation that run with prd Ship Map File.
        iteration_sim_df: Iteration simulation that run with different SMF from prd.

    Returns:
        pd.DataFrame: DataFrame containing FC and carrier change metrics between 
                     baseline and iteration simulation.
    """
    # Merge baseline and iteration simulations
    compare_df = baseline_sim_df.merge(
        iteration_sim_df,
        on=['order_id', 'shipment_tracking_number', 'order_placed_date', 'zip5']
    )

    # Calculate total package count per zip
    package_count_by_zip = compare_df.groupby(['zip5'])['shipment_tracking_number'].nunique().reset_index()
    package_count_by_zip.columns = ['zip5', 'package_count']

    # Calculate FC switch count (packages that changed FC)
    fc_switch = compare_df.groupby(
        ['zip5', 'base_fc_name', 'sim_fc_name']
    )['shipment_tracking_number'].nunique().reset_index()
    fc_switch.columns = ['zip5', 'base_fc_name', 'sim_fc_name', 'package_count']
    fc_switch['fc_switch_package_count'] = fc_switch['package_count']
    fc_switch.loc[
        fc_switch['base_fc_name'] == fc_switch['sim_fc_name'],
        'fc_switch_package_count'
    ] = 0
    fc_switch = fc_switch.groupby(['zip5'])['fc_switch_package_count'].sum().reset_index()

    # Calculate carrier switch count (packages that changed carrier)
    carrier_switch = compare_df.groupby(
        ['zip5', 'base_carrier_code', 'sim_carrier_code']
    )['shipment_tracking_number'].nunique().reset_index()
    carrier_switch.columns = ['zip5', 'base_carrier_code', 'sim_carrier_code', 'package_count']
    carrier_switch['carrier_switch_package_count'] = carrier_switch['package_count']
    carrier_switch.loc[
        carrier_switch['base_carrier_code'] == carrier_switch['sim_carrier_code'],
        'carrier_switch_package_count'
    ] = 0
    carrier_switch = carrier_switch.groupby(['zip5'])['carrier_switch_package_count'].sum().reset_index()

    # Combine metrics
    fc_carrier_switch = package_count_by_zip.merge(fc_switch, on='zip5', how='left')
    fc_carrier_switch['fc_switch_package_count'] = fc_carrier_switch['fc_switch_package_count'].fillna(0)
    fc_carrier_switch['fc_switch_package_per_temp'] = (
        fc_carrier_switch['fc_switch_package_count'] / fc_carrier_switch['package_count']
    )

    fc_carrier_switch = fc_carrier_switch.merge(carrier_switch, on='zip5', how='left')
    fc_carrier_switch['carrier_switch_package_count_temp'] = (
        fc_carrier_switch['carrier_switch_package_count'].fillna(0)
    )

    fc_carrier_switch = fc_carrier_switch[[
        'zip5',
        'fc_switch_package_per_temp',
        'carrier_switch_package_count_temp'
    ]]

    return fc_carrier_switch


def calculate_tnt_by_carrier_dow(baseline_sim_df, smf_expansion_df):  # pylint: disable=redefined-outer-name
    """
    Calculate average adjusted transit time (adjtnt) by day of week, zip, and carrier.

    Args:
        baseline_sim_df: Baseline simulation that run with prd Ship Map File.
        smf_expansion_df: DataFrame that includes expansion SMF by date, zip, carrier.

    Returns:
        pd.DataFrame: DataFrame containing average adjtnt by day of week, zip, carrier.
    """
    # Add day of week columns
    baseline_sim_df['day_of_week'] = pd.to_datetime(baseline_sim_df['order_placed_date']).dt.dayofweek
    smf_expansion_df['day_of_week'] = pd.to_datetime(smf_expansion_df['shipdate']).dt.dayofweek

    # Aggregate package count by day of week, zip, and FC
    base_df_agg = baseline_sim_df.groupby(
        ['day_of_week', 'zip5', 'act_fc_name']
    )['shipment_tracking_number'].nunique().reset_index()
    base_df_agg.columns = ['day_of_week', 'zip5', 'fcname', 'package_count']

    # Merge with SMF expansion data
    adjtnt_df_avg = smf_expansion_df.merge(
        base_df_agg,
        on=['day_of_week', 'zip5', 'fcname'],
        how='left'
    )
    adjtnt_df_avg['package_count'] = adjtnt_df_avg['package_count'].fillna(0)

    # Calculate weighted average adjtnt
    adjtnt_df_avg['weighted_adjtnt'] = adjtnt_df_avg['package_count'] * adjtnt_df_avg['adjtnt']
    adjtnt_df_avg = adjtnt_df_avg.groupby(['day_of_week', 'zip5', 'mode']).agg({
        'weighted_adjtnt': 'sum',
        'package_count': 'sum',
        'adjtnt': 'mean'
    }).reset_index()
    
    adjtnt_df_avg.columns = [
        'day_of_week', 'zip5', 'carrier_code', 'weighted_adjtnt', 'package_count', 'adjtnt_avg'
    ]
    adjtnt_df_avg['weighted_adjtnt_avg'] = (
        adjtnt_df_avg['weighted_adjtnt'] / adjtnt_df_avg['package_count']
    )
    adjtnt_df_avg.loc[
        adjtnt_df_avg['weighted_adjtnt_avg'].isnull(),
        'weighted_adjtnt_avg'
    ] = adjtnt_df_avg['adjtnt_avg']

    return adjtnt_df_avg


def generate_reason_dea(dea_df, dea_threshold):  # pylint: disable=redefined-outer-name
    """
    Generate DEA reason codes based on unpadded EDD DEA metrics.

    Args:
        dea_df: DataFrame that includes unpadded EDD DEA by delivery date, zip, fc, carrier.
        dea_threshold: unpadded EDD DEA threshold.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for DEA.
    """
    # Count unique days per zip-carrier combination
    dea_day_count = dea_df.groupby(['zip5', 'carrier_code'])['unpadded_edd'].nunique().reset_index()
    dea_day_count.columns = ['zip5', 'carrier_code', 'dea_day_count']

    # Aggregate DEA counts and package counts
    dea_df_agg = dea_df.groupby(['zip5', 'carrier_code'])[
        ['unpadded_edd_dea_count', 'package_count']
    ].sum().reset_index()

    # Calculate DEA rate
    dea_df_agg['unpadded_edd_dea'] = (
        dea_df_agg['unpadded_edd_dea_count'] / dea_df_agg['package_count']
    )

    # Merge with day count and filter for zips with at least 3 days of data
    dea_df_agg = dea_df_agg.merge(dea_day_count, on=['zip5', 'carrier_code'])
    dea_df_agg = dea_df_agg.loc[dea_df_agg['dea_day_count'] >= 3]

    dea_df_agg['act_package_count'] = dea_df_agg['package_count']

    # Pivot to get carrier-specific columns
    dea_df_agg_p = dea_df_agg.pivot_table(
        index='zip5',
        columns='carrier_code',
        values=['unpadded_edd_dea', 'act_package_count'],
        aggfunc='first'
    ).reset_index()

    dea_df_agg_p.columns = ['zip5'] + [
        f'{col[1]}_{col[0]}' for col in dea_df_agg_p.columns.values if col[0] != 'zip5'
    ]

    # Ensure both carrier columns exist
    if 'ONTRGD_unpadded_edd_dea' not in dea_df_agg_p.columns:
        dea_df_agg_p['ONTRGD_unpadded_edd_dea'] = None
        dea_df_agg_p['ONTRGD_act_package_count'] = None
    if 'FDXHD_unpadded_edd_dea' not in dea_df_agg_p.columns:
        dea_df_agg_p['FDXHD_unpadded_edd_dea'] = None
        dea_df_agg_p['FDXHD_act_package_count'] = None

    # Apply DEA reason code logic
    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] < dea_threshold)
        & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] < dea_threshold),
        'reason_code_dea'
    ] = 'ok'

    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] >= dea_threshold)
        & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] >= dea_threshold),
        'reason_code_dea'
    ] = 'ok'

    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] < dea_threshold)
        & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] >= dea_threshold),
        'reason_code_dea'
    ] = 'deactivate'

    # TODO: Activate logic (currently commented out)
    # dea_df_agg_p.loc[
    #     (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] >= dea_threshold)
    #     & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] < dea_threshold)
    #     ,'reason_code_dea'] = 'activate'

    return dea_df_agg_p


def generate_reason_backlog(  # pylint: disable=redefined-outer-name
        baseline_sim_df,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,  # pylint: disable=unused-argument
        smf_expansion_df
        ):
    """
    Generate backlog reason codes based on days behind metrics.

    Args:
        baseline_sim_df: Baseline simulation that run with prd Ship Map File.
        backlog_df: DataFrame that includes backlog by date, zip, carrier.
        backlog_threshold: Backlog threshold.
        clear_date_threshold: Clear date threshold (currently unused).
        smf_expansion_df: DataFrame that includes expansion SMF by date, zip, carrier.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for Backlog.
    """

    # Find the latest date with both FDXHD and ONTRGD present in carrier_code
    latest_date_with_both = (
        backlog_df.groupby('date')['carrier_code']
        .apply(lambda x: set(['FDXHD', 'ONTRGD']).issubset(set(x)))
        .loc[lambda s: s].index.max()
    )

    # Get latest backlog date
    backlog_df = backlog_df.loc[backlog_df['date'] == latest_date_with_both]
    backlog_df['day_of_week'] = pd.to_datetime(backlog_df['date']).dt.dayofweek

    # Calculate average adjtnt by day of week, zip, carrier
    adjtnt_df_avg = calculate_tnt_by_carrier_dow(baseline_sim_df, smf_expansion_df)

    # Merge backlog and average adjtnt
    clear_df = backlog_df.merge(
        adjtnt_df_avg,
        on=['day_of_week', 'zip5', 'carrier_code'],
        how='left'
    )
    clear_df['weighted_adjtnt_avg'] = clear_df['weighted_adjtnt_avg'].fillna(2)

    # TODO: get max of estimate clear date and date+adjtnt_avg
    # clear_df['estimated_clear_date_max'] = clear_df.apply(
    #     lambda row: max(
    #         pd.to_datetime(row['estimated_clear_date']),
    #         pd.to_datetime(row['date']) + pd.to_timedelta(row['weighted_adjtnt_avg'], unit='D')
    #     ),
    #     axis=1
    # )

    # Pivot to get carrier-specific columns
    clear_df_p = clear_df.pivot_table(
        index='zip5',
        columns='carrier_code',
        values=['days_behind', 'weighted_adjtnt_avg'],
        aggfunc='first'
    ).reset_index()

    clear_df_p.columns = ['zip5'] + [
        f'{col[1]}_{col[0]}' for col in clear_df_p.columns.values if col[0] != 'zip5'
    ]

    # Ensure both carrier columns exist
    if 'ONTRGD_days_behind' not in clear_df_p.columns:
        clear_df_p['ONTRGD_days_behind'] = None
        clear_df_p['ONTRGD_estimated_clear_date_max'] = None
    if 'FDXHD_days_behind' not in clear_df_p.columns:
        clear_df_p['FDXHD_days_behind'] = None
        clear_df_p['FDXHD_estimated_clear_date_max'] = None

    # Apply backlog reason code logic
    clear_df_p.loc[
        (clear_df_p['ONTRGD_days_behind'] < backlog_threshold)
        & (clear_df_p['FDXHD_days_behind'] < backlog_threshold),
        'reason_code_backlog'
    ] = 'ok'

    clear_df_p.loc[
        clear_df_p['ONTRGD_days_behind'] >= backlog_threshold,
        'reason_code_backlog'
    ] = 'deactivate'

    clear_df_p.loc[
        clear_df_p['FDXHD_days_behind'] >= backlog_threshold,
        'reason_code_backlog'
    ] = 'activate'

    # TODO: Alternative backlog logic using clear_date_threshold
    # clear_df_p.loc[
    #     (clear_df_p['ONTRGD_days_behind'] >= backlog_threshold)
    #     & (clear_df_p['ONTRGD_estimated_clear_date_max'] - clear_df_p['FDXHD_estimated_clear_date_max'] >= clear_date_threshold)
    #     ,'reason_code_backlog'] = 'deactivate'
    #
    # clear_df_p.loc[
    #     (clear_df_p['FDXHD_days_behind'] >= backlog_threshold)
    #     & (clear_df_p['FDXHD_estimated_clear_date_max'] - clear_df_p['ONTRGD_estimated_clear_date_max'] >= clear_date_threshold)
    #     ,'reason_code_backlog'] = 'activate'
    #
    # print(clear_df_p['FDXHD_estimated_clear_date_max'] - clear_df_p['ONTRGD_estimated_clear_date_max'])
    # print(clear_df_p['FDXHD_estimated_clear_date_max'] - clear_df_p['ONTRGD_estimated_clear_date_max'] >= clear_date_threshold)
    # print(clear_df_p)

    return clear_df_p


def generate_reason_shutdown(  # pylint: disable=redefined-outer-name
        run_date,
        shutdown_df,
        clear_date_threshold,
        baseline_sim_df,
        smf_expansion_df
    ):
    """
    Generate shutdown reason codes based on estimated clear dates.

    Args:
        run_date: Date of the run.
        shutdown_df: DataFrame that includes shutdown by start date, end date, zip, carrier.
        clear_date_threshold: Clear date threshold.
        baseline_sim_df: Baseline simulation that run with prd Ship Map File.
        smf_expansion_df: DataFrame that includes expansion SMF by date, zip, carrier.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for Shutdown.
    """
    # Get tomorrow's shutdown data if any
    tomorrow_date = (pd.to_datetime(run_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    shutdown_df = shutdown_df.loc[shutdown_df['start_date'] == tomorrow_date]

    if shutdown_df.shape[0] > 0:
        shutdown_zips = pd.DataFrame()
        shutdown_zips_temp = shutdown_df[['zip5']].drop_duplicates()
        for carrier in ['FDXHD', 'ONTRGD']:
            shutdown_zips_temp_copy = shutdown_zips_temp.copy()
            shutdown_zips_temp_copy['carrier_code'] = carrier
            shutdown_zips = pd.concat([shutdown_zips, shutdown_zips_temp_copy], ignore_index=True)

        shutdown_zips = shutdown_zips.merge(
            shutdown_df[['zip5','carrier_code','start_date','end_date']],
            on=['zip5','carrier_code'],
            how='left'
        )
        shutdown_zips.loc[
            (shutdown_zips['end_date'].isnull())
            & (shutdown_zips['start_date'].isnull()),
            'end_date'] = run_date

        shutdown_zips.loc[
            (shutdown_zips['end_date'].isnull())
            & (~shutdown_zips['start_date'].isnull()),
            'end_date'] = '9999-12-31'

        shutdown_zips['run_date'] = run_date
        shutdown_zips['day_of_week'] = pd.to_datetime(shutdown_zips['run_date']).dt.dayofweek

        # Calculate average adjtnt by day of week, zip, carrier
        adjtnt_df_avg = calculate_tnt_by_carrier_dow(baseline_sim_df, smf_expansion_df)

        # Merge shutdown and average adjtnt
        clear_df = shutdown_zips.merge(
            adjtnt_df_avg,
            on=['day_of_week', 'zip5', 'carrier_code'],
            how='left'
        )
        clear_df['weighted_adjtnt_avg'] = clear_df['weighted_adjtnt_avg'].fillna(2)

        # Calculate estimated clear date (max of end_date or run_date + adjtnt)
        clear_df['estimated_clear_date_max'] = clear_df.apply(
            lambda row: max(
                pd.to_datetime(row['end_date']),
                pd.to_datetime(row['run_date']) + pd.to_timedelta(row['weighted_adjtnt_avg'], unit='D')
            ),
            axis=1
        )

        # Pivot to get carrier-specific columns
        clear_df_p = clear_df.pivot_table(
            index='zip5',
            columns='carrier_code',
            values=['end_date', 'estimated_clear_date_max'],
            aggfunc='first'
        ).reset_index()
        clear_df_p.columns = ['zip5'] + [
            f'{col[1]}_{col[0]}' for col in clear_df_p.columns.values if col[0] != 'zip5'
        ]

        # Apply shutdown reason code logic
        clear_df_p['reason_code_shutdown'] = 'ok'

        clear_df_p.loc[
            (clear_df_p['ONTRGD_estimated_clear_date_max'] - 
             clear_df_p['FDXHD_estimated_clear_date_max'] >= clear_date_threshold),
            'reason_code_shutdown'
        ] = 'deactivate'

        clear_df_p.loc[
            (clear_df_p['FDXHD_estimated_clear_date_max'] - 
             clear_df_p['ONTRGD_estimated_clear_date_max'] >= clear_date_threshold),
            'reason_code_shutdown'
        ] = 'activate'

        return clear_df_p
    
    # Return empty DataFrame if no shutdown data
    return pd.DataFrame()


def resolve_current_decision(row):
    """
    Resolve current recommendation based on reason codes with priority order.

    Priority: Shutdown > DEA > Backlog > OK

    Args:
        row: Row of DataFrame containing zip codes to activate or deactivate.

    Returns:
        pd.Series: Series containing current recommendation and reasons.
    """
    reason_dea = row.get('reason_code_dea', 'None')
    reason_shutdown = row.get('reason_code_shutdown', 'None')
    reason_backlog = row.get('reason_code_backlog', 'None')

    # Shutdown ALWAYS takes precedence
    if reason_shutdown in ['activate', 'deactivate']:
        return pd.Series({
            'current_recommendation': reason_shutdown,
            'current_reason_dea': 'None',
            'current_reason_shutdown': reason_shutdown,
            'current_reason_backlog': 'None'
        })

    # DEA next (ONLY IF shutdown is not activate/deactivate)
    if reason_dea in ['activate', 'deactivate']:
        return pd.Series({
            'current_recommendation': reason_dea,
            'current_reason_dea': reason_dea,
            'current_reason_shutdown': 'None',
            'current_reason_backlog': 'None'
        })

    # Backlog next
    if reason_backlog in ['activate', 'deactivate']:
        return pd.Series({
            'current_recommendation': reason_backlog,
            'current_reason_dea': 'None',
            'current_reason_shutdown': 'None',
            'current_reason_backlog': reason_backlog
        })

    # Any 'ok' can be propagated (priority: DEA > Shutdown > Backlog)
    decision = 'None'
    rdea = 'None'
    rshutdown = 'None'
    rbacklog = 'None'

    if reason_dea == 'ok':
        decision, rdea = 'ok', 'ok'
    elif reason_shutdown == 'ok':
        decision, rshutdown = 'ok', 'ok'
    elif reason_backlog == 'ok':
        decision, rbacklog = 'ok', 'ok'

    return pd.Series({
        'current_recommendation': decision,
        'current_reason_dea': rdea,
        'current_reason_shutdown': rshutdown,
        'current_reason_backlog': rbacklog
    })


def determine_final_decision(row):
    """
    Args:
        row: Row of DataFrame containing zip codes to activate or deactivate.

    Returns:
        pd.Series: Series containing final recommendation and reasons.
    """
    status = row.get('active')
    curr = row.get('current_recommendation')
    last = row.get('last_recommendation')
    last_dea = row.get('last_reason_dea')
    last_shutdown = row.get('last_reason_shutdown')
    last_backlog = row.get('last_reason_backlog')
    curr_dea = row.get('current_reason_dea')
    curr_shutdown = row.get('current_reason_shutdown')
    curr_backlog = row.get('current_reason_backlog')

    # If status is 0 and last is activate, then make last None
    # If status is 1 and last is deactivate, then make last None
    # If last decision was a switchback to usual and not executed, then make last before that
    if status == 0 and last == 'activate' and \
        not (last_dea == 'ok' and last_shutdown == 'ok' and last_backlog == 'ok'):
        last = 'None'
        last_dea = 'None'
        last_shutdown = 'None'
        last_backlog = 'None'
    elif status == 1 and last == 'deactivate' and \
        not (last_dea == 'ok' and last_shutdown == 'ok' and last_backlog == 'ok'):
        last = 'None'
        last_dea = 'None'
        last_shutdown = 'None'
        last_backlog = 'None'
    elif status == 0 and last == 'activate' and \
        last_dea == 'ok' and last_shutdown == 'ok' and last_backlog == 'ok':
        last = 'deactivate'
        last_dea = 'None'
        last_shutdown = 'None'
        last_backlog = 'None'
    elif status == 1 and last == 'deactivate' and \
        last_dea == 'ok' and last_shutdown == 'ok' and last_backlog == 'ok':
        last = 'activate'
        last_dea = 'None'
        last_shutdown = 'None'
        last_backlog = 'None'

    # If deactivated due to dea before, keep it deactivated
    if last in ['deactivate'] and last_dea == 'deactivate':
        return pd.Series({
            'final_recommendation': 'None',
            'final_reason_dea': 'None',
            'final_reason_shutdown': 'None',
            'final_reason_backlog': 'None'
        })

    # If current recommendation is activate or deactivate, use it
    if curr in ['activate', 'deactivate']:
        return pd.Series({
            'final_recommendation': curr,
            'final_reason_dea': curr_dea,
            'final_reason_shutdown': curr_shutdown,
            'final_reason_backlog': curr_backlog
        })
    # If last decision was a switchback to usual
    elif last in ['activate','deactivate'] \
        and last_dea == 'ok' and last_shutdown == 'ok' and last_backlog == 'ok':
        return pd.Series({
            'final_recommendation': 'None',
            'final_reason_dea': 'None',
            'final_reason_shutdown': 'None',
            'final_reason_backlog': 'None'
        })
    # If current is ok and last is activate, then deactivate and reasons 'ok'
    elif curr == 'ok' and last == 'activate':
        return pd.Series({
            'final_recommendation': 'deactivate',
            'final_reason_dea': 'ok',
            'final_reason_shutdown': 'ok',
            'final_reason_backlog': 'ok'
        })
    # If current is ok and last is deactivate, then activate and reasons 'ok'
    elif curr == 'ok' and last == 'deactivate':
        return pd.Series({
            'final_recommendation': 'activate',
            'final_reason_dea': 'ok',
            'final_reason_shutdown': 'ok',
            'final_reason_backlog': 'ok'
        })
    # Otherwise, also use current decision/reasons (safe fallback)
    else:
        return pd.Series({
            'final_recommendation': 'None',
            'final_reason_dea': 'None',
            'final_reason_shutdown': 'None',
            'final_reason_backlog': 'None'
        })


def calculate_priority_score(df):
    """
    Calculate priority score for zip codes based on package count and DEA metrics.

    Priority score = 0.65 * (scaled package count) + 0.25 * (inverted scaled DEA) + 0.1 * (scaled days behind)

    Args:
        df: DataFrame containing zip codes to calculate priority score.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with priority score, sorted descending.
    """
    if df.shape[0] > 0:
        scaler = MinMaxScaler()

        df = fillna_with_mean(df, 'act_package_count')
        df['act_package_count_scaled'] = scaler.fit_transform(df[['act_package_count']])

        df = fillna_with_mean(df, 'unpadded_edd_dea')
        df['unpadded_edd_dea_scaled'] = scaler.fit_transform(df[['unpadded_edd_dea']])
        df['inverted_dea'] = 1 - df['unpadded_edd_dea_scaled']

        df = fillna_with_mean(df, 'days_behind')
        df['days_behind_scaled'] = scaler.fit_transform(df[['days_behind']])

        df['priority_score'] = (
            df['act_package_count_scaled'] * 0.65 \
                + df['inverted_dea'] * 0.25 \
                    + df['days_behind_scaled'] * 0.1
        )

        df = df.sort_values('priority_score', ascending=False)
    else:
        df['priority_score'] = None

    return df


def get_zip_recommendation(  # pylint: disable=redefined-outer-name
        PREFIX,
        RUN_DATE,
        RUN_NAME,
        zip_status_df,
        baseline_sim_df,
        remediate_sim_df,
        expand_sim_df,  # pylint: disable=unused-argument
        dea_df,
        dea_threshold,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,
        smf_expansion_df,
        last_zip_recommendation,
        fc_switch_threshold=0.1,
        zip_volume_floor=25
    ):
    """
    Args:
        zip_status_df: DataFrame that includes zip status.
        baseline_sim_df: Baseline simulation 
            that run with prd Ship Map File.
        remediate_sim_df: DataFrame that includes remediate simulation 
            that run with FDXHD Ship Map File.
        expand_sim_df: DataFrame that includes expand simulation 
            that run with prd Ship Map File + eligible ONTRGD Ship Map File.
        dea_df: DataFrame that includes unpadded EDD DEA by delivery date, zip, fc.
        dea_threshold: unpadded EDD DEA threshold.
        backlog_df: DataFrame that includes backlog by date, zip, carrier.
        backlog_threshold: Backlog threshold.
        clear_date_threshold: Clear date threshold.
        smf_expansion_df: DataFrame that includes expansion SMF by date, zip, carrier.
        last_zip_recommendation: DataFrame that includes last zip recommendation.
        fc_switch_threshold: FC switch threshold.
        zip_volume_floor: Zip volume floor.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for remediation and expansion.
    """
    ## STEP 1: Calculate base metrics and reason codes

    # Calculate FC-carrier switch for remediation
    fc_carrier_switch_remediate = calculate_fc_carrier_switch(
        baseline_sim_df,
        remediate_sim_df
    )

    # TODO: change to expand_df
    # fc_carrier_switch_expand = calculate_fc_carrier_switch(
    #     baseline_sim_df,
    #     expand_sim_df
    # )

    # Calculate zip daily volume
    zip_volume_daily = calculate_zip_volume_daily(baseline_sim_df)

    # Get DEA reason code
    zips_to_recommend_dea = generate_reason_dea(dea_df, dea_threshold)
    zips_to_recommend_dea = zips_to_recommend_dea[
        ['zip5', 'FDXHD_unpadded_edd_dea', 'ONTRGD_unpadded_edd_dea',
         'reason_code_dea', 'FDXHD_act_package_count', 'ONTRGD_act_package_count']
    ]

    # Get Backlog reason code
    zips_to_recommend_backlog = generate_reason_backlog(
        baseline_sim_df,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,
        smf_expansion_df
    )
    zips_to_recommend_backlog = zips_to_recommend_backlog[
        ['zip5', 'FDXHD_days_behind', 'ONTRGD_days_behind', 'reason_code_backlog']
    ]

    # shutdown rule for expansion / remediation
    # (Currently not implemented)

    ## STEP 2: Build master table with all reason codes

    # TODO: change to expand_df
    # zips_to_recommend = zip_status_df.merge(
    # fc_carrier_switch_expand, on='zip5', how='left')
    #
    # zips_to_recommend.loc[
    #     zips_to_recommend['active'] == 0,
    #     'fc_switch_package_per'
    # ] = zips_to_recommend.loc[
    #     zips_to_recommend['active'] == 0,
    #     'fc_switch_package_per_temp'
    # ]
    # zips_to_recommend = zips_to_recommend.drop(
    #     ['fc_switch_package_per_temp','carrier_switch_package_count_temp'], 
    # axis=1)

    # Merge zip status with FC-carrier switch for remediation
    zips_to_recommend = zip_status_df.merge(
        fc_carrier_switch_remediate, on='zip5', how='left')

    zips_to_recommend.loc[
        zips_to_recommend['active'] == 1,
        'fc_switch_package_per'
    ] = zips_to_recommend.loc[
        zips_to_recommend['active'] == 1,
        'fc_switch_package_per_temp'
    ]

    # Clean up temporary columns
    zips_to_recommend = zips_to_recommend.drop(
        ['fc_switch_package_per_temp', 'carrier_switch_package_count_temp'],
        axis=1
    )

    # Merge all reason code dataframes
    zips_to_recommend = zips_to_recommend.merge(zip_volume_daily, on='zip5', how='left')
    zips_to_recommend = zips_to_recommend.merge(zips_to_recommend_dea, on='zip5', how='left')
    zips_to_recommend = zips_to_recommend.merge(
        zips_to_recommend_backlog, on='zip5', how='left')

    ## STEP 3: Apply decision logic

    current_recommendation = zips_to_recommend.apply(resolve_current_decision, axis=1)
    zips_to_recommend = pd.concat([zips_to_recommend, current_recommendation], axis=1)

    # Merge last recommendation and determine final decision
    zips_to_recommend = zips_to_recommend.merge(
        last_zip_recommendation, on='zip5', how='left')
    zips_to_recommend['last_recommendation'] = zips_to_recommend['last_recommendation'].fillna('None')
    zips_to_recommend['last_reason_dea'] = zips_to_recommend['last_reason_dea'].fillna('None')
    zips_to_recommend['last_reason_shutdown'] = zips_to_recommend['last_reason_shutdown'].fillna('None')
    zips_to_recommend['last_reason_backlog'] = zips_to_recommend['last_reason_backlog'].fillna('None')

    final_recommendation = zips_to_recommend.apply(determine_final_decision, axis=1)
    zips_to_recommend = pd.concat([zips_to_recommend, final_recommendation], axis=1)

    ## STEP 4: Save intermediate results for debugging purposes

    if not hf.path_exists(f'{PREFIX}/results/execution/{RUN_NAME}/metadata/{RUN_DATE}'):
        hf.ensure_directory_exists(f'{PREFIX}/results/execution/{RUN_NAME}/metadata/{RUN_DATE}')
    zips_to_recommend.to_parquet(
        f'{PREFIX}/results/execution/{RUN_NAME}/metadata/{RUN_DATE}/zips_to_recommend.parquet')

    ## STEP 5: Apply additional constraints

    if zip_volume_floor:
        zips_to_recommend = zips_to_recommend.loc[
            zips_to_recommend['daily_package_count_avg'] >= zip_volume_floor
        ]

    if fc_switch_threshold:
        zips_to_recommend = zips_to_recommend.loc[
            zips_to_recommend['fc_switch_package_per'] <= fc_switch_threshold
        ]

    # apply exclusion list
    # (Currently not implemented)

    ## STEP 6: Split into remediation and expansion groups

    zips_to_recommend_remediate = zips_to_recommend.loc[
        (zips_to_recommend['active'] == 1)
        & (zips_to_recommend['final_recommendation'] == 'deactivate')
    ]
    zips_to_recommend_expand = zips_to_recommend.loc[
        (zips_to_recommend['active'] == 0)
        & (zips_to_recommend['final_recommendation'] == 'activate')
    ]

    ## STEP 7: Format output columns

    cols = [
        'zip5', 'active', 'daily_package_count_avg',
        'current_recommendation', 'current_reason_dea', 
        'current_reason_shutdown', 'current_reason_backlog',
        'last_recommendation', 'last_reason_dea', 
        'last_reason_shutdown', 'last_reason_backlog',
        'final_recommendation', 'final_reason_dea', 
        'final_reason_shutdown', 'final_reason_backlog'
    ]

    # Format remediation output
    zips_to_recommend_remediate = zips_to_recommend_remediate[
        cols + ['ONTRGD_unpadded_edd_dea', 'ONTRGD_act_package_count', 'ONTRGD_days_behind']
    ]  # ,'ONTRGD_days_behind'
    zips_to_recommend_remediate.columns = cols + [
        'unpadded_edd_dea', 'act_package_count', 'days_behind'
    ]  # ,'days_behind'

    # Format expansion output
    zips_to_recommend_expand = zips_to_recommend_expand[
        cols + ['FDXHD_unpadded_edd_dea', 'FDXHD_act_package_count', 'FDXHD_days_behind']
    ]  # ,'FDXHD_days_behind'
    zips_to_recommend_expand.columns = cols + [
        'unpadded_edd_dea', 'act_package_count', 'days_behind'
    ]  # ,'days_behind'

    ## STEP 8: Calculate priority scores and finalize outputs

    zips_to_recommend_remediate = calculate_priority_score(zips_to_recommend_remediate)
    zips_to_recommend_expand = calculate_priority_score(zips_to_recommend_expand)

    # Select final columns and add informative columns
    zips_to_recommend_remediate = zips_to_recommend_remediate[cols + ['priority_score']]
    zips_to_recommend_expand = zips_to_recommend_expand[cols + ['priority_score']]

    info_cols = ['zip5','fc_switch_package_per',
                 'FDXHD_act_package_count','ONTRGD_act_package_count',
                 'FDXHD_unpadded_edd_dea','ONTRGD_unpadded_edd_dea',
                 'FDXHD_days_behind','ONTRGD_days_behind']
    zips_to_recommend = zips_to_recommend[info_cols]

    zips_to_recommend_remediate = zips_to_recommend_remediate.merge(
        zips_to_recommend, on='zip5', how='left')
    zips_to_recommend_expand = zips_to_recommend_expand.merge(
        zips_to_recommend, on='zip5', how='left')

    return zips_to_recommend_remediate, zips_to_recommend_expand


def add_execution_parameters_to_df(df, execution_parameters):
    """
    Add execution parameter names and values as JSON arrays to a DataFrame.

    Args:
        df: DataFrame to add parameters to.
        execution_parameters: Dictionary containing execution parameters.

    Returns:
        pd.DataFrame: DataFrame with 'parameter_names' and 'parameter_values' columns added.
    """
    df = df.copy()
    df['parameter_names'] = json.dumps(list(execution_parameters.keys()))
    df['parameter_values'] = json.dumps(list(execution_parameters.values()))
    return df


def select_zips_and_get_simulation(simulation_df, zip_df, zip_count):  # pylint: disable=redefined-outer-name
    """
    Select top N zip codes by priority score and return corresponding simulation data.

    Args:
        simulation_df: DataFrame of selected simulation.
        zip_df: DataFrame that contains priority score of each zip code.
        zip_count: Zip count threshold to turn on / off ONTRGD.

    Returns:
        tuple: (simulation_df, zip_df) - Selected simulation data and zip codes.
    """
    # Sort by priority score and select top N
    zip_df = zip_df.sort_values('priority_score', ascending=False)
    zip_df = zip_df[0:zip_count]
    
    # TODO: Alternative selection logic
    # zip_df = zip_df.loc[
    #     (zip_df['carrier_switch_package_count_cumsum'] <= package_count)
    # ]
    # zip_df = zip_df[
    #     ['zip5', 'priority_score', 'reason_code_dea', 'reason_code_backlog']
    # ].drop_duplicates()
    
    zip_df['selected'] = 1
    
    # Merge with simulation data
    simulation_df = simulation_df.merge(zip_df, on='zip5')
    simulation_df = simulation_df[[
       'order_id',
       'order_placed_date',
       'shipment_tracking_number',
       'zip5',
       'sim_fc_name',
       'sim_carrier_code',
       'sim_route',
       'sim_tnt',
       'sim_transit_cost'
       ]]

    return simulation_df, zip_df


if __name__ == '__main__':

    # Load configuration parameters
    with open("./configs.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    PREFIX = config['ENVIRONMENT']['prefix']
    RUN_DATE = date.today().strftime('%Y-%m-%d') # '2025-11-15'
    RUN_DTTM = datetime.today().strftime('%Y-%m-%d %H:%M:%S') # '2025-11-15 00:00:00'
    RUN_NAME = config['EXECUTION']['run_name']
    SAVE_TO_SNOWFLAKE = config['EXECUTION']['save_to_snowflake']
    SAVE_TO_LOCAL = config['EXECUTION']['save_to_local']

    EXPANSION = config['EXECUTION']['expansion']
    REMEDIATION = config['EXECUTION']['remediation']

    BASELINE_SCENARIO = config['EXECUTION']['baseline_scenario']
    EXPANSION_SCENARIO = config['EXECUTION']['expansion_scenario']
    REMEDIATION_SCENARIO = config['EXECUTION']['remediation_scenario']

    LOOKBACK_DAY_COUNT = config['EXECUTION']['lookback_day_count']
    START_DATE = config['EXECUTION']['start_date']
    END_DATE = config['EXECUTION']['end_date']

    RECOMMENDATION_COUNT_LIST = config['EXECUTION']['recommendation_count_list']
    ZIP_VOLUME_FLOOR = config['EXECUTION']['zip_volume_floor']
    FC_SWITCH_THRESHOLD = config['EXECUTION']['fc_switch_threshold']

    DEA_THRESHOLD = config['EXECUTION']['dea_threshold']
    DEA_LOOKBACK_DAY_COUNT = config['EXECUTION']['dea_lookback_day_count']

    BACKLOG_THRESHOLD = config['EXECUTION']['backlog_threshold']
    CLEAR_DATE_THRESHOLD = config['EXECUTION']['clear_date_threshold']

    # set output paths
    OUTPUT_PATH = os.path.join(f'{PREFIX}/results/execution', RUN_NAME)
    METRICS_PATH = os.path.join(OUTPUT_PATH, 'metrics', RUN_DATE)
    SIM_PATH = os.path.join(OUTPUT_PATH, 'simulation_output', RUN_DATE)
    REMEDIATION_PATH = os.path.join(OUTPUT_PATH, 'zips_to_remediate', RUN_DATE)
    EXPANSION_PATH = os.path.join(OUTPUT_PATH, 'zips_to_expand', RUN_DATE)

    # set start and end date for simulation data
    if START_DATE and END_DATE:
        pass
    else:
        END_DATE = date.today() - timedelta(days=1)
        START_DATE = END_DATE - timedelta(days=LOOKBACK_DAY_COUNT)
        END_DATE = END_DATE.strftime('%Y-%m-%d')
        START_DATE = START_DATE.strftime('%Y-%m-%d')

    logger.info('Run date is set to: %s', RUN_DATE)
    logger.info('Getting simulation data for: %s - %s ...', START_DATE, END_DATE)

    # LOAD DATA
    # baseline - remediation - expansion simulation - based on sim start andend date
    baseline_sim_df = read_helper(
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
    remediate_sim_df = read_helper(
        os.path.join(f'{PREFIX}/data/simulations', REMEDIATION_SCENARIO),
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
    expand_sim_df = read_helper(
        os.path.join(f'{PREFIX}/data/simulations', EXPANSION_SCENARIO),
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

    # dea / backlog / shutdown - based on RUN_DATE
    dea_df = read_helper(
        f'{PREFIX}/data/execution_data/unpadded_dea',
        start_date=(
            pd.to_datetime(RUN_DATE) - timedelta(days=DEA_LOOKBACK_DAY_COUNT))\
                .strftime('%Y-%m-%d'),
        end_date=(
            pd.to_datetime(RUN_DATE) - timedelta(days=1))\
                .strftime('%Y-%m-%d'),
        date_col_name='unpadded_edd',
        cols=['unpadded_edd',
              'ffmcenter_name',
              'carrier_code',
              'zip5',
              'package_count',
              'unpadded_edd_dea_count'
              ]
    )
    backlog_df = read_helper(
        f'{PREFIX}/data/execution_data/backlog',
        start_date=(pd.to_datetime(RUN_DATE) - timedelta(days=3)).strftime('%Y-%m-%d'),
        end_date=RUN_DATE,
        date_col_name='date'
    )

    # smf baseline - expansion - based on sim start and end date
    smf_baseline_df = read_helper(
        os.path.join(f'{PREFIX}/data/smf', BASELINE_SCENARIO),
        start_date=START_DATE,
        end_date=END_DATE,
        date_col_name='shipdate'
    )
    smf_expansion_df = read_helper(
        os.path.join(f'{PREFIX}/data/smf', EXPANSION_SCENARIO),
        start_date=START_DATE,
        end_date=END_DATE,
        date_col_name='shipdate'
    )

    # last recommendation if any - based on RUN_DATE
    last_zip_recommendation = get_last_recommendation(
        PREFIX,
        RUN_DATE,
        RUN_NAME
    )
    # print(last_zip_recommendation.head())

    # exclusion list if any
    # exclusion_list = read_helper()

    # Generate zip ONTRGD status - based on sim end date
    # TODO: change to smf_expansion_df
    zip_status_df = get_zip_status(
        smf_baseline_df,
        smf_baseline_df,  # smf_expansion_df
        check_date=END_DATE
    )

    logger.info(
        'ONTRGD zip count by active status: \n%s', 
        zip_status_df.groupby('active')['zip5'].count().to_string())

    # APPLY DECISION RULES AND GET CANDIDATE ZIPS TO REMDIATE / EXPAND FOR ONTRGD
    # TODO: change to smf_expansion_df
    zips_to_remediate, zips_to_expand = get_zip_recommendation(
        PREFIX,
        RUN_DATE,
        RUN_NAME,
        zip_status_df,
        baseline_sim_df,
        remediate_sim_df,
        expand_sim_df,
        dea_df,
        DEA_THRESHOLD,
        backlog_df,
        BACKLOG_THRESHOLD,
        CLEAR_DATE_THRESHOLD,
        smf_baseline_df,  # smf_expansion_df
        last_zip_recommendation,
        FC_SWITCH_THRESHOLD,
        ZIP_VOLUME_FLOOR
    )

    # Determine maximum zip count and filter recommendation count list
    ZIP_COUNT_MAX = 0
    if zips_to_remediate.shape[0] > 0 and REMEDIATION:
        REMEDIATION = True
        ZIP_COUNT_MAX = max(ZIP_COUNT_MAX, zips_to_remediate.shape[0])
    else:
        REMEDIATION = False

    if zips_to_expand.shape[0] > 0 and EXPANSION:
        EXPANSION = True
        ZIP_COUNT_MAX = max(ZIP_COUNT_MAX, zips_to_expand.shape[0])
    else:
        EXPANSION = False

    filtered_list = [
        item for item in RECOMMENDATION_COUNT_LIST if item <= ZIP_COUNT_MAX
    ]
    filtered_list.append(999999)

    # SELECT ZIPS AND GET SIMULATION RESULT BASED ON RECOMMENDATION ZIP COUNT
    for zip_count in filtered_list:

        logger.info('scoping zip count: %i', zip_count)

        REMEDIATION_ZIP_COUNT = 0
        EXPANSION_ZIP_COUNT = 0

        baseline_sim_df_no_change = baseline_sim_df.copy()

        # SELECTING ZIPS AND GETTING SIMULATION RESULT BASED ON RECOMMENDATION ZIP COUNT

        if REMEDIATION:
            remediate_sim_df_temp, remediate_zips_temp = select_zips_and_get_simulation(
                remediate_sim_df,
                zips_to_remediate,
                zip_count
            )
            logger.info('remediated zip count: %i', remediate_zips_temp.shape[0])
            REMEDIATION_ZIP_COUNT = remediate_zips_temp.shape[0]

            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                remediate_zips_temp,
                on='zip5',
                how='left')
            remediate_zips_temp = remediate_zips_temp.drop('selected', axis=1)

            logger.info('carrier switch package count: %i',
                baseline_sim_df_no_change.loc[
                    (baseline_sim_df_no_change['base_carrier_code'] == 'ONTRGD')
                    & (baseline_sim_df_no_change['selected'] == 1)].groupby(
                    ['base_carrier_code','selected'])['shipment_tracking_number'].nunique().item()
                    )

            baseline_sim_df_no_change = baseline_sim_df_no_change.loc[
                baseline_sim_df_no_change['selected'].isnull()
            ]
            baseline_sim_df_no_change = baseline_sim_df_no_change.drop('selected', axis=1)

        if EXPANSION:
            expand_sim_df_temp, expand_zips_temp = select_zips_and_get_simulation(
                expand_sim_df,
                zips_to_expand,
                zip_count
            )
            logger.info('expanded zip count: %i', expand_zips_temp.shape[0])
            EXPANSION_ZIP_COUNT = expand_zips_temp.shape[0]
            
            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                expand_zips_temp,
                on='zip5',
                how='left')
            expand_zips_temp = expand_zips_temp.drop('selected', axis=1)

            logger.info('carrier switch package count: %i',
                expand_sim_df_temp.loc[expand_sim_df_temp['sim_carrier_code'] == 'ONTRGD'].groupby(
                    ['sim_carrier_code'])['shipment_tracking_number'].nunique().item()
                    )

            baseline_sim_df_no_change = baseline_sim_df_no_change.loc[
                baseline_sim_df_no_change['selected'].isnull()
            ]
            baseline_sim_df_no_change = baseline_sim_df_no_change.drop('selected', axis=1)

        # Generate final simulation result with expand / remediate decisions
        final_sim_df = baseline_sim_df_no_change[[
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'zip5',
            'base_fc_name',
            'base_carrier_code',
            'base_route',
            'base_tnt',
            'base_transit_cost'
        ]]
        final_sim_df.columns = [
            'order_id',
            'order_placed_date',
            'shipment_tracking_number',
            'zip5',
            'sim_fc_name',
            'sim_carrier_code',
            'sim_route',
            'sim_tnt',
            'sim_transit_cost'
        ]

        if REMEDIATION:
            final_sim_df = pd.concat([final_sim_df, remediate_sim_df_temp])
        if EXPANSION:
            final_sim_df = pd.concat([final_sim_df, expand_sim_df_temp])

        # CALCULATE METRICS

        # Estimate network level ONTRGD % before WMS proxy
        carrier_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_carrier_code': 'sim_carrier_code'},
                inplace=False
            ),
            final_sim_df,
            ['sim_carrier_code'],
            'shipment_tracking_number',
            'nunique'
        )
        logger.info('NETWORK LEVEL ONTRGD %% - BEFORE WMS PROXY\n%s', carrier_change.to_string())

        # Estimate network level ONTRGD % after WMS proxy
        baseline_sim_df_proxy = apply_wms_proxy(
            baseline_sim_df.rename(
                columns={
                    'base_carrier_code': 'sim_carrier_code',
                    'base_fc_name': 'sim_fc_name'
                },
                inplace=False
            ),
            'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1109_1122;'
        )
        final_sim_df_proxy = apply_wms_proxy(
            final_sim_df,
            'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_1109_1122;'
        )

        carrier_change_proxy = calculate_package_distribution_change_by_groups(
            baseline_sim_df_proxy,
            final_sim_df_proxy,
            ['carrier'],
            'package_count',
            'sum'
        )
        logger.info('NETWORK LEVEL ONTRGD %% - AFTER WMS PROXY\n%s',
        carrier_change_proxy.to_string())

        # Estimate cost saving
        cost_saving_by_zip = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_transit_cost': 'sim_transit_cost'},
                inplace=False
            ),
            final_sim_df,
            ['zip5'],
            'sim_transit_cost',
            'sum'
        )
        cost_saving_by_zip['cost_change'] = cost_saving_by_zip['iter_value'] \
            - cost_saving_by_zip['base_value']
        cost_saving_by_zip = cost_saving_by_zip[['zip5', 'base_value', 'iter_value', 'cost_change']]
        cost_saving = pd.DataFrame(
            {'cost_saving_iter-base': [cost_saving_by_zip['cost_change'].sum()]})

        logger.info('COST SAVING (ITER - BASE)\n%s', cost_saving)
        logger.info('COST SAVING by ZIP (ITER - BASE)\n%s', cost_saving_by_zip)

        # Estimate FC charge changes
        fc_charge_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_fc_name': 'sim_fc_name'},
                inplace=False
            ),
            final_sim_df,
            ['sim_fc_name'],
            'shipment_tracking_number',
            'nunique'
        )

        # Add FC charge change to baseline
        fc_charge_change['change_to_baseline'] = (
            fc_charge_change['iter_value'] - fc_charge_change['base_value'])
        fc_charge_change['percent_change_to_baseline'] = (
            fc_charge_change['iter_value'] - fc_charge_change['base_value']) / fc_charge_change['base_value']
        logger.info('FC CHARGE CHANGES\n%s', fc_charge_change.to_string())

        # Estimate FC Flow & FC-Carrier FLow
        fc_carrier_flow = baseline_sim_df.merge(
            final_sim_df,
        on=['order_id','shipment_tracking_number','order_placed_date','zip5']
        )
        fc_flow = hf.calculate_package_distribution_by_groups(
            fc_carrier_flow,
            ['base_fc_name', 'sim_fc_name'],
            'shipment_tracking_number',
            'nunique',
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
            'shipment_tracking_number',
            'nunique',
            ['base_fc_name', 'base_carrier_code']
        )
        carrier_flow = carrier_flow.sort_values(
            ['base_fc_name', 'base_carrier_code', 'value'], 
            ascending=[True, True, False])
        carrier_flow['id'] = carrier_flow.groupby(['base_fc_name', 'base_carrier_code']).cumcount()
        carrier_flow = carrier_flow[carrier_flow['id'] < 4]
        carrier_flow = carrier_flow[
            ['base_fc_name', 'base_carrier_code', 'sim_fc_name', 'sim_carrier_code', 'value','percent']
            ]
        logger.info('FC CARRIER FLOW\n%s', carrier_flow.to_string())


        # SAVING OUTPUTS

        logger.info('Saving outputs...')

        # ADD PARAMETERS TO OUTPUTS

        execution_parameters = {
            'REMEDIATION_ZIP_COUNT': REMEDIATION_ZIP_COUNT,
            'EXPANSION_ZIP_COUNT': EXPANSION_ZIP_COUNT,
            'DEA_THRESHOLD': DEA_THRESHOLD,
            'DEA_LOOKBACK_DAY_COUNT': DEA_LOOKBACK_DAY_COUNT,
            'BACKLOG_THRESHOLD': BACKLOG_THRESHOLD,
            'CLEAR_DATE_THRESHOLD': CLEAR_DATE_THRESHOLD,
            'FC_SWITCH_THRESHOLD': FC_SWITCH_THRESHOLD,
            'LOOKBACK_DAY_COUNT': LOOKBACK_DAY_COUNT
        }

        carrier_change = add_execution_parameters_to_df(
            carrier_change, execution_parameters)
        carrier_change_proxy = add_execution_parameters_to_df(
            carrier_change_proxy, execution_parameters)
        cost_saving = add_execution_parameters_to_df(
            cost_saving, execution_parameters)
        cost_saving_by_zip = add_execution_parameters_to_df(
            cost_saving_by_zip, execution_parameters)
        fc_charge_change = add_execution_parameters_to_df(
            fc_charge_change, execution_parameters)
        fc_flow = add_execution_parameters_to_df(
            fc_flow, execution_parameters)
        carrier_flow = add_execution_parameters_to_df(
            carrier_flow, execution_parameters)
        final_sim_df = add_execution_parameters_to_df(
            final_sim_df, execution_parameters)
        baseline_sim_df = add_execution_parameters_to_df(
            baseline_sim_df, execution_parameters)

        carrier_change['run_dttm'] = RUN_DTTM
        carrier_change_proxy['run_dttm'] = RUN_DTTM
        cost_saving['run_dttm'] = RUN_DTTM
        cost_saving_by_zip['run_dttm'] = RUN_DTTM
        fc_charge_change['run_dttm'] = RUN_DTTM
        fc_flow['run_dttm'] = RUN_DTTM
        carrier_flow['run_dttm'] = RUN_DTTM
        final_sim_df['run_dttm'] = RUN_DTTM
        baseline_sim_df['run_dttm'] = RUN_DTTM

        carrier_change['run_name'] = RUN_NAME
        carrier_change_proxy['run_name'] = RUN_NAME
        cost_saving['run_name'] = RUN_NAME
        cost_saving_by_zip['run_name'] = RUN_NAME
        fc_charge_change['run_name'] = RUN_NAME
        fc_flow['run_name'] = RUN_NAME
        carrier_flow['run_name'] = RUN_NAME
        final_sim_df['run_name'] = RUN_NAME
        baseline_sim_df['run_name'] = RUN_NAME

        # SAVE METRICS TO SNOWFLAKE

        if SAVE_TO_SNOWFLAKE:

            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=carrier_change,
                table_name='ZIPCAR_CARRIER_CHANGE_BEFORE_WMS_PROXY',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=carrier_change_proxy,
                table_name='ZIPCAR_CARRIER_CHANGE_AFTER_WMS_PROXY',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=cost_saving,
                table_name='ZIPCAR_COST_SAVING',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=cost_saving_by_zip,
                table_name='ZIPCAR_COST_SAVING_BY_ZIP',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=fc_charge_change,
                table_name='ZIPCAR_FC_CHARGE_CHANGE',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=fc_flow,
                table_name='ZIPCAR_FC_CHANGE_DISTRIBUTION',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=carrier_flow,
                table_name='ZIPCAR_FC_CARRIER_CHANGE_DISTRIBUTION',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=final_sim_df,
                table_name='ZIPCAR_ITERATION_SIMULATION',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')
            success, nchunks, nrows, _ = insert_data_to_snowflake(
                df=baseline_sim_df,
                table_name='ZIPCAR_BASELINE_SIMULATION',
                database='EDLDB',
                schema='SC_PROMISE_SANDBOX')

            if REMEDIATION:
                remediate_zips_temp = add_execution_parameters_to_df(
                    remediate_zips_temp, execution_parameters)
                remediate_zips_temp['run_dttm'] = RUN_DTTM
                remediate_zips_temp['run_name'] = RUN_NAME
                remediate_zips_temp['zip_count'] = REMEDIATION_ZIP_COUNT
                if zip_count == 999999:
                    remediate_zips_temp['zip_count_tag'] = 'all'
                else:
                    remediate_zips_temp['zip_count_tag'] = 'partial'
                success, nchunks, nrows, _ = insert_data_to_snowflake(
                    df=remediate_zips_temp,
                    table_name='ZIPCAR_REMEDIATION_ZIPS',
                    database='EDLDB',
                    schema='SC_PROMISE_SANDBOX')

            if EXPANSION:
                expand_zips_temp = add_execution_parameters_to_df(
                    expand_zips_temp, execution_parameters)
                expand_zips_temp['run_dttm'] = RUN_DTTM
                expand_zips_temp['run_name'] = RUN_NAME
                expand_zips_temp['zip_count'] = EXPANSION_ZIP_COUNT
                if zip_count == 999999:
                    expand_zips_temp['zip_count_tag'] = 'all'
                else:
                    expand_zips_temp['zip_count_tag'] = 'partial'
                success, nchunks, nrows, _ = insert_data_to_snowflake(
                    df=expand_zips_temp,
                    table_name='ZIPCAR_EXPANSION_ZIPS',
                    database='EDLDB',
                    schema='SC_PROMISE_SANDBOX')

        # SAVE METRICS TO LOCAL EXCEL FILES

        if SAVE_TO_LOCAL:

            if not hf.path_exists(OUTPUT_PATH):
                hf.ensure_directory_exists(OUTPUT_PATH)
            if not hf.path_exists(METRICS_PATH):
                hf.ensure_directory_exists(METRICS_PATH)

            excel_file_path = os.path.join(METRICS_PATH, 'metrics.xlsx')
            with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                carrier_change.to_excel(
                    writer,
                    sheet_name='carrier_net_before_wms_proxy',
                    index=False,
                    na_rep=''
                )
                carrier_change_proxy.to_excel(
                    writer,
                    sheet_name='carrier_net_after_wms_proxy',
                    index=False,
                    na_rep=''
                )
                cost_saving.to_excel(
                    writer,
                    sheet_name='cost_saving',
                    index=False,
                    na_rep=''
                )
                cost_saving_by_zip.to_excel(
                    writer,
                    sheet_name='cost_saving_by_zip',
                    index=False,
                    na_rep=''
                )
                fc_charge_change.to_excel(
                    writer,
                    sheet_name='fc_charge_changes',
                    index=False,
                    na_rep=''
                )
                fc_flow.to_excel(
                    writer,
                    sheet_name='fc_flow',
                    index=False,
                    na_rep=''
                )
                carrier_flow.to_excel(
                    writer,
                    sheet_name='fc_carrier_flow',
                    index=False,
                    na_rep=''
                )

            # Save final simulation
            if not hf.path_exists(SIM_PATH):
                hf.ensure_directory_exists(SIM_PATH)
            final_sim_df.to_parquet(
                os.path.join(
                    SIM_PATH,
                    f'simulation_output_rem_{REMEDIATION_ZIP_COUNT}_exp_{EXPANSION_ZIP_COUNT}.parquet')
            )

            ALL_ZIP_TAG = ''
            if zip_count == 999999:
                ALL_ZIP_TAG = '_all'

            # Save zip recommendations
            if EXPANSION:
                if not hf.path_exists(EXPANSION_PATH):
                    hf.ensure_directory_exists(EXPANSION_PATH)

                expand_zips_temp = add_execution_parameters_to_df(
                    expand_zips_temp, execution_parameters)
                expand_zips_temp['run_date'] = RUN_DATE
                expand_zips_temp['run_name'] = RUN_NAME

                expand_zips_temp.to_parquet(
                    os.path.join(EXPANSION_PATH, f'exp_{EXPANSION_ZIP_COUNT}{ALL_ZIP_TAG}.parquet')
                )
                expand_zips_temp.to_csv(
                    os.path.join(EXPANSION_PATH, f'exp_{EXPANSION_ZIP_COUNT}{ALL_ZIP_TAG}.csv'),
                    index=False
                )

            if REMEDIATION:
                if not hf.path_exists(REMEDIATION_PATH):
                    hf.ensure_directory_exists(REMEDIATION_PATH)

                remediate_zips_temp = add_execution_parameters_to_df(
                    remediate_zips_temp, execution_parameters)
                remediate_zips_temp['run_date'] = RUN_DATE
                remediate_zips_temp['run_name'] = RUN_NAME

                remediate_zips_temp.to_parquet(
                    os.path.join(REMEDIATION_PATH, f'rem_{REMEDIATION_ZIP_COUNT}{ALL_ZIP_TAG}.parquet')
                )
                remediate_zips_temp.to_csv(
                    os.path.join(REMEDIATION_PATH, f'rem_{REMEDIATION_ZIP_COUNT}{ALL_ZIP_TAG}.csv'),
                    index=False
                )
