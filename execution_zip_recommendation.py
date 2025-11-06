import os
import logging
from datetime import date, timedelta
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from metric_helper import read_helper, calculate_package_distribution_change_by_groups, apply_wms_proxy

# pylint: disable=abstract-class-instantiated

# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


def get_zip_status(
        smf_baseline_df,
        smf_expansion_df,
        date
    ):
    """
    Args:
        smf_baseline_df: DataFrame containing baseline SMF.
        smf_expansion_df: DataFrame containing expansion SMF.
        date: Date to check zip status.
    Returns:
        pd.DataFrame: DataFrame containing zip status.
    """
    smf_baseline_df = smf_baseline_df.loc[
        smf_baseline_df['mode'] == 'ONTRGD',
        ['zip5']].drop_duplicates()
    smf_baseline_df['active'] = 1

    smf_expansion_df = smf_expansion_df.loc[
        smf_expansion_df['mode'] == 'ONTRGD',
        ['zip5']].drop_duplicates()

    eligible_ontrgd = smf_expansion_df.merge(
        smf_baseline_df,
        on='zip5',
        how='left'
    )
    eligible_ontrgd['active'] = eligible_ontrgd['active'].fillna(0)

    return eligible_ontrgd


def get_last_recommendation(
        run_name,
        run_date,
        dea_threshold,
        fc_switch_threshold,
        zc=999999
    ):
    """
    Args:
        run_name: Name of the run.
        run_date: Date of the run.
        dea_threshold: DEA threshold.
        fc_switch_threshold: FC switch threshold.
        zc: Zip count.
    
    Returns:
        pd.DataFrame: DataFrame containing previous zips to remediate and expand.
    """

    recommendation_path = os.path.join('./results/execution', run_name)
    run_date_dt = pd.to_datetime(run_date)
    start_date_dt = run_date_dt - timedelta(days=30)
    prev_zip_recommendation = pd.DataFrame(
        columns=['zip5', 'last_recommendation', 'recommendation_date',
                'last_reason_dea', 'last_reason_shutdown', 'last_reason_backlog']
    )
    filename = f'dea_th_{dea_threshold}_fc_switch_{fc_switch_threshold}_rec_count_{zc}.parquet'
    output_cols = ['zip5', 'last_recommendation', 'recommendation_date',
                   'last_reason_dea', 'last_reason_shutdown', 'last_reason_backlog']
    input_cols = ['zip5', 'final_recommendation', 'recommendation_date',
                  'final_reason_dea', 'final_reason_shutdown', 'final_reason_backlog']

    # read previous zips from both remediation and expansion directories
    for directory in ['zips_to_remediate', 'zips_to_expand']:
        directory_path = os.path.join(recommendation_path, directory)

        if not os.path.exists(directory_path):
            continue

        try:
            # List all dates in directory
            date_list = [
                f for f in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, f))
            ]

            # Filter dates within lookback window
            prev_dates = [
                d for d in date_list
                if (pd.to_datetime(d) < run_date_dt)
                and (pd.to_datetime(d) >= start_date_dt)
            ]

            if not prev_dates:
                continue

            # Read all parquet files for valid dates
            prev_zips = pd.DataFrame()
            for prev_date in prev_dates:
                file_path = os.path.join(directory_path, prev_date, filename)
                if os.path.exists(file_path):
                    temp_df = pd.read_parquet(file_path)
                    temp_df['recommendation_date'] = prev_date
                    prev_zips = pd.concat([prev_zips, temp_df], ignore_index=True)

            if prev_zips.empty:
                continue

            # Get latest recommendation date
            prev_zips['latest_recommendation_date'] = prev_zips\
                .groupby('zip5')['recommendation_date']\
                    .transform('max')
            prev_zips = prev_zips.loc[
                prev_zips['recommendation_date'] == prev_zips['latest_recommendation_date']]

            # Select and rename columns
            prev_zips = prev_zips[input_cols]
            prev_zips.columns = output_cols

            prev_zip_recommendation = pd.concat(
                [prev_zip_recommendation, prev_zips], ignore_index=True
            )

        except Exception as e:
            logger.error("Error reading previous recommendations from %s: %s", directory, e)

    return prev_zip_recommendation


def calculate_zip_volume_daily(
        base_df
    ):
    """
    Args:
        base_df: Baseline simulation that run with prd Ship Map File.

    Returns:
        pd.DataFrame: DataFrame containing zip average daily volume
        in selected time period
    """
    demand_by_zip = base_df\
        .groupby(['zip5'])['shipment_tracking_number']\
        .nunique()\
        .reset_index()
    demand_by_zip.columns = ['zip5','package_count']

    date_count = len(base_df['order_placed_date'].drop_duplicates())
    demand_by_zip['daily_package_count_avg'] = demand_by_zip['package_count'] / date_count
    demand_by_zip = demand_by_zip[['zip5','daily_package_count_avg']]

    return demand_by_zip


def calculate_fc_carrier_switch(
        base_df,
        iter_df
        ):
    """
    Args:
        base_df: Baseline simulation that run with prd Ship Map File.
        iter_df: Iteration simulation that run with 
                    different SMF from prd.

    Returns:
        pd.DataFrame: DataFrame containing fc and carrier change 
                    between baseline and iteration simulation

    """

    compare_df = base_df.merge(
        iter_df,
        on=['order_id','shipment_tracking_number','order_placed_date','zip5']
        )

    package_count_by_zip = compare_df\
        .groupby(['zip5'])['shipment_tracking_number']\
        .nunique()\
        .reset_index()
    package_count_by_zip.columns = ['zip5','package_count']


    fc_switch = compare_df\
        .groupby(['zip5','base_fc_name', 'sim_fc_name'])['shipment_tracking_number']\
        .nunique()\
        .reset_index()
    fc_switch.columns = ['zip5','base_fc_name','sim_fc_name','package_count']
    fc_switch['fc_switch_package_count'] = fc_switch['package_count']
    fc_switch.loc[
        fc_switch['base_fc_name'] == fc_switch['sim_fc_name']
    ,'fc_switch_package_count'] = 0
    fc_switch = fc_switch\
        .groupby(['zip5'])['fc_switch_package_count']\
        .sum()\
        .reset_index()


    carrier_switch = compare_df\
        .groupby(['zip5','base_carrier_code', 'sim_carrier_code'])['shipment_tracking_number']\
        .nunique()\
        .reset_index()
    carrier_switch.columns = ['zip5','base_carrier_code','sim_carrier_code','package_count']
    carrier_switch['carrier_switch_package_count'] = carrier_switch['package_count']
    carrier_switch.loc[
        carrier_switch['base_carrier_code'] == carrier_switch['sim_carrier_code']
    ,'carrier_switch_package_count'] = 0
    carrier_switch = carrier_switch\
        .groupby(['zip5'])['carrier_switch_package_count']\
        .sum()\
        .reset_index()

    fc_carrier_switch = package_count_by_zip.merge(fc_switch,on='zip5',how='left')
    fc_carrier_switch['fc_switch_package_count'] = fc_carrier_switch['fc_switch_package_count'].fillna(0)
    fc_carrier_switch['fc_switch_package_per_temp'] = fc_carrier_switch['fc_switch_package_count'] / fc_carrier_switch['package_count']
    fc_carrier_switch = fc_carrier_switch.merge(carrier_switch,on='zip5',how='left')
    fc_carrier_switch['carrier_switch_package_count_temp'] = fc_carrier_switch['carrier_switch_package_count'].fillna(0)

    fc_carrier_switch = fc_carrier_switch[[
        'zip5',
        'fc_switch_package_per_temp',
        'carrier_switch_package_count_temp'
        ]]

    return fc_carrier_switch


def generate_reason_dea(
        dea_df,
        dea_threshold
        ):
    """
    Args:
        dea_df: DataFrame that includes unpadded EDD DEA by 
                delivery date, zip, fc, carrier.
        dea_threshold: unpadded EDD DEA threshold.
        dea_lookback_day_count: lookback day count for unpadded EDD DEA calculation.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for DEA.

    """

    dea_day_count = dea_df\
        .groupby(['zip5','carrier_code'])['delivery_date']\
        .nunique()\
        .reset_index()
    dea_day_count.columns = ['zip5','carrier_code','dea_day_count']

    dea_df_agg = dea_df\
        .groupby(['zip5','carrier_code'])[['unpadded_edd_dea_count','package_count']]\
        .sum()\
        .reset_index()

    dea_df_agg['unpadded_edd_dea'] = dea_df_agg[
        'unpadded_edd_dea_count'] / dea_df_agg[
            'package_count']

    dea_df_agg = dea_df_agg.merge(
        dea_day_count,
        on=['zip5','carrier_code'])

    dea_df_agg = dea_df_agg.loc[
        dea_df_agg['dea_day_count'] >= 3
    ]

    # dea_df_agg.to_parquet('./archieve/dea_df_agg.parquet')

    dea_df_agg_p = dea_df_agg.pivot(
        index='zip5',
        columns='carrier_code',
        values='unpadded_edd_dea')\
        .reset_index()

    # dea_df_agg_p.to_parquet('./archieve/dea_df_agg_p.parquet')

    dea_df_agg_p.columns = ['zip5'] \
        + [f'{col}_unpadded_edd_dea' for col in dea_df_agg_p.columns.values if col != 'zip5']

    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] < dea_threshold)
        & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] < dea_threshold)
        ,'reason_code_dea'] = 'ok'

    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] >= dea_threshold)
        & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] >= dea_threshold)
        ,'reason_code_dea'] = 'ok'

    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] < dea_threshold)
        & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] >= dea_threshold)
        ,'reason_code_dea'] = 'deactivate'

    #dea_df_agg_p.loc[
    #    (dea_df_agg_p['ONTRGD_unpadded_edd_dea'] >= dea_threshold)
    #    & (dea_df_agg_p['FDXHD_unpadded_edd_dea'] < dea_threshold)
    #    ,'reason_code_dea'] = 'activate'

    return dea_df_agg_p


def generate_reason_backlog(
        base_df,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,
        smf_df
        ):
    """
    Args:
        base_df: Baseline simulation that run with prd Ship Map File.
        backlog_df: DataFrame that includes backlog by date, zip, carrier.
        backlog_threshold: Backlog threshold.
        clear_date_threshold: Clear date threshold.
        smf_df: DataFrame that includes expansion SMF by date, zip, carrier.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for Backlog.

    """
    # get latest backlog date
    backlog_df = backlog_df.loc[
                backlog_df['date'] == backlog_df['date'].max()]

    # calculate average adjtnt by date, zip, carrier
    base_df_agg = base_df\
        .groupby(['order_placed_date','zip5','act_fc_name'])['shipment_tracking_number']\
            .nunique()\
                .reset_index()
    base_df_agg.columns = ['shipdate','zip5','fcname','package_count']

    adjtnt_df_avg = smf_df.merge(
        base_df_agg,
        on=['shipdate','zip5','fcname'],
        how='left'
    )
    adjtnt_df_avg['package_count'] = adjtnt_df_avg['package_count'].fillna(0)

    adjtnt_df_avg['weighted_adjtnt'] = adjtnt_df_avg['package_count'] * adjtnt_df_avg['adjtnt']
    adjtnt_df_avg = adjtnt_df_avg\
        .groupby(['shipdate','zip5','mode'])\
            .agg({
                'weighted_adjtnt': 'sum',
                'package_count': 'sum',
                'adjtnt': 'mean'
            })\
                 .reset_index()
    adjtnt_df_avg.columns = [
        'date','zip5','carrier_code','weighted_adjtnt','package_count','adjtnt_avg']
    adjtnt_df_avg['weighted_adjtnt_avg'] = adjtnt_df_avg['weighted_adjtnt'] / adjtnt_df_avg['package_count']
    adjtnt_df_avg.loc[
        adjtnt_df_avg['weighted_adjtnt_avg'].isnull()
        ,'weighted_adjtnt_avg'] = adjtnt_df_avg['adjtnt_avg']

    # merge backlog and average adjtnt
    clear_df = backlog_df.merge(
        adjtnt_df_avg,
        on=['date','zip5','carrier_code'],
        how='left')
    clear_df['weighted_adjtnt_avg'] = clear_df['weighted_adjtnt_avg'].fillna(2)

    # get max of estimate clear date and date+adjtnt_avg
    clear_df['estimated_clear_date_max'] = clear_df.apply(
        lambda row: max(
            pd.to_datetime(row['estimated_clear_date']),
            pd.to_datetime(row['date']) + pd.to_timedelta(row['weighted_adjtnt_avg'], unit='D')
        ),
        axis=1
    )

    clear_df_p = clear_df.pivot(
        index='zip5',
        columns='carrier_code',
        values=['days_behind','estimated_clear_date_max'])\
        .reset_index()

    clear_df_p.columns = ['zip5'] \
        + [f'{col[1]}_{col[0]}' for col in clear_df_p.columns.values if col[0] != 'zip5']

    clear_df_p['reason_code_backlog'] = 'ok'

    """
    clear_df_p.loc[
        (clear_df_p['ONTRGD_days_behind'] >= backlog_threshold)
        & (clear_df_p['ONTRGD_estimated_clear_date_max'] - clear_df_p['FDXHD_estimated_clear_date_max'] >= clear_date_threshold)
        ,'reason_code_backlog'] = 'deactivate'

    clear_df_p.loc[
        (clear_df_p['FDXHD_days_behind'] >= backlog_threshold)
        & (clear_df_p['FDXHD_estimated_clear_date_max'] - clear_df_p['ONTRGD_estimated_clear_date_max'] >= clear_date_threshold)
        ,'reason_code_backlog'] = 'activate'

    print(clear_df_p['FDXHD_estimated_clear_date_max'] - clear_df_p['ONTRGD_estimated_clear_date_max'])
    print(clear_df_p['FDXHD_estimated_clear_date_max'] - clear_df_p['ONTRGD_estimated_clear_date_max'] >= clear_date_threshold)
    print(clear_df_p)

    """

    return clear_df_p


def resolve_current_decision(row):

    """
    Args:
        row: Row of DataFrame containing zip codes to activate or deactivate.

    Returns:
        pd.Series: Series containing current recommendation and reasons.
    """

    reason_dea = row.get('reason_code_dea', None)
    reason_shutdown = row.get('reason_code_shutdown', None)
    reason_backlog = row.get('reason_code_backlog', None)

    # dea ALWAYS takes precedence
    if reason_dea in ['activate', 'deactivate']:
        return pd.Series({
            'current_recommendation': reason_dea,
            'current_reason_dea': reason_dea,
            'current_reason_shutdown': None,
            'current_reason_backlog': None
        })
    # shutdown next (ONLY IF dea is not activate/deactivate)
    elif reason_shutdown in ['activate', 'deactivate']:
        return pd.Series({
            'current_recommendation': reason_shutdown,
            'current_reason_dea': None,
            'current_reason_shutdown': reason_shutdown,
            'current_reason_backlog': None
        })
    # backlog next
    elif reason_backlog in ['activate', 'deactivate']:
        return pd.Series({
            'current_recommendation': reason_backlog,
            'current_reason_dea': None,
            'current_reason_shutdown': None,
            'current_reason_backlog': reason_backlog
        })
    else:
        # Any ok can be propagated
        decision = None
        rdea = None
        rshutdown = None
        rbacklog = None
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
    if status == 0 and last == 'activate':
        last = None
        last_dea = None
        last_shutdown = None
        last_backlog = None
    elif status == 1 and last == 'deactivate':
        last = None
        last_dea = None
        last_shutdown = None
        last_backlog = None

    # If deactivated due to dea before, keep it deactivated
    if last in ['deactivate'] and last_dea == 'deactivate':
        return pd.Series({
            'final_recommendation': None,
            'final_reason_dea': None,
            'final_reason_shutdown': None,
            'final_reason_backlog': None
        })

    # If last decision was a switchback to usual
    if last in ['activate','deactivate'] \
        and last_dea == 'ok' and last_shutdown == 'ok' and last_backlog == 'ok':
        return pd.Series({
            'final_recommendation': None,
            'final_reason_dea': None,
            'final_reason_shutdown': None,
            'final_reason_backlog': None
        })

    # If current recommendation is activate or deactivate, use it
    if curr in ['activate', 'deactivate']:
        return pd.Series({
            'final_recommendation': curr,
            'final_reason_dea': curr_dea,
            'final_reason_shutdown': curr_shutdown,
            'final_reason_backlog': curr_backlog
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
            'final_recommendation': None,
            'final_reason_dea': None,
            'final_reason_shutdown': None,
            'final_reason_backlog': None
        })


def calculate_priority_score(df):
    """
    Args:
        df: DataFrame containing zip codes to calculate priority score.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with priority score.
    """
    if df.shape[0] > 0:
        scaler = MinMaxScaler()
        df['daily_package_count_avg_scaled'] = scaler.fit_transform(
                df[['daily_package_count_avg']])
        df['unpadded_edd_dea_scaled'] = scaler.fit_transform(
                df[['unpadded_edd_dea']])
        df['inverted_dea'] = 1 - df['unpadded_edd_dea_scaled']

        df['priority_score'] = (df['daily_package_count_avg_scaled'] * 0.3) + \
                                (df['inverted_dea'] * 0.7)

        df = df.sort_values(
            'priority_score', 
            ascending=False)
    else:
        df['priority_score'] = None

    return df


def get_zip_recommendation(
        zip_status_df,
        base_df,
        remediate_df,
        expand_df,
        dea_df,
        dea_threshold,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,
        smf_df,
        last_zip_recommendation,
        fc_switch_th=0.1,
        zip_volume_floor=25
    ):
    """
    Args:
        zip_status_df: DataFrame that includes zip status.
        base_df: Baseline simulation that run with prd Ship Map File.
        remediate_df: DataFrame that includes remediate simulation that run with FDXHD Ship Map File.
        expand_df: DataFrame that includes expand simulation that run with prd Ship Map File + eligible ONTRGD Ship Map File.
        dea_df: DataFrame that includes unpadded EDD DEA by delivery date, zip, fc.
        dea_threshold: unpadded EDD DEA threshold.
        dea_lookback_day_count: lookback day count for unpadded EDD DEA calculation.
        backlog_df: DataFrame that includes backlog by date, zip, carrier.
        backlog_threshold: Backlog threshold.
        clear_date_threshold: Clear date threshold.
        smf_df: DataFrame that includes expansion SMF by date, zip, carrier.
        last_zip_recommendation: DataFrame that includes last zip recommendation.
        fc_switch_th: FC switch threshold.
        zip_volume_floor: Zip volume floor.

    Returns:
        pd.DataFrame: DataFrame containing zip codes with reason code for remediation and expansion.
    """

    # Calculate fc-carrier switch
    fc_carrier_switch_remediate = calculate_fc_carrier_switch(
        base_df,
        remediate_df
    )

    fc_carrier_switch_expand = calculate_fc_carrier_switch(
        base_df,
        expand_df
    )

    # Calculate zip daily volume
    zip_volume_daily = calculate_zip_volume_daily(
        base_df
    )

    # DEA rule for remediation
    zips_to_recommend_dea = generate_reason_dea(
        dea_df,
        dea_threshold
        )
    zips_to_recommend_dea = zips_to_recommend_dea[
        ['zip5','FDXHD_unpadded_edd_dea','ONTRGD_unpadded_edd_dea','reason_code_dea']]

    # Backlog rule for remediation
    zips_to_recommend_backlog = generate_reason_backlog(
        base_df,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,
        smf_df
        )
    zips_to_recommend_backlog = zips_to_recommend_backlog[
        ['zip5','FDXHD_days_behind','reason_code_backlog']] # 'ONTRGD_days_behind',

    # shutdown rule for expansion / remediation

    # master table for zips to remediate # FIX THIS
    zips_to_recommend = zip_status_df.merge(fc_carrier_switch_expand, on='zip5', how='left')
    zips_to_recommend.loc[
        zips_to_recommend['active'] == 0,
        'fc_switch_package_per'] = zips_to_recommend.loc[
        zips_to_recommend['active'] == 0,
        'fc_switch_package_per_temp']
    zips_to_recommend = zips_to_recommend.drop(
        ['fc_switch_package_per_temp','carrier_switch_package_count_temp'], axis=1)


    zips_to_recommend = zips_to_recommend.merge(fc_carrier_switch_remediate, on='zip5', how='left')
    zips_to_recommend.loc[
        zips_to_recommend['active'] == 1,
        'fc_switch_package_per'] = zips_to_recommend.loc[
        zips_to_recommend['active'] == 1,
        'fc_switch_package_per_temp']
    zips_to_recommend = zips_to_recommend.drop(
        ['fc_switch_package_per_temp','carrier_switch_package_count_temp'], axis=1)

    zips_to_recommend = zips_to_recommend.merge(zip_volume_daily, on='zip5', how='left')
    zips_to_recommend = zips_to_recommend.merge(zips_to_recommend_dea, on='zip5', how='left')
    zips_to_recommend = zips_to_recommend.merge(zips_to_recommend_backlog, on='zip5', how='left')
    # If reason_dea is activate or deactivate, it ALWAYS takes priority,
    # regardless of any shutdown or backlog value.
    # Only if dea is not activate/deactivate, then shutdown is checked; otherwise, backlog.
    current_recommendation = zips_to_recommend.apply(resolve_current_decision, axis=1)
    zips_to_recommend = pd.concat([zips_to_recommend, current_recommendation], axis=1)

    zips_to_recommend = zips_to_recommend.merge(last_zip_recommendation, on='zip5', how='left')
    final_recommendation = zips_to_recommend.apply(determine_final_decision, axis=1)
    zips_to_recommend = pd.concat([zips_to_recommend, final_recommendation], axis=1)

    # zips_to_recommend.to_parquet('./archieve/zips_to_recommend.parquet')

    # apply additional constraint if any
    if zip_volume_floor:
        zips_to_recommend = zips_to_recommend.loc[
            zips_to_recommend['daily_package_count_avg'] >= zip_volume_floor
        ]

    if fc_switch_th:
        zips_to_recommend = zips_to_recommend.loc[
            zips_to_recommend['fc_switch_package_per'] <= fc_switch_th]

    # zips_to_recommend.to_parquet('./archieve/zips_to_recommend.parquet')

    # split remediation and expansion to two groups
    zips_to_recommend_remediate = zips_to_recommend.loc[
        (zips_to_recommend['active'] == 1)
        & (zips_to_recommend['final_recommendation'] == 'deactivate')]
    zips_to_recommend_expand = zips_to_recommend.loc[
        (zips_to_recommend['active'] == 0)
        & (zips_to_recommend['final_recommendation'] == 'activate')]

    cols = ['zip5','active',
            'daily_package_count_avg',
            'current_recommendation',
            'current_reason_dea',
            'current_reason_shutdown',
            'current_reason_backlog',
            'last_recommendation',
            'last_reason_dea',
            'last_reason_shutdown',
            'last_reason_backlog',
            'final_recommendation',
            'final_reason_dea',
            'final_reason_shutdown',
            'final_reason_backlog'
            ]
    zips_to_recommend_remediate = zips_to_recommend_remediate[
        cols + ['ONTRGD_unpadded_edd_dea']] # ,'ONTRGD_days_behind'
    zips_to_recommend_remediate.columns = cols + ['unpadded_edd_dea'] # ,'days_behind'
    zips_to_recommend_expand = zips_to_recommend_expand[
        cols + ['FDXHD_unpadded_edd_dea']] # ,'FDXHD_days_behind'
    zips_to_recommend_expand.columns = cols + ['unpadded_edd_dea'] # ,'days_behind'

    # apply exclusion list

    # calculate priority score
    zips_to_recommend_remediate = calculate_priority_score(zips_to_recommend_remediate)
    zips_to_recommend_expand = calculate_priority_score(zips_to_recommend_expand)

    zips_to_recommend_remediate = zips_to_recommend_remediate[cols + ['priority_score']]
    zips_to_recommend_expand = zips_to_recommend_expand[cols + ['priority_score']]

    # zips_to_recommend_remediate.to_parquet('./archieve/zips_to_recommend_remediate.parquet')
    # zips_to_recommend_expand.to_parquet('./archieve/zips_to_recommend_expand.parquet')

    return zips_to_recommend_remediate, zips_to_recommend_expand


def select_zips_and_get_simulation(
          simulation_df,
          zip_df,
          zip_count):
    """
    Args:
        simulation_df: DataFrame of selected simulation.
        zip_score_df: DataFrame that contains priority score of each zip code.
        zip_count: Zip count threshold to turn on / off ONTRGD.
        # package_count: Package count threshold to switch one carrier to another.

    Returns:
        pd.DataFrame: DataFrame of selected simulation for selected zips.
    """
    zip_df = zip_df.sort_values(
        'priority_score', 
        ascending=False)
    zip_df = zip_df[0:zip_count]
    # zip_df = zip_df.loc[
    #         (zip_df['carrier_switch_package_count_cumsum'] <= package_count)
    #         ]
    # zip_df = zip_df[
    #     ['zip5','priority_score','reason_code_dea','reason_code_backlog']]\
    #         .drop_duplicates()
    zip_df['selected'] = 1
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

    # parameters
    with open("./configs.yaml") as f:
        config = yaml.safe_load(f)

    RUN_DATE = '2025-10-11' # today().strftime('%Y-%m-%d')
    RUN_NAME = config['EXECUTION']['run_name']

    expansion = config['EXECUTION']['expansion']
    remediation = config['EXECUTION']['remediation']

    baseline_scenario = config['EXECUTION']['baseline_scenario']
    expansion_scenario = config['EXECUTION']['expansion_scenario']
    remediation_scenario = config['EXECUTION']['remediation_scenario']

    lookback_day_count = config['EXECUTION']['lookback_day_count']
    start_date = config['EXECUTION']['start_date']
    end_date = config['EXECUTION']['end_date']

    recommendation_count_list = config['EXECUTION']['recommendation_count_list']
    zip_volume_floor = config['EXECUTION']['zip_volume_floor']
    fc_switch_threshold = config['EXECUTION']['fc_switch_threshold']

    dea_threshold = config['EXECUTION']['dea_threshold']
    dea_lookback_day_count = config['EXECUTION']['dea_lookback_day_count']

    backlog_threshold = config['EXECUTION']['backlog_threshold']
    clear_date_threshold = config['EXECUTION']['clear_date_threshold']

    # set output paths
    output_path = os.path.join('./results/execution', RUN_NAME)
    metrics_path = os.path.join(output_path, 'metrics', RUN_DATE)
    sim_path = os.path.join(output_path, 'simulation_output', RUN_DATE)
    remediation_path = os.path.join(output_path, 'zips_to_remediate', RUN_DATE)
    expansion_path = os.path.join(output_path, 'zips_to_expand', RUN_DATE)

    # set start and end date for simulation data
    if start_date and end_date:
        pass
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_day_count)
        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')

    # load data
    # baseline - remediation - expansion simulation
    baseline_sim_df = read_helper(
        os.path.join('./data/simulations', baseline_scenario),
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
        start_date=start_date,
        end_date=end_date,
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
        os.path.join('./data/simulations', remediation_scenario),
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
        start_date=start_date,
        end_date=end_date
        )
    expand_sim_df = read_helper(
        os.path.join('./data/simulations', expansion_scenario),
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
        start_date=start_date,
        end_date=end_date
        )

    # dea / backlog / shutdown
    dea_df = read_helper(
        './data/execution_data/unpadded_dea',
        start_date=(
            pd.to_datetime(RUN_DATE) - timedelta(days=dea_lookback_day_count))\
                .strftime('%Y-%m-%d'),
        end_date=RUN_DATE,
        date_col_name='delivery_date',
        cols=['delivery_date',
              'ffmcenter_name',
              'carrier_code',
              'zip5',
              'package_count',
              'unpadded_edd_dea_count'
              ]
        )
    backlog_df = read_helper(
        './data/execution_data/backlog',
        start_date=(pd.to_datetime(RUN_DATE) - timedelta(days=3)).strftime('%Y-%m-%d'),
        end_date=RUN_DATE,
        date_col_name='date'
        )

    # smf baseline - expansion
    smf_baseline_df = read_helper(
        './data/smf/baseline',
        start_date=start_date,
        end_date=(pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d'),
        date_col_name='shipdate'
        )
    smf_expansion_df = read_helper(
        './data/smf/expansion',
        start_date=start_date,
        end_date=(pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d'),
        date_col_name='shipdate'
        )

    # last recommendation if any
    last_zip_recommendation = get_last_recommendation(
       RUN_NAME,
       RUN_DATE,
       dea_threshold,
       fc_switch_threshold,
       zc=999999
        )

    # exclusion list if any
    # exclusion_list = read_helper()

    # generate zip ONTRGD status
    zip_status_df = get_zip_status(
        smf_baseline_df,
        smf_expansion_df,
        date=end_date
        )
    logger.info(
        'ONTRGD zip count by active status: \n%s', 
        zip_status_df.groupby('active')['zip5'].count().to_string())

    # apply decision rules and get candidate zips to remediate / expand for ONTRGD
    zips_to_remediate, zips_to_expand = get_zip_recommendation(
        zip_status_df,
        baseline_sim_df,
        remediate_sim_df,
        expand_sim_df,
        dea_df,
        dea_threshold,
        backlog_df,
        backlog_threshold,
        clear_date_threshold,
        smf_expansion_df,
        last_zip_recommendation,
        )

    ZIP_COUNT_MAX = 0
    if zips_to_remediate.shape[0] > 0:
        REMEDIATION = True
        ZIP_COUNT_MAX = max(ZIP_COUNT_MAX, zips_to_remediate.shape[0])
    else:
        REMEDIATION = False
    if zips_to_expand.shape[0] > 0:
        EXPANSION = True
        ZIP_COUNT_MAX = max(ZIP_COUNT_MAX, zips_to_expand.shape[0])
    else:
        EXPANSION = False
    filtered_list = [
        item for item in recommendation_count_list
            if item <= ZIP_COUNT_MAX]
    filtered_list.append(999999)


    # select zips and get simulation result based on recommendation zip count
    for zc in filtered_list:

        logger.info('recommendation zip count: %i', zc)

        SETTING_NAME = f'dea_th_{dea_threshold}_fc_switch_{fc_switch_threshold}_rec_count_{zc}'

        baseline_sim_df_no_change = baseline_sim_df.copy()

        if REMEDIATION:
            remediate_sim_df_temp, remediate_zips_temp = select_zips_and_get_simulation(
                remediate_sim_df,
                zips_to_remediate,
                zc)
            logger.info('remediated zip count: %i', remediate_zips_temp.shape[0])

            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                remediate_zips_temp,
                on='zip5',
                how='left')
            logger.info('carrier switch package count: %i',
                baseline_sim_df_no_change.loc[
                    (baseline_sim_df_no_change['base_carrier_code'] == 'ONTRGD')
                    & (baseline_sim_df_no_change['selected'] == 1)].groupby(
                    ['base_carrier_code','selected'])['shipment_tracking_number'].nunique().item()
                    )

            baseline_sim_df_no_change = baseline_sim_df_no_change.loc[
                baseline_sim_df_no_change['selected'].isnull()]
            baseline_sim_df_no_change = baseline_sim_df_no_change.drop('selected',axis=1)

        if EXPANSION:
            expand_sim_df_temp, expand_zips_temp = select_zips_and_get_simulation(
                expand_sim_df,
                zips_to_expand,
                zc)
            logger.info('expanded zip count: %i', expand_zips_temp.shape[0])

            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                expand_zips_temp,
                on='zip5',
                how='left')

            logger.info('carrier switch package count: %i',
                expand_sim_df_temp.loc[expand_sim_df_temp['sim_carrier_code'] == 'ONTRGD'].groupby(
                    ['sim_carrier_code'])['shipment_tracking_number'].nunique().item()
                    )

            baseline_sim_df_no_change = baseline_sim_df_no_change.loc[
                baseline_sim_df_no_change['selected'].isnull()]
            baseline_sim_df_no_change = baseline_sim_df_no_change.drop('selected',axis=1)


        # generate final simulation result with expand / remediate decisions
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
            'sim_transit_cost']

        if REMEDIATION:
            final_sim_df = pd.concat([final_sim_df, remediate_sim_df_temp])
        if EXPANSION:
            final_sim_df = pd.concat([final_sim_df, expand_sim_df_temp])

        # calculate & save all metrics
        # save recommendations

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)

        # estimate network level ONTRGD % before wms proxy
        carrier_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_carrier_code': 'sim_carrier_code'}, inplace=False),
            final_sim_df,
            ['sim_carrier_code'],
            'shipment_tracking_number',
            'nunique'
            )
        print('NETWORK LEVEL ONTRGD % - BEFORE WMS PROXY')
        print(carrier_change)

        # estimate network level ONTRGD % after wms proxy
        baseline_sim_df_proxy = apply_wms_proxy(
            baseline_sim_df.rename(
                columns={'base_carrier_code': 'sim_carrier_code',
                        'base_fc_name': 'sim_fc_name'
                        }, inplace=False),
            'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_0925_v2;')
        final_sim_df_proxy = apply_wms_proxy(
            final_sim_df,
            'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_0925_v2;')

        carrier_change_proxy = calculate_package_distribution_change_by_groups(
            baseline_sim_df_proxy,
            final_sim_df_proxy,
            ['carrier'],
            'package_count',
            'sum'
            )
        print('NETWORK LEVEL ONTRGD % - AFTER WMS PROXY')
        print(carrier_change_proxy)

        # estimate fc charge changes
        print('FC CHARGE CHANGES')
        fc_charge_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_fc_name': 'sim_fc_name'}, inplace=False),
            final_sim_df,
            ['sim_fc_name'],
            'shipment_tracking_number',
            'nunique')
        print(fc_charge_change)

        excel_file_path = os.path.join(metrics_path, SETTING_NAME + '.xlsx')
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            carrier_change.to_excel(
                writer, 
                sheet_name='carrier_network_before_wms_proxy', 
                index=False)
            carrier_change_proxy.to_excel(
                writer, 
                sheet_name='carrier_network_after_wms_proxy', 
                index=False)
            fc_charge_change.to_excel(
                writer, 
                sheet_name='fc_charge_changes', 
                index=False)


        # Save final simulation
        if not os.path.exists(sim_path):
            os.makedirs(sim_path)

        final_sim_df.to_parquet(
            os.path.join(sim_path,
                        f'{SETTING_NAME}.parquet'
                        )
            )

        # Save zip recommendations
        if EXPANSION:
            if not os.path.exists(expansion_path):
                os.makedirs(expansion_path)

            expand_zips_temp.to_parquet(
                os.path.join(expansion_path,
                            f'{SETTING_NAME}.parquet'
                            )
            )

        if REMEDIATION:
            if not os.path.exists(remediation_path):
                os.makedirs(remediation_path)

            remediate_zips_temp.to_parquet(
                os.path.join(remediation_path,
                            f'{SETTING_NAME}.parquet'
                            )
            )
