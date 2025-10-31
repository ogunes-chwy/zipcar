import pandas as pd
import numpy as np
import os
from metric_helper import read_helper, calculate_package_distribution_change_by_groups, apply_wms_proxy
import logging
import yaml
from datetime import date, timedelta


# Get a logger instance
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s', # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S' # Define the timestamp format
)
logger = logging.getLogger(__name__)


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
    fc_carrier_switch['fc_switch_package_per'] = fc_carrier_switch['fc_switch_package_count'] / fc_carrier_switch['package_count']
    fc_carrier_switch = fc_carrier_switch.merge(carrier_switch,on='zip5',how='left')
    fc_carrier_switch['carrier_switch_package_count'] = fc_carrier_switch['carrier_switch_package_count'].fillna(0)
    # fc_carrier_switch['carrier_switch_package_per'] = fc_carrier_switch['carrier_switch_package_count'] / fc_carrier_switch['package_count']

    return fc_carrier_switch


def generate_reason_dea(
        dea_df,
        dea_threshold,
        dea_lookback_day_count
        ):
    """
    Args:
        dea_df: .
        dea_threshold: .
        dea_lookback_day_count: .

    Returns:
        pd.DataFrame: .

    """

    max_date = pd.to_datetime(dea_df['delivery_date'].max())
    dea_start_date = (max_date - timedelta(days=dea_lookback_day_count)).strftime('%Y-%m-%d')

    dea_df_agg = dea_df.loc[
        (dea_df['delivery_date'] >= dea_start_date)
    ]\
        .groupby(['zip5','carrier_code'])[['unpadded_edd_dea_count','package_count']]\
        .sum()\
        .reset_index()

    dea_df_agg['unpadded_edd_dea'] = dea_df_agg[
        'unpadded_edd_dea_count'] / dea_df_agg[
            'package_count']

    dea_df_agg_p = dea_df_agg.pivot(
        index='zip5',
        columns='carrier_code',
        values='unpadded_edd_dea')\
        .reset_index()

    dea_df_agg_p = dea_df_agg_p[~dea_df_agg_p['ONTRGD'].isnull()]
    dea_df_agg_p = dea_df_agg_p[~dea_df_agg_p['FDXHD'].isnull()]

    dea_df_agg_p['ontrgd_dea_flag'] = 0
    dea_df_agg_p.loc[
        (dea_df_agg_p['ONTRGD'] < dea_threshold),
        'ontrgd_dea_flag'] = 1

    dea_df_agg_p['fdxhd_dea_flag'] = 0
    dea_df_agg_p.loc[
        (dea_df_agg_p['FDXHD'] < dea_threshold),
        'fdxhd_dea_flag'] = 1

    dea_df_agg_p['reason_code_dea'] = 'ok'
    dea_df_agg_p.loc[
        (dea_df_agg_p['ontrgd_dea_flag'] == 1)
        & (dea_df_agg_p['fdxhd_dea_flag'] == 0)
        ,'reason_code_dea'] = 'deactivate'

    return dea_df_agg_p


def generate_reason_backlog(
        backlog_df,
        backlog_threshold,
        smf_df
        ):
    """
    Args:
        backlog_df: .
        backlog_threshold: .

    Returns:
        pd.DataFrame: .

    """

    backlog_df['max_date'] = backlog_df['date'].max()
    backlog_df = backlog_df.loc[backlog_df['date'] == backlog_df['max_date']]

    backlog_df['backlog_flag'] = 0
    backlog_df.loc[
        backlog_df['days_behind'] >= backlog_threshold,
    'backlog_flag'] = 1

    smf_df_agg = smf_df\
        .groupby(['shipdate','zip5','mode'])['adjtnt']\
            .mean()\
                .reset_index()
    smf_df_agg.columns = ['date','zip5','carrier_code','adjtnt']

    clear_df = backlog_df.merge(
        smf_df_agg,
        on=['date','zip5','carrier_code'])

    clear_df['estimated_clear_backlog_delivery_date'] = pd.to_datetime(
        clear_df['estimated_clear_date']
        ) + pd.to_timedelta(
            clear_df['adjtnt'],
            unit='D'
            )
    clear_df = clear_df[[
        'zip5',
        'carrier_code',
        'days_to_current',
        'estimated_clear_date',
        'adjtnt',
        'estimated_clear_backlog_delivery_date'
        ]]

    # print(clear_df)

    return 0


def get_remediation_zips(
        base_df,
        remediate_df,
        dea_df,
        dea_threshold,
        dea_lookback_day_count,
        backlog_df,
        backlog_threshold,
        smf_df,
        zip_volume_floor= 25,
        fc_switch_th = None
):
    """
    Applies remedation rules for ONTRGD zip codes 
    that had at least one package delivered by FDXHD & ONTRGD.

    Args:
        base_df: Baseline simulation that run with prd Ship Map File.
        remediate_df: Remediation simulation that run with 
                    prd Ship Map File (-) ONTRGD.
        dea_df: DataFrame that includes unpadded EDD DEA by 
                delivery date, zip, fc, carrier.
        dea_threshold: unpadded EDD DEA threshold for ONTRGD remediation.
        dea_lookback_day_count: lookback day count for unpadded EDD DEA calculation.

    Returns:
        pd.DataFrame: DataFrame containing zip codes to remediate.
    """

    # Calculate fc-carrier switch
    fc_carrier_switch = calculate_fc_carrier_switch(
        base_df,
        remediate_df
    )


    # Calculate zip daily volume
    zip_volume_daily = calculate_zip_volume_daily(
        base_df
    )
    zip_volume_daily = zip_volume_daily.loc[
        zip_volume_daily['daily_package_count_avg'] >= zip_volume_floor
    ]


    # DEA rule for remediation
    zips_to_remediate_dea = generate_reason_dea(
        dea_df,
        dea_threshold,
        dea_lookback_day_count
        )

    zips_to_remediate_dea = zips_to_remediate_dea.loc[
        (zips_to_remediate_dea['reason_code_dea'] == 'deactivate'),
    ['zip5','ONTRGD']]
    zips_to_remediate_dea.columns = ['zip5','unpadded_edd_dea']


    # backlog rule for remediation
    zips_to_remediate_backlog = generate_reason_backlog(
        backlog_df,
        backlog_threshold,
        smf_df
        )





    # shutdown rule for remediation

    # check last decision

    # master table for zips to remediate
    zips_to_remediate = fc_carrier_switch.merge(zip_volume_daily, on='zip5')
    zips_to_remediate = zips_to_remediate.merge(zips_to_remediate_dea, on='zip5')

    # apply additional constraint if any
    if fc_switch_th:
        zips_to_remediate = zips_to_remediate.loc[
            zips_to_remediate['fc_switch_package_per'] <= fc_switch_th]

    # order by dea % - priority score next
    zips_to_remediate = zips_to_remediate.sort_values('unpadded_edd_dea')

    # cumsum of carrier switch package counts to decide where to stop
    zips_to_remediate[
        'carrier_switch_package_count_cumsum'] = zips_to_remediate[
            'carrier_switch_package_count'].cumsum()

    # zips_to_remediate.to_parquet('./archieve/zips_to_remediate.parquet')

    zips_to_remediate = zips_to_remediate[[
        'zip5',
        'unpadded_edd_dea',
        'fc_switch_package_per',
        'carrier_switch_package_count',
        'carrier_switch_package_count_cumsum'
    ]]

    return zips_to_remediate


def get_expansion_zips(
        base_df,
        expand_df,
        baseline_scenario,
        expansion_scenario,
        end_date,
        dea_df,
        dea_threshold,
        dea_lookback_day_count,
        zip_volume_floor,
        fc_switch_th = None
):
    """
    Applies remedation rules for ONTRGD zip codes 
    that had at least one package delivered by FDXHD & ONTRGD.

    Args:
        base_df: Baseline simulation that run with prd Ship Map File.
        expand_df: Expansion simulation that run with 
                    prd Ship Map File (+) eligible ONTRGD.
        baseline_scenario: baseline scenario.
        expansion_scenario: expansion scenario.
        end_date: simulation end date.
        dea_df: DataFrame that includes unpadded EDD DEA by 
                delivery date, zip, fc, carrier.
        dea_threshold: unpadded EDD DEA threshold for ONTRGD remediation.
        dea_lookback_day_count: lookback day count for unpadded EDD DEA calculation.
        fc_switch_th: Threshold of fc switching with carrier network change.

    Returns:
        pd.DataFrame: DataFrame containing zip codes to remediate.
    """

    # Get inactive ONTRGD zips to expand
    baseline_smf = read_helper(
        os.path.join('./data/smf', baseline_scenario),
        cols=['zip5','mode'],
        start_date=end_date,
        end_date=end_date,
        date_col_name='shipdate'
        )

    expand_smf = read_helper(
        os.path.join('./data/smf', expansion_scenario),
        cols=['zip5','mode'],
        start_date=end_date,
        end_date=end_date,
        date_col_name='shipdate'
        )

    baseline_smf = baseline_smf.loc[
        baseline_smf['mode'] == 'ONTRGD',
        ['zip5']].drop_duplicates()
    baseline_smf['active'] = 1

    expand_smf = expand_smf.loc[
        expand_smf['mode'] == 'ONTRGD',
        ['zip5']].drop_duplicates()

    eligible_ontrgd = expand_smf.merge(
        baseline_smf,
        on='zip5',
        how='left'
        )

    # Calculate fc-carrier switch
    fc_carrier_switch = calculate_fc_carrier_switch(
        base_df,
        expand_df
    )

    # Calculate zip daily volume
    zip_volume_daily = calculate_zip_volume_daily(
        base_df
    )
    zip_volume_daily = zip_volume_daily.loc[
        zip_volume_daily['daily_package_count_avg'] >= zip_volume_floor
    ]

    # DEA rule for expansion
    max_date = pd.to_datetime(dea_df['delivery_date'].max())
    dea_start_date = (max_date - timedelta(days=dea_lookback_day_count)).strftime('%Y-%m-%d')

    dea_df_agg = dea_df.loc[
        (dea_df['delivery_date'] >= dea_start_date)
    ]\
        .groupby(['zip5','carrier_code'])[['unpadded_edd_dea_count','package_count']]\
        .sum()\
        .reset_index()

    dea_df_agg['unpadded_edd_dea'] = dea_df_agg[
        'unpadded_edd_dea_count'] / dea_df_agg[
            'package_count']

    dea_df_agg = dea_df_agg.loc[
        dea_df_agg['carrier_code'] == 'FDXHD']

    dea_df_agg['fdxhd_dea_flag'] = 0
    dea_df_agg.loc[
        (dea_df_agg['unpadded_edd_dea'] < dea_threshold),
        'fdxhd_dea_flag'] = 1

    zips_to_expand_dea = dea_df_agg.loc[
        (dea_df_agg['fdxhd_dea_flag'] == 1),
    ['zip5','unpadded_edd_dea']]
    zips_to_expand_dea.columns = ['zip5','unpadded_edd_dea']

    # TnT rule for expansion
    compare_df = base_df.merge(
         expand_df,
         on=['order_id','shipment_tracking_number','order_placed_date','zip5'])

    zips_to_expand_tnt = compare_df.groupby(
        ['zip5'])[['base_tnt','sim_tnt']].mean().reset_index()
    zips_to_expand_tnt[
         'tnt_change'] = zips_to_expand_tnt[
              'sim_tnt']-zips_to_expand_tnt[
                   'base_tnt']

    zips_to_expand_tnt = zips_to_expand_tnt.loc[
        zips_to_expand_tnt['tnt_change'] <= 0,
    ['zip5','tnt_change']]

    # master table for zips to remediate
    zips_to_expand = fc_carrier_switch.merge(zip_volume_daily, on='zip5')
    zips_to_expand = zips_to_expand.merge(zips_to_expand_dea, on='zip5')
    zips_to_expand = zips_to_expand.merge(zips_to_expand_tnt, on='zip5')

    # apply additional constraint if any
    zips_to_expand = zips_to_expand.merge(
        eligible_ontrgd,
        on=['zip5'],
        how='left')
    zips_to_expand.loc[
        zips_to_expand['active'].isnull(),
        'active'] = 0
    zips_to_expand = zips_to_expand.loc[
            zips_to_expand['active'] == 0]

    if fc_switch_th:
        zips_to_expand = zips_to_expand.loc[
            zips_to_expand['fc_switch_package_per'] <= fc_switch_th]

    # order by dea % - priority score next
    zips_to_expand = zips_to_expand.sort_values('unpadded_edd_dea')

    # cumsum of carrier switch package counts to decide where to stop
    zips_to_expand[
        'carrier_switch_package_count_cumsum'] = zips_to_expand[
            'carrier_switch_package_count'].cumsum()

    # zips_to_expand.to_parquet('./archieve/zips_to_expand.parquet')

    zips_to_expand = zips_to_expand[[
        'zip5',
        'unpadded_edd_dea',
        'tnt_change',
        'fc_switch_package_per',
        'carrier_switch_package_count',
        'carrier_switch_package_count_cumsum'
    ]]

    return zips_to_expand


def select_zips_and_get_simulation(
          simulation_df,
          zip_df,
          package_count):
    """
    Args:
        simulation_df: DataFrame of selected simulation.
        zip_score_df: DataFrame that contains priority score of each zip code.
        package_count: Package count threshold to switch one carrier to another.

    Returns:
        pd.DataFrame: DataFrame of selected simulation for selected zips.
    """

    zip_df = zip_df.loc[
            (zip_df['carrier_switch_package_count_cumsum'] <= package_count)]
    zip_df = zip_df[['zip5']].drop_duplicates()
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

    run_name = config['EXECUTION']['run_name']
    run_date = '2025-10-12' # today().strftime('%Y-%m-%d')

    expansion = config['EXECUTION']['expansion']
    remediation = config['EXECUTION']['remediation']

    baseline_scenario = config['EXECUTION']['baseline_scenario']
    expansion_scenario = config['EXECUTION']['expansion_scenario']
    remediation_scenario = config['EXECUTION']['remediation_scenario']

    lookback_day_count = config['EXECUTION']['lookback_day_count']
    start_date = config['EXECUTION']['start_date']
    end_date = config['EXECUTION']['end_date']

    dea_threshold = config['EXECUTION']['dea_threshold']
    dea_lookback_day_count = config['EXECUTION']['dea_lookback_day_count']

    backlog_threshold = config['EXECUTION']['backlog_threshold']

    carrier_switch_package_count_cap = config['EXECUTION']['carrier_switch_package_count_cap']
    zip_volume_floor = config['EXECUTION']['zip_volume_floor']
    fc_switch_threshold = config['EXECUTION']['fc_switch_threshold']
    fc_charge_change_allowance = config['EXECUTION']['fc_charge_change_allowance']


    if start_date and end_date:
        pass
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_day_count)
        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')


    # load baseline simulation / dea / backlog / shutdown data
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
    dea_df = read_helper(
        './data/execution_data/unpadded_dea',
        start_date=start_date,
        end_date=end_date,
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
        start_date=end_date,
        end_date=(pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d'),
        date_col_name='date'
    )
    smf_df = read_helper(
        './data/smf/expansion',
        start_date=end_date,
        end_date=(pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d'),
        date_col_name='shipdate'
    )

    # apply decision rules and get candidate zips to remediate / expand for ONTRGD
    if remediation:
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
        zips_to_remediate = get_remediation_zips(
            baseline_sim_df,
            remediate_sim_df,
            dea_df,
            dea_threshold,
            dea_lookback_day_count,
            backlog_df,
            backlog_threshold,
            smf_df,
            zip_volume_floor,
            fc_switch_threshold
        )

    if expansion:
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
        zips_to_expand = get_expansion_zips(
            base_df=baseline_sim_df,
            expand_df=expand_sim_df,
            baseline_scenario=baseline_scenario,
            expansion_scenario=expansion_scenario,
            end_date=end_date,
            dea_df=dea_df,
            dea_threshold=dea_threshold,
            dea_lookback_day_count=dea_lookback_day_count,
            zip_volume_floor=zip_volume_floor,
            fc_switch_th=fc_switch_threshold
        )


    package_count_change_list = []
    package_count_change_list.append(carrier_switch_package_count_cap)


    for pc in package_count_change_list:

        print('PACKAGE COUNT SWITCH TARGET', pc)
        baseline_sim_df_no_change = baseline_sim_df.copy()

        if remediation:
            print('ONTRGD remediation')
            remediate_sim_df_temp, remediate_zips_temp = select_zips_and_get_simulation(
                remediate_sim_df,
                zips_to_remediate,
                pc)
            print('remediated zip count: ', remediate_zips_temp.shape[0])

            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                remediate_zips_temp,
                on='zip5',
                how='left')
            print('carrier switch package count: ',
                baseline_sim_df_no_change.loc[
                    (baseline_sim_df_no_change['base_carrier_code'] == 'ONTRGD')
                    & (baseline_sim_df_no_change['selected'] == 1)].groupby(
                    ['base_carrier_code','selected'])['shipment_tracking_number'].nunique().item()
                    )

            baseline_sim_df_no_change = baseline_sim_df_no_change.loc[
                baseline_sim_df_no_change['selected'].isnull()]
            baseline_sim_df_no_change = baseline_sim_df_no_change.drop('selected',axis=1)

        if expansion:
            print('ONTRGD expansion')
            expand_sim_df_temp, expand_zips_temp = select_zips_and_get_simulation(
                expand_sim_df,
                zips_to_expand,
                pc)
            print('expanded zip count: ', expand_zips_temp.shape[0])

            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                expand_zips_temp,
                on='zip5',
                how='left')

            print('carrier switch package count: ',
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

        if remediation:
            final_sim_df = pd.concat([final_sim_df, remediate_sim_df_temp])
        if expansion:
            final_sim_df = pd.concat([final_sim_df, expand_sim_df_temp])


        # check package count change & ONTRGD % at network level
        # calculate & print all metrics

        # check network level ONTRGD %
        # rule 1
        # before wms proxy
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

        # after wms proxy
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

        # check fc charge changes for stop condition
        # rule 2
        print('FC CHARGE CHANGES')
        fc_charge_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_fc_name': 'sim_fc_name'}, inplace=False),
            final_sim_df,
            ['sim_fc_name'],
            'shipment_tracking_number',
            'nunique')
        print(fc_charge_change)


        # if rules not passed, decrease carrier switch package count to 3/4 and rerun
        if any(x > fc_charge_change_allowance for x in abs(fc_charge_change['abs_percent_change'])):
            logger.info('Decreasing carrier switch package count allowance ...')

            package_count_change_list.append(pc*3/4)

        else:
            logger.info('Saving ...')

            # Save output
            s_d = final_sim_df['order_placed_date'].min()
            e_d = final_sim_df['order_placed_date'].max()

            r_target = np.round(carrier_change.loc[
                carrier_change['sim_carrier_code'] == 'ONTRGD',
                'iter_percent'].item(),1)


            if not os.path.exists('./results/execution' + f'/{run_name}/{run_date}/simulation'):
                os.makedirs('./results/execution' + f'/{run_name}/{run_date}/simulation')

            final_sim_df.to_parquet(
                os.path.join('./results/execution',
                            run_name,
                            run_date,
                            'simulation',
                            f'{s_d}_{e_d}_dea_th_{dea_threshold}_fc_switch_{fc_switch_threshold}_carrier_switch_pc_{pc}.parquet'
                            )
                )

            if not os.path.exists('./results/execution' + f'/{run_name}/{run_date}/zips_to_expand'):
                os.makedirs('./results/execution' + f'/{run_name}/{run_date}/zips_to_expand')

            if expansion:
                expand_zips_temp.to_parquet(
                    os.path.join('./results/execution',
                                run_name,
                                run_date,
                                'zips_to_expand',
                                f'{s_d}_{e_d}_dea_th_{dea_threshold}_fc_switch_{fc_switch_threshold}_carrier_switch_pc_{pc}.parquet'
                                )
                )

            if not os.path.exists('./results/execution' + f'/{run_name}/{run_date}/zips_to_remediate'):
                os.makedirs('./results/execution' + f'/{run_name}/{run_date}/zips_to_remediate')

            if remediation:
                remediate_zips_temp.to_parquet(
                    os.path.join('./results/execution',
                                run_name,
                                run_date,
                                'zips_to_remediate',
                                f'{s_d}_{e_d}_dea_th_{dea_threshold}_fc_switch_{fc_switch_threshold}_carrier_switch_pc_{pc}.parquet'
                                )
                )