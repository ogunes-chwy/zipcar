import pandas as pd
import numpy as np
import os
from metric_helper import calculate_package_distribution_change_by_groups, apply_wms_proxy, read_helper
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


def get_remediation_score(
          base_df):
    """
    Calculates remedation priority score for ONTRGD zip codes 
    that had at least one package delivered by FDXHD & ONTRGD.

    Args:
        base_df: Baseline simulation that run with prd Ship Map File.
        remediation_path: Path to remediation simulation that 
                                     run with FDXHD Ship Map File only.

    Returns:
        pd.DataFrame: DataFrame containing remediation scores of zip codes.
        pd.DataFrame: DataFrame containing remediation simulation 
                        for zip codes eligible for remediation.
    """

    # filter out shipments not delivered yet
    base_df_delivered = base_df.loc[
         ~base_df['std'].isnull()]

    # collect metrics to calculate remediation scores
    remediation_scores_packages = base_df_delivered.groupby(
    ['zip5', 'act_carrier_code']
    )['shipment_tracking_number'].nunique().reset_index()
    remediation_scores_packages.columns = ['zip5', 'act_carrier_code', 'package_count']

    remediation_scores_cost = base_df_delivered.groupby(
        ['zip5', 'act_carrier_code']
        )['act_transit_cost'].mean().reset_index()
    remediation_scores_cost.columns = ['zip5', 'act_carrier_code', 'act_transit_cost_avg']

    remediation_scores_std = base_df_delivered.groupby(
        ['zip5', 'act_carrier_code']
        )['std'].mean().reset_index()
    remediation_scores_std.columns = ['zip5', 'act_carrier_code', 'std_avg']

    remediation_scores_dea = base_df_delivered.groupby(
        ['zip5', 'act_carrier_code']
        )['dea_flag'].sum().reset_index()
    remediation_scores_dea.columns = ['zip5', 'act_carrier_code', 'dea_count']

    # for comparison both carrier must be available
    ontrgd_zips = base_df_delivered.loc[
        base_df_delivered['act_carrier_code'] == 'ONTRGD',
    ['zip5']].drop_duplicates()

    fdxhd_zips = base_df_delivered.loc[
        base_df_delivered['act_carrier_code'] == 'FDXHD',
    ['zip5']].drop_duplicates()

    remediation_scores_df = remediation_scores_packages.merge(
         remediation_scores_cost,
         on=['zip5','act_carrier_code']
         )
    remediation_scores_df = remediation_scores_df.merge(
         remediation_scores_std,
         on=['zip5','act_carrier_code']
         )
    remediation_scores_df = remediation_scores_df.merge(
         remediation_scores_dea,
         on=['zip5','act_carrier_code']
         )
    remediation_scores_df = remediation_scores_df.merge(
         ontrgd_zips,
         on=['zip5']
         )
    remediation_scores_df = remediation_scores_df.merge(
         fdxhd_zips,
         on=['zip5'])

    remediation_scores_df[
         'dea_per'] = np.round(remediation_scores_df[
              'dea_count'].astype(int) / remediation_scores_df[
                   'package_count'].astype(int), 4)

    remediation_scores_df = remediation_scores_df.pivot(
        index='zip5',
        columns='act_carrier_code',
        values=['dea_per','act_transit_cost_avg','std_avg']
    ).reset_index()

    remediation_scores_df[
         'dea_per_change'] = remediation_scores_df[
              'dea_per']['ONTRGD'] - remediation_scores_df[
                   'dea_per']['FDXHD']
    remediation_scores_df[
         'cost_change'] = remediation_scores_df[
              'act_transit_cost_avg']['FDXHD'] - remediation_scores_df[
                   'act_transit_cost_avg']['ONTRGD']
    remediation_scores_df[
         'std_change'] = remediation_scores_df[
              'std_avg']['FDXHD'] - remediation_scores_df[
                   'std_avg']['ONTRGD']

    remediation_scores_df = remediation_scores_df[
         ['zip5','dea_per_change','cost_change','std_change']].reset_index(drop=True)
    remediation_scores_df = remediation_scores_df.droplevel('act_carrier_code',axis=1)

    # assign priority score
    # dea change negative -> ontrgd worse
    # cost change negative -> ontrgd worse
    # std change negative -> ontrgd worse
    # low priority score -> likely to remediate

    # Normalize dea_per_change
    remediation_scores_df["normalized_dea_per_change"] = (
        (remediation_scores_df[
             "dea_per_change"] - remediation_scores_df[
                  "dea_per_change"].min()) /
        (remediation_scores_df[
             "dea_per_change"].max() - remediation_scores_df[
                  "dea_per_change"].min())
    )

    # Normalize cost_change
    remediation_scores_df["normalized_cost_change"] = (
        (remediation_scores_df[
             "cost_change"] - remediation_scores_df[
                  "cost_change"].min()) /
        (remediation_scores_df[
             "cost_change"].max() - remediation_scores_df[
                  "cost_change"].min())
    )

    # Normalize tnt_change
    remediation_scores_df["normalized_std_change"] = (
        (remediation_scores_df[
             "std_change"] - remediation_scores_df[
                  "std_change"].min()) /
        (remediation_scores_df[
             "std_change"].max() - remediation_scores_df[
                  "std_change"].min())
    )

    w1, w2, w3 = 0.7, 0.2, 0.1
    remediation_scores_df["priority_score"] = (
        w1 * remediation_scores_df["normalized_dea_per_change"] +
        w2 * remediation_scores_df["normalized_cost_change"] +
        w3 * remediation_scores_df["normalized_std_change"]
    )

    package_by_zip = base_df_delivered.groupby(
         'zip5')['shipment_tracking_number'].nunique().reset_index()
    package_by_zip.columns = ['zip5', 'package_count']
    remediation_scores_df = remediation_scores_df.merge(package_by_zip, on='zip5')

    package_to_fdxhd = base_df_delivered.loc[
        base_df_delivered['base_carrier_code'] == 'ONTRGD'
    ].groupby('zip5')['shipment_tracking_number'].nunique().reset_index()
    package_to_fdxhd.columns = ['zip5', 'carrier_switch_package_count']
    remediation_scores_df = remediation_scores_df.merge(package_to_fdxhd, on='zip5')
    remediation_scores_df[
        'carrier_switch_package_count'] = remediation_scores_df[
            'carrier_switch_package_count'].fillna(0)

    remediation_scores_df = remediation_scores_df.sort_values('priority_score')
    remediation_scores_df = remediation_scores_df.reset_index(drop=True)

    remediation_scores_df[
            'carrier_switch_package_count_cumsum'] = remediation_scores_df[
                'carrier_switch_package_count'].cumsum()

    remediation_scores_df[
            'package_count_cumsum_per'] = remediation_scores_df[
                 'package_count'].cumsum() / remediation_scores_df[
                      'package_count'].sum()

    return remediation_scores_df


def get_expansion_score(
          base_df,
          expand_df,
          fc_switch_th,
          baseline_scenario,
          expansion_scenario,
          s,
          e):
    """
    Calculates remedation priority score for ONTRGD zip codes 
    that had at least one package delivered by FDXHD & ONTRGD.

    Args:
        base_df: Baseline simulation that run with prd Ship Map File.
        expand_df: Expansion simulation that run with 
                    prd Ship Map File + eligible ONTRGD Ship Map File.
        fc_switch_th: Minimum threshold for selecting same FC 
                             before and after ONTRGD activation.
        baseline_scenario: Baseline scenario.
        expansion_scenario: Expansion scenario.

    Returns:
        pd.DataFrame: DataFrame containing expansion scores of zip codes.
        pd.DataFrame: DataFrame containing expansion simulation 
                        for zip codes eligible for expansion.
    """

    baseline_smf = read_helper(
        os.path.join('./data/smf', baseline_scenario),
        cols=['zip5','mode'],
        start_date=s,
        end_date=e,
        date_col_name='shipdate'
        )

    expand_smf = read_helper(
        os.path.join('./data/smf', expansion_scenario),
        cols=['zip5','mode'],
        start_date=s,
        end_date=e,
        date_col_name='shipdate'
        )

    baseline_smf = baseline_smf.loc[
        baseline_smf['mode'] == 'ONTRGD',
        ['zip5']].drop_duplicates()
    baseline_smf['active'] = 1

    expand_smf = expand_smf.loc[
        expand_smf['mode'] == 'ONTRGD',
        ['zip5']].drop_duplicates()

    eligible_ontrgd = expand_smf.merge(baseline_smf, on='zip5', how='left')

    compare_df = base_df.merge(
         expand_df,
         on=['order_id','shipment_tracking_number','order_placed_date','zip5'])

    compare_df = compare_df.merge(eligible_ontrgd,
                                on=['zip5'],
                                how='left')
    compare_df.loc[compare_df['active'].isnull(),
                'active'] = 0

    # cost and TnT changes with ONTRGD introduction by zip
    expansion_scores_df = compare_df.groupby(
         ['zip5','active'])[['shipment_tracking_number']].nunique().reset_index()
    expansion_scores_df.columns = ['zip5','active','package_count']

    expansion_scores_cost = compare_df.groupby(
         ['zip5'])[['base_transit_cost','sim_transit_cost']].sum().reset_index()
    expansion_scores_cost[
         'cost_change'] = expansion_scores_cost[
              'sim_transit_cost']-expansion_scores_cost[
                   'base_transit_cost']

    expansion_scores_tnt = compare_df.groupby(
        ['zip5'])[['base_tnt','sim_tnt']].mean().reset_index()
    expansion_scores_tnt[
         'tnt_change'] = expansion_scores_tnt[
              'sim_tnt']-expansion_scores_tnt[
                   'base_tnt']

    # calculate FC-carrier switch percent by zip once ONTRGD activation
    fc_carrier_switch = calculate_fc_carrier_switch(
        base_df,
        expand_df
    )
    fc_carrier_switch = fc_carrier_switch.drop('package_count', axis=1)


    expansion_scores_df = expansion_scores_df.merge(
        expansion_scores_cost,
        on='zip5')

    expansion_scores_df = expansion_scores_df.merge(
        expansion_scores_tnt,
        on='zip5')

    expansion_scores_df = expansion_scores_df.merge(
        fc_carrier_switch,
        on='zip5',
        how='left')

    expansion_scores_df[
        'carrier_switch_package_count'] = expansion_scores_df[
            'carrier_switch_package_count'].fillna(0)

    expansion_scores_df[
        'fc_switch_package_per'] = expansion_scores_df[
            'fc_switch_package_per'].fillna(0)

    # assign priority score
    # cost change negative -> sim better / ontrgd better
    # tnt change negative -> sim better / ontrgd better
    # low priority score -> likely to expand

    # Normalize cost_change
    expansion_scores_df["normalized_cost_change"] = (
        (expansion_scores_df["cost_change"] - expansion_scores_df["cost_change"].min()) /
        (expansion_scores_df["cost_change"].max() - expansion_scores_df["cost_change"].min())
    )

    # Normalize tnt_change
    expansion_scores_df["normalized_tnt_change"] = (
        (expansion_scores_df["tnt_change"] - expansion_scores_df["tnt_change"].min()) /
        (expansion_scores_df["tnt_change"].max() - expansion_scores_df["tnt_change"].min())
    )

    w1, w2 = 0.7, 0.3
    expansion_scores_df["priority_score"] = (
        w1 * expansion_scores_df["normalized_cost_change"] +
        w2 * expansion_scores_df["normalized_tnt_change"]
    )

    # apply additional constraints
    expansion_scores_df = expansion_scores_df.loc[
         (expansion_scores_df['fc_switch_package_per'] <= fc_switch_th)]

    expansion_scores_df = expansion_scores_df.loc[
         (expansion_scores_df['active'] == 0)]

    expansion_scores_df = expansion_scores_df.sort_values('priority_score')
    expansion_scores_df = expansion_scores_df.reset_index(drop=True)

    expansion_scores_df[
        'carrier_switch_package_count_cumsum'] = expansion_scores_df[
            'carrier_switch_package_count'].cumsum()

    expansion_scores_df[
        'package_count_cumsum_per'] = expansion_scores_df[
             'package_count'].cumsum() / expansion_scores_df[
                  'package_count'].sum()

    return expansion_scores_df


def select_zips_and_get_simulation(
          simulation_df,
          zip_score_df,
          package_count):
    """
    Args:
        simulation_df: DataFrame of selected simulation.
        zip_score_df: DataFrame that contains priority score of each zip code.
        package_count: Package Count threshold to switch one carrier to another.

    Returns:
        pd.DataFrame: DataFrame of selected simulation for selected zips.
        pd.DataFrame: DataFrame of selected zips subject to package count.
    """

    zip_score_df = zip_score_df.loc[
            (zip_score_df['carrier_switch_package_count_cumsum'] <= package_count)]
    zip_score_df = zip_score_df[['zip5']].drop_duplicates()
    zip_score_df['selected'] = 1
    simulation_df = simulation_df.merge(zip_score_df, on='zip5')
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

    return simulation_df, zip_score_df


if __name__ == '__main__':

    # parameters
    with open("./configs.yaml") as f:
        config = yaml.safe_load(f)

    run_name = config['PLANNING']['run_name']

    expansion = config['PLANNING']['expansion']
    remediation = config['PLANNING']['remediation']

    baseline_scenario = config['PLANNING']['baseline_scenario']
    expansion_scenario = config['PLANNING']['expansion_scenario']
    remediation_scenario = config['PLANNING']['remediation_scenario']

    fc_switch_threshold = config['PLANNING']['fc_switch_threshold']
    ontrgd_target = config['PLANNING']['ontrgd_target']
    package_count_change_list = config['PLANNING']['package_count_change_list']

    lookback_day_count = config['PLANNING']['lookback_day_count']
    start_date = config['PLANNING']['start_date']
    end_date = config['PLANNING']['end_date']


    if start_date and end_date:
        pass
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_day_count)
        end_date = end_date.strftime('%Y-%m-%d')
        start_date = start_date.strftime('%Y-%m-%d')


    # load baseline simulation
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


    # get priority scores of all eligible zips for remediation
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

        remediation_scores = get_remediation_score(
             baseline_sim_df
             )



    # get priority scores of all eligible zips for expansion
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

        expansion_scores = get_expansion_score(
             baseline_sim_df,
             expand_sim_df,
             fc_switch_threshold,
             baseline_scenario,
             expansion_scenario,
             start_date,
             end_date
             )



    # select zip codes to remediate / expand based on target ONTRGD %

    # calculate package count change based on target
    if package_count_change_list is None:
        baseline_carrier_charge = baseline_sim_df.groupby(
            ['base_carrier_code']
            )['shipment_tracking_number'].nunique().reset_index()
        baseline_carrier_charge.columns = ['carrier','package_count']

        ontrgd_package_count = baseline_carrier_charge.loc[
            baseline_carrier_charge['carrier'] == 'ONTRGD',
            'package_count'].sum()
        ontrgd_target_package_count = (baseline_carrier_charge['package_count'].sum())*ontrgd_target

        target_package_count_change = int(
            np.round(
                abs(ontrgd_target_package_count - ontrgd_package_count)))
        logger.info('target package count change: %i', target_package_count_change)

        # package_count_change_list = range(
        #     target_package_count_change-int(np.round((target_package_count_change*1/4))),
        #     target_package_count_change+10000,
        #     10000
        # )

        package_count_change_list = [target_package_count_change]


    for pc in package_count_change_list:

        logger.info('grid package count change: %i',pc)
        baseline_sim_df_no_change = baseline_sim_df.copy()

        # select zips to remediate based on target - pc & pull sim result for that zips
        if remediation:
            print('ONTRGD remediation')
            remediate_sim_df_temp, remediate_zips_temp = select_zips_and_get_simulation(
                remediate_sim_df,
                remediation_scores,
                pc)
            print('remediated zip count: ', remediate_zips_temp.shape)

            baseline_sim_df_no_change = baseline_sim_df_no_change.merge(
                remediate_zips_temp,
                on='zip5',
                how='left')
            print(
                baseline_sim_df_no_change.groupby(
                    ['base_carrier_code','selected'])['shipment_tracking_number'].nunique()
                    )

            baseline_sim_df_no_change = baseline_sim_df_no_change.loc[
                baseline_sim_df_no_change['selected'].isnull()]
            baseline_sim_df_no_change = baseline_sim_df_no_change.drop('selected',axis=1)

        # select zips to expand based on target - pc & pull sim result for that zips
        if expansion:
            print('ONTRGD expansion')
            expand_sim_df_temp, expand_zips_temp = select_zips_and_get_simulation(
                expand_sim_df,
                expansion_scores,
                pc)
            print('expanded zip count: ', expand_zips_temp.shape)


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


        # calculate & print all metrics

        print('BEFORE WMS PROXY')
        #  before WMS proxy
        # check network level ONTRGD % (point estimate) before WMS proxy
        # rule 1
        carrier_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_carrier_code': 'sim_carrier_code'}, inplace=False),
            final_sim_df,
            ['sim_carrier_code'],
            'shipment_tracking_number',
            'nunique'
            )
        print(carrier_change)

        # check FC level ONTRGD % (point estimate) before WMS proxy
        # rule 2
        fc_carrier_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_carrier_code': 'sim_carrier_code',
                         'base_fc_name': 'sim_fc_name'
                         }, inplace=False),
            final_sim_df,
            ['sim_fc_name','sim_carrier_code'],
            'shipment_tracking_number',
            'nunique',
            ['sim_fc_name'])
        print(fc_carrier_change)

        print('AFTER WMS PROXY')
        # after WMS proxy
        baseline_sim_df_proxy = apply_wms_proxy(
            baseline_sim_df.rename(
                columns={'base_carrier_code': 'sim_carrier_code',
                         'base_fc_name': 'sim_fc_name'
                         }, inplace=False),
            'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_0925_v2;')
        final_sim_df_proxy = apply_wms_proxy(
            final_sim_df,
            'select * from edldb_dev.sc_promise_sandbox.sim_wms_proxy_0925_v2;')

        # check network level ONTRGD % (point estimate) after WMS proxy
        # rule 1
        carrier_change_proxy = calculate_package_distribution_change_by_groups(
            baseline_sim_df_proxy,
            final_sim_df_proxy,
            ['carrier'],
            'package_count',
            'sum'
            )
        print(carrier_change_proxy)

        # check FC level ONTRGD % (point estimate) after WMS proxy
        # rule 2
        fc_carrier_change_proxy = calculate_package_distribution_change_by_groups(
            baseline_sim_df_proxy,
            final_sim_df_proxy,
            ['fc','carrier'],
            'package_count',
            'sum',
            ['fc'])
        print(fc_carrier_change_proxy)


        print('FC CHARGE CHANGE')
        # check FC charge change
        # rule 3
        fc_charge_change = calculate_package_distribution_change_by_groups(
            baseline_sim_df.rename(
                columns={'base_fc_name': 'sim_fc_name'}, inplace=False),
            final_sim_df,
            ['sim_fc_name'],
            'shipment_tracking_number',
            'nunique')
        print(fc_charge_change)


        # save output
        s_d = baseline_sim_df['order_placed_date'].min()
        e_d = baseline_sim_df['order_placed_date'].max()
        r_target = np.round(carrier_change.loc[
            carrier_change['sim_carrier_code'] == 'ONTRGD',
            'iter_percent'].item(),1)

        if not os.path.exists('./results/planning' + f'/{run_name}'):
            os.makedirs('./results/planning' + f'/{run_name}')

        final_sim_df.to_parquet(
            os.path.join('./results/planning',
                        run_name,
                        f'simulation_dates_{s_d}_{e_d}_ontrgd_target_{ontrgd_target}_ontrgd_realized_{r_target}.parquet'
                        )
            )

        if expansion:
            expand_zips_temp.to_parquet(
                os.path.join('./results/planning',
                            run_name,
                            f'zips_to_expand_dates_{s_d}_{e_d}_ontrgd_target_{ontrgd_target}_ontrgd_realized_{r_target}.parquet'
                            )
            )
        if remediation:
            remediate_zips_temp.to_parquet(
                os.path.join('./results/planning',
                            run_name,
                            f'zips_to_remediate_dates_{s_d}_{e_d}_ontrgd_target_{ontrgd_target}_ontrgd_realized_{r_target}.parquet'
                            )
            )
