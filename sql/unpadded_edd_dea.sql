select 
    unpadded_edd
    ,fc as ffmcenter_name
    ,carrier_code as carrier_code
    ,zip as zip5
    ,sum(package_total_unpadded) - sum(package_failures_unpadded_carrier_controllable) as unpadded_edd_dea_count
    ,sum(package_total_unpadded) as package_count
from
    edldb.sc_operations_sandbox.obp_package_failures_unpadded_summary
where 
    fc in ('AVP1','AVP2','CFC1','CLT1','DAY1','DFW1','MCI1','MCO1','MDT1','PHX1','RNO1','BNA1','HOU1')
    and unpadded_edd >= date({start_date})
    and unpadded_edd <= date({end_date})
group by
    1,2,3,4
;

/*
with base as (
    select distinct --st.order_id,
           st.shipment_tracking_number,
           st.ffmcenter_name,
           st.carrier_code,
           st.shipment_ship_route,
           st.postcode,
           -- st.order_placed_dttm_est,
           smf.cutoff,
           smf.pulltime,
           convert_timezone('UTC', 'America/New_York', to_timestamp_ntz(tpc.actual_ship_date)) as actual_ship_dttm_est,
           -- st.shipment_shipped_dttm_est as actual_ship_dttm_est,
           coalesce(st.initial_delivery_attempt_dttm_est, st.bulk_track_delivery_dttm_est) as delivery_dttm_est,
           st.shipment_estimated_delivery_date as padded_edd
    from edldb.chewybi.shipment_transactions st
    join edldb.aad.t_pick_container tpc
        on tpc.tracking_number = st.shipment_tracking_number
    join edldb.sc_promise_sandbox.ship_map_file smf
        on smf.zip5 = st.postcode
        and smf.fcname = st.ffmcenter_name
        and smf.routeid = st.shipment_ship_route
        and smf.shipdate = DATE(st.release_dttm_est)
    where 
        -- tpc.arrive_date >= date($start_date)
        -- and tpc.arrive_date <= date($end_date)
        st.ffmcenter_name in ('AVP1','AVP2','CFC1','CLT1',
                              'DAY1','DFW1','MCI1',
                              'MCO1','MDT1','PHX1','RNO1',
                              'BNA1','HOU1')
        and smf.fcname in ('AVP1','AVP2','CFC1','CLT1',
                              'DAY1','DFW1','MCI1',
                              'MCO1','MDT1','PHX1','RNO1',
                              'BNA1','HOU1')
        and date(st.initial_delivery_attempt_dttm_est) >= date({start_date})
        and date(st.initial_delivery_attempt_dttm_est) <= date({end_date})

)
,metadata as (
    select *
            , case when (cutoff::time < pulltime::time
                    and actual_ship_dttm_est::time <= cutoff::time) then 'regular - before cutoff'
                    
                   when (cutoff::time < pulltime::time
                    and actual_ship_dttm_est::time > cutoff::time
                    and actual_ship_dttm_est::time <= pulltime::time) then 'regular - cutoff to pulltime'
    
                   when (cutoff::time < pulltime::time
                    and actual_ship_dttm_est::time > pulltime::time) then 'regular - after pulltime'
    
    
                  when (cutoff::time > pulltime::time
                    and actual_ship_dttm_est::time > pulltime::time
                    and actual_ship_dttm_est::time <= cutoff::time) then 'midnight - before cutoff'
    
                  when (cutoff::time > pulltime::time
                    and actual_ship_dttm_est::time >= cutoff::time
                    and actual_ship_dttm_est::time > pulltime::time) then 'midnight - cutoff to pulltime - cutoff date'
                
                  when (cutoff::time > pulltime::time
                    and actual_ship_dttm_est::time < cutoff::time
                    and actual_ship_dttm_est::time <= pulltime::time) then 'midnight - cutoff to pulltime - cpt date'
                    
            end as ship_tag
    from base
)
,metadata2 as (
    select *,
           case when ship_tag in ('regular - before cutoff','regular - cutoff to pulltime')
                then date(actual_ship_dttm_est)
                when ship_tag in ('regular - after pulltime')
                then dateadd(day,1,date(actual_ship_dttm_est))
                when ship_tag in ('midnight - before cutoff')
                then date(actual_ship_dttm_est)
                when ship_tag in ('midnight - cutoff to pulltime - cutoff date')
                then date(actual_ship_dttm_est)
                when ship_tag in ('midnight - cutoff to pulltime - cpt date')
                then dateadd(day,-1,date(actual_ship_dttm_est))
                end as ship_cutoffdate
            ,
           case when ship_tag in ('regular - before cutoff','regular - cutoff to pulltime')
                then date(actual_ship_dttm_est)
                when ship_tag in ('regular - after pulltime')
                then dateadd(day,1,date(actual_ship_dttm_est))
                when ship_tag in ('midnight - before cutoff')
                then dateadd(day,1,date(actual_ship_dttm_est))
                when ship_tag in ('midnight - cutoff to pulltime - cutoff date')
                then dateadd(day,1,date(actual_ship_dttm_est))
                when ship_tag in ('midnight - cutoff to pulltime - cpt date')
                then date(actual_ship_dttm_est)
                end as ship_cptdate
    from metadata
)
,granular as (
    select --m.order_id,
           m.shipment_tracking_number,
           m.ffmcenter_name,
           m.carrier_code,
           --m.shipment_ship_route,
           m.postcode as zip5,
           --m.actual_ship_dttm_est,
           --m.ship_tag,
           --m.padded_edd,
           dateadd(day,adjtnt,ship_cutoffdate) as unpadded_edd,
           date(delivery_dttm_est) as delivery_date,
           case when unpadded_edd <= delivery_date
            then 1 else 0 end as unpadded_edd_dea
    from metadata2 m
    join edldb.sc_promise_sandbox.ship_map_file smf
        on m.ffmcenter_name = smf.fcname
        and m.shipment_ship_route = smf.routeid
        and m.postcode = smf.zip5
        and m.ship_cutoffdate = smf.shipdate

)

select delivery_date,
       ffmcenter_name,
       carrier_code,
       zip5,
       count(distinct shipment_tracking_number) as package_count,
       sum(unpadded_edd_dea) as unpadded_edd_dea_count
from granular
group by delivery_date,ffmcenter_name,carrier_code,zip5
;
*/