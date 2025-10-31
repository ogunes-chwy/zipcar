with base as (

    select 
        b.run_date
        ,b.dest_facility_no
        ,b.days_behind
        ,b.days_to_current
        ,b.estimated_clear_date
        ,b.carrier
        ,b.latest_station_status
    from 
        edldb.sc_sandbox.estimated_clear_date_data b
    where
        run_date >= date({start_date})
        and run_date <= date({end_date})
        
)

,zip_station_mapping as (

    select 
            zipcode,
            master_term,
            snapshot_date,
            max(snapshot_date) over (
                partition by zipcode, master_term) as max_snapshot_date
    from
        edldb.srm.srm_fdx_ref_cov_base 
        
)

select 
    b.run_date as date
    ,b.dest_facility_no
    ,sz.zipcode as zip5
    ,b.days_behind
    ,b.days_to_current
    ,b.estimated_clear_date
    ,b.latest_station_status
    ,COALESCE(b.carrier, 'FDXHD') as carrier_code
from 
    base b
join 
    (select * from zip_station_mapping where snapshot_date = max_snapshot_date) sz
    on b.dest_facility_no = sz.master_term
;