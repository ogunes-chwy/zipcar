with base as (

    select 
        run_date
        ,carrier
        ,dest_facility_num
        ,days_behind
        ,row_created_date
        ,row_number() over (partition by run_date, carrier, dest_facility_num order by row_created_date desc) as rn
    from 
        edldb.sc_transportation_sandbox.cndimpactedstations
    where
        carrier is not null
        and date(run_date) >= date({start_date})
        and date(run_date) <= date({end_date})
        
)

,zip_station_mapping_fdx as (

    select distinct
            'FedEx' as carrier,
            LPAD(zipcode, 5, '0') as zip5,
            master_term as dest_facility_num,
            snapshot_date,
            row_number() over (partition by zipcode order by snapshot_date desc) as rn_fdx
            --max(snapshot_date) over (
            --    partition by zipcode) as max_snapshot_date
    from
        edldb.srm.srm_fdx_ref_cov_base 
        
)

,zip_station_mapping_ontrgd as (

    select distinct
            'OnTrac' as carrier,
            LPAD(customer_zip, 5, '0') as zip5,
            dest_facility_num as dest_facility_num
    from
        edldb.sc_operations_sandbox.obp_ontrac_mapping_station_zip
        
)

select distinct
    date(b.run_date) as date
    ,b.dest_facility_num
    ,COALESCE(fdx.zip5, ontrgd.zip5) as zip5
    ,b.days_behind
    ,CASE 
        when b.carrier = 'FedEx' then 'FDXHD' 
        when b.carrier = 'OnTrac' then 'ONTRGD' 
        ELSE NULL 
    END as carrier_code
from 
    base b
left join 
    (select * from zip_station_mapping_fdx where rn_fdx = 1) fdx
    on b.dest_facility_num = fdx.dest_facility_num
    and b.carrier = fdx.carrier
left join 
    (select * from zip_station_mapping_ontrgd) ontrgd
    on b.dest_facility_num = ontrgd.dest_facility_num
    and b.carrier = ontrgd.carrier
where 
    b.rn = 1
;


/*
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
*/