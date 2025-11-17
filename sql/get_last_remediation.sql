with remediation as (

    select 
        *,
        row_number() over (partition by zip5 order by run_dttm desc) as rn
    from 
        EDLDB.SC_PROMISE_SANDBOX.ZIPCAR_REMEDIATION_ZIPS
    where
        date(run_dttm) >= date({start_date})
        and date(run_dttm) < date({end_date})
        -- and run_name = 'default'
        and zip_count_tag = 'all'
)

    select 
        *
    from   
        remediation
    where
        rn = 1
;