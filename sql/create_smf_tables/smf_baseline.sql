
create or replace temporary table edldb_dev.sc_promise_sandbox.simulation_smf as
with base as (

    select 
        ship_date as shipdate
        ,fc as fcname
        ,zip5
        ,mode
        ,routeid
        ,zone
        ,case when cutoff >= 240000 then LPAD(cast(cutoff - 240000 as varchar(10)),6,'0') else LPAD(cast(cutoff as varchar(10)),6,'0') end as cutoff
        ,tnt
        ,adjtnt
        ,nextadjtnt
        ,generate_date
        ,max(generate_date) over (partition by ship_date, fc, zip5, mode) as max_generate_date
    from 
        EDLDB.BT_SC_TRANSPORTATION.SHIPMAP
    where
        fc in ('AVP1','AVP2','MDT1','BNA1',
                'CLT1','MCO1','MCI1','DAY1',
                'CFC1','PHX1','RNO1','DFW1',
                'HOU1')
        and mode in ('FDXHD','ONTRGD')
        and orsitemtype = 'N'
        and ship_date >= date({start_date})
        and ship_date <= dateadd('day', 3, date({end_date}))

)

    select
        shipdate
        ,fcname
        ,zip5
        ,mode
        ,routeid
        ,zone
        ,concat(
            left(cast(cutoff as varchar(10)),2),
            ':',
            SUBSTRING(cast(cutoff as varchar(10)),3,2),
            ':',
            right(cast(cutoff as varchar(10)),2)
            ) as cutoff
        ,tnt
        ,adjtnt
        ,nextadjtnt
    from
        base
    where
        generate_date = max_generate_date
    ;


