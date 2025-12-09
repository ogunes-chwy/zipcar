/* CUSTOM SMF TABLE */
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
        ,max(generate_date) over (partition by ship_date, fc, zip5) as max_generate_date
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
        --and date(generate_date) < date('2025-12-05')

)

, current_smf as (
    select
        base.shipdate
        ,base.fcname
        ,base.zip5
        ,base.mode
        ,base.routeid
        ,base.zone
        ,concat(
            left(cast(cutoff as varchar(10)),2),
            ':',
            SUBSTRING(cast(cutoff as varchar(10)),3,2),
            ':',
            right(cast(cutoff as varchar(10)),2)
            ) as cutoff
        ,base.tnt
        ,base.adjtnt
        ,base.nextadjtnt
    from
        base
    left join
        (
            select distinct zip5, 'ONTRGD' as mode,'deactivate' as final_recommendation
            from EDLDB_DEV.SC_PROMISE_SANDBOX.ontrgd_first_pass_removal_adhoc_20251201 
            ) rem
            on base.zip5 = LPAD(rem.zip5, 5, '0')
            and base.mode = rem.mode
     -- join with custom zips here to remediate
    /*left join
        (
            select distinct zip, 'ONTRGD' as mode,'deactivate' as final_recommendation
            from EDLDB_DEV.SC_PROMISE_SANDBOX.ONTRGD_ZIP_REMOVAL_ALLFC_ADHOC_20251204 
            ) allfc_rem
            on base.zip5 = LPAD(allfc_rem.zip, 5, '0')
            and base.mode = allfc_rem.mode
    left join
        (
            select distinct customer_zip, fc, 'ONTRGD' as mode,'deactivate' as final_recommendation
            from EDLDB_DEV.SC_PROMISE_SANDBOX.ONTRGD_ZIP_REMOVAL_BYFC_ADHOC_20251204 
            ) byfc_rem
            on base.zip5 = LPAD(byfc_rem.customer_zip, 5, '0')
            and base.mode = byfc_rem.mode
            and base.fcname = byfc_rem.fc */
    where
        generate_date = max_generate_date
        and rem.final_recommendation is null
        --and allfc_rem.final_recommendation is null
        --and byfc_rem.final_recommendation is null
)

    select distinct
        *
    from current_smf
    
    ;


