-- baseline OR remediation


INSERT INTO edldb_dev.sc_promise_sandbox.simulation_smf
select
        shipdate,
        fcname,
        zip5,
        mode,
        routeid,
        zone,
        cutoff,
        tnt,
        adjtnt,
        nextadjtnt
from 
    edldb.sc_promise_sandbox.ship_map_file
where 
    mode in ('FDXHD','ONTRGD') -- 'FDXHD','ONTRGD'
    and fcname in ('AVP1','AVP2','CFC1','CLT1',
                       'DAY1','DFW1','MCI1',
                       'MCO1','MDT1','PHX1','RNO1',
                       'BNA1','HOU1')
    and shipdate >= {start_date}
    and shipdate <= {end_date};


/*
-- modified SMF due to expansion etc. 


INSERT INTO edldb_dev.sc_promise_sandbox.simulation_smf
with base as (

    select
        shipdate,
        fcname,
        zip5,
        mode,
        routeid,
        zone,
        cutoff,
        tnt,
        adjtnt,
        nextadjtnt
from 
    edldb.sc_promise_sandbox.ship_map_file smf
where 
    smf.mode in ('FDXHD','ONTRGD') 
    and smf.fcname in ('AVP1','AVP2','CFC1','CLT1',
                       'DAY1','DFW1','MCI1',
                       'MCO1','MDT1','PHX1','RNO1',
                       'BNA1','HOU1')
    and shipdate >= {start_date}
    and shipdate < {end_date}
    
)

, ontrgd_selected as (

    select distinct
        smf.shipdate
        ,smf.fcname
        ,smf.zip5
        ,smf.mode
        ,smf.routeid
        ,smf.zone
        ,concat(
            left(cast(smf.cutoff as varchar(10)),2),
            ':',
            SUBSTRING(cast(smf.cutoff as varchar(10)), 3, 2),
            ':',
            right(cast(smf.cutoff as varchar(10)),2)
            ) as cutoff
        ,smf.tnt
        ,smf.adjtnt
        ,smf.nextadjtnt
    from
        edldb_dev.sc_promise_sandbox.ontrgd_eligible_zip_smf_MMruleApplied_0831_1010 smf 
    --join
        --edldb_dev.sc_promise_sandbox.selected_zip_test_v2 sz -- selected zips for expansion
        --on smf.zip5 = sz.zip5
    where 
        smf.fcname != 'HOU1'
        and shipdate >= {start_date}
        and shipdate < {end_date}
)

    select distinct
        *
    from (
        (select 
            * 
        from 
            base)
        UNION
        (select 
            *
        from
            ontrgd_selected)
    )
    ;
*/