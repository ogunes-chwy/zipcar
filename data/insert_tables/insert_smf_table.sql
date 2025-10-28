-- baseline OR remediation

-- old way of creating SMF

/*
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
*/


-- new way of creating SMF

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
        BT_SC_TRANSPORTATION.SHIPMAP
    where
        fc in ('AVP1','AVP2','MDT1','BNA1',
                'CLT1','MCO1','MCI1','DAY1',
                'CFC1','PHX1','RNO1','DFW1',
                'HOU1')
        and mode in ('FDXHD','ONTRGD')
        and orsitemtype = 'N'
        and ship_date >= {start_date}
        and ship_date <= {end_date}

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




-- modified SMF due to expansion etc. 

/*
INSERT INTO edldb_dev.sc_promise_sandbox.simulation_smf
with base as (

    select 
        ship_date as shipdate
        ,fc as fcname
        ,zip5
        ,mode
        ,routeid
        ,zone
        ,case when cutoff >= 240000 then LPAD(cutoff - 240000,6,'0') else LPAD(cutoff,6,'0') end as cutoff
        ,tnt
        ,adjtnt
        ,nextadjtnt
        ,generate_date
        ,max(generate_date) over (partition by ship_date, fc, zip5, mode) as max_generate_date
    from 
        BT_SC_TRANSPORTATION.SHIPMAP
    where
        fc in ('AVP1','AVP2','MDT1','BNA1',
                'CLT1','MCO1','MCI1','DAY1',
                'CFC1','PHX1','RNO1','DFW1',
                'HOU1')
        and mode in ('FDXHD','ONTRGD')
        and orsitemtype = 'N'
        and ship_date >= {start_date}
        and ship_date <= {end_date}

)

, current_smf as (
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
)

, ontrgd_eligible_smf as (

    select distinct
        shipdate
        ,fcname
        ,zip5
        ,mode
        ,routeid
        ,zone
        ,concat(
            left(cast(cutoff as varchar(10)),2),
            ':',
            SUBSTRING(cast(cutoff as varchar(10)), 3, 2),
            ':',
            right(cast(cutoff as varchar(10)),2)
            ) as cutoff
        ,tnt
        ,adjtnt
        ,nextadjtnt
    from
        edldb_dev.sc_promise_sandbox.ontrgd_eligible_zip_smf_MMruleApplied_0831_1010 smf 
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
            current_smf)
        UNION
        (select 
            *
        from
            ontrgd_eligible_smf)
    )
    */
    ;
