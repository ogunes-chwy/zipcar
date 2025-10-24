INSERT INTO edldb_dev.sc_promise_sandbox.simulation_inventory
with scope as (

    select distinct
        st.order_id
        ,postcode as zip5
        ,order_placed_dttm_est as order_dttm
        ,st.shipment_tracking_number
        ,order_line_id
        ,order_line_quantity
    from 
        edldb.chewybi.shipment_transactions st
    join
        edldb.chewybi.order_line_cost_measures olcm
    on 
        st.order_id = olcm.order_order_id
        and st.shipment_tracking_number = olcm.shipment_tracking_number
    where 
        st.ffmcenter_name in ('AVP1','AVP2','CFC1','CLT1',
                              'DAY1','DFW1','MCI1',
                              'MCO1','MDT1','PHX1','RNO1',
                              'BNA1','HOU1')
        and date(st.order_placed_dttm_est) >= date({start_date})
        and date(st.order_placed_dttm_est) <= date({end_date})
        and date(olcm.order_placed_date) >= date({start_date})
        and date(olcm.order_placed_date) <= date({end_date})
), 

base as (

  select distinct scope.order_id,
                  shipment_tracking_number,
                  scope.order_line_id,
                  zip5,
                  order_dttm,
                  inv.fc_name,
                  inv.quantity_available,
    from 
        scope
    join
        edldb.chewybi.order_routing_inventory_level inv
        on  scope.order_id = inv.order_id
        and scope.order_line_id = inv.order_line_id
    where 
            quantity_available >= order_line_quantity         
)

,base2 as(
    select order_id,
           order_dttm,
           zip5,
           shipment_tracking_number,
           order_line_id,
           fc_name,
           count(distinct order_line_id) OVER (PARTITION BY order_id, shipment_tracking_number, fc_name) AS part_c_by_fc,
           count(distinct order_line_id) OVER (PARTITION BY order_id, shipment_tracking_number) AS part_c
    from base
)

    select distinct
        order_id,
        order_dttm,
        zip5,
        shipment_tracking_number,
        fc_name,
    from 
        base2
    where 
        part_c_by_fc = part_c
;


