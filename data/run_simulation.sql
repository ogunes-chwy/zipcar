with base as (

        select inv.order_id,
               inv.shipment_tracking_number,
               inv.fc_name,
               inv.zip5,
               inv.order_dttm,
               st.release_dttm_est as estimated_release_dttm
                        
        from  
            edldb_dev.sc_promise_sandbox.simulation_inventory inv
        join  
            edldb.chewybi.shipment_transactions st
            on st.order_id = inv.order_id
            and st.shipment_tracking_number = inv.shipment_tracking_number
    )
    
    ,metadata1 as(
    
        select base.order_id,
                base.shipment_tracking_number,
                base.fc_name,
                base.order_dttm,
                base.zip5,

                smf.routeid,
                smf.mode,
                smf.zone,
                
                case when timestamp_ntz_from_parts(order_dttm::date,smf.cutoff::time) >= base.estimated_release_dttm
                    then smf.adjtnt else smf.nextadjtnt end as tnt_selected
   
            from 
                base
            join 
                edldb_dev.sc_promise_sandbox.simulation_smf smf
                on smf.zip5 = base.zip5
                and smf.fcname = base.fc_name
                and smf.shipdate = DATE(base.order_dttm) 
    )

    ,top_fc_metadata as (
    
        select m.order_id,
               m.shipment_tracking_number,
               order_dttm,
               zip5,

               m.fc_name,
               routeid,

               c.total_amount,
               m.zone,
               case when m.fc_name in ('AVP2','MCI1','RNO1','BNA1','HOU1') 
                 then 1 else 0 
                 end as fc_type,
               CASE 
                    WHEN tnt_selected = 1 THEN 2
                    ELSE tnt_selected END AS tnt_selected_2,
               tnt_selected,

               case when tnt_selected <= 2 then 2.1
                    when tnt_selected = 3 then 2.4
                    when tnt_selected = 4 then 2.8
                    when tnt_selected = 5 then 3.6
                    when tnt_selected = 6 then 35.1
                    when tnt_selected = 7 then 36.1
                    when tnt_selected = 8 then 37.1
                    when tnt_selected >= 9 then 38.1
                    end as tnt_penalty,
               -- tnt_costs: {1:2.1, 2:2.1, 3:2.4, 4:2.8, 5:3.6, 6:35.1, 7:36.1, 8:37.1, 9:38.1}

               coalesce(geo.dist_in_miles_float,10000) as distance_mi,
               m.mode as carrier_code
                     
        from 
            metadata1 m
        left join 
            edldb.sc_promise_sandbox.fc_zip_distance geo
            on m.zip5 = geo.postcode    
            and m.fc_name = geo.fc_name  
        left join
            edldb_dev.sc_promise_sandbox.simulation_cost_estimation c
            on m.order_id = c.order_id
            and m.shipment_tracking_number = c.shipment_tracking_number
            and m.fc_name = c.fc_name
            and m.mode = c.mode
        
    )

    ,top_fc as (
        select order_id,
               shipment_tracking_number,
               order_dttm,
               m.zip5,
               
               m.fc_name,
               m.routeid,
               m.zone,
               m.carrier_code,

               m.total_amount,
               tnt_penalty,
               tnt_selected_2,
               fc_type,
               tnt_selected,
               distance_mi, 
               
               
               ROW_NUMBER() OVER (
                        PARTITION BY order_id, shipment_tracking_number 
                        ORDER BY m.total_amount, -fc_type, tnt_selected_2, -tnt_selected, distance_mi, m.carrier_code
                                 ) AS row_n           
        from 
            top_fc_metadata m
  
    ),

    cost_neutral_md as (
    
        select distinct 
                order_id,
                shipment_tracking_number,
                fc_name,
                carrier_code,
                total_amount+tnt_penalty as cost,
                max(
                    case when row_n=1 then total_amount+tnt_penalty else 0 end) 
                    over (
                        partition by order_id,shipment_tracking_number) as selected_cost 
        from top_fc

    )
    
    , cost_neutral as (
        select distinct
            order_id,
            shipment_tracking_number,
            fc_name,
            carrier_code,
            ARRAY_AGG(distinct fc_name) 
                within group (order by fc_name) 
                over (partition by order_id, shipment_tracking_number) as cost_neutral_fc
        from 
            cost_neutral_md
        where 
            cost = selected_cost  
    )

        select distinct top_fc.order_id,
                        top_fc.shipment_tracking_number,
                        top_fc.zip5,
                        date(top_fc.order_dttm) as order_placed_date,

                        top_fc.fc_name as sim_fc_name,
                        top_fc.carrier_code as sim_carrier_code,
                        top_fc.routeid as sim_route,
                        top_fc.total_amount as sim_transit_cost,
                        top_fc.tnt_penalty as sim_tnt_penalty,
                        top_fc.zone as sim_zone,
                        top_fc.tnt_selected as sim_tnt,

                        cn.cost_neutral_fc,
                        
                        -- actual records / ORS decisions
                        st.ffmcenter_name as act_fc_name,
                        st.carrier_code as act_carrier_code,
                        st.shipment_ship_route as act_route,
                        c.total_amount as act_transit_cost,
                        date(st.actual_ship_date) as ship_date,
                        
                        --st.shipment_tracking_number,
                        st.shipment_count_of_items_in_box as units,

                        datediff(day, st.order_placed_dttm_est , st.initial_delivery_attempt_dttm_est) as ctd,
                        datediff(day, st.actual_ship_date , st.initial_delivery_attempt_dttm_est) as std,
                        case 
                            when date(st.shipment_estimated_delivery_date) >= date(st.initial_delivery_attempt_dttm_est)
                            then 1 
                            when st.initial_delivery_attempt_dttm_est is null
                            then null 
                            else 0 end as dea_flag                                         
        from 
            top_fc 
        join 
            edldb.chewybi.shipment_transactions st
            on top_fc.order_id = st.order_id
            and top_fc.shipment_tracking_number = st.shipment_tracking_number 
        left join
            edldb_dev.sc_promise_sandbox.simulation_cost_estimation c
            on top_fc.order_id = c.order_id
            and top_fc.shipment_tracking_number = c.shipment_tracking_number
            and st.ffmcenter_name = c.fc_name
            and st.carrier_code = c.mode
        left join
            cost_neutral cn
            on cn.order_id = top_fc.order_id
            and cn.shipment_tracking_number = top_fc.shipment_tracking_number
        where 
            top_fc.row_n = 1
            ;
