create or replace temporary table edldb_dev.sc_promise_sandbox.simulation_inventory as
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



create or replace temporary table edldb_dev.sc_promise_sandbox.simulation_cost_estimation as
with base as (
    select distinct
        i.*
        ,substr(i.zip5,1,3)::varchar(10) as zip3
        ,smf.mode
        ,smf.zone
        ,st.warehouse_expected_length as length
        ,st.warehouse_expected_width as width
        ,st.warehouse_expected_height as height
        ,st.warehouse_expected_weight as weight 
        ,date(actual_ship_date) as ship_date
        ,date_trunc('week', actual_ship_date) as ship_week
    from 
        edldb_dev.sc_promise_sandbox.simulation_inventory i
    join 
        (select distinct shipdate, fcname, zip5, mode, zone from edldb_dev.sc_promise_sandbox.simulation_smf) smf
        on i.zip5 = smf.zip5
        and i.fc_name = smf.fcname
        and date(i.order_dttm) = smf.shipdate
    join
        edldb.chewybi.shipment_transactions st
        on i.order_id = st.order_id
        and i.shipment_tracking_number = st.shipment_tracking_number
    where
        date(st.order_placed_dttm_est) >= date({start_date})
        and date(st.order_placed_dttm_est) <= date({end_date})
)

,calculate_dims as (

    select 
        b.*
        ,greatest(length,width,height) as longest_side
       ,least(length,width,height) as third_longest_side
       ,case 
            when (greatest(length,width,height) = length and least(length,width,height) = width) 
                or (greatest(length,width,height) = width and least(length,width,height) = length) then height
            when (greatest(length,width,height) = width and least(length,width,height) = height) 
                or (greatest(length,width,height) = height and least(length,width,height) = width) then length
            when (greatest(length,width,height) = height and least(length,width,height) = length) 
                or (greatest(length,width,height) = length and least(length,width,height) = height) then width
        end as second_longest_side
       ,greatest(length,width,height) + 2*(
                                case 
                                    when (greatest(length,width,height) = length and least(length,width,height) = width) 
                                        or (greatest(length,width,height) = width and least(length,width,height) = length) then height
                                    when (greatest(length,width,height) = width and least(length,width,height) = height) 
                                        or (greatest(length,width,height) = height and least(length,width,height) = width) then length
                                    when (greatest(length,width,height) = height and least(length,width,height) = length) 
                                    or (greatest(length,width,height) = length and least(length,width,height) = height) then width
                                end + least(length,width,height)
                                ) 
        as length_plus_girth
       ,weight as actual_weight
       ,ceil((length*width*height)/166) as dim_weight
       ,ceil(greatest(weight,(length*width*height)/166)) as bill_weight
       ,case 
            when greatest(length, width, height) between 96.001 and 108 
                or (    greatest(length, width, height) +2*(least(length, width, height) + (
                        case 
                            when (greatest(length, width, height) = length and least(length, width, height) = height) 
                                or (greatest(length, width, height) = height and least(length, width, height) = length) then width  
                            when (greatest(length, width, height) = width and least(length, width, height) = height) 
                                or (greatest(length, width, height) = height and least(length, width, height) = width) then length 
                            when (greatest(length, width, height)= length and least(length, width, height) = width) 
                                or (greatest(length, width, height) = width and least(length, width, height) = length) then height 
                         end)
                         )
                    ) between 130.001 and 165 
             then  
                   (case 
                        when ceil(greatest(weight,(length*width*height)/166)) > 90 then ceil(greatest(weight,(length*width*height)/166)) 
                        else 90 end)
             when greatest(length, width, height) between 48.001 and 96
             or (   case 
                        when (greatest(length, width, height) = length and least(length, width, height) = height) 
                            or (greatest(length, width, height) = height and least(length, width, height) = length) then width  
                        when (greatest(length, width, height) = width and least(length, width, height) = height) 
                            or (greatest(length, width, height) = height and least(length, width, height) = width) then length 
                        when (greatest(length, width, height)= length and least(length, width, height) = width) 
                            or (greatest(length, width, height) = width and least(length, width, height) = length) then height 
                 end
                 ) > 30 
             or (greatest(length, width, height) +2*(
                    least(length, width, height) + (
                        case 
                            when (greatest(length, width, height) = length and least(length, width, height) = height) 
                                or (greatest(length, width, height) = height and least(length, width, height) = length) then width  
                            when (greatest(length, width, height) = width and least(length, width, height) = height) 
                                or (greatest(length, width, height) = height and least(length, width, height) = width) then length 
                            when (greatest(length, width, height)= length and least(length, width, height) = width) 
                                or (greatest(length, width, height) = width and least(length, width, height) = length) then height 
                                                                                     end
                                                                                     )
                                                                                     )
                  ) between 105.001 and 130 
             then case 
                     when ship_date >= '2025-01-13' then
                           (case 
                                when ceil(greatest(weight,(length*width*height)/166)) > 40 then ceil(greatest(weight,(length*width*height)/166)) 
                                else 40 
                           end)
                     else ceil(greatest(weight,(length*width*height)/166))
                 end    
             else ceil(greatest(weight,(length*width*height)/166))  
       end as bill_weight_oversize_and_adh_dims_included
    from 
        base b
    
)

, all_amount as (

select 
    b.*
    ,br.base_charge as base_rate -- considering the bill_weight_oversize_and_adh_dims_included rule
    ,zeroifnull(r.resi_amount) as resi_amount
    ,zeroifnull(das.amount) as das_amount
    ,0.01::numeric(3,2) as address_correction_amount -- No way to estimate this, adding average address correction CPP of 0.02 per package
    ,zeroifnull(ls.amount) as handling_longest_side_amt
    ,ls.charge_type as ls_charge_type 
    ,zeroifnull(sls.amount) as handling_second_longest_side_amt     
    ,sls.charge_type as sls_charge_type
    ,zeroifnull(lg.amount) as handling_length_plus_girth_amt
    ,lg.charge_type as lg_charge_type 
    ,zeroifnull(weight_thresholds.amount) as handling_weight_amt
    ,weight_thresholds.charge_type as aw_charge_type
    ,zeroifnull(pc.peak_surcharge) as peak_surcharge

    ,case 
        when ls_charge_type = 'unauthorized_oversize' 
            or lg_charge_type = 'unauthorized_oversize' 
            or aw_charge_type = 'unauthorized_oversize' 
        then 1 else 0 end as Uov_flag_temp
    ,case 
        when ls_charge_type = 'oversize' 
            or  lg_charge_type = 'oversize' 
            then 1 else 0 end as Oversize_flag_temp 
    ,case 
        when ls_charge_type = 'additional_handling' 
            or  lg_charge_type = 'additional_handling' 
            or sls_charge_type = 'additional_handling' 
            or  aw_charge_type = 'additional_handling' 
        then 1 else 0 end as add_handling_flag_temp
       ,zeroifnull(
                greatest(handling_longest_side_amt, 
                        handling_second_longest_side_amt, 
                        handling_length_plus_girth_amt, 
                        handling_weight_amt)
                        ) as handling_charge

        ,case 
            when Uov_flag_temp = 1 then handling_charge 
            else 0 end as uov_amount
        ,case 
            when Uov_flag_temp = 0 
                and Oversize_flag_temp = 1 then handling_charge 
            else 0 end as oversize_amount
        ,case 
            when Uov_flag_temp = 0 
                and Oversize_flag_temp = 0 
                and  add_handling_flag_temp = 1 then handling_charge 
            else 0 end as add_handling_amount
            
        ,fsc.fsc_per  
from 
    calculate_dims b  
left join 
    sc_transportation_sandbox.split_packages br -- Base Rates table
     on  b.ship_date between br.start_date and br.end_date
     and b.mode = br.carrier_code
     and b.zone = br.zone
     and b.bill_weight_oversize_and_adh_dims_included = br.bill_weight 

left join 
    sc_transportation_sandbox.resi_amounts r -- Resi Table
    on b.ship_date between r.start_date and r.end_date
    and b.mode = r.carrier_code

left join 
     sc_transportation_sandbox.das_amounts das -- Das Table
     on b.ship_date between das.start_date and das.end_date 
     and b.mode = das.carrier_code
     and b.zip5 = das.zipcode

left join 
     sc_transportation_sandbox.package_weight weight_thresholds -- Weight Table
     on  b.ship_date between weight_thresholds.start_date and weight_thresholds.end_date
     and b.mode = weight_thresholds.carrier_code
     and b.zone = weight_thresholds.zone
     and b.weight > weight_thresholds.weight_lower and  b.weight<= weight_thresholds.weight_upper
     
left join 
     sc_transportation_sandbox.package_longest_side ls -- Longest Side Table
     on  b.ship_date between ls.start_date and ls.end_date
     and b.mode = ls.carrier_code
     and b.zone = ls.zone
     and b.longest_side > ls.longest_side_lower 
     and  b.longest_side<= ls.longest_side_upper
     
left join 
     sc_transportation_sandbox.package_second_longest_side sls -- Second Longest Side Table
     on  b.ship_date between sls.start_date and sls.end_date
     and b.mode = sls.carrier_code
     and b.zone = sls.zone
     and b.second_longest_side > sls.second_longest_side_lower 
     and  b.second_longest_side<= sls.second_longest_side_upper
     
left join  
     sc_transportation_sandbox.package_length_plus_girth lg -- Length+Girth
     on  b.ship_date between lg.start_date and lg.end_date
     and b.mode = lg.carrier_code
     and b.zone = lg.zone
     and b.length_plus_girth > lg.length_plus_girth_lower 
     and  b.length_plus_girth <= lg.length_plus_girth_upper 
     
left join  
     sc_transportation_sandbox.shipment_peak_charges pc
     on b.ship_week = peak_surcharge_application_week
     and b.mode = pc.shipmode

left join 
    sc_transportation_sandbox.ob_carrier_fsc fsc
    on b.mode = fsc.carrier_code 
    and b.ship_date between fsc.start_dt and fsc.end_dt


)

select distinct
        order_id
        ,shipment_tracking_number
        ,fc_name
        ,mode
        ,zone
        ,base_rate
        + resi_amount
        + das_amount
        + address_correction_amount
        + uov_amount
        + oversize_amount
        + add_handling_amount
        + peak_surcharge 
        + ( base_rate
            + resi_amount
            + das_amount
            + uov_amount
            + oversize_amount
            + add_handling_amount
            + peak_surcharge
            + address_correction_amount
            )*fsc_per as total_amount  
            
from 
    all_amount

;


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
