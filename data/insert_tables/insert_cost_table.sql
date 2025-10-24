INSERT INTO edldb_dev.sc_promise_sandbox.simulation_cost_estimation
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

