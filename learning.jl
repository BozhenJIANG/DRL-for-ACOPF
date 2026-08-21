module Learning

import PowerModels
import Ipopt
import Statistics
import Optim

# balance_ref_id = 15
# gen_num = 33
# bus_num = 24

function _update_para(_balance_ref_id,_gen_num,_bus_num)
    global balance_ref_id, gen_num, bus_num

    balance_ref_id = _balance_ref_id
    gen_num = _gen_num
    bus_num = _bus_num  
end


function _python_data_to_PMs_data(python_data)

    case = Dict{String,Any}()

    if haskey(python_data, "name")
        case["name"] = python_data["name"]
    end

    if haskey(python_data, "source_version")
        case["source_version"] = python_data["source_version"]
    else
        Memento.warn(_LOGGER, string("no case version found in matpower file.  The file seems to be missing \"mpc.version = ...\""))
        case["source_version"] = "0.0.0+"
    end

    if haskey(python_data, "baseMVA")
        case["baseMVA"] = python_data["baseMVA"]
    else
        Memento.warn(_LOGGER, string("no baseMVA found in matpower file.  The file seems to be missing \"mpc.baseMVA = ...\""))
        case["baseMVA"] = 1.0
    end


    if haskey(python_data, "bus")
        buses = []
        for bus_row in python_data["bus"]
            _bus_data_row = Dict{String,Any}()
            if haskey(bus_row,"source_id")
                _bus_data_row["source_id"] = bus_row["source_id"]
            end
            if haskey(bus_row,"pd")
                _bus_data_row["pd"] = bus_row["pd"]
            end
            if haskey(bus_row,"qd")
                _bus_data_row["qd"] = bus_row["qd"]
            end
            if haskey(bus_row,"gs")
                _bus_data_row["gs"] = bus_row["gs"]
            end
            if haskey(bus_row,"bs")
                _bus_data_row["bs"] = bus_row["bs"]
            end
            if haskey(bus_row,"zone")
                _bus_data_row["zone"] = bus_row["zone"]
            end
            if haskey(bus_row,"area")
                _bus_data_row["area"] = bus_row["area"]
            end
            if haskey(bus_row,"vmin")
                _bus_data_row["vmin"] = bus_row["vmin"]
            end
            if haskey(bus_row,"vmax")
                _bus_data_row["vmax"] = bus_row["vmax"]
            end
            if haskey(bus_row,"bus_i")
                _bus_data_row["bus_i"] = bus_row["bus_i"]
            end
            if haskey(bus_row,"index")
                _bus_data_row["index"] = bus_row["index"]
            end
            if haskey(bus_row,"bus_type")
                _bus_data_row["bus_type"] = bus_row["bus_type"]
            end
            if haskey(bus_row,"va")
                _bus_data_row["va"] = bus_row["va"]
            end
            if haskey(bus_row,"vm")
                _bus_data_row["vm"] = bus_row["vm"]
            end
            if haskey(bus_row,"base_kv")
                _bus_data_row["base_kv"] = bus_row["base_kv"]
            end
            push!(buses, _bus_data_row)
        end
        case["bus"] = buses
    else
        Memento.error(string("no bus table found in matpower file.  The file seems to be missing \"mpc.bus = [...];\""))
    end

    if haskey(python_data, "gen")
        gens = []
        for gen_row in python_data["gen"]
            _gen_data_row = Dict{String,Any}()
            if haskey(gen_row,"source_id")
                _gen_data_row["source_id"] = gen_row["source_id"]
            end
            if haskey(gen_row,"pg")
                _gen_data_row["pg"] = gen_row["pg"]
            end
            if haskey(gen_row,"qg")
                _gen_data_row["qg"] = gen_row["qg"]
            end
            if haskey(gen_row,"gen_bus")
                _gen_data_row["gen_bus"] = gen_row["gen_bus"]
            end
            if haskey(gen_row,"pmax")
                _gen_data_row["pmax"] = gen_row["pmax"]
            end            
            if haskey(gen_row,"qmax")
                _gen_data_row["qmax"] = gen_row["qmax"]
            end
            if haskey(gen_row,"vg")
                _gen_data_row["vg"] = gen_row["vg"]
            end
            if haskey(gen_row,"mbase")
                _gen_data_row["mbase"] = gen_row["mbase"]
            end
            if haskey(gen_row,"index")
                _gen_data_row["index"] = gen_row["index"]
            end
            if haskey(gen_row,"gen_status")
                _gen_data_row["gen_status"] = gen_row["gen_status"]
            end
            if haskey(gen_row,"qmin")
                _gen_data_row["qmin"] = gen_row["qmin"]
            end
            if haskey(gen_row,"pmin")
                _gen_data_row["pmin"] = gen_row["pmin"]
            end
            push!(gens, _gen_data_row)
        end
        case["gen"] = gens
    else
        Memento.error(string("no gen table found in matpower file.  The file seems to be missing \"mpc.gen = [...];\""))
    end

    if haskey(python_data, "branch")
        branches = []
        for branch_row in python_data["branch"]
            _branch_data_row = Dict{String,Any}()
            if haskey(branch_row,"source_id")
                _branch_data_row["source_id"] = branch_row["source_id"]
            end
            if haskey(branch_row,"br_r")
                _branch_data_row["br_r"] = branch_row["br_r"]
            end
            if haskey(branch_row,"br_b")
                _branch_data_row["br_b"] = branch_row["br_b"]
            end
            if haskey(branch_row,"br_x")
                _branch_data_row["br_x"] = branch_row["br_x"]
            end
            if haskey(branch_row,"br_status")
                _branch_data_row["br_status"] = branch_row["br_status"]
            end            
            if haskey(branch_row,"f_bus")
                _branch_data_row["f_bus"] = branch_row["f_bus"]
            end
            if haskey(branch_row,"t_bus")
                _branch_data_row["t_bus"] = branch_row["t_bus"]
            end
            if haskey(branch_row,"rate_a")
                _branch_data_row["rate_a"] = branch_row["rate_a"]
            end
            if haskey(branch_row,"rate_b")
                _branch_data_row["rate_b"] = branch_row["rate_b"]
            end
            if haskey(branch_row,"rate_c")
                _branch_data_row["rate_c"] = branch_row["rate_c"]
            end
            if haskey(branch_row,"index")
                _branch_data_row["index"] = branch_row["index"]
            end
            if haskey(branch_row,"shift")
                _branch_data_row["shift"] = branch_row["shift"]
            end
            if haskey(branch_row,"tap")
                _branch_data_row["tap"] = branch_row["tap"]
            end
            if haskey(branch_row,"angmin")
                _branch_data_row["angmin"] = branch_row["angmin"]
            end
            if haskey(branch_row,"angmax")
                _branch_data_row["angmax"] = branch_row["angmax"]
            end
            push!(branches, _branch_data_row)
        end
        case["branch"] = branches
    else
        Memento.error(string("no branch table found in matpower file.  The file seems to be missing \"mpc.branch = [...];\""))
    end

    if haskey(python_data, "gencost")
        gencost = []
        for gencost_row in python_data["gencost"]
            _gencost_data_row = Dict{String,Any}()
            if haskey(gencost_row,"source_id")
                _gencost_data_row["source_id"] = gencost_row["source_id"]
            end
            if haskey(gencost_row,"cost")
                _gencost_data_row["cost"] = gencost_row["cost"]
            end
            if haskey(gencost_row,"model")
                _gencost_data_row["model"] = gencost_row["model"]
            end
            if haskey(gencost_row,"shutdown")
                _gencost_data_row["shutdown"] = gencost_row["shutdown"]
            end
            if haskey(gencost_row,"startup")
                _gencost_data_row["startup"] = gencost_row["startup"]
            end            
            if haskey(gencost_row,"ncost")
                _gencost_data_row["ncost"] = gencost_row["ncost"]
            end
            if haskey(gencost_row,"index")
                _gencost_data_row["index"] = gencost_row["index"]
            end 
            push!(gencost, _gencost_data_row)
        end
        case["gencost"] = gencost
    end

    if haskey(python_data,"per_unit")
        case["per_unit"] = python_data["per_unit"]
    end

    return case
end


# function _cal_loss(PM_network_data,opf_result,print_summary="false")
#     # weight=[0.5,0.5,1,10,0.5]
#     if haskey(PM_network_data["branch"]["1"],"rate_a")
#         # current limitation loss    
#         loss_current_ac =  Dict(name => ( round.(data["pf"]^2+data["qf"]^2,digits=3) <= round.(PM_network_data["branch"][name]["rate_a"]^2,digits=3) ? 1 : 0) for (name, data) in opf_result["branch"])
#         loss_current_ac_reward = []
#         for (name,data) in loss_current_ac
#             push!(loss_current_ac_reward, data)
#         end
#     end

#     # voltage limitation loss
#     loss_vol_ac =  Dict(name => (data["vm"] <= PM_network_data["bus"][name]["vmax"] && data["vm"] >= PM_network_data["bus"][name]["vmin"] ? 1 : 0) for (name, data) in opf_result["bus"])
#     loss_vol_reward = []
#     for (name,data) in loss_vol_ac
#         push!(loss_vol_reward, data)
#     end
#     # print(loss_vol_ac)
#     # PowerModels.print_summary(opf_result)

#     # line loss
#     loss_line_ac =  Dict(name => data["pt"]+data["pf"] for (name, data) in opf_result["branch"])
#     loss_line_ac_reward = []
#     for (name,data) in loss_line_ac
#         push!(loss_line_ac_reward, data)
#     end

#     # balance gen loss
#     balance_index = []
#     loss_balance_gen = Dict{String,Int}()
#     for (name,data) in opf_result["bus"]
#         if PM_network_data["bus"][name]["bus_type"] == 3 
#             push!(balance_index, name)
#         end
#     end

#     total_balance_active_power = 0
#     total_balance_active_power_upper = 0
#     total_balance_active_power_lower = 0

#     for (name,data) in opf_result["gen"] 

#         if string(PM_network_data["gen"][name]["gen_bus"]) in balance_index
#             total_balance_active_power += round.(data["pg"],digits=3)
#             total_balance_active_power_upper += round.(PM_network_data["gen"][name]["pmax"],digits=3)+0.05
#             total_balance_active_power_lower += round.(PM_network_data["gen"][name]["pmin"],digits=3)-0.05
            
#             # 可能存在多台平衡机组
#             # if round.(data["pg"],digits=3) <= round.(PM_network_data["gen"][name]["pmax"],digits=3) && 
#             #     round.(data["pg"],digits=3) >= round.(PM_network_data["gen"][name]["pmin"],digits=3)

#             #     loss_balance_gen[name] = 1
#             # else
#             #     loss_balance_gen[name] = -1
#             # end
#         end
#     end
#     # 0.9-1.1之间
#     if total_balance_active_power <= 1.1*total_balance_active_power_upper && total_balance_active_power >= 0.9*total_balance_active_power_lower
#         loss_balance_gen_reward = 1
#         if print_summary == "true"
#             println("balance pg is feasible ", round.(total_balance_active_power,digits=3)," ",
#             round.(total_balance_active_power_lower,digits=3)," ",
#             round.(total_balance_active_power_upper,digits=3))
#         end
#     else
#         loss_balance_gen_reward = -1
#         if print_summary == "true"
#             println("balance pg is not feasible ", round.(total_balance_active_power,digits=3)," ",
#             round.(total_balance_active_power_lower,digits=3)," ",
#             round.(total_balance_active_power_upper,digits=3))
#         end
#     end


#     # Gen Qvar limitation loss
#     loss_qg_limit = 1
#     gen_bus_set = group_by_bus(PM_network_data["gen"])
#     for gen_bus in gen_bus_set
#         qg_sum_min = 0
#         qg_sum_max = 0
#         qg_sum = 0
#         for (name,data) in PM_network_data["gen"]
#             if data["gen_bus"] == gen_bus
#                 qg_sum_min += data["qmin"]
#                 qg_sum_max += data["qmax"]
#                 qg_sum += data["qg"]
#             end
#         end
#         if ((round.(qg_sum,digits=3)+0.1  < round.(qg_sum_min,digits=3)) || (round.(qg_sum,digits=3)-0.1 > round.(qg_sum_max,digits=3)))
#             if print_summary == "true"
#                 println(gen_bus," ","qg is not feasible "," ",round.(qg_sum,digits=3)," ",round.(qg_sum_min,digits=3)," ",round.(qg_sum_max,digits=3))
#             end
#             loss_qg_limit = -1
#             break
#         end
#     end

#     # operating cost
#     cost_gen = Dict{String,Float64}()
#     for (name,data) in opf_result["gen"]
#         if PM_network_data["gen"][name]["gen_status"] == 1 #gen is operating
#             if haskey(data,"pg_cost")
#                 cost_gen[name] = data["pg_cost"]
#             end
#         end
#     end
#     cost_gen_reward = []
#     for (name,data) in cost_gen
#         push!(cost_gen_reward,data)
#     end

#     # println(print_summary," loss_qg_limit: ",loss_qg_limit)
#     if haskey(PM_network_data["branch"]["1"],"rate_a")
#         # println(Statistics.mean(loss_current_ac_reward)," ", exp(-100*sum(loss_line_ac_reward))-1," ",Statistics.mean(loss_balance_gen_reward)," ",exp(-sum(cost_gen_reward))-1)
#         reward = (0.5*Statistics.mean(loss_current_ac_reward) + 0.5*Statistics.mean(loss_vol_reward) + 1*exp(-sum(loss_line_ac_reward)) + 3*( loss_balance_gen_reward < 0 ? -2 : loss_qg_limit < 0 ? -1 : 2) + 0.5*exp(-sum(cost_gen_reward)/100000000))/(0.5+0.5+1+3+0.5)
#         if print_summary == "true"
#             println("current_limit_reward:", 0.5*Statistics.mean(loss_current_ac_reward),
#             " loss_vol_reward:", 0.5*Statistics.mean(loss_vol_reward),
#             " line_loss_reward:", exp(-sum(loss_line_ac_reward)), 
#             " XOR:", 10*( loss_balance_gen_reward < 0 ? -2 : loss_qg_limit < 0 ? -1 : 2),
#             " balanced_gen_limit_reward:", loss_balance_gen_reward,
#             " loss_qg_limit_reward:", loss_qg_limit,
#             " cost_reward:", exp(-sum(cost_gen_reward)/100000000),
#             " Total_reward:", reward)
#         end
#     else 
#         reward = (0.5*Statistics.mean(loss_vol_reward) + 1*exp(-sum(loss_line_ac_reward)) + 3*( loss_balance_gen_reward < 0 ? -2 : loss_qg_limit < 0 ? -1 : 2) + 0.5*exp(-sum(cost_gen_reward)/100000000))/(0.5+1+3+0.5)
#         if print_summary == "true"
#             println("loss_vol_reward:", 0.5*Statistics.mean(loss_vol_reward),
#             " line_loss_reward:", exp(-sum(loss_line_ac_reward)), 
#             " XOR:", 10*( loss_balance_gen_reward < 0 ? -2 : loss_qg_limit < 0 ? -1 : 2),
#             " balanced_gen_limit_reward:", loss_balance_gen_reward,
#             " loss_qg_limit_reward:", loss_qg_limit,
#             " cost_reward:", exp(-sum(cost_gen_reward)/100000000),
#             " Total_reward:", reward)
#         end
        

#     end

#     return reward
# end



function _cal_loss(PM_network_data,opf_result,print_summary="false")
    # weight=[0.5,0.5,1,10,0.5]
    if haskey(PM_network_data["branch"]["1"],"rate_a")
        # current limitation loss    
        loss_current_ac =  Dict(name => ( round.(data["pf"]^2+data["qf"]^2,digits=3) <= round.(PM_network_data["branch"][name]["rate_a"]^2,digits=3) ? 1 : 0) for (name, data) in opf_result["branch"])
        loss_current_ac_reward = []
        for (name,data) in loss_current_ac
            push!(loss_current_ac_reward, data)
        end
    end

    loss_vol_reward = []
    # voltage limitation loss
    for (name, data) in opf_result["bus"]
        if data["vm"]-0.001 > PM_network_data["bus"][name]["vmax"]
            if print_summary == "true"
                println(name," ","vol upper vialation: ",data["vm"]," ", PM_network_data["bus"][name]["vmax"])
            end
            push!(loss_vol_reward, (PM_network_data["bus"][name]["vmax"]-data["vm"])/PM_network_data["bus"][name]["vmax"])
        elseif data["vm"]+0.001 < PM_network_data["bus"][name]["vmin"]
            if print_summary == "true"
	            println(name," ","vol lower vialation: ",data["vm"]," ",PM_network_data["bus"][name]["vmin"])
            end
            push!(loss_vol_reward, (data["vm"]-PM_network_data["bus"][name]["vmin"])/PM_network_data["bus"][name]["vmin"])
        else
            push!(loss_vol_reward, 0)
        end
    end
    # print(loss_vol_ac)
    # PowerModels.print_summary(opf_result)

    # line loss
    loss_line_ac =  Dict(name => abs(data["pt"]+data["pf"]) for (name, data) in opf_result["branch"])
    loss_line_ac_reward = []
    for (name,data) in loss_line_ac
        push!(loss_line_ac_reward, data)
    end
    # if print_summary == "true"
    #     println(loss_line_ac_reward)
    # end

    # balance gen loss
    balance_index = []
    loss_balance_gen = Dict{String,Int}()
    for (name,data) in opf_result["bus"]
        if PM_network_data["bus"][name]["bus_type"] == 3 
            push!(balance_index, name)
        end
    end

    total_balance_active_power = 0
    total_balance_active_power_upper = 0
    total_balance_active_power_lower = 0

    for (name,data) in opf_result["gen"] 

        if string(PM_network_data["gen"][name]["gen_bus"]) in balance_index
            total_balance_active_power += round.(data["pg"],digits=3)
            total_balance_active_power_upper += round.(PM_network_data["gen"][name]["pmax"],digits=3)+0.05
            total_balance_active_power_lower += round.(PM_network_data["gen"][name]["pmin"],digits=3)-0.05
            
            # 可能存在多台平衡机组
            # if round.(data["pg"],digits=3) <= round.(PM_network_data["gen"][name]["pmax"],digits=3) && 
            #     round.(data["pg"],digits=3) >= round.(PM_network_data["gen"][name]["pmin"],digits=3)

            #     loss_balance_gen[name] = 1
            # else
            #     loss_balance_gen[name] = -1
            # end
        end
    end
    # 0.9-1.1之间
    if total_balance_active_power > total_balance_active_power_upper 
        loss_balance_gen_reward = (total_balance_active_power_upper-total_balance_active_power)/(total_balance_active_power_upper) 
        if print_summary == "true"
            println("balance pg is not feasible ", round.(total_balance_active_power,digits=3)," ",
            round.(total_balance_active_power_lower,digits=3)," ",
            round.(total_balance_active_power_upper,digits=3))
        end
    elseif total_balance_active_power < total_balance_active_power_lower
        loss_balance_gen_reward = (total_balance_active_power-total_balance_active_power_lower)/(total_balance_active_power_lower)
        if print_summary == "true"
            println("balance pg is not feasible ", round.(total_balance_active_power,digits=3)," ",
            round.(total_balance_active_power_lower,digits=3)," ",
            round.(total_balance_active_power_upper,digits=3))
        end
    else
        loss_balance_gen_reward = 0
        if print_summary == "true"
            println("balance pg is feasible ", round.(total_balance_active_power,digits=3)," ",
            round.(total_balance_active_power_lower,digits=3)," ",
            round.(total_balance_active_power_upper,digits=3))
        end
    end


    # Gen Qvar limitation loss
    loss_qg_limit = []
    gen_bus_set = group_by_bus(PM_network_data["gen"])
    for gen_bus in gen_bus_set
        qg_sum_min = 0
        qg_sum_max = 0
        qg_sum = 0
        for (name,data) in PM_network_data["gen"]
            if data["gen_bus"] == gen_bus
                qg_sum_min += data["qmin"]
                qg_sum_max += data["qmax"]
                qg_sum += data["qg"]
            end
        end
        if round.(qg_sum,digits=3)+0.1  < round.(qg_sum_min,digits=3)
	    
            if print_summary == "true"
                println(gen_bus," ","qg is not feasible "," ",round.(qg_sum,digits=3)," ",round.(qg_sum_min,digits=3)," ",round.(qg_sum_max,digits=3))
            end
 	    #println(round.(qg_sum,digits=3)," ",round.(qg_sum_min,digits=3))
            push!(loss_qg_limit,-abs((round.(qg_sum,digits=3)+0.1-round.(qg_sum_min,digits=3))/(round.(qg_sum_min,digits=3)+0.1)))
            # break
        elseif round.(qg_sum,digits=3)-0.1 > round.(qg_sum_max,digits=3)
            if print_summary == "true"
                println(gen_bus," ","qg is not feasible "," ",round.(qg_sum,digits=3)," ",round.(qg_sum_min,digits=3)," ",round.(qg_sum_max,digits=3))
            end
	    #println(round.(qg_sum,digits=3)," ",round.(qg_sum_max,digits=3))
            push!(loss_qg_limit,-abs((round.(qg_sum_max,digits=3)-round.(qg_sum,digits=3)-0.1)/(round.(qg_sum_max,digits=3)-0.1)))
            # break
        else
            push!(loss_qg_limit,0)
        end
    end

    # operating cost
    cost_gen = Dict{String,Float64}()
    for (name,data) in opf_result["gen"]
        if PM_network_data["gen"][name]["gen_status"] == 1 #gen is operating
            if haskey(data,"pg_cost")
                cost_gen[name] = data["pg_cost"]
            end
        end
    end
    cost_gen_reward = []
    for (name,data) in cost_gen
        push!(cost_gen_reward,data)
    end
    # println(print_summary," loss_qg_limit: ",loss_qg_limit)
    if haskey(PM_network_data["branch"]["1"],"rate_a")
        #println(Statistics.mean(loss_current_ac_reward)," ", exp(-100*sum(loss_line_ac_reward))-1," ",Statistics.mean(loss_balance_gen_reward)," ",exp(-sum(cost_gen_reward))-1)
        volt_reward     = 0.5*(exp(sum(loss_vol_reward))-1)
        branch_reward   = 0.5*Statistics.mean(loss_current_ac_reward)
        lineloss_reward = 1*exp(-sum(loss_line_ac_reward))
        balance_reward  = 10*(exp(loss_balance_gen_reward)-1)
        cost_reward     = 10*exp(-sum(cost_gen_reward)/100000000)
        qg_reward       = 0.05*(exp(sum(loss_qg_limit))-1)
        _reward = branch_reward + volt_reward + lineloss_reward + balance_reward + cost_reward
        reward = (_reward, _reward + qg_reward)
        if print_summary == "true"
            #println(loss_qg_limit," ", exp(sum(loss_qg_limit)))
            println("current_limit_reward:", branch_reward,
            " loss_vol_reward:", volt_reward,
            " line_loss_reward:", lineloss_reward, 
            " balanced_gen_limit_reward:", balance_reward,
            " loss_qg_limit_reward:", qg_reward,
            " cost_reward:", cost_reward,
            " Total_reward:", reward)
        end
    else 
        volt_reward     = 0.5*(exp(sum(loss_vol_reward))-1)
        branch_reward   = 0.0
        lineloss_reward = 1*exp(-sum(loss_line_ac_reward))
        balance_reward  = 10*(exp(loss_balance_gen_reward)-1)
        cost_reward     = 10*exp(-sum(cost_gen_reward)/100000000)
        qg_reward       = 0.05*(exp(sum(loss_qg_limit))-1)
        _reward = lineloss_reward + volt_reward + balance_reward + cost_reward
        reward = (_reward, _reward + qg_reward)
        if print_summary == "true"
            println(" loss_vol_reward:", volt_reward,
            " line_loss_reward:", lineloss_reward, 
            " balanced_gen_limit_reward:", balance_reward,
            " loss_qg_limit_reward:", qg_reward,
            " cost_reward:", cost_reward,
            " Total_reward:", reward)
        end
    end

    # Also return per-component rewards for logging
    component_rewards = Dict{String,Float64}(
        "volt_reward"     => volt_reward,
        "branch_reward"   => branch_reward,
        "lineloss_reward" => lineloss_reward,
        "balance_reward"  => balance_reward,
        "qg_reward"       => qg_reward,
        "cost_reward"     => cost_reward
    )
    return reward, component_rewards
end

function  _sort_according_index(_data,_list,_act_dim)
    _index = sortperm(_list)
    
    # sort_data = []
    # for _i in range(1,_act_dim,_act_dim)
    #     _temp_index = 1
    #     for _j in _list
    #         if _i == _j 
    #             push!(sort_data,round.(_data[_temp_index],digits=3))
    #         else 
    #             _temp_index += 1
    #         end
    #     end
    # end
    return Float64.(_data[_index])    
    
end

function _get_load(case_name)
    network_data = PowerModels._parse_matpower_string(read(string("./",case_name,".m"), String))
    _ref_data = []
    for (_index,_data) in enumerate(network_data["bus"])
        # if _data["pd"] != 0 || _data["qd"] != 0
        #     push!(_ref_data, [_data["bus_i"], _data["pd"],_data["qd"]])
        # end
        push!(_ref_data, [_data["bus_i"], _data["pd"],_data["qd"]])
    end
    # load_p = [108.0,97.0,180.0,74.0,71.0,136.0,125.0,171.0,175.0,195.0,265.0,194.0,317.0,100.0,333.0,181.0,128.0]
    # load_q = [22.0,20.0,37.0,15.0,14.0,28.0,25.0,35.0,36.0,40.0,54.0,39.0,64.0,20.0,68.0,37.0,26.0]
    # bus_id = [1,2,3,4,5,6,7,8,9,10,13,14,15,16,18,19,20]
    # bus_load = np.vstack((np.vstack((bus_id,load_p)),load_q)).T
   
    return _ref_data
end

# function _get_bus_num(bus_data)
#     _bus_num = 0
#     for (name,data) in bus_data
#         _bus_num+=1
#     end   
#     return _bus_num
# end

# function _get_gen_num(gen_data)
#     _gen_num = 0
#     for (name,data) in gen_data
#         _gen_num+=1
#     end   
#     return _gen_num
# end

# function _get_balance_ref_id(PM_network_data)
#     _ref_gen_id = get_balance_gen(PM_network_data)   
#     return _ref_gen_id[Int64(length(_ref_gen_id)//2)]
# end

function create_env(case_name,bus_id,loadp,loadq)
    PowerModels.silence()
    # network_data = PowerModels.parse_file("./PowerModels.jl-master/test/data/matpower/case24.m")
    network_data = PowerModels._parse_matpower_string(read(string("./",case_name,".m"), String))
    
    for _data in network_data["bus"]
        if _data["bus_i"] in bus_id
            for (_i,_j) in enumerate(bus_id)
                if _j == _data["bus_i"]
                    _data["pd"] = loadp[_i]
                    _data["qd"] = loadq[_i]
                end
            end
        end
    end

    python_tomatpower_network_data = _python_data_to_PMs_data(network_data)
    PM_network_data = PowerModels._matpower_to_powermodels!(python_tomatpower_network_data)
    PowerModels.correct_network_data!(PM_network_data)
    balance_gen_id = get_balance_gen(PM_network_data)

    for (name, data) in PM_network_data["gen"]
        # print(typeof(name))
        if name in balance_gen_id
            # println(name,data)
            data["pmin"] = data["pmin"]+0.05 #避免平衡机在下边界运行
            data["pmax"] = data["pmax"]-0.05 #避免平衡机在上边界运行
        end
    end

    ACPP_model = PowerModels.instantiate_model(PM_network_data, PowerModels.ACPPowerModel, PowerModels.build_opf)

    opf_result = PowerModels.optimize_model!(ACPP_model, optimizer=Ipopt.Optimizer)

    # PowerModels.update_data!(PM_network_data, opf_result["solution"])

    # # 需要将母线电压赋值到发电机端电压上
    # for (name,data) in opf_result["solution"]["bus"]
    #     for (_name,_data) in PM_network_data["gen"]
    #         if _data["gen_bus"] == parse(Int64,name)
    #             _data["vg"] = data["vm"]
    #         end
    #     end
    # end

    # # PM_network_data["bus"]["13"]["vm"] = 1.005
    # pf_result = PowerModels.solve_ac_pf(PM_network_data, Ipopt.Optimizer)
    # PowerModels.update_data!(PM_network_data, pf_result["solution"])

    # flows = PowerModels.calc_branch_flow_ac(PM_network_data)
    
    # pf_result["solution"]["branch"] = flows["branch"]
    

    return network_data, opf_result["solution"]["branch"]
end

function get_PV_bus_set(PM_network_data)

    bus_set = []
    for (i, bus) in PM_network_data["bus"]
        if bus["bus_type"] == 2 || bus["bus_type"] == 3
            push!(bus_set, parse(Int64, i))
        end
    end

    bus_set = sort(unique(bus_set))
    return bus_set
end

function env_reset(case_name,bus_id,loadp,loadq)
    return create_env(case_name,bus_id,loadp,loadq)
end

function group_by_bus(PM_network_data_gen)

    gen_bus_set = []
    for (name,data) in PM_network_data_gen
        push!(gen_bus_set, data["gen_bus"])
    end
    gen_bus_set = unique(gen_bus_set)

    return gen_bus_set
end

function get_balance_gen(PM_network_data)

    balance_gen = []
    for (name,data) in PM_network_data["gen"]
        for (_name,_data) in PM_network_data["bus"]
            if parse(Int64, _name) == data["gen_bus"]
                if _data["bus_type"] == 3
                    push!(balance_gen, name)
                end
            end
        end
    end

    return balance_gen
end

function actor_solve_and_step(network_data,dp,dq,action,_env,print_summary="false") 
    # println(print_summary," ",typeof(print_summary))
    PowerModels.silence() 
    python_tomatpower_network_data = _python_data_to_PMs_data(network_data)
    _gencost = python_tomatpower_network_data["gencost"]
    PM_network_data = PowerModels._matpower_to_powermodels!(python_tomatpower_network_data)
    PowerModels.correct_network_data!(PM_network_data)

    network_data["gencost"] = _env["gencost"]

    balance_gen_id = get_balance_gen(PM_network_data)

    for (name, data) in PM_network_data["gen"]
        # print(typeof(name))
        if name in balance_gen_id
            # println(name,data)
            data["pmin"] = data["pmin"]+0.05 #避免平衡机在下边界运行
            data["pmax"] = data["pmax"]-0.05 #避免平衡机在上边界运行
        end
    end

    # println(PM_network_data["gen"]["12"])

    # load
    for (i, load) in PM_network_data["load"]
        load["pd"] = dp[parse(Int64, i)]/100.0
        load["qd"] = dq[parse(Int64, i)]/100.0
    end

    for (i, gen) in PM_network_data["gen"]
        if i in balance_gen_id
            gen["pg"] = 0
        else

            gen["pg"] = action[parse(Int64, i)]/100.0
        end
        # gen["qg"] = action[parse(Int64, i)+33]/100.0
    end

    bus_set = get_PV_bus_set(PM_network_data)

    _temp_index = 1
    for bus_index in bus_set
        for (i, bus) in PM_network_data["bus"]
            if parse(Int64, i) == bus_index
                bus["vm"] = action[_temp_index+gen_num]
                _temp_index = _temp_index+1
            end
        end
    end

    # _temp_index = 1
    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 2
    #         bus["vm"] = action[_temp_index+33]
    #         _temp_index = _temp_index+1
    #     end
    #     # if parse(Int64, i) == 24
    #     #     println(action[parse(Int64, i)+33])
    #     #     print(bus["vm"])
    #     # end
    #     # bus["va"] = 0
    # end

    # for (i, bus) in PM_network_data["bus"]
    #     # if bus["bus_type"] == 1
    #     println(bus["vm"])
    #     # end
    # end

    # for (_i,_data) in PM_network_data["bus"]
    #     if parse(Int64, _i) == 24 
    #         println(_i,_data)
    #     end
    # end

    # println(PM_network_data["bus"])
    # Ipopt.Optimizer
    solver = PowerModels.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)    
    result = PowerModels.solve_ac_pf(PM_network_data, solver) 

    # for (_i,_data) in PM_network_data["bus"]
    #     if parse(Int64, _i) == 24 
    #         println(_i,_data)
    #     end
    # end

    # if print_summary == "true"
    #     println(PM_network_data["gen"])
    # end

    #后续需要考虑无功误差，重新算潮流以及线路状态
    # println(result["solution"]["gen"]["24"])
    # println(result["solution"]["bus"]["9"])

    # check that the solver converged
    # print(PM_network_data["gen"])
    PowerModels.update_data!(PM_network_data, result["solution"])
    # for (_i,_data) in PM_network_data["bus"]
    #     if parse(Int64, _i) == 24 
    #         println(_i,_data)
    #     end
    # end
    # for (i, bus) in result["solution"]["bus"]
    #     println(bus["vm"])
    # end

    if print_summary == "true"
        PowerModels.print_summary(result["solution"])
    end
    # print(PM_network_data["gen"])
    # println(PM_network_data["gen"]["1"]["qg"])
    # println(result["solution"]["gen"]["1"]["qg"])
    flows = PowerModels.calc_branch_flow_ac(PM_network_data)
    PowerModels.update_data!(PM_network_data, flows)
    # println(PM_network_data["gen"])
    # println(result["solution"]["gen"]["1"]["qg"])
    # println(flows["branch"])
    result["solution"]["branch"] = flows["branch"]
    # print(result["solution"]["branch"])

    # result["solution"]["gen"]["gencost"] = _gencost

    # for (name,data) in result["solution"]["gen"]
    #     println(name," ",data["pg"])
    # end

    # println(PM_network_data["gen"])
    for (name,data) in result["solution"]["gen"]
        if PM_network_data["gen"][name]["gen_status"] == 1 #gen is operating
            # if name=="3"
            #     println(_gencost[parse(Int64, name)]["cost"])
            #     println(PM_network_data["gen"][name]["pg"])
            #     println(data)
            #     println(_gencost[parse(Int64, name)]["cost"][1]*PM_network_data["gen"][name]["pg"]^2
            #     + _gencost[parse(Int64, name)]["cost"][2]*PM_network_data["gen"][name]["pg"]
            #     + _gencost[parse(Int64, name)]["cost"][3])
            # end
            if !haskey(data,"pg_cost")
                _temp_cost = _gencost[parse(Int64, name)]["cost"][1]*(data["pg"])^2+ _gencost[parse(Int64, name)]["cost"][2]*(data["pg"])+ _gencost[parse(Int64, name)]["cost"][3]
                # println(data["pg"]," ",_gencost[parse(Int64, name)]["cost"][1]," ",_gencost[parse(Int64, name)]["cost"][2]," ",_gencost[parse(Int64, name)]["cost"][3])
                result["solution"]["gen"][name]["pg_cost"] = _temp_cost
                # if result["solution"]["gen"][name]["pg_cost"] < 0
                #     println(name,PM_network_data["gen"],_gencost[parse(Int64, name)]["cost"])
                # end
            end
        end
    end

    # println(result["solution"]["gen"]["21"]["pg_cost"])

    # for (name,data) in result["solution"]["gen"]
    #     println(name," ",data["pg_cost"])
    # end
    # loss_current_ac, loss_line_ac, loss_balance_gen, cost_gen =_cal_loss(PM_network_datsa,result["solution"])
    # println("!!")
    # print(PM_network_data["branch"])
    
    (reward_without_Q, reward), _ =_cal_loss(PM_network_data,result["solution"], "false")
    # println("!!")
    if print_summary == "true"
        # println(network_data["bus"][24])
        # println(PM_network_data["bus"]["24"])

        # PowerModels.print_summary(result["solution"])
        _temp = _cal_loss(PM_network_data,result["solution"], print_summary)
    end
    
    # network_data["gen"] = [data for (name ,data) in PM_network_data["gen"]]
    # network_data["bus"] = [data for (name ,data) in PM_network_data["bus"]]
    # println(PM_network_data["bus"])
    for (_name,_PM_data) in PM_network_data["bus"]
        for _data in network_data["bus"]
            if _data["source_id"][2] == parse(Int64,_name)  
                _data["va"] = _PM_data["va"]
                _data["vm"] = _PM_data["vm"]
                break
            end
        end
    end

    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 1
    #         println(bus["vm"])
    #     end
    # end
    # println(network_data["bus"])

    # print(PM_network_data["gen"])

    for (_name,_PM_data) in PM_network_data["gen"]
        for _data in network_data["gen"]
            if _data["source_id"][2] == parse(Int64,_name) 
                _data["pg"] = round.(_PM_data["pg"],digits=3)*100.0
                _data["qg"] = round.(_PM_data["qg"],digits=3)*100.0
                _data["vg"] = round.(_PM_data["vg"],digits=3)
                break
            end
        end
    end  

    done = false
    # print(PM_network_data["gen"])


    for (name,data) in PM_network_data["gen"]
        if name in balance_gen_id # 14，12，13均是平衡机组 再求PF时，PowerModel仅仅将12作为平衡机组
            balance_pg_sum_min = 0
            balance_pg_sum_max = 0
            balance_pg_sum = 0

            for (_temp_name,_temp_data) in PM_network_data["gen"]
                if _temp_name in balance_gen_id
                    balance_pg_sum_min += _temp_data["pmin"]
                    balance_pg_sum_max += _temp_data["pmax"]
                    balance_pg_sum += _temp_data["pg"]
                end
            end

            if balance_pg_sum <= balance_pg_sum_max + length(balance_gen_id)*0.05 && 
                balance_pg_sum >= balance_pg_sum_min - length(balance_gen_id)*0.05

                done = false
            else
                if print_summary == "true"
                    # _b = round.(PM_network_data["gen"]["14"]["pg"] + PM_network_data["gen"]["12"]["pg"] + PM_network_data["gen"]["13"]["pg"],digits=3)
                    # if _b < 0
                    #     println(PM_network_data["gen"]["12"]["pg"]," ",PM_network_data["gen"]["13"]["pg"]," ",PM_network_data["gen"]["14"]["pg"])
                    #     println(action[begin:34])
                    #     println(action[34:end])
                    #     PowerModels.print_summary(result["solution"])
                    #     # println(action)
                    # end
                    println("balance pg is not feasible ", 
                    round.(balance_pg_sum,digits=3)," ", 
                    round.(balance_pg_sum_min - length(balance_gen_id)*0.05,digits=3)," ",
                    round.(balance_pg_sum_max + length(balance_gen_id)*0.05 ,digits=3))
                    
                    # println(action[begin:34])
                    # println(action[34:end])
                    # PowerModels.print_summary(result["solution"])
                end
                # if print_summary == "true"
                #     # _b = round.(PM_network_data["gen"]["14"]["pg"] + PM_network_data["gen"]["12"]["pg"] + PM_network_data["gen"]["13"]["pg"],digits=3)
                #     # if _b < 0
                #     #     println(PM_network_data["gen"]["12"]["pg"]," ",PM_network_data["gen"]["13"]["pg"]," ",PM_network_data["gen"]["14"]["pg"])
                #     #     println(action[begin:34])
                #     #     println(action[34:end])
                #     #     PowerModels.print_summary(result["solution"])
                #     #     # println(action)
                #     # end
                #     println("balance pg is feasible ", 
                #     round.(balance_pg_sum,digits=3)," ", 
                #     round.(balance_pg_sum_min - length(balance_gen_id)*0.05,digits=3)," ",
                #     round.(balance_pg_sum_max + length(balance_gen_id)*0.05 ,digits=3))
                    
                #     # println(action[begin:34])
                #     # println(action[34:end])
                #     # PowerModels.print_summary(result["solution"])
                # end
                done = true
                break
            end
        else
            if (round.(data["pg"],digits=3)-0.001 > data["pmax"]) || (round.(data["pg"],digits=3)+0.001 < data["pmin"]) #判断其他机组有功是否越限
                if print_summary == "true"
                    println(name," ","pg is not feasible "," ",round.(data["pg"],digits=3)," ",data["pmin"]," ",data["pmax"]) 
                end
                done = true
                break
            end
        end
    end


    # #PowerModel这里有个Bug,无功是按照母线进行分配的，需要按照母线上连接的所有的发电机组无功上下限的总和进行判断               
    # gen_bus_set = group_by_bus(PM_network_data["gen"])

    # for gen_bus in gen_bus_set
    #     qg_sum_min = 0
    #     qg_sum_max = 0
    #     qg_sum = 0
    #     for (name,data) in PM_network_data["gen"]
    #         if data["gen_bus"] == gen_bus
    #             qg_sum_min += data["qmin"]
    #             qg_sum_max += data["qmax"]
    #             qg_sum += data["qg"]
    #         end
    #     end
    #     if ((round.(qg_sum,digits=3)+0.1 < round.(qg_sum_min,digits=3)) || (round.(qg_sum,digits=3)-0.1 > round.(qg_sum_max,digits=3)))
    #         if print_summary == "true"
    #             println("#######################")
    #             println("Julia Agent")
    #             println(gen_bus," ","qg is not feasible "," ",round.(qg_sum,digits=3)," ",round.(qg_sum_min,digits=3)," ",round.(qg_sum_max,digits=3))
    #         end
    #         done = true
    #         break
    #     end
    # end

    # 端电压也不能越线！！！！！！！！！！！！！！！！！！！！！！！！！


    # println("Julia"," ",done)
    if done 
        info = "False"
    else
        info = "True"
    end

    # return  network_data,[loss_current_ac, loss_line_ac, loss_balance_gen, cost_gen], done, info
    return  (network_data,flows["branch"]), (reward_without_Q, reward), done, info


end


function opf_solve_and_step(network_data,dp,dq,_env,print_summary="false")  
    # PowerModels.logger_config!("trace")
    PowerModels.silence()
    python_tomatpower_network_data = _python_data_to_PMs_data(network_data)
    PM_network_data = PowerModels._matpower_to_powermodels!(python_tomatpower_network_data)
    PowerModels.correct_network_data!(PM_network_data)
    balance_gen_id = get_balance_gen(PM_network_data)

    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 3
    #         println("Before Solver Env BUS: ", bus["vm"])
    #     end
    # end

    # for (i, gen) in PM_network_data["gen"]
    #     if gen["gen_bus"] == 13
    #         println("After Solver Env GEN: ",i," ", gen["vg"])
    #     end
    # end

    network_data["gencost"] = _env["gencost"]

    # # There meybe a bug, The function "PowerModels.correct_network_data!" revise the gencost in network_data
    # for _data in network_data["gencost"]
    #     _data["cost"][1] = round.(_data["cost"][1]/100.0,digits=3)
    #     _data["cost"][2] = round.(_data["cost"][2]/100.0,digits=3)
    # end

    for (name, load) in PM_network_data["load"]
        load["pd"] = round.(dp[parse(Int64, name)]/100.0,digits=3)
        load["qd"] = round.(dq[parse(Int64, name)]/100.0,digits=3)
    end

    for (name, data) in PM_network_data["gen"]
        # print(typeof(name))
        if name in balance_gen_id
            # println(name,data)
            data["pmin"] = data["pmin"]+0.05 #避免平衡机在下边界运行
            data["pmax"] = data["pmax"]-0.05 #避免平衡机在上边界运行
        end
    end

    # # print(PM_network_data["gen"])
    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 3
    #         println("Before Solver Env BUS 1: ", bus["vm"])
    #     end
    # end

    # for (i, gen) in PM_network_data["gen"]
    #     if gen["gen_bus"] == 13
    #         println("After Solver Env GEN 1: ",i," ", gen["vg"])
    #     end
    # end
    # println(PM_network_data["load"]["11"])

    ACPP_model = PowerModels.instantiate_model(PM_network_data, PowerModels.ACPPowerModel, PowerModels.build_opf)
    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 3
    #         println("Before Solver Env BUS 2: ", bus["vm"])
    #     end
    # end

    # for (i, gen) in PM_network_data["gen"]
    #     if gen["gen_bus"] == 13
    #         println("After Solver Env GEN 2: ",i," ", gen["vg"])
    #     end
    # end

    opf_result = PowerModels.optimize_model!(ACPP_model, optimizer=Ipopt.Optimizer)

    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 3
    #         println("Before Solver Env BUS 3: ", bus["vm"])
    #     end
    # end
    # println(opf_result["termination_status"])
    # for (i, gen) in PM_network_data["gen"]
    #     if gen["gen_bus"] == 13
    #         println("After Solver Env GEN 3: ",i," ", gen["vg"])
    #     end
    # end
    PowerModels.update_data!(PM_network_data, opf_result["solution"])

    # for (i, bus) in PM_network_data["bus"]
    #     if bus["bus_type"] == 3
    #         println("After Solver Env: ", bus["vm"])
    #     end
    # end

    # for (i, gen) in PM_network_data["gen"]
    #     if gen["gen_bus"] == 13
    #         println("After Solver Env GEN: ",i," ", gen["vg"])
    #     end
    # end

    # PowerModels.print_summary(opf_result["solution"])


    # PowerModels.print_summary(opf_result["solution"])

    # print(PM_network_data["gen"])
    # loss_current_ac, loss_line_ac, loss_balance_gen, cost_gen =_cal_loss(PM_network_data,opf_result["solution"])
    
    # println(opf_result["solution"]["gen"])
    # print(PM_network_data["gen"]["12"]["pmin"])
    
    # check any gens turn off
    # for (name,data) in PM_network_data["gen"]
    #     if data["gen_status"] == 0
    #         println("########################")
    #         println("The gen turns off ",name)
    #         println("########################")
    #     end
    # end
    if string(opf_result["termination_status"]) != "LOCALLY_SOLVED"
        println(opf_result["termination_status"])
        println("Solver NOT converged !!!!!!!!!!!!")
    end

    # println(PM_network_data["gen"])
    # if print_summary == "true"
    #     if string(opf_result["termination_status"]) != "LOCALLY_SOLVED"
    #         println(opf_result["termination_status"])
    #         println("Solver NOT converged !!!!!!!!!!!!")
    #     end
    # end
    # if print_summary == "true"
    #     PowerModels.print_summary(opf_result["solution"])
    # end

    # PowerModels.print_summary(opf_result["solution"])

    # 需要将母线电压赋值到发电机端电压上
    for (name,data) in opf_result["solution"]["bus"]
        for (_name,_data) in PM_network_data["gen"]
            if _data["gen_bus"] == parse(Int64,name)
                _data["vg"] = data["vm"]
            end
        end
    end

    (reward_without_Q, reward), components = _cal_loss(PM_network_data,opf_result["solution"],print_summary)

    # PM_network_data["bus"]["13"]["vm"] = 1.005
    # Ipopt.Optimizer
    solver = PowerModels.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)
    pf_result = PowerModels.solve_ac_pf(PM_network_data, solver)        
    PowerModels.update_data!(PM_network_data, pf_result["solution"])

    flows = PowerModels.calc_branch_flow_ac(PM_network_data)
    pf_result["solution"]["branch"] = flows["branch"]


    # if print_summary == "true"
    #     PowerModels.print_summary(pf_result["solution"])
    # end
    

    # for (i, gen) in PM_network_data["gen"]
    #     if i in ["14","12","13"]
    #         gen["pg"] = 0
    #     end
    #     # gen["qg"] = action[parse(Int64, i)+33]/100.0
    # end

    # pf_result = PowerModels.solve_ac_pf(PM_network_data, Ipopt.Optimizer)
    # PowerModels.print_summary(pf_result["solution"])

    sort_index = []
    action = []
    _action = []
    # active power
    for (name,data) in opf_result["solution"]["gen"]
        #需要按name从大到小排序！！！
        if name == string(balance_ref_id) # 机组是平衡机组 
            push!(sort_index,parse(Int64,name))     
            push!(action,(data["pg"]-PM_network_data["gen"][name]["pmin"])/(PM_network_data["gen"][name]["pmax"]+0.001-PM_network_data["gen"][name]["pmin"])) #(action-Pmin)/(Pmax-Pmin) \in [0,1]
            push!(_action,data["pg"]*100)
            # println(data["pg"]*100)
        else
            push!(sort_index,parse(Int64,name)) 
            push!(action,(data["pg"]-PM_network_data["gen"][name]["pmin"])/(PM_network_data["gen"][name]["pmax"]-PM_network_data["gen"][name]["pmin"]))
            push!(_action,data["pg"]*100)
            # println(data["pg"]*100)
        end

    end

    # # inactive power
    
    # for (name,data) in opf_result["solution"]["gen"]

    #     push!(sort_index,parse(Int64,name)+Int64(33))
    #     push!(action,(data["qg"]-PM_network_data["gen"][name]["qmin"])/
    #     (PM_network_data["gen"][name]["qmax"]-PM_network_data["gen"][name]["qmin"]))
    #     push!(_action,data["qg"]*100)
    # end    

    # bus vm
    for (name,data) in opf_result["solution"]["bus"]
        if PM_network_data["bus"][name]["bus_type"] == 2 || PM_network_data["bus"][name]["bus_type"] == 3
            push!(sort_index,parse(Int64,name)+Int64(gen_num))
            push!(action,(data["vm"]-PM_network_data["bus"][name]["vmin"])/(PM_network_data["bus"][name]["vmax"]-PM_network_data["bus"][name]["vmin"]))
            # println(name," ",data["vm"]," ",(data["vm"]-PM_network_data["bus"][name]["vmin"])/(PM_network_data["bus"][name]["vmax"]-PM_network_data["bus"][name]["vmin"])," ",PM_network_data["bus"][name]["vmin"]," ",PM_network_data["bus"][name]["vmax"])
            push!(_action,data["vm"])
        end
    end   

    _action = _sort_according_index(_action, sort_index, gen_num+bus_num)
    action = _sort_according_index(action, sort_index, gen_num+bus_num)
    # println("ACOPF:",_action[55:end])
    # println("Normal_ACOPF:",action[55:end])
    # println("ACOPF:",length(_action))
    # println("Normal_ACOPF:",length(action))

    # println([action[_i] for _i in sortperm(sort_index,rev=true)])


    
    # for (name,data) in opf_result["solution"]["gen"]
    #     println(name," ",data["pg"]," ",PM_network_data["gen"][name]["pmax"])
    # end

    # network_data["gen"] = [data for (name ,data) in PM_network_data["gen"]]
    # network_data["bus"] = [data for (name ,data) in PM_network_data["bus"]]

    # 这里出问题了，输入数据没对上！！！！！！！！！！！！
    #network_data是个list，需要遍历进行检查然后赋值

    for (_name,_PM_data) in PM_network_data["bus"]
        for _data in network_data["bus"]
            if _data["source_id"][2] == parse(Int64,_name)  
                _data["va"] = _PM_data["va"]
                _data["vm"] = _PM_data["vm"]
                break
            end
        end
    end

    # for _data in network_data["gen"]
    #     if _data["source_id"][2] == 1
    #         println(_data["pg"])
    #     end
    # end

    for (_name,_PM_data) in PM_network_data["gen"]
        for _data in network_data["gen"]
            if _data["source_id"][2] == parse(Int64,_name) 
                _data["pg"] = round.(_PM_data["pg"],digits=3)*100.0
                _data["qg"] = round.(_PM_data["qg"],digits=3)*100.0
                _data["vg"] = round.(_PM_data["vg"],digits=3)
                break
            end
        end
    end  

    # for _data in network_data["gen"]
    #     if _data["source_id"][2] == 1
    #         println(_data["pg"])
    #     end
    # end

    done = false
    # if step 
    # println(done)
    #判断有功是否越线
    for (name,data) in PM_network_data["gen"]
        if round.(data["pg"],digits=3) > data["pmax"] || round.(data["pg"],digits=3) < data["pmin"]
            done = true
            break
        end
    end
    # println(done)

    #PowerModel这里有个Bug,无功是按照母线进行分配的，需要按照母线上连接的所有的发电机组无功上下限的总和进行判断               
    gen_bus_set = group_by_bus(PM_network_data["gen"])

    for gen_bus in gen_bus_set
        qg_sum_min = 0
        qg_sum_max = 0
        qg_sum = 0
        for (name,data) in PM_network_data["gen"]
            if data["gen_bus"] == gen_bus
                qg_sum_min += data["qmin"]
                qg_sum_max += data["qmax"]
                qg_sum += data["qg"]
            end
        end
        if ((round.(qg_sum,digits=3)+0.1 < round.(qg_sum_min,digits=3)) || (round.(qg_sum,digits=3)-0.1 > round.(qg_sum_max,digits=3)))
            if print_summary == "true"
                println("#######################")
                println("Julia Solver")
                println(gen_bus," ","qg is not feasible "," ",round.(qg_sum,digits=3)," ",round.(qg_sum_min,digits=3)," ",round.(qg_sum_max,digits=3))
            end
            done = true
            break
        end
    end

    # println(done)

    if done 
        info = "False"
    else
        info = "True"
    end


    # return  network_data,[loss_current_ac, loss_line_ac, loss_balance_gen, cost_gen], done, info
    return  (network_data,flows["branch"]), action, (reward_without_Q, reward), done, info


end


function cal_detailed_metrics(PM_network_data, opf_result)
    # Return raw engineering metrics (not reward-scaled) for evaluation/reporting
    metrics = Dict{String, Any}()

    # 1. Voltage violations
    n_volt_upper = 0; n_volt_lower = 0; max_volt_vio = 0.0
    for (name, data) in opf_result["bus"]
        vm = data["vm"]
        vmax = PM_network_data["bus"][name]["vmax"]
        vmin = PM_network_data["bus"][name]["vmin"]
        if vm > vmax
            n_volt_upper += 1
            max_volt_vio = max(max_volt_vio, vm - vmax)
        elseif vm < vmin
            n_volt_lower += 1
            max_volt_vio = max(max_volt_vio, vmin - vm)
        end
    end
    metrics["n_volt_upper"] = n_volt_upper
    metrics["n_volt_lower"] = n_volt_lower
    metrics["max_volt_vio"] = max_volt_vio

    # 2. Branch flow violations
    n_branch_vio = 0; max_branch_vio = 0.0
    if haskey(PM_network_data["branch"]["1"], "rate_a")
        for (name, data) in opf_result["branch"]
            flow_mva = sqrt(data["pf"]^2 + data["qf"]^2)
            rate_a = PM_network_data["branch"][name]["rate_a"]
            if flow_mva > rate_a
                n_branch_vio += 1
                max_branch_vio = max(max_branch_vio, flow_mva - rate_a)
            end
        end
    end
    metrics["n_branch_vio"] = n_branch_vio
    metrics["max_branch_vio"] = max_branch_vio

    # 3. Balance generator PG feasibility
    balance_index = []
    for (name, data) in opf_result["bus"]
        if PM_network_data["bus"][name]["bus_type"] == 3
            push!(balance_index, name)
        end
    end
    total_balance_pg = 0.0
    total_balance_pmax = 0.0
    total_balance_pmin = 0.0
    for (name, data) in PM_network_data["gen"]
        if string(data["gen_bus"]) in balance_index
            total_balance_pg += data["pg"]
            total_balance_pmax += data["pmax"]
            total_balance_pmin += data["pmin"]
        end
    end
    bal_vio = 0.0
    if total_balance_pg > total_balance_pmax
        bal_vio = total_balance_pg - total_balance_pmax
    elseif total_balance_pg < total_balance_pmin
        bal_vio = total_balance_pmin - total_balance_pg
    end
    metrics["balance_pg_vio"] = bal_vio
    metrics["total_balance_pg"] = total_balance_pg * 100.0

    # 4. QG violations (per bus aggregated)
    n_qg_vio = 0; max_qg_vio = 0.0
    gen_bus_set = group_by_bus(PM_network_data["gen"])
    for gen_bus in gen_bus_set
        qg_sum_min = 0.0; qg_sum_max = 0.0; qg_sum = 0.0
        for (name, data) in PM_network_data["gen"]
            if data["gen_bus"] == gen_bus
                qg_sum_min += data["qmin"]
                qg_sum_max += data["qmax"]
                qg_sum += data["qg"]
            end
        end
        if qg_sum < qg_sum_min
            n_qg_vio += 1
            max_qg_vio = max(max_qg_vio, qg_sum_min - qg_sum)
        elseif qg_sum > qg_sum_max
            n_qg_vio += 1
            max_qg_vio = max(max_qg_vio, qg_sum - qg_sum_max)
        end
    end
    metrics["n_qg_vio"] = n_qg_vio
    metrics["max_qg_vio"] = max_qg_vio

    # 5. Line losses (MW)
    total_loss = 0.0
    for (name, data) in opf_result["branch"]
        total_loss += abs(data["pt"] + data["pf"])
    end
    metrics["total_line_loss"] = total_loss * 100.0

    # 6. Generation cost — compute directly in Julia from PM_network_data pg values
    # (Avoid PyObject arithmetic issues by using PM_network_data, not raw _gencost)
    total_cost = 0.0
    # Get OPF objective as proxy for cost (only available in OPF, not PF)
    # For PF results, sum pg_cost if available
    for (name, data) in opf_result["gen"]
        if PM_network_data["gen"][name]["gen_status"] == 1
            if haskey(data, "pg_cost") && isfinite(data["pg_cost"])
                total_cost += data["pg_cost"]
            end
        end
    end
    metrics["total_cost"] = total_cost

    # 7. Aggregate violations summary
    metrics["total_violations"] = n_volt_upper + n_volt_lower + n_branch_vio + n_qg_vio + (bal_vio > 0.001 ? 1 : 0)

    return metrics
end


function evaluate_agent_step(network_data, dp, dq, action, _env)
    PowerModels.silence()
    python_tomatpower_network_data = _python_data_to_PMs_data(network_data)
    _gencost = python_tomatpower_network_data["gencost"]
    PM_network_data = PowerModels._matpower_to_powermodels!(python_tomatpower_network_data)
    PowerModels.correct_network_data!(PM_network_data)

    balance_gen_id = get_balance_gen(PM_network_data)
    for (name, data) in PM_network_data["gen"]
        if name in balance_gen_id
            data["pmin"] = data["pmin"] + 0.05
            data["pmax"] = data["pmax"] - 0.05
        end
    end

    for (i, load) in PM_network_data["load"]
        load["pd"] = dp[parse(Int64, i)] / 100.0
        load["qd"] = dq[parse(Int64, i)] / 100.0
    end

    for (i, gen) in PM_network_data["gen"]
        if i in balance_gen_id
            gen["pg"] = 0
        else
            gen["pg"] = action[parse(Int64, i)] / 100.0
        end
    end

    bus_set = get_PV_bus_set(PM_network_data)
    _temp_index = 1
    for bus_index in bus_set
        for (i, bus) in PM_network_data["bus"]
            if parse(Int64, i) == bus_index
                bus["vm"] = action[_temp_index + gen_num]
                _temp_index = _temp_index + 1
            end
        end
    end

    solver = PowerModels.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)
    result = PowerModels.solve_ac_pf(PM_network_data, solver)
    PowerModels.update_data!(PM_network_data, result["solution"])

    for (name, data) in result["solution"]["gen"]
        if PM_network_data["gen"][name]["gen_status"] == 1
            if !haskey(data, "pg_cost")
                local _idx = parse(Int64, name)
                local _cost_arr = _gencost[_idx]["cost"]
                # Convert PyObject to Julia Float64 to avoid arithmetic issues
                data["pg_cost"] = Float64(_cost_arr[1]) * (Float64(data["pg"]))^2 +
                                   Float64(_cost_arr[2]) * Float64(data["pg"]) +
                                   Float64(_cost_arr[3])
            end
        end
    end

    flows = PowerModels.calc_branch_flow_ac(PM_network_data)
    PowerModels.update_data!(PM_network_data, flows)
    result["solution"]["branch"] = flows["branch"]

    # Check feasibility
    done = false
    for (name, data) in PM_network_data["gen"]
        if name in balance_gen_id
            balance_pg_sum_min = 0.0; balance_pg_sum_max = 0.0; balance_pg_sum = 0.0
            for (_temp_name, _temp_data) in PM_network_data["gen"]
                if _temp_name in balance_gen_id
                    balance_pg_sum_min += _temp_data["pmin"]
                    balance_pg_sum_max += _temp_data["pmax"]
                    balance_pg_sum += _temp_data["pg"]
                end
            end
            if !(balance_pg_sum <= balance_pg_sum_max + length(balance_gen_id) * 0.05 &&
                 balance_pg_sum >= balance_pg_sum_min - length(balance_gen_id) * 0.05)
                done = true
                break
            end
        else
            if (round(data["pg"], digits = 3) - 0.001 > data["pmax"]) ||
               (round(data["pg"], digits = 3) + 0.001 < data["pmin"])
                done = true
                break
            end
        end
    end

    info = done ? "False" : "True"
    metrics = cal_detailed_metrics(PM_network_data, result["solution"])
    ((reward_without_Q, reward), components) = _cal_loss(PM_network_data, result["solution"], "false")
    metrics["reward_without_Q"] = reward_without_Q
    metrics["reward"] = reward
    # Also save per-component rewards
    for (k, v) in components
        metrics[k] = v
    end

    # Update network_data for next step
    for (_name, _PM_data) in PM_network_data["bus"]
        for _data in network_data["bus"]
            if _data["source_id"][2] == parse(Int64, _name)
                _data["va"] = _PM_data["va"]
                _data["vm"] = _PM_data["vm"]
                break
            end
        end
    end
    for (_name, _PM_data) in PM_network_data["gen"]
        for _data in network_data["gen"]
            if _data["source_id"][2] == parse(Int64, _name)
                _data["pg"] = round(_PM_data["pg"], digits = 3) * 100.0
                _data["qg"] = round(_PM_data["qg"], digits = 3) * 100.0
                _data["vg"] = round(_PM_data["vg"], digits = 3)
                break
            end
        end
    end

    return (network_data, flows["branch"]), metrics, (reward_without_Q, reward), done, info
end


end
