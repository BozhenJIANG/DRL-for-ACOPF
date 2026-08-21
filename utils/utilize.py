import numpy as np

def creat_dataset(bus_load):
    bus_id = bus_load[:,0].astype(int)
    # case24 0.75-0.95
    # case118 0.9-1.1 1.4-1.8
    # np.array([1.5,1.4,1.4,1.4,1.5,1.55,1.6,1.65,1.7,1.75,1.78,1.79,1.8,1.78,1.7,1.6,1.6,1.65,1.7,1.75,1.76,1.7,1.6,1.5])
    # base_rate_p = np.array([1,1,1,1.01,1.02,1.02,1.05,1.06,1.07,1.07,1.075,1.077,1.08,1.10,1.08,1.06,1.04,1.06,1.07,1.075,1.07,1.07,1.06,1.05])
    # base_rate_q = np.array([1,1,1,1.01,1.02,1.02,1.05,1.06,1.07,1.07,1.075,1.077,1.08,1.10,1.08,1.06,1.04,1.06,1.07,1.075,1.07,1.07,1.06,1.05])
    base_rate_p = np.array([1,1,1,1.01,1.02,1.02,1.05,1.06,1.07,1.07,1.075,1.077,1.08,1.10,1.08,1.06,1.04,1.06,1.07,1.075,1.07,1.07,1.06,1.05])*0.85
    base_rate_q = np.array([1,1,1,1.01,1.02,1.02,1.05,1.06,1.07,1.07,1.075,1.077,1.08,1.10,1.08,1.06,1.04,1.06,1.07,1.075,1.07,1.07,1.06,1.05])*0.83
    rate_p = []
    rate_q = []
    for i in range(1000):
        rate_p.append(base_rate_p+np.random.normal(loc=0, scale=0.0005, size=24))
        rate_q.append(base_rate_q+np.random.normal(loc=0, scale=0.0005, size=24))
    rate_p = np.array(rate_p)
    rate_q = np.array(rate_q)
    base_load_data_p = np.einsum("ij,k->ijk",rate_p,bus_load[:,1])
    base_load_data_q = np.einsum("ij,k->ijk",rate_q,bus_load[:,2])

    return bus_id,base_load_data_p,base_load_data_q

def train_and_test_data_split(loadp,loadq):
    train_dataset = np.hstack((loadp[:800,:,:],loadq[:800,:,:]))
    test_dataset = np.hstack((loadp[800:,:,:],loadq[800:,:,:]))
    return train_dataset, test_dataset


def julia_data_to_python_data(julia_data,flows,timestep,expand_state,normal=False):
    _gen_state = []
    _branch_state = []
    _bus_state = []
    _flows_state = []
    # for _i in julia_data["gen"]:
    #     _gen_state.append([_i["pg"]/100.0,_i["qg"]/100.0,_i["vg"],_i["gen_status"],_i["pmax"]/100.0,_i["qmax"]/100.0
    #                        ,_i["source_id"][1]
    #                        ])

    # for _i in julia_data["branch"]:
    #     _branch_state.append([
    #         # _i["br_r"],_i["br_b"],_i["br_x"],
    #         _i["rate_a"]/100.0
    #         # ,_i["br_status"]
    #         # ,_i["source_id"][1]
    #                           ])
    if normal:
        for _i in julia_data["bus"]:
            # if _i["bus_type"] == 1:
            _bus_state.append([_i["pd"],_i["qd"]
                            #    ,_i["va"],_i["vm"]
                            #    ,_i["gs"],_i["bs"]
                            ,_i["source_id"][1]
                            ])    
    else:
        for _i in julia_data["bus"]:
            # if _i["bus_type"] == 1:
            _bus_state.append([_i["pd"]/100,_i["qd"]/100
                            #    ,_i["va"],_i["vm"]
                            #    ,_i["gs"],_i["bs"]
                            ,_i["source_id"][1]
                            ])          
    
    _bus_state.sort(key=lambda x: x[-1]) 
        
    if expand_state:
        flows_key = [_i for _i in flows.keys()]
        
        for _i in flows_key:
            _flows_state.append([flows[_i]["qf"]
                                ,flows[_i]["qt"]
                                ,flows[_i]["pf"]
                                ,flows[_i]["pt"],int(_i)])
        _flows_state.sort(key=lambda x: x[-1])
        states = np.hstack((
            (np.array(_bus_state)[:,:-1]).T.reshape(-1),
            (np.array(_flows_state)[:,:-1]).reshape(-1)))
    else:
        states = np.array(_bus_state)[:,:-1].T.reshape(-1)
    # _gen_state.sort(key=lambda x: x[-1])
    # _branch_state.sort(key=lambda x: x[-1])
     
    # print(np.array(_gen_state)[:,:-1])   
    # print(np.array(_branch_state)[:,:-1]) 
    # print(np.array(_bus_state)[:,:-1])  

    # states = np.hstack((np.hstack((
    #     (np.array(_gen_state)[:,:-1]).reshape(-1),
    #     (np.array(_branch_state)[:,:-1]).reshape(-1))),
    #     (np.array(_bus_state)[:,:-1]).reshape(-1)))

    
    return np.append(states, timestep)
    # return states


def inverse_a2action(a,env,gen_num,PV_bus_set):
    pg_action = []
    qg_action = []
    vm_action = []
    for _i in range(gen_num):
        for _j in env["gen"]:
            if _j["source_id"][1] == _i+1: #list index start from 1 injulia
                pg_action.append(((_j["pmax"]-_j["pmin"])*a[_i]+_j["pmin"])) #"*(a*（pgmax-pgmin)+pgmin)" 暂时先不考虑gen_status
                # qg_action.append(((_j["qmax"]-_j["qmin"])*a[_i+33]+_j["qmin"])*state[_i*6+3]) #"gen_status*(a*（pgmax-pgmin)+pgmin)"
    _temp_i = 0

    for _i in PV_bus_set:
        for _j in env["bus"]:
            if  _j["bus_type"] == 2 or _j["bus_type"] == 3: #判断是否是PV节点

                #问题在这，
                if _j["source_id"][1] == _i: #list index start from 1 in julia
                    vm_action.append((_j["vmax"]-_j["vmin"])*a[_temp_i+gen_num]+_j["vmin"])
                    _temp_i += 1

    return pg_action + vm_action

def get_PV_bus_set(julia_data):
    PV_bus_set = []
    for _i in julia_data["bus"]:
            if _i["bus_type"] == 2 or 3:
                PV_bus_set.append(_i["source_id"][1])

    return sorted(PV_bus_set)