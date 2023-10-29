import copy
import numpy as np

def get_neighbors(state, dim):
    find_blank = np.where(state == 0)
    agent_ix = list(zip(find_blank[0], find_blank[1]))[0]
    n_d = [agent_ix[0] + 1,agent_ix[1]]
    n_u = [agent_ix[0] - 1,agent_ix[1]]
    n_r = [agent_ix[0],agent_ix[1] + 1]
    n_l = [agent_ix[0],agent_ix[1] - 1]

    ns = []
    act_no = []
    cost = []
    if n_d[0] >= 0 and n_d[0] <= dim-1 and n_d[1] >= 0 and n_d[1] <= dim-1:
        new_map = copy.deepcopy(state)
        new_map[agent_ix[0]][agent_ix[1]] = new_map[n_d[0]] [n_d[1]]
        new_map[n_d[0]] [n_d[1]] = 0#agent goes down
        ns.append(new_map)
        act_no.append(1)
        cost.append(1)

      
    if n_u[0] >= 0 and n_u[0] <= dim-1 and n_u[1] >= 0 and n_u[1] <= dim-1:
        new_map = copy.deepcopy(state)
        new_map[agent_ix[0]][agent_ix[1]] = new_map[n_u[0]][n_u[1]]
        new_map[n_u[0]] [n_u[1]] = 0
        ns.append(new_map)
        act_no.append(2)
        cost.append(1)
        
    if n_r[0] >= 0 and n_r[0] <= dim-1 and n_r[1] >= 0 and n_r[1] <= dim-1:
        new_map = copy.deepcopy(state)
        new_map[agent_ix[0]][agent_ix[1]] = new_map[n_r[0]] [n_r[1]]
        new_map[n_r[0]] [n_r[1]] = 0
        ns.append(new_map)
        act_no.append(3)
        cost.append(1)
        
    if n_l[0] >= 0 and n_l[0] <=dim-1  and n_l[1] >= 0 and n_l[1] <= dim-1:
        new_map = copy.deepcopy(state)
        new_map[agent_ix[0]][agent_ix[1]] = new_map[n_l[0]][n_l[1]] 
        new_map[n_l[0]] [n_l[1]] = 0
        ns.append(new_map)
        act_no.append(4)
        cost.append(1)
       
    return ns, act_no, cost
    
    
 
    
