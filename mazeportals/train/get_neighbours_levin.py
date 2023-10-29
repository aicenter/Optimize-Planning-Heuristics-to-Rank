import copy
import numpy as np

def get_neighbors(state, portal, dim):

    def check_return(state,portal,pos):
        for key, val in portal.items():
            #print(pos,val)
            if list(pos) == list(val[0]) or list(pos) == list(val[1]):
                if list(val[0]) == list(pos):
                    return int(key),list(val[1])
                if list(val[1]) == list(pos):
                    return int(key),list(val[0])
        return False, []


    find_player = np.where(state == 2)
    agent_ix = list(zip(find_player[0], find_player[1]))[0]
    n_d = [agent_ix[0] + 1,agent_ix[1]]
    n_u = [agent_ix[0] - 1,agent_ix[1]]
    n_r = [agent_ix[0],agent_ix[1] + 1]
    n_l = [agent_ix[0],agent_ix[1] - 1]

    ns = []
    act_no = []
    cost = []
    if n_d[0] >= 0 and n_d[0] <= dim-1 and n_d[1] >= 0 and n_d[1] <= dim-1:
        new_map = copy.deepcopy(state)
        k,p = check_return(new_map,portal,agent_ix)
        if k !=False:
            new_map[agent_ix[0]][agent_ix[1]] = k#agent is sitting at goal
        else:
            new_map[agent_ix[0]][agent_ix[1]] = 1
        n_m = copy.deepcopy(new_map)

        if state[n_d[0]][n_d[1]] == 1 or state[n_d[0]][n_d[1]] == 4 or state[n_d[0]][n_d[1]] == 5 or state[n_d[0]][n_d[1]] == 6 or state[n_d[0]][n_d[1]] == 7:#agent moving to empty floor.
            new_map[n_d[0]][n_d[1]] = 2#agent goes down
            new_state =  new_map
            ns.append(new_state)
            act_no.append(2)
            cost.append(1)
                   #agent pos                    box_pos

        if state[n_d[0]][n_d[1]] != 0 and state[n_d[0]][n_d[1]] != 1:
            k_d, p_d = check_return(n_m,portal,n_d)
            n_m[p_d[0]][p_d[1]] = 2
            new_state = n_m
            ns.append(new_state)
            act_no.append(2)
            cost.append(1)

    if n_u[0] >= 0 and n_u[0] <= dim-1 and n_u[1] >= 0 and n_u[1] <= dim-1:
        new_map = copy.deepcopy(state)
        k,p = check_return(new_map,portal,agent_ix)
        if k !=False:
            new_map[agent_ix[0]][agent_ix[1]] = k#agent is sitting at goal
        else:
            new_map[agent_ix[0]][agent_ix[1]] = 1
        n_m = copy.deepcopy(new_map)
        if state[n_u[0]][n_u[1]] == 1 or state[n_u[0]][n_u[1]] == 4 or state[n_u[0]][n_u[1]] == 5 or state[n_u[0]][n_u[1]] == 6 or state[n_u[0]][n_u[1]] == 7:#agent moving to empty floor.
            new_map[n_u[0]][n_u[1]] = 2#agent goes down
            new_state =  new_map
            ns.append(new_state)
            act_no.append(1)
            cost.append(1)

        if state[n_u[0]][n_u[1]] != 0 and state[n_u[0]][n_u[1]] != 1:#box is down Checking if it can be moved further down.
            k_d, p_d = check_return(n_m,portal,n_u)
            n_m[p_d[0]][p_d[1]] = 2
            new_state = n_m
            ns.append(new_state)
            act_no.append(1)
            cost.append(1)

    if n_r[0] >= 0 and n_r[0] <= dim-1 and n_r[1] >= 0 and n_r[1] <= dim-1:
       new_map = copy.deepcopy(state)
       k,p = check_return(new_map,portal,agent_ix)
       if k !=False:
           new_map[agent_ix[0]][agent_ix[1]] = k#agent is sitting at goal
       else:
           new_map[agent_ix[0]][agent_ix[1]] = 1
       n_m = copy.deepcopy(new_map)
       if state[n_r[0]][n_r[1]] == 1 or state[n_r[0]][n_r[1]] == 4 or state[n_r[0]][n_r[1]] == 5 or state[n_r[0]][n_r[1]] == 6 or state[n_r[0]][n_r[1]] == 7:#agent moving to empty floor.
           new_map[n_r[0]][n_r[1]] = 2#agent goes down
           new_state =  new_map
           ns.append(new_state)
           act_no.append(4)
           cost.append(1)
                  #agent pos

       if state[n_r[0]][n_r[1]] != 0 and state[n_r[0]][n_r[1]] != 1:#box is down Checking if it can be moved further down.
           k_d, p_d = check_return(n_m,portal,n_r)
           n_m[p_d[0]][p_d[1]] = 2
           new_state = n_m
           ns.append(new_state)
           act_no.append(4)
           cost.append(1)

    if n_l[0] >= 0 and n_l[0] <=dim-1  and n_l[1] >= 0 and n_l[1] <= dim-1:
        new_map = copy.deepcopy(state)
        k,p = check_return(new_map,portal,agent_ix)
        if k !=False:
            new_map[agent_ix[0]][agent_ix[1]] = k#agent is sitting at goal
        else:
            new_map[agent_ix[0]][agent_ix[1]] = 1
        n_m = copy.deepcopy(new_map)

        if state[n_l[0]][n_l[1]] == 1 or state[n_l[0]][n_l[1]] == 4 or state[n_l[0]][n_l[1]] == 5 or state[n_l[0]][n_l[1]] == 6 or state[n_l[0]][n_l[1]] == 7:#agent moving to empty floor.
            new_map[n_l[0]][n_l[1]] = 2#agent goes down
            new_state =  new_map
            ns.append(new_state)
            act_no.append(3)
            cost.append(1)
                   #agent pos                    box_pos
        new_map = copy.deepcopy(state)
        if state[n_l[0]][n_l[1]] != 0 and state[n_l[0]][n_l[1]] != 1:
            k_d, p_d = check_return(n_m,portal,n_l)
            n_m[p_d[0]][p_d[1]] = 2
            new_state = n_m
            ns.append(new_state)
            act_no.append(3)
            cost.append(1)

    return ns, act_no, cost
