import copy
import numpy as np

def get_neighbors(state, goal_coords, dim):
    find_player = np.where(state == 3)
    agent_ix = list(zip(find_player[0], find_player[1]))[0]
    n_d = [agent_ix[0] + 1,agent_ix[1]]
    n_u = [agent_ix[0] - 1,agent_ix[1]]
    n_r = [agent_ix[0],agent_ix[1] + 1]
    n_l = [agent_ix[0],agent_ix[1] - 1]

    ns = []
    act_no = []
    cost = []
    if n_d[0] >= 0 and n_d[0] <= dim-1 and n_d[1] >= 0 and n_d[1] <= dim-1:
        if state[n_d[0]] [n_d[1]] == 1 or state[n_d[0]][n_d[1]] == 2:#agent moving to floor or goal.
            new_map = copy.deepcopy(state)
            if agent_ix in goal_coords:
                new_map[agent_ix[0]][agent_ix[1]] = 2#agent is sitting at goal
            else:
                new_map[agent_ix[0]][agent_ix[1]] = 1

            new_map[n_d[0]] [n_d[1]] = 3#agent goes down
            new_state =  new_map
            ns.append(new_state)
            act_no.append(6)
            cost.append(1)
                   #agent pos                    box_pos
        elif state[n_d[0]][n_d[1]] == 4 and n_d[0] + 1 >= 0 and n_d[0] + 1 <= dim-1:#box is down Checking if it can be moved further down.
            if state[n_d[0] + 1] [n_d[1]] == 1 or state[n_d[0] + 1] [n_d[1]] == 2: #moving location empty or goal_coords
                new_map = copy.deepcopy(state)
                if agent_ix in goal_coords:
                    new_map[agent_ix[0]][agent_ix[1]] = 2
                else:
                    new_map[agent_ix[0]][agent_ix[1]] = 1

                new_map[n_d[0]][n_d[1]] = 3
                new_map[n_d[0] + 1][n_d[1]] = 4#box going here

                new_state = new_map
                ns.append(new_state)
                act_no.append(2)
                cost.append(1)
    if n_u[0] >= 0 and n_u[0] <= dim-1 and n_u[1] >= 0 and n_u[1] <= dim-1:
        if state[n_u[0]] [n_u[1]] == 1 or state[n_u[0]][n_u[1]] == 2:
            new_map = copy.deepcopy(state)
            if agent_ix in goal_coords:
                new_map[agent_ix[0]][agent_ix[1]] = 2
            else:
                new_map[agent_ix[0]][agent_ix[1]] = 1

            new_map[n_u[0]][n_u[1]] = 3

            new_state =new_map
            ns.append(new_state)
            act_no.append(5)
            cost.append(1)
        elif state[n_u[0]] [n_u[1]] == 4 and n_u[0] - 1 >= 0 and n_u[0] - 1 <= dim-1:
            if state[n_u[0] - 1][n_u[1]] == 1 or state[n_u[0] - 1] [n_u[1]] == 2:
                new_map = copy.deepcopy(state)
                if agent_ix in goal_coords:
                    new_map[agent_ix[0]][agent_ix[1]] = 2
                else:
                    new_map[agent_ix[0]][agent_ix[1]] = 1

                new_map[n_u[0]] [n_u[1]] = 3
                new_map[n_u[0] - 1] [n_u[1]] = 4
                new_state =  new_map
                ns.append(new_state)
                act_no.append(1)
                cost.append(1)
    if n_r[0] >= 0 and n_r[0] <= dim-1 and n_r[1] >= 0 and n_r[1] <= dim-1:
        if state[n_r[0]][n_r[1]] == 1 or state[n_r[0]] [n_r[1]] == 2:
            new_map = copy.deepcopy(state)
            if agent_ix in goal_coords:
                new_map[agent_ix[0]][agent_ix[1]] = 2
            else:
                new_map[agent_ix[0]][agent_ix[1]] = 1

            new_map[n_r[0]] [n_r[1]] = 3
            new_state =  new_map
            ns.append(new_state)
            act_no.append(8)
            cost.append(1)
        elif state[n_r[0]] [n_r[1]] == 4 and n_r[1] + 1 >= 0 and n_r[1] + 1 <= dim-1:
            if state[n_r[0]] [n_r[1] + 1] == 1 or state[n_r[0]] [n_r[1] + 1] == 2:
                new_map = copy.deepcopy(state)
                if agent_ix in goal_coords:
                    new_map[agent_ix[0]][agent_ix[1]] = 2
                else:
                    new_map[agent_ix[0]][agent_ix[1]] = 1

                new_map[n_r[0]] [n_r[1]] = 3
                new_map[n_r[0]] [n_r[1] + 1] = 4
                new_state = new_map
                ns.append(new_state)
                act_no.append(4)
                cost.append(1)
    if n_l[0] >= 0 and n_l[0] <=dim-1  and n_l[1] >= 0 and n_l[1] <= dim-1:
        if state[n_l[0]] [n_l[1]] == 1 or state[n_l[0]] [n_l[1]] == 2:
            new_map = copy.deepcopy(state)
            if agent_ix in goal_coords:
                new_map[agent_ix[0]][agent_ix[1]] = 2
            else:
                new_map[agent_ix[0]][agent_ix[1]] = 1

            new_map[n_l[0]][n_l[1]] = 3
            new_state = new_map
            ns.append(new_state)
            act_no.append(7)
            cost.append(1)
        elif state[n_l[0]] [n_l[1]] == 4 and n_l[1] - 1 >= 0 and n_l[1] - 1 <= dim-1:
            if state[n_l[0]] [n_l[1] - 1] == 1 or state[n_l[0]] [n_l[1] - 1] == 2:
                new_map = copy.deepcopy(state)
                if agent_ix in goal_coords:
                    new_map[agent_ix[0]][agent_ix[1]] = 2
                else:
                    new_map[agent_ix[0]][agent_ix[1]] = 1

                new_map[n_l[0]] [n_l[1]] = 3
                new_map[n_l[0]] [n_l[1] - 1] = 4
                new_state = new_map
                ns.append(new_state)
                act_no.append(3)
                cost.append(1)
    return ns, act_no, cost
