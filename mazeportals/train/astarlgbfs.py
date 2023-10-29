import numpy as np
import copy
import heapq
import networkx as nx
import time

from nn_struct_old import NN
from train_sok import to_categorical_tensor
from get_neighbours import get_neighbors

dim = 15 # dimensions set to 15*15 during training

dqn = NN(dim)
dqn.model.load_weights('por5')

G = nx.DiGraph() #attributes of G are global
key_states = []

def check_goal(stateName):
    state = evaluate_state(stateName)
    find_player = np.where(state == 2)
    player_pos = list(zip(find_player[0], find_player[1]))[0]
    if (player_pos[0]-1>=0 and state[player_pos[0]-1][player_pos[1]] == 3) or (player_pos[0]+1<=dim-1 and state[player_pos[0]+1][player_pos[1]] == 3) or (player_pos[1]-1 >=0 and state[player_pos[0]][player_pos[1]-1] == 3) or (player_pos[1]+1 <=dim-1 and state[player_pos[0]][player_pos[1]+1] == 3):
        return True
    else:
       return False

def evaluate_state(stateName):
    arr=[]
    sub = stateName[1:-1]
    for i in sub:
        if i.isdigit():
            arr.append(int(i))
    arr = np.array(arr).reshape(dim,dim)
    return arr

def findNN(state, portal):
    state = evaluate_state(state)
    find_player = np.where(state == 2)
    player_pos = list(zip(find_player[0], find_player[1]))[0]
    if (player_pos[0]-1>=0 and state[player_pos[0]-1][player_pos[1]] == 3) or (player_pos[0]+1<=dim-1 and state[player_pos[0]+1][player_pos[1]] == 3) or (player_pos[1]-1 >=0 and state[player_pos[0]][player_pos[1]-1] == 3) or (player_pos[1]+1 <=dim-1 and state[player_pos[0]][player_pos[1]+1] == 3):
        return 0
    old_state = state
    state = to_categorical_tensor(old_state,portal,dim)
    val = dqn.model.predict(state.reshape(1,dim,dim,8),verbose = 0)[1][0][0]
    return val

class PriorityQ:
    def __init__(self):
        self.elements = []
    def insert(self,value,element):
        heapq.heappush(self.elements, (value, element))
    def getMin(self):
        return heapq.heappop(self.elements)[1]
    def length(self):
        return len(self.elements)

def Astar_lgbfs(init):
    start = time.time()
    closedSet={}
    openSet={}
    dontExpand=[]
    heap=PriorityQ()
    find_player = np.where(init == 2)
    player_pos = list(zip(find_player[0], find_player[1]))
    portal = {}
    find_4 = np.where(init == 4)
    find_4 = list((zip(find_4[0], find_4[1])))
    portal['4'] = find_4

    find_5 = np.where(init == 5)
    find_5 = list((zip(find_5[0], find_5[1])))
    portal['5'] = find_5

    find_6 = np.where(init == 6)
    find_6 = list((zip(find_6[0], find_6[1])))
    portal['6'] = find_6

    find_7 = np.where(init == 7)
    find_7 = list((zip(find_7[0], find_7[1])))
    portal['7'] = find_7

    copy_init=copy.deepcopy(init)
    init = np.reshape(init, (1,dim*dim))[0]
    h=findNN(str(init),portal)

    openSet[str(init)]=[0,h,h,[],[]]
    state = openSet[str(init)] #Stores the g,h,f,[],[]
    stateName = str(init)
    G.add_node(str(init).replace('\n', ''), o=1, g = 0)

    states_expanded = 0
    while True:
        if (time.time() - start) > 600:
            return None, float("inf")
        if check_goal(stateName):
            end = time.time()
            G_state = stateName
            actions=[]
            closedSet[str(stateName)]=state
            while stateName!=str(init):
                  G.nodes[stateName.replace('\n', '')]['o'] = 1
                  actions.append(closedSet[str(stateName)][4])
                  stateName=closedSet[str(stateName)][3]
            key_states = find_keystates(init,G_state)#find_no_of_states
            constraints,raw_h_matrix = find_constraints(key_states)
            h_matrix = create_hmatrix(key_states,raw_h_matrix)
            X_Train = create_minibatch(key_states,portal)
            return X_Train, h_matrix
            break
        closedSet[stateName]= openSet[stateName] #add state to closedSet

        temp_state = evaluate_state(stateName)
        op, act_no, cost = get_neighbors(temp_state, portal, dim)
        for i in  range(0,len(op)):

            newList=str(op[i].reshape(1,dim*dim)[0])
            if newList in closedSet:

                if closedSet[newList][0]>state[0]+cost[i]:#reopening closed states
                    closedSet[newList][0]=state[0]+cost[i]
                    G.nodes[newList.replace('\n', '')]['g'] = state[0]+cost[i]
                    old_pred=list(G.predecessors(newList.replace('\n', '')))[0]
                    G.remove_edge(old_pred.replace('\n', ''), newList.replace('\n', ''))
                    G.add_edge(stateName.replace('\n', ''), newList.replace('\n', ''))
                    closedSet[newList][2]=state[0]+cost[i]+closedSet[newList][1]
                    closedSet[newList][3]=stateName
                    closedSet[newList][4]=act_no[i] #or full operator?
                    openSet[newList]=closedSet[newList]
                    heap.insert(closedSet[newList][2],newList)
                    states_expanded+=1

            elif newList in openSet:
               if openSet[newList][0]>state[0]+cost[i]:
                   openSet[newList][0]=state[0]+cost[i]
                   G.nodes[newList.replace('\n', '')]['g'] = state[0]+cost[i]
                   old_pred=list(G.predecessors(newList.replace('\n', '')))[0]
                   G.remove_edge(old_pred.replace('\n', ''), newList.replace('\n', ''))
                   G.add_edge(stateName.replace('\n', ''), newList.replace('\n', ''))
                   openSet[newList][2]=state[0]+cost[i]+openSet[newList][1]
                   openSet[newList][3]=stateName
                   openSet[newList][4]=act_no[i] #or full operator?
                   heap.insert(openSet[newList][2],newList)

            else:
                h= findNN(str(newList), portal)
                openSet[newList]=[state[0]+cost[i],h,state[0]+h+cost[i],stateName,act_no[i]]
                heap.insert(openSet[newList][2],newList)
                G.add_node(newList.replace('\n', ''), o=0, g = state[0]+cost[i])
                G.add_edge(stateName.replace('\n', ''), newList.replace('\n', ''))
                #print("val", openSet[newList][2])
                states_expanded+=1

        if heap.length()==0:
            return float("inf")
        stateName = heap.getMin()
        state = openSet[stateName]

def find_keystates(init_key,G_state):
    goal_key = G_state.replace("\n",'')
    while len(list(G.predecessors(goal_key)))!= 0:
        key_states.append(goal_key)
        pred = list(G.predecessors(goal_key))[0]
        child_of_pred = list(G.successors(pred.replace("\n",'')))
        for child in child_of_pred:
            if G.nodes[child]['o']==1:
                continue
            else:
                key_states.append(child)
        goal_key = pred
    return key_states

def find_constraints(key_states):
    constraints=0
    for i in range(len(key_states)):
        for j in range(i+1,len(key_states)):
            if (G.nodes[key_states[i]]['o'] == 1 and G.nodes[key_states[j]]['o'] == 0) :
                if (G.nodes[key_states[i]]['g'] == G.nodes[key_states[j]]['g']):
                    #print(evaluate_state(key_states[i]),G.nodes[key_states[i]]['g'], evaluate_state(key_states[j]),G.nodes[key_states[j]]['g'],"\n")
                    constraints+=1

            elif (G.nodes[key_states[i]]['o'] == 0 and G.nodes[key_states[j]]['o'] == 1) :
                if (G.nodes[key_states[i]]['g'] == G.nodes[key_states[j]]['g']):
                    #print(evaluate_state(key_states[i]),G.nodes[key_states[i]]['g'], evaluate_state(key_states[j]),G.nodes[key_states[j]]['g'],"\n")
                    constraints+=1

    h_matrix = [[0] * constraints for i in range(len(key_states))]
    return constraints, h_matrix

def create_hmatrix(key_states,h_matrix):
    c=0
    for i in range(len((key_states))):
        for j in range(i+1,len((key_states))):
            if  G.nodes[key_states[i]]['o'] == 1   and G.nodes[key_states[j]]['o'] == 0:
                if G.nodes[key_states[i]]['g'] == G.nodes[key_states[j]]['g']:
                    h_matrix[i][c] = -1
                    h_matrix[j][c] = 1
                    c+=1

            elif G.nodes[key_states[j]]['o'] == 1   and G.nodes[key_states[i]]['o'] == 0:
                if G.nodes[key_states[i]]['g'] == G.nodes[key_states[j]]['g']:
                    h_matrix[j][c] = -1
                    h_matrix[i][c] = 1
                    c+=1
    return h_matrix

def create_minibatch(key_states, portal):
    X_Train = []
    for k in key_states:
        X_Train.append(to_categorical_tensor(evaluate_state(k),portal,dim))
    return np.array(X_Train) #minibatch
