import numpy as np
import copy
import time
import heapq
from nn_struct_old import NN
from math import inf
import tensorflow as tf
import networkx as nx
from tensorflow.keras.activations import softplus
from train_sok import to_categorical_tensor
from get_neighbours import get_neighbors

dim = 10
nn = NN(dim)

G = nx.DiGraph() #attributes of G are global
key_states = []

nn.model.load_weights('finalSok3')

def find_goal_state(goal_state, box_tar):
    for row in range(0,dim):
        for col in range(0,dim):
            if goal_state[row][col] == 3:
                goal_state[row][col] = 1
            if goal_state[row][col] == 4:
                goal_state[row][col] = 1
    for row in range(0,dim):
        for col in range(0,dim):
            if goal_state[row][col] == 2:
                goal_state[row][col] = 4
    return(to_categorical_tensor(goal_state,box_tar,dim,dim))

def check_goal(stateName, goal_coords):
    state = evaluate_state(stateName)
    find_box_pos = np.where(state == 4)
    box_pos=list(zip(find_box_pos[0], find_box_pos[1]))
    tot_box = len(box_pos)
    for i in range(0,tot_box):
        if box_pos[i] not in goal_coords:
            return False
    return True

def evaluate_state(stateName):
    arr=[]
    sub = stateName[1:-1]
    for i in sub:
        if i.isdigit():
            arr.append(int(i))
    arr = np.array(arr).reshape(dim,dim)
    return arr

def findNN(stateName, box_tar, goal_state):#Strips state

    state = evaluate_state(stateName)
    box_on_T=[]

    for i in range(len(box_tar)):
        if state[box_tar[i][0]][box_tar[i][1]] == 4:
            box_on_T.append([box_tar[i][0],box_tar[i][1]])

    if len(box_on_T) == len(box_tar):
        return 0

    state = to_categorical_tensor(state,box_tar,dim,dim)
    h = nn.model.predict([state.reshape(1,dim,dim,5),goal_state.reshape(1,dim,dim,5)])[1][0][0]
    #action_prob = nn.model.predict([state.reshape(1,dim,dim,5),goal_state.reshape(1,dim,dim,5)])[0][0]
    return h

class BFSNode:
    def __init__(self, stateName, path_cost = None, parentName=None, parent_act = None):
        self.stateName = stateName
        self.path_cost = path_cost
        self.parentName = parentName
        self.parent_act = parent_act


class PriorityQ:
    def __init__(self):
        self.elements = []
    def insert(self,value,element):
        heapq.heappush(self.elements, (value, element))
    def getMin(self):
        return heapq.heappop(self.elements)[1]
    def length(self):
        return len(self.elements)
    def search(self, element):
        for elem in self.elements:
            if elem[1] == element:
                return True
        return False
    def replace(self, h, element):
        p_c = 0
        pos = -1
        for elem in self.elements:
            pos+=1
            if elem[1] == element:

                break
        del self.elements[pos]
        heapq.heappush(self.elements,(h,element))
        return

def Gbfs_lrt(init, box_tar):
    g_state = copy.deepcopy(init)
    goal_state = find_goal_state(g_state, box_tar) #in one-hot encoding
    start = time.time()

    closedSet={}
    openSet={}
    dontExpand=[]
    heap=PriorityQ()

    init = np.reshape(init, (1,dim*dim))[0]
    h=findNN(str(init), box_tar, goal_state)

    openSet[str(init)]=[0,h,h,[],[]]
    state = openSet[str(init)] #Stores the g,h,f,[],[]
    stateName = str(init)
    G.add_node(str(init).replace('\n', ''), o=1, g = 0)

    states_expanded = 0
    while True:
        if (time.time() - start) > 600:
            return None, float("inf")
        if check_goal(stateName,box_tar):
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
            X_Train, Y_Train = create_minibatch(key_states,box_tar,goal_state)
            return X_Train, Y_Train, h_matrix
            break
        closedSet[stateName]= openSet[stateName] #add state to closedSet

        temp_state = evaluate_state(stateName)
        op, act_no, cost = get_neighbors(temp_state, box_tar, dim)
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
                    heap.insert(closedSet[newList][1],newList)
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
                   heap.insert(openSet[newList][1],newList)

            else:
                h= findNN(str(newList), box_tar, goal_state)
                openSet[newList]=[state[0]+cost[i],h,state[0]+h+cost[i],stateName,act_no[i]]
                heap.insert(openSet[newList][1],newList)
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
        goal_key = pred
    key_states.append(str(init_key).replace("\n",''))
    return key_states

def find_constraints(key_states):
    constraints= int(len(key_states)*(len(key_states)-1)/2)
    h_matrix = [[0] * constraints for i in range(len(key_states))]
    return constraints, h_matrix

def create_hmatrix(key_states,h_matrix):
    c=0
    for i in range(len((key_states))):
        for j in range(i+1,len((key_states))):
            if G.nodes[key_states[i]]['g'] > G.nodes[key_states[j]]['g']:
                h_matrix[j][c] = -1
                h_matrix[i][c] = 1
                c+=1
            else:
                h_matrix[i][c] = -1
                h_matrix[j][c] = 1
                c+=1
    return h_matrix

def create_minibatch(key_states, box_tar, goal_state):
    X_Train = []
    Y_Train = []
    for k in key_states:
        X_Train.append(to_categorical_tensor(evaluate_state(k),box_tar,dim,dim))
        Y_Train.append(goal_state)
    return np.array(X_Train), np.array(Y_Train)  #minibatch
