import numpy as np
import copy
import heapq
import networkx as nx
import time

from nn_struct_old import NN
from train_sok import to_categorical_tensor
from get_neighbours import get_neighbors

dim = 5  # dimensions set to 5*5 during training

nn = NN(dim)
nn.model.load_weights('stile')

G = nx.DiGraph()  # attributes of G are global
key_states = []


def find_goal_state():
    r, c = (dim, dim)
    arr = [[0 for i in range(c)] for j in range(r)]
    k = 1
    for row in range(0, dim):
        for col in range(0, dim):
            if row == dim - 1 and col == dim - 1:
                arr[row][col] == 0
                continue
            arr[row][col] = k
            k = k + 1
    return (to_categorical_tensor(np.array(arr), dim)), np.array(arr)


def check_goal(stateName):
    state = evaluate_state(stateName)
    _, goal = find_goal_state()
    if np.array_equal(state, goal):
        return True
    else:
        return False


def evaluate_state(stateName):
    arr = []
    sub = stateName[1:-1]
    for i in sub.split(" "):
        if i.replace('\n', '').isdigit():
            arr.append(int(i))     
    arr = np.array(arr).reshape(dim, dim)
    return arr


def findNN(stateName, goal_state):  # Strips state
    if check_goal(stateName):
        return 0
    state = evaluate_state(stateName)

    old_state = state
    state = to_categorical_tensor(old_state, dim)
    val = nn.model.predict([state.reshape(1, dim, dim, 25), goal_state.reshape(1, dim, dim, 25)],verbose=0)[1][0][0]
    print(val)
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

def Gbfs_l2(init):
    goal_state, _ = find_goal_state()
    start = time.time()

    closedSet = {}
    openSet = {}
    heap = PriorityQ()

    init = np.reshape(init, (1, dim * dim))[0]
    copy_init = copy.deepcopy(init)
    h = findNN(str(init), goal_state)

    openSet[str(copy_init)] = [0, h, h, [], []]
    state = openSet[str(copy_init)]  # Stores the g,h,f,[],[]
    stateName = str(copy_init)
    G.add_node(str(copy_init).replace('\n', ''), o=1, g=0)

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
            X_Train, Y_Train, h_val = create_onehot(key_states,goal_state)
            return X_Train, Y_Train, h_val
            break
        closedSet[stateName]= openSet[stateName] #add state to closedSet

        temp_state = evaluate_state(stateName)
        op, act_no, cost = get_neighbors(temp_state, dim)
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
                h= findNN(str(newList), goal_state)
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

def create_onehot(key_states, goal_state):
    X_Train = []
    Y_Train = []
    H_val = np.arange(0,len(key_states))
    for k in key_states:
        X_Train.append(to_categorical_tensor(evaluate_state(k),dim))
        Y_Train.append(goal_state)
    return np.array(X_Train), np.array(Y_Train), np.array(H_val)  #minibatch
