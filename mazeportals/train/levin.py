import numpy as np
import copy
import time
import heapq
from nn_struct_old import NN
from math import inf
import tensorflow as tf
from levin_nn import LevinN
from tensorflow.keras.activations import softplus
from train_sok import to_categorical_tensor
from get_neighbours_levin import get_neighbors
from getMazeData import get_data
# Load a data set

states = get_data()

dim = 15
nn = NN(dim)
levin_n = LevinN(dim)
nn.model.load_weights('por5')

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

def findNN(stateName, portal):#Strips state
    if check_goal(stateName):
       return 0, ""
    state = evaluate_state(stateName)
    old_state = state
    state = to_categorical_tensor(old_state,portal,dim)
    h = nn.model.predict([state.reshape(1,dim,dim,8)],verbose=1)[1][0][0]
    action_prob = nn.model.predict([state.reshape(1,dim,dim,8)],verbose=1)[0][0]
    return h, action_prob

def get_final_actions(actions):
    act=[]
    for i in actions:
        if i == 1: #go up
          a = [0] * 4
          a[0]=1

        if i == 2: #go down
          a =[1] * 4
          a[1]=1

        if i== 3: #go left
          a = [0] * 4
          a[2]=1

        if i == 4: #go right
          a = [0] * 4
          a[3]=1
          
        act.append(a)

    return act



class BFSNode:
    def __init__(self, stateName, path_cost = None, log_path_prob = None, log_action_prob = None, parentName=None, parent_act = None):
        self.stateName = stateName
        self.path_cost = path_cost
        self.parentName = parentName
        self.log_path_prob = log_path_prob
        self.log_action_prob = log_action_prob
        self.parent_act = parent_act


class PriorityQ:
    def __init__(self):
        self.elements = []
    def insert(self,value,h,element):
        heapq.heappush(self.elements, (value, h, element))
    def getMin(self):
        return heapq.heappop(self.elements)[2]
    def length(self):
        return len(self.elements)
    def search(self, element):
        for elem in self.elements:
            if elem[2] == element:
                return True
        return False
    def replace(self, cost_diff, element):
        p_c = 0
        pos = -1
        for elem in self.elements:
            pos+=1
            if elem[2] == element:
                p_c=elem[0]-cost_diff
                h = elem[1]
                break
        del self.elements[pos]
        heapq.heappush(self.elements,(p_c,h,element))
        return

def levin(init, portal):

    start = time.time()
    find_player = np.where(init == 2)
    player_pos = list(zip(find_player[0], find_player[1]))[0]

    closedSet={}
    openSet={}
    dontExpand=[]
    search_dict = {}
    heap=PriorityQ()

    #Initialise
    copy_init=copy.deepcopy(init)
    init = np.reshape(init, (1,dim*dim))[0]
    h, log_action_prob = findNN(str(init), portal)

    state = BFSNode(str(init), path_cost = 0, log_path_prob = 0, log_action_prob = log_action_prob)
    search_dict[str(init)] = BFSNode(str(init), path_cost = 0, log_path_prob = 0, log_action_prob = log_action_prob)

    stateName = str(init)

    while True:
        if (time.time() - start) > 600:
            return None, float("inf")
        #print(stateName,"\n")
        if check_goal(stateName):
            end = time.time()
            states=[]
            actions = []
            while stateName!=str(init):
                  states.append(evaluate_state(search_dict[stateName].parentName))
                  actions.append(search_dict[stateName].parent_act)
                  stateName=search_dict[stateName].parentName
            Actions = get_final_actions(actions)
            X_Train = create_minibatch(states,portal)
            return X_Train, np.array(Actions), np.arange(1,len(X_Train)+1)
            break

        temp_state = evaluate_state(stateName)
        op, act_no, cost = get_neighbors(temp_state, portal, dim)
        
        for i in  range(0,len(op)):

            newList=str(op[i].reshape(1,dim*dim)[0]).replace("\n",'')   #next_state_id
            path_cost = state.path_cost + cost[i]
            log_path_prob = state.log_path_prob + state.log_action_prob[act_no[i]-1]
            pred_h, next_action_prob = findNN(newList, portal)
            #print(pred_h,next_action_prob)
            #if pred_h == 0:
            #    x = []
            #    x.append(evaluate_state(newList))
            #    while stateName != str(init):
            #        x.append(evaluate_state(stateName))
            #        stateName = search_dict[stateName].parentName
            #    x.append(evaluate_state(str(init)))
            #    return x
            #print(evaluate_state(newList))
            #action_prob_fn= lambda x: np.log(x)
            #next_action_prob = action_prob_fn(next_action_prob)
            next_state= search_dict.setdefault(newList, BFSNode(str(newList), path_cost = inf, log_path_prob = log_path_prob, log_action_prob = next_action_prob))
            cost_diff = next_state.path_cost - path_cost
            #print(evaluate_state(newList),"\n")# search_dict[newList].path_cost,cost_diff,search_dict[newList].log_path_prob
            if cost_diff > 0:
                next_state.parentName = state.stateName
                next_state.path_cost = path_cost
                next_state.parent_act = act_no[i]
                if heap.search(newList) == False:
                    levin_cost = get_levin_cost(path_cost, log_path_prob, pred_h)
                    heap.insert(levin_cost,pred_h,newList)
                else:
                    heap.replace(cost_diff,newList)

        stateName = heap.getMin()
        state = search_dict[stateName]

def get_levin_cost(path_cost, log_path_prob, predicted_h):
    predicted_h = max(0, predicted_h)
    return np.log(predicted_h + path_cost) - log_path_prob

def create_minibatch(states, portal):
    X_Train = []
    for k in states:
        X_Train.append(to_categorical_tensor(k,portal,dim))
    return np.array(X_Train)  #minibatch


optimizer = tf.keras.optimizers.Adam()


def levin_train(X_Train,act,h):
    levin_n.model.compile(optimizer='adam',
                         loss=['categorical_crossentropy','mse'],
                          metrics=['accuracy'])
    levin_n.model.fit(X_Train, [act,h], epochs=1, batch_size=len(X_Train))
    nn.model.save_weights('maze_param_levin')

for i in range(5000):
    index = np.random.permutation(200)[:200]#32016
    sample_states = [states[p] for p in index]
    #sample_actions = [actions[i] for i in index]

    for sample in range(200):
        init = sample_states[sample]
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
        
        X_Train, act, heur = levin(init, portal)
        levin_train(X_Train,act,heur)


