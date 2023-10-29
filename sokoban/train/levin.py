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
from get_neighbours import get_neighbors
from getSokobanData import get_data
# Load a data set

states, actions = get_data()
#sample 2000


dim = 10
nn = NN(dim)
levin_n = LevinN(dim)
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
        return 0, ""

    state = to_categorical_tensor(state,box_tar,dim,dim)
    h = nn.model.predict([state.reshape(1,dim,dim,5),goal_state.reshape(1,dim,dim,5)])[1][0][0]
    action_prob = nn.model.predict([state.reshape(1,dim,dim,5),goal_state.reshape(1,dim,dim,5)])[0][0]
    return h, action_prob

def get_final_actions(actions):
    act=[]
    for i in actions:
        if i == 1: #push up
          a = [0] * 8
          a[0]=1

        if i == 2: #push down
          a =[1] * 8
          a[1]=1

        if i== 3: #push left
          a = [0] * 8
          a[2]=1

        if i == 4: #push right
          a = [0] * 8
          a[3]=1

        if i== 5: #move up
            a = [0] * 8
            a[4]=1

        if i ==6: #move down
          a = [0] * 8
          a[5]=1

        if i == 7: #move left
          a = [0] * 8
          a[6] = 1

        if i == 8: #move right
          a = [0] * 8
          a[7]=1
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

def levin(init, box_tar):
    g_state = copy.deepcopy(init)
    goal_state = find_goal_state(g_state, box_tar) #in one-hot encoding
    start = time.time()
    find_player = np.where(init == 3)
    player_pos = list(zip(find_player[0], find_player[1]))[0]

    closedSet={}
    openSet={}
    dontExpand=[]
    search_dict = {}
    heap=PriorityQ()

    find_box_pos = np.where(init == 4)
    box_pos=list(zip(find_box_pos[0], find_box_pos[1]))
    #Initialise
    copy_init=copy.deepcopy(init)
    init = np.reshape(init, (1,dim*dim))[0]
    h, log_action_prob = findNN(str(init), box_tar, goal_state)

    state = BFSNode(str(init), path_cost = 0, log_path_prob = 0, log_action_prob = log_action_prob)
    search_dict[str(init)] = BFSNode(str(init), path_cost = 0, log_path_prob = 0, log_action_prob = log_action_prob)

    stateName = str(init)#mutable?

    while True:
        if (time.time() - start) > 600:
            return None, float("inf")
        #print(stateName,"\n")
        if check_goal(stateName,box_tar):
            end = time.time()
            states=[]
            actions = []
            while stateName!=str(init):
                  states.append(evaluate_state(search_dict[stateName].parentName))
                  actions.append(search_dict[stateName].parent_act)
                  stateName=search_dict[stateName].parentName
            Actions = get_final_actions(actions)
            X_Train, Y_Train = create_minibatch(states,box_tar,goal_state)
            return X_Train, Y_Train, np.array(Actions), np.arange(1,len(X_Train)+1)
            break

        temp_state = evaluate_state(stateName)
        op, act_no, cost = get_neighbors(temp_state, box_tar, dim)

        for i in  range(0,len(op)):

            newList=str(op[i].reshape(1,dim*dim)[0]).replace("\n",'')   #next_state_id
            path_cost = state.path_cost + cost[i]
            log_path_prob = state.log_path_prob + state.log_action_prob[act_no[i]-1]
            pred_h, next_action_prob = findNN(newList, box_tar, goal_state)
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

def create_minibatch(states, box_tar, goal_state):
    X_Train = []
    Y_Train = []
    for k in states:
        X_Train.append(to_categorical_tensor(k,box_tar,dim,dim))
        Y_Train.append(goal_state)
    return np.array(X_Train), np.array(Y_Train)  #minibatch




optimizer = tf.keras.optimizers.Adam()


def levin_train(X_Train,Y_Train,act,h):
    levin_n.model.compile(optimizer='adam',
                         loss=['categorical_crossentropy','mse'],
                          metrics=['accuracy'])
    levin_n.model.fit([X_Train,Y_Train], [act,h], epochs=1, batch_size=len(X_Train))
    #nn.model.save_weights('sokoban_param_levin')

for i in range(5000):
    index = np.random.permutation(200)[:200]#32016
    sample_states = [states[p] for p in index]
    #sample_actions = [actions[i] for i in index]

    for sample in range(200):
        state = sample_states[sample]
        find_box_tar = np.where(state == 2)
        box_tar=list(zip(find_box_tar[0], find_box_tar[1]))
        X_Train, Y_Train, act, heur = levin(state, box_tar)
        levin_train(X_Train,Y_Train,act,heur)

#h = tf.convert_to_tensor(np.array([1, 2, 3, 4, 5]),dtype = tf.float32)
#act = tf.convert_to_tensor(np.array([[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0]]),dtype=tf.int64)
