import copy
import numpy as np
import heapq
import random
import time
import math
from nn_struct_new import NN
from get_neighbours import get_neighbors
from train_sok import to_categorical_tensor


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

    old_state = state
    state = to_categorical_tensor(old_state,box_tar,dim,dim)
    val = heur_model.model.predict([state.reshape(1,dim,dim,5),goal_state.reshape(1,dim,dim,5)])[1][0][0]

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

def astar(init, box_tar):
    nn = NN(10)
    #nn.model.load_weights('sokoban_parameters')
    g_state = copy.deepcopy(init)
    goal_state = find_goal_state(g_state, box_tar) #in one-hot encoding
    start = time.time()
    find_player = np.where(init == 3)
    player_pos = list(zip(find_player[0], find_player[1]))[0]

    closedSet={}
    openSet={}
    dontExpand=[]
    heap=PriorityQ()

    find_box_pos = np.where(init == 4)
    box_pos=list(zip(find_box_pos[0], find_box_pos[1]))

    copy_init=copy.deepcopy(init)
    init = np.reshape(init, (1,dim*dim))[0]
    h=findNN(str(init), box_tar, goal_state)

    openSet[str(init)]=[0,h,h,[],[]]
    state = openSet[str(init)] #Stores the g,h,f,[],[]
    stateName = str(init)

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
                  actions.append(closedSet[str(stateName)][4])
                  stateName=closedSet[str(stateName)][3]

            return states_expanded, len(actions)
        closedSet[stateName]= openSet[stateName] #add state to closedSet

        temp_state = evaluate_state(stateName)
        op, act_no, cost = get_neighbors(temp_state, box_tar, dim)
        for i in  range(0,len(op)):

            newList=str(op[i].reshape(1,dim*dim)[0])
            if newList in closedSet:

                if closedSet[newList][0]>state[0]+cost[i]:#reopening closed states
                    closedSet[newList][0]=state[0]+cost[i]
                    closedSet[newList][2]=state[0]+cost[i]+closedSet[newList][1]
                    closedSet[newList][3]=stateName
                    closedSet[newList][4]=act_no[i] #or full operator?
                    openSet[newList]=closedSet[newList]
                    heap.insert(closedSet[newList][2],newList)
                    states_expanded+=1

            elif newList in openSet:
               if openSet[newList][0]>state[0]+cost[i]:
                   openSet[newList][0]=state[0]+cost[i]
                   openSet[newList][2]=state[0]+cost[i]+openSet[newList][1]
                   openSet[newList][3]=stateName
                   openSet[newList][4]=act_no[i] #or full operator?
                   heap.insert(openSet[newList][2],newList)

            else:
                h= findNN(str(newList), box_tar, goal_state)
                openSet[newList]=[state[0]+cost[i],h,state[0]+h+cost[i],stateName,act_no[i]]
                heap.insert(openSet[newList][2],newList)
                #print("val", openSet[newList][2])
                states_expanded+=1

        if heap.length()==0:
            return float("inf")
        stateName = heap.getMin()
        state = openSet[stateName]
