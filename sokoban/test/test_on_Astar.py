import copy
import numpy as np
import heapq
import time
from getSokobanData import get_test_data
from nn_struct_new import NN
from get_neighbours import get_neighbors
from train_sok import to_categorical_tensor

class Astar:

    def __init__(self, dim, loss):
        self._dim = dim
        self._loss = loss
        self._nn = NN(dim)
        self._nn.model.load_weights("sok_astar_"+loss)
        self._states = get_test_data()

    def find_goal_state(self, goal_state, box_tar):
        for row in range(0, self._dim):
            for col in range(0, self._dim):
                if goal_state[row][col] == 3:
                    goal_state[row][col] = 1
                if goal_state[row][col] == 4:
                    goal_state[row][col] = 1
        for row in range(0, self._dim):
            for col in range(0, self._dim):
                if goal_state[row][col] == 2:
                    goal_state[row][col] = 4
        return (to_categorical_tensor(goal_state, box_tar, self._dim, self._dim)), goal

    def check_goal(self, stateName, goal_coords):
        state = self.evaluate_state(stateName)
        find_box_pos = np.where(state == 4)
        box_pos = list(zip(find_box_pos[0], find_box_pos[1]))
        tot_box = len(box_pos)
        for i in range(0, tot_box):
            if box_pos[i] not in goal_coords:
                return False
        return True

    def evaluate_state(self,stateName):
        arr = []
        sub = stateName[1:-1]
        for i in sub:
            if i.isdigit():
                arr.append(int(i))
        arr = np.array(arr).reshape(self._dim, self._dim)
        return arr

    def findNN(self, stateName, box_tar, goal_state):  # Strips state

        state = self.evaluate_state(stateName)
        box_on_T = []

        for i in range(len(box_tar)):
            if state[box_tar[i][0]][box_tar[i][1]] == 4:
                box_on_T.append([box_tar[i][0], box_tar[i][1]])

        if len(box_on_T) == len(box_tar):
            return 0

        old_state = state
        state = to_categorical_tensor(old_state, box_tar, self._dim, self._dim)
        val = self._nn.model.predict([state.reshape(1, self._dim, self._dim, 5), goal_state.reshape(1, self._dim, self._dim, 5)])[1][0][0]
        return val

    def astar(self,init):
        find_box_pos = np.where(init == 2)
        box_tar = list(zip(find_box_pos[0], find_box_pos[1]))

        g_state = copy.deepcopy(init)
        goal_state = self.find_goal_state(g_state, box_tar)  # in one-hot encoding
        start = time.time()

        closedSet = {}
        openSet = {}
        heapf = PriorityQ()
        heapb = PriorityQ()
        init = np.reshape(init, (1, self._dim * self._dim))[0]
        h = self.findNN(str(init), box_tar, goal_state)

        openSet[str(init)] = [0, h, h, [], []]
        state = openSet[str(init)]  # Stores the g,h,f,[],[]
        stateName = str(init)

        states_expanded = 0
        while True:
            if (time.time() - start) > 600:
                return None, float("inf")
            if self.check_goal(stateName, box_tar):
                end = time.time()
                actions = []
                closedSet[str(stateName)] = state
                while stateName != str(init):
                    actions.append(closedSet[str(stateName)][4])
                    stateName = closedSet[str(stateName)][3]

                return states_expanded, len(actions)
            closedSet[stateName] = openSet[stateName]  # add state to closedSet

            temp_state = self.evaluate_state(stateName)
            op, act_no, cost = get_neighbors(temp_state, box_tar, self._dim)
            
            for i in range(0, len(op)):

                newList = str(op[i].reshape(1, self._dim * self._dim)[0])
                if newList in closedSet:

                    if closedSet[newList][0] > state[0] + cost[i]:  # reopening closed states
                        closedSet[newList][0] = state[0] + cost[i]
                        closedSet[newList][2] = state[0] + cost[i] + closedSet[newList][1]
                        closedSet[newList][3] = stateName
                        closedSet[newList][4] = act_no[i]  # or full operator?
                        openSet[newList] = closedSet[newList]
                        heap.insert(closedSet[newList][2], newList)
                        states_expanded += 1

                elif newList in openSet:
                    if openSet[newList][0] > state[0] + cost[i]:
                        openSet[newList][0] = state[0] + cost[i]
                        openSet[newList][2] = state[0] + cost[i] + openSet[newList][1]
                        openSet[newList][3] = stateName
                        openSet[newList][4] = act_no[i]  # or full operator?
                        heap.insert(openSet[newList][2], newList)

                else:
                    h = self.findNN(str(newList), box_tar, goal_state)
                    openSet[newList] = [state[0] + cost[i], h, state[0] + h + cost[i], stateName, act_no[i]]
                    heap.insert(openSet[newList][2], newList)
                    # print("val", openSet[newList][2])
                    states_expanded += 1

            if heap.length() == 0:
                return float("inf")
            stateName = heap.getMin()
            state = openSet[stateName]

    def run_Astar(self):
        unsolved = []
        for i in range(len(self._states)):
            expanded, plan_len = self.astar(self._states[i])
            print(plan_len)
            if expanded == None:
                unsolved.append(i)
        print("solved Sokoban mazes", 200 - len(unsolved))

class PriorityQ:
    def __init__(self):
        self.elements = []
    def insert(self,value,element):
        heapq.heappush(self.elements, (value, element))
    def getMin(self):
        return heapq.heappop(self.elements)[1]
    def length(self):
        return len(self.elements)
        


        
        

