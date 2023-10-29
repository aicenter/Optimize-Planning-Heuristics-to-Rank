import copy
import numpy as np
import heapq
import time
from getMazeData import get_test_data15, get_test_data50, get_test_data60
from nn_struct_new import NN
from get_neighbours import get_neighbors
from train_sok import to_categorical_tensor

class Astar:
    def __init__(self, dim, loss):
        self._dim = dim #Higher dim solutions will be added later.
        self._loss = loss
        self._nn = NN(dim)
        self._nn.model.load_weights("stp_gbfs_" + loss)
        if dim == 15:
           self._states = get_test_data15()
        if dim == 50:
           self._states = get_test_data50()   
        if dim == 60:
           self._states = get_test_data60()   
        self._portal = {}

    def check_goal(self,stateName):
        state = self.evaluate_state(stateName)
        find_player = np.where(state == 2)
        player_pos = list(zip(find_player[0], find_player[1]))[0]
        if (player_pos[0]-1>=0 and state[player_pos[0]-1][player_pos[1]] == 3) or (player_pos[0]+1<=self._dim-1 and state[player_pos[0]+1][player_pos[1]] == 3) or (player_pos[1]-1 >=0 and state[player_pos[0]][player_pos[1]-1] == 3) or (player_pos[1]+1 <=self._dim-1 and state[player_pos[0]][player_pos[1]+1] == 3):
           return True
        else:
           return False

    def evaluate_state(self, stateName):
        arr = []
        sub = stateName[1:-1]
        for i in sub:
            if i.isdigit():
                arr.append(int(i))
        arr = np.array(arr).reshape(self._dim, self._dim)
        return arr

    def findNN(self, stateName):  # Strips state

        if self.check_goal(stateName):
            return 0
        state = self.evaluate_state(stateName)
        old_state = state
        state = to_categorical_tensor(old_state, self._portal, self._dim)
        val = self._nn.model.predict([state.reshape(1, self._dim, self._dim, 8)],verbose=0)[1][0][0]
        return val
        
    def portals(self,init):
        find_4 = np.where(init == 4)
        find_4 = list((zip(find_4[0], find_4[1])))
        self._portal['4'] = find_4

        find_5 = np.where(init == 5)
        find_5 = list((zip(find_5[0], find_5[1])))
        self._portal['5'] = find_5

        find_6 = np.where(init == 6)
        find_6 = list((zip(find_6[0], find_6[1])))
        self._portal['6'] = find_6

        find_7 = np.where(init == 7)
        find_7 = list((zip(find_7[0], find_7[1])))
        self._portal['7'] = find_7    

    def gbfs(self, init):
        start = time.time()

        closedSet = {}
        openSet = {}
        heap = PriorityQ()

        init = np.reshape(init, (1, self._dim * self._dim))[0]
        copy_init = copy.deepcopy(init)
        h = self.findNN(str(init), g_state)

        openSet[str(copy_init)] = [0, h, h, [], []]
        state = openSet[str(copy_init)]  # Stores the g,h,f,[],[]
        stateName = str(copy_init)

        states_expanded = 0
        while True:
            if (time.time() - start) > 600:
                return None, float("inf")
            if self.check_goal(stateName):
                end = time.time()
                actions = []
                closedSet[str(stateName)] = state
                while stateName != str(init):
                    actions.append(closedSet[str(stateName)][4])
                    stateName = closedSet[str(stateName)][3]

                return states_expanded, len(actions)
            closedSet[stateName] = openSet[stateName]  # add state to closedSet

            temp_state = self.evaluate_state(stateName)
            op, act_no, cost = get_neighbors(temp_state, self._portal, self._dim)
            for i in range(0, len(op)):

                newList = str(op[i].reshape(1, self._dim * self._dim)[0])
                if newList in closedSet:

                    if closedSet[newList][0] > state[0] + cost[i]:  # reopening closed states
                        closedSet[newList][0] = state[0] + cost[i]
                        closedSet[newList][2] = state[0] + cost[i] + closedSet[newList][1]
                        closedSet[newList][3] = stateName
                        closedSet[newList][4] = act_no[i]  # or full operator?
                        openSet[newList] = closedSet[newList]
                        heap.insert(closedSet[newList][1], newList)
                        states_expanded += 1

                elif newList in openSet:
                    if openSet[newList][0] > state[0] + cost[i]:
                        openSet[newList][0] = state[0] + cost[i]
                        openSet[newList][2] = state[0] + cost[i] + openSet[newList][1]
                        openSet[newList][3] = stateName
                        openSet[newList][4] = act_no[i]  # or full operator?
                        heap.insert(openSet[newList][1], newList)

                else:
                    h = self.findNN(str(newList))
                    openSet[newList] = [state[0] + cost[i], h, state[0] + h + cost[i], stateName, act_no[i]]
                    heap.insert(openSet[newList][1], newList)
                    states_expanded += 1

            if heap.length() == 0:
                return float("inf")
            stateName = heap.getMin()
            state = openSet[stateName]

    def run_Gbfs(self):
        unsolved = []
        for i in range(len(self._states)):
            self._portal = self.portals(self._states[i])
            expanded, plan_len = self.gbfs(self._states[i])
            self._portal.clear()
            if expanded == None:
                unsolved.append(i)
        print("solved Stp mazes", 200 - len(unsolved))


class PriorityQ:
    def __init__(self):
        self.elements = []

    def insert(self, value, element):
        heapq.heappush(self.elements, (value, element))

    def getMin(self):
        return heapq.heappop(self.elements)[1]

    def length(self):
        return len(self.elements)


