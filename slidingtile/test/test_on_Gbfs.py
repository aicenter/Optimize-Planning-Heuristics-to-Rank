import copy
import numpy as np
import heapq
import time
from gettiledata import get_test_data5, get_test_data6, get_test_data7
from nn_struct_new import NN
from get_neighbours import get_neighbors
from train_sok import to_categorical_tensor


class Astar:
    def __init__(self, dim, loss):
        self._dim = dim #Higher dim solutions will be added later.
        self._loss = loss
        self._nn = NN(dim)
        self._nn.model.load_weights("stp_gbfs_" + loss)
        if dim == 5:
           self._states = get_test_data5()
        if dim == 6:
           self._states = get_test_data6()   
        if dim == 7:
           self._states = get_test_data7()

    def find_goal_state(self):
        arr = [[0 for i in range(self._dim)] for j in range(self._dim)]
        k = 1
        for row in range(0, self._dim):
            for col in range(0, self._dim):
                if row == self._dim - 1 and col == self._dim - 1:
                    arr[row][col] == 0
                    continue
                arr[row][col] = k
                k = k + 1
        return (to_categorical_tensor(np.array(arr), self._dim)), np.array(arr)

    def check_goal(self, stateName):
        state = self.evaluate_state(stateName)
        _, goal = self.find_goal_state()
        if np.array_equal(state, goal):
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

    def findNN(self, stateName, goal_state):  # Strips state

        if self.check_goal(stateName):
            return 0
        state = self.evaluate_state(stateName)
        old_state = state
        state = to_categorical_tensor(old_state, self._dim)
        val = self._nn.model.predict(
            [state.reshape(1, self._dim, self._dim, 25), goal_state.reshape(1, self._dim, self._dim, 25)])[1][0][0]
        return val

    def gbfs(self, init):
        g_state, _ = np.array(self.find_goal_state())  # in one-hot encoding
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
            op, act_no, cost = get_neighbors(temp_state, self._dim)
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
                    h = self.findNN(str(newList), g_state)
                    openSet[newList] = [state[0] + cost[i], h, state[0] + h + cost[i], stateName, act_no[i]]
                    heap.insert(openSet[newList][1], newList)
                    # print("val", openSet[newList][2])
                    states_expanded += 1

            if heap.length() == 0:
                return float("inf")
            stateName = heap.getMin()
            state = openSet[stateName]

    def run_Gbfs(self):
        unsolved = []
        for i in range(len(self._states)):
            expanded, plan_len = self.gbfs(self._states[i])
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


