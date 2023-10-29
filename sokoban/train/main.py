#problem-sokoban

# 0 - wall, 1 - empty spaces, 2- box targets, 3- sokoban , 4 - box positions.
import numpy as np
import argparse
from train import expand_search_algs
from getSokobanData import get_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', action='store', dest='heur_search_algorithm',
                        help='Name of the search algorithm (astar, gbfs')
    parser.add_argument('--loss', action='store', dest='loss',
                        help='Name of the loss (lstar, lgbfs, lrt, l2, bellman)')
    parameters = parser.parse_args()

    states = get_data()
    print("Loading data done!")
    #sample 2000
    index = np.random.permutation(1)[:1]#32016
    sample_states = [states[i] for i in index]

    expand_search_algs(sample_states,parameters.heur_search_algorithm,parameters.loss)

if __name__ == "__main__":
    main()
