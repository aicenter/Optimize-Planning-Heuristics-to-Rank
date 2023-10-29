#problem-sliding tile

import argparse
from train import expand_search_algs
from gettiledata import get_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', action='store', dest='heur_search_algorithm',
                        help='Name of the search algorithm (astar, gbfs')
    parser.add_argument('--loss', action='store', dest='loss',
                        help='Name of the loss (lstar, lgbfs, lrt, l2, bellman)')
    parameters = parser.parse_args()
    print("Training is on 5 x 5 grid size.")
    states = get_data()
    print("Loading states done!")

    expand_search_algs(states,parameters.heur_search_algorithm,parameters.loss)

if __name__ == "__main__":
    main()
