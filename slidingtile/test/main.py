#problem- sliding tile

import numpy as np
import argparse
from test_on_Astar import Astar
from test_on_Gbfs import Gbfs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', action='store', dest='heur_search_algorithm',
                        help='Name of the search algorithm (astar, gbfs')
    parser.add_argument('--loss', action='store', dest='loss',
                        help='Name of the loss (lstar, lgbfs, lrt, bellman, levin)')
    parser.add_argument('--dim', action='store', dest='dimension',
                        help='Name of the loss (5, 6, 7)', default=5)
    parameters = parser.parse_args()

    #print("Default testing is on 5 x 5 grid size.")

    if parameters.heur_search_algorithm == "astar":
        results = Astar(parameters.dimension, parameters.loss)
        results.run_Astar()

    if parameters.heur_search_algorithm == "gbfs":
       results = Gbfs(parameters.dimension, parameters.loss)
       results.run_Gbfs()

if __name__ == "__main__":
    main()
