# grid-domains
Code for NeurIPS 2023 paper Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal

This repository contains the grid domains, namely, Sokoban, Maze-with-teleports and Sliding tile. 

Dependencies

Tensorflow: Version: 2.14.0
Numpy version: 1.21.1

Note: Astar/Gbfs with bellman loss might fail for some higher pip versions.  Try to downgrade pip if that is the case.


Run $pip install networkx to create the search tree graph.


Example usage:

$ cd sokoban\
$ cd train\
$ python3 main.py --alg astar --loss lstar
