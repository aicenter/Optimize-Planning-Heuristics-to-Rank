import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import to_categorical

def to_categorical_tensor(x3d,portal,dim) :#portal is a dictionary
      x1d = x3d.ravel()
      y1d = to_categorical(x1d, 8)
      y4d = y1d.reshape([dim, dim, 8])
      #4
      find_p = portal['4']
      y4d[find_p[0][0]][find_p[0][1]][4] = 1
      y4d[find_p[1][0]][find_p[1][1]][4] = 1
      #5
      find_p = portal['5']
      y4d[find_p[0][0]][find_p[0][1]][5] = 1
      y4d[find_p[1][0]][find_p[1][1]][5] = 1
      #6
      find_p = portal['6']
      y4d[find_p[0][0]][find_p[0][1]][6] = 1
      y4d[find_p[1][0]][find_p[1][1]][6] = 1
      #7
      find_p = portal['7']
      y4d[find_p[0][0]][find_p[0][1]][7] = 1
      y4d[find_p[1][0]][find_p[1][1]][7] = 1

      return y4d
      
