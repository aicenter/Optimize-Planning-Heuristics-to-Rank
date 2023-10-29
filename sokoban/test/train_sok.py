import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import to_categorical

def to_categorical_tensor(x3d,Tar,dim1,dim2) :
      find_box_pos = np.where(x3d == 4)
      pos=list(zip(find_box_pos[0], find_box_pos[1]))
      x1d = x3d.ravel()

      y1d = to_categorical( x1d, 5 )

      y4d = y1d.reshape( [dim1, dim2, 5])

      for i in range(len(pos)):
          y4d[Tar[i][0]][Tar[i][1]][2] = 1
          y4d[pos[i][0]][pos[i][1]][2] = 1
      return y4d
