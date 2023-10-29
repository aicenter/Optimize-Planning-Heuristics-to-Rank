import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import to_categorical

def to_categorical_tensor(x3d,dim) :#portal is a dictionary
      x1d = x3d.ravel()
      y1d = to_categorical(x1d, 25)
      y4d = y1d.reshape([dim, dim, 25])
      

      return y4d
      
