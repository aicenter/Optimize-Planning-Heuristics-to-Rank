import numpy as np
def get_test_data15():
    all_states=[]

    f=open("mazeportal15statesval.txt", "r")

    for line in f:
        arr = [int(x) for x in line.split()]
        arr=np.asarray(arr)
        temp=np.reshape(arr, (15,15))
        all_states.append(temp)

    f.close()

    return all_states 
    
    
def get_test_data60():
    all_states=[]

    f=open("mazeportal60states", "r")

    for line in f:
        arr = [int(x) for x in line.split()]
        arr=np.asarray(arr)
        temp=np.reshape(arr, (15,15))
        all_states.append(temp)

    f.close()

    return all_states     
    
    
    
def get_test_data50():
    all_states=[]

    f=open("mazeportal50states.txt", "r")

    for line in f:
        arr = [int(x) for x in line.split()]
        arr=np.asarray(arr)
        temp=np.reshape(arr, (15,15))
        all_states.append(temp)

    f.close()

    return all_states     
