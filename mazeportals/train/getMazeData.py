import numpy as np
def get_data():
    all_states=[]
    #array_a = []

    f=open("mazeportal15states.txt", "r")
    #g=open("mazeportal15actions.txt", "r")

    for line in f:
        arr = [int(x) for x in line.split()]
        arr=np.asarray(arr)
        temp=np.reshape(arr, (15,15))
        all_states.append(temp)

    #for line in g:
    #    array_a.append([int(x) for x in line.split()])

    f.close()
    #g.close()

    return all_states
    
    
