import numpy as np

def get_test_data():
    all_states=[]
    f=open("states10test.txt", "r")
    array_s=[]
    duplicate = []
    index = []

    i=0
    for line in f:
        array_s.append([int(x) for x in line.split()])
        #print(i)
        if array_s[i] not in duplicate:
            arr=np.asarray(array_s[i])
            temp=np.reshape(arr, (10,10))
            all_states.append(temp)
            duplicate.append(array_s[i])
            index.append(i)

        i+=1

    return all_states
    f.close()
