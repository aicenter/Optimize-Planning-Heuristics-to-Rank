import numpy as np

def get_data():
    all_states=[]
    all_actions= []
    f=open("stile.txt", "r")
    #g=open("stileact.txt", "r") #1 - tile up, 2 - tile down, 3 - tile left, 4 - tile right
    array_s=[]
    array_a=[]
    duplicate = []
    index = []

    i=0
    for line in f:
        array_s.append([int(x) for x in line.split()])
        #print(i)
        if array_s[i] not in duplicate:
            arr=np.asarray(array_s[i])
            temp=np.reshape(arr, (5,5))
            all_states.append(temp)
            duplicate.append(array_s[i])
            index.append(i)

        i+=1
    #i = 0
    #for line in g:
    #    if i in index:
    #       print(i)
    #       array_a.append([int(x) for x in line.split()])
    #    i+=1



    return all_states #, array_a
    f.close()
    #g.close()

