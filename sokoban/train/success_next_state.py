def next_state(old_state, action, box_tar, box_on_T, player_pos, dim):
    no_of_boxes = len(box_tar)
    #print(old_state)
    if action == 1:#Push Up
        if player_pos[0]-2 >=0 and (old_state[player_pos[0]-2][player_pos[1]] == 1 or old_state[player_pos[0]-2][player_pos[1]] == 2) and old_state[player_pos[0]-1][player_pos[1]] == 4:
            if old_state[player_pos[0]-2][player_pos[1]] == 2:
                if [player_pos[0]-2,player_pos[1]] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0]-2,player_pos[1]])
                    #print(box_on_T)
                if  [player_pos[0]-1,player_pos[1]] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0]-1,player_pos[1]])
                    #print(box_tar[0])

                if old_state[player_pos[0]-2][player_pos[1]] == 2:
                    box_on_T.append([player_pos[0]-2,player_pos[1]])
                    #print(box_on_T)
                #old_state[player_pos[0]-2][player_pos[1]] = 0
            if  [player_pos[0]-1,player_pos[1]] in box_on_T:
                #on_target-=1
                box_on_T.remove([player_pos[0]-1,player_pos[1]])
                #print(box_tar[0])
            old_state[player_pos[0]-2][player_pos[1]] = 4

            old_state[player_pos[0]-1][player_pos[1]] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1
            #check if over

            #create_inter(state)
            #time.sleep(2)

    elif action == 2:#push down
        if player_pos[0]+2 <=dim-1 and (old_state[player_pos[0]+2][player_pos[1]] == 1 or old_state[player_pos[0]+2][player_pos[1]] == 2) and old_state[player_pos[0]+1][player_pos[1]] == 4:
            if old_state[player_pos[0]+2][player_pos[1]] == 2:
                if [player_pos[0]+2,player_pos[1]] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0]+2,player_pos[1]])
                    #print(box_on_T)
                if [player_pos[0]+1,player_pos[1]] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0]+1,player_pos[1]])
                    #print(box_on_T)

                if old_state[player_pos[0]+2][player_pos[1]] == 2:
                    box_on_T.append([player_pos[0]+2,player_pos[1]])
                    #print(box_on_T)
            if [player_pos[0]+1,player_pos[1]] in box_on_T:
                #on_target-=1
                box_on_T.remove([player_pos[0]+1,player_pos[1]])
                #print(box_on_T)
            old_state[player_pos[0]+2][player_pos[1]] = 4
            old_state[player_pos[0]+1][player_pos[1]] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1
            #check if over

            #create_inter(state)
            #time.sleep(2)

    elif action == 3:#push left
        if player_pos[1]-2 >=0 and (old_state[player_pos[0]][player_pos[1]-2] == 1 or old_state[player_pos[0]][player_pos[1]-2] == 2) and old_state[player_pos[0]][player_pos[1]-1] == 4:
            if old_state[player_pos[0]][player_pos[1]-2] == 2:
                if [player_pos[0],player_pos[1]-2] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0],player_pos[1]-2])
                    #print(box_on_T)
                if [player_pos[0],player_pos[1]-1] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0],player_pos[1]-1])
                    #print(box_on_T)

                if old_state[player_pos[0]][player_pos[1]-2] == 2:
                    box_on_T.append([player_pos[0],player_pos[1]-2])
                    #print(box_on_T)
                #old_state[player_pos[0]][player_pos[1]-2] = 0
            if [player_pos[0],player_pos[1]-1] in box_on_T:
                #on_target-=1
                box_on_T.remove([player_pos[0],player_pos[1]-1])
                #print(box_on_T)
            old_state[player_pos[0]][player_pos[1]-2] = 4
            old_state[player_pos[0]][player_pos[1]-1] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1
            #check if over

            #create_inter(state)
            #time.sleep(2)

    elif action == 4:#push right
        if player_pos[1]+2 <=dim-1 and (old_state[player_pos[0]][player_pos[1]+2] == 1 or old_state[player_pos[0]][player_pos[1]+2] == 2) and old_state[player_pos[0]][player_pos[1]+1] == 4:
            if old_state[player_pos[0]][player_pos[1]+2] == 2:
                if [player_pos[0],player_pos[1]+2] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0],player_pos[1]+2])
                    #print(box_on_T)
                if [player_pos[0],player_pos[1]+1] in box_on_T:
                    #on_target-=1
                    box_on_T.remove([player_pos[0],player_pos[1]+1])
                    #print(box_on_T)

                if old_state[player_pos[0]][player_pos[1]+2] == 2:
                    box_on_T.append([player_pos[0],player_pos[1]+2])
                    #print(box_on_T)
                #old_state[player_pos[0]][player_post[1]+2] = 0
            if [player_pos[0],player_pos[1]+1] in box_on_T:
                #on_target-=1
                box_on_T.remove([player_pos[0],player_pos[1]+1])
                #print(box_on_T)
            old_state[player_pos[0]][player_pos[1]+2] = 4
            old_state[player_pos[0]][player_pos[1]+1] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1
            #check if over

            #create_inter(state)
            #time.sleep(2)

    elif action == 5:#Move up
        if player_pos[0]-1 >=0 and (old_state[player_pos[0]-1][player_pos[1]] == 1 or old_state[player_pos[0]-1][player_pos[1]] == 2):
            old_state[player_pos[0]-1][player_pos[1]] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1
            #check if over

            #create_inter(state)
            #time.sleep(2)

    elif action == 6: #Move down
        if player_pos[0]+1 <=dim-1 and (old_state[player_pos[0]+1][player_pos[1]] == 1 or old_state[player_pos[0]+1][player_pos[1]] == 2):
            old_state[player_pos[0]+1][player_pos[1]] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1
            #check if over

            #create_inter(state)
            #time.sleep(2)

    elif action == 7:
        if player_pos[1]-1 >=0 and (old_state[player_pos[0]][player_pos[1]-1] == 1 or old_state[player_pos[0]][player_pos[1]-1] == 2):
            old_state[player_pos[0]][player_pos[1]-1] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1


    elif action == 8:
        if player_pos[1]+1 <=dim-1 and (old_state[player_pos[0]][player_pos[1]+1] == 1 or old_state[player_pos[0]][player_pos[1]+1] == 2):
            old_state[player_pos[0]][player_pos[1]+1] = 3
            if player_pos in box_tar:
                old_state[player_pos[0]][player_pos[1]] = 2
            else:
                old_state[player_pos[0]][player_pos[1]] = 1

    if no_of_boxes==3:
       if list(box_tar[0]) in box_on_T and list(box_tar[1]) in box_on_T and list(box_tar[2]) in box_on_T:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          return old_state, box_on_T, True
       return old_state, box_on_T, False
    else:
       if list(box_tar[0]) in box_on_T and list(box_tar[1]) in box_on_T and list(box_tar[2])in box_on_T and list(box_tar[3]) in box_on_T:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          return old_state, box_on_T, True
       return old_state, box_on_T, False
