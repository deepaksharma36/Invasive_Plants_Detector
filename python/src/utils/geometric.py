
def check_overlapping(coor_1, coor_2):
    dx = 0
    dy = 0
    #bb_map[panoramaId] = (tlX,tlY,brX,brY)
    if (coor_1[2] > coor_2[0] and coor_1[2] < coor_2[2]):
        dx = coor_1[2] - max(coor_2[0], coor_1[0])

    elif (coor_1[0] > coor_2[0] and coor_1[0] < coor_2[2]):
        dx = min(coor_2[2], coor_1[2]) - coor_1[0]
    else:
        #print("first ok")
        return False, dx*dy
        #dx = coor_1[2] - max(coor_2[0], coor_1[0])

    if (coor_1[3] > coor_2[1] and coor_1[3] < coor_2[3]):
        #return False, dx*dy
        dy = coor_1[3] - max(coor_2[1], coor_1[1])

    elif (coor_1[1] > coor_2[1] and coor_1[1] < coor_2[3]):
        dy = min(coor_2[3], coor_1[3]) - coor_1[1]
        #return False, dx*dy
    else:
        return False, dx*dy


    return True, dx*dy
