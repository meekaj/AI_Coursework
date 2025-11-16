#write a* alg here
'''
for every square on grid, a* calcs
    1. g(n) how far have i travelled from start to get here
    2. h(n) how far am i from the goal
    3. f(n) = g(n) = h(n) total score for the square,
a* will choose square with lowest f(n)
'''
import heapq

def manhattan (a, b):
    #row and coloumn 
    (r1, c1) = a
    (r2, c2) = b
    return abs(r1-r2) + abs(c1-c2)

def build_portal_map(portals):
    #creating a portal dictionary, only used by algorithm
    portal_map = {}

    for key, (r, c) in portals.items():
        if key.startswith("S"): #find start portals
            end_key = "E" + key[1:] #matching end key
            if end_key in portals:
                portal_map [(r,c)] = portals[end_key]

    return portal_map


def get_neighbors(cell, desc, portal_map):
    #return cells we can move to (no walls or portal start?)
    (r, c) = cell
    nrow, ncol = desc.shape #dimensions 

    #cell is a portal start, outcome is teleport
    if (r, c) in portal_map:
        return [portal_map[(r, c)]]
    #otherwised, normal 4 directional moves
    neighbours = []
    candidates = [
        (r, c-1), #left
        (r, c+1), #right
        (r+1, c), #down
        (r-1, c) #up
    ]

    for (nr, nc) in candidates:
        #check if in grid boundary 
        if 0 <= nr < nrow and 0 <= nc <ncol:
            if desc[nr,nc] != "W":
                neighbours.append ((nr, nc))
    
    return neighbours

def reconstruct_path(came_from, current):
    #to rebuild shortest path after a* finds the goal.
    #a star does not build path while searching it only records parents so once goal reached follow parent links backwards and reverse the list
    path = [current]

    while current in came_from:
        current = came_from[current] #move to parent
        path.append(current)
    path.reverse() #reverse list to have in order
    return path

def astar_search (desc, portals):
    #wish to return list of cells from start to goal or nothing if we can't find a path

    nrow, ncol = desc.shape

    start =(0, 0) #start state I is the top-left
    goal = (nrow-1, ncol -1) #end goal is the bottom-right cell

    portal_map = build_portal_map(portals)

    #a queue of cells we might explore 
    open_heap=[]
    heapq.heappush(open_heap, (0, 0, start))
    came_from = {}
    g_score = {start: 0}

    #alg start here, explore while there are cells in the open set
    while open_heap:
        #pop cell with smallest f score
        f,g, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(came_from, current)
        
        #otherwise we look at all the neighbours of our current cell
        for neighbour in get_neighbors(current, desc, portal_map):
            temp_g = g_score[current] + 1 #each move has a cost of 1
            #if we found a cheaper path, update record or if we have not seen it 
            if neighbour not in g_score or temp_g < g_score[neighbour]:
                g_score[neighbour] = temp_g
                f_score = temp_g + manhattan(neighbour, goal) #calc estimated cost f = g+ h 
                came_from[neighbour] = current #remember how we got to this neighbour for contructing our path
                heapq.heappush(open_heap, (f_score, temp_g, neighbour))
    
    return None #if we exit this loop, there couldnt of been a path to the goal