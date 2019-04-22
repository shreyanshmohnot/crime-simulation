import numpy as np
from CrimeWorld import middleMatrix

def greedy_action(grid,radius=-1):
# given grid of rewards, return action 0-4 moving towards highest reward
# radius reduces the grid to focus on center only    
    
    if radius > -1:
        grid = middleMatrix(grid,radius)
    
    ind = np.argmax(grid)
    y,x = np.unravel_index(ind, grid.shape)
    
    # center point
    cx = int(grid.shape[1]/2)
    cy = int(grid.shape[0]/2)
    dx = x-cx
    dy = y-cy
    
    # already at max, do nothing
    if dx ==0 and dy==0:
        return 4
    
    if dx > 0:
        # consider right
        nx = cx+1
        res_x = 1
    elif dx < 0:
        # consider left
        nx = cx-1
        res_x = 3
    else:
        # consider neither
        nx = -1
        
    if dy > 0:
        # consider down
        ny = cy+1
        res_y = 2
    elif dy < 0:
        # consider up
        ny = cy-1
        res_y = 0
    else:
        # only one choice (move left or right)
        return res_x
    
    # also only one choice (up or down)
    if nx == -1:
        return res_y
    
    # otherwise select which neighbor is best
    if grid[ny,cx] > grid[cy,nx]:
        return res_y
    else:
        return res_x
