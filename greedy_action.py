def greedy_action(grid):
    ind = np.argmax(grid)
    y,x = np.unravel_index(ind, grid.shape)
    c = int(grid.shape[0]/2)
    dx = x-c
    dy = y-c
    if dx ==0 and dy==0:
        return 4
    nx = c-1
    res_x = 3
    if dx>0:
        nx = c+1
        res_x = 1
    
    ny = c-1
    res_y = 2
    if dy>0:
        ny = c+1
        res_y = 0
    if grid[nx, c] > grid[c, ny]:
        return res_x
    else:
        return res_y
