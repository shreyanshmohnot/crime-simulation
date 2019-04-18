import numpy as np

# helper function
def neighborMat(A):
    # 4-neighbor sum (edge repeated)
    B = np.zeros(A.shape)
    B[1:,] += A[:-1,]       # up
    B[:-1,] += A[1:,]       # down
    B[:,1:] += A[:,:-1]     # left
    B[:,:-1] += A[:,1:]     # right

#     # correct edges (move out of bound => return to same spot)
#     B[0,] += A[0,]          # up out of bounds
#     B[-1,] += A[-1,]        # down out of bounds
#     B[:,0] += A[:,0]        # left out of bounds
#     B[:,-1] += A[:,-1]      # right out of bounds

    # correct edges (periodic)
    B[0,] += A[-1,]         # up out of bounds
    B[-1,] += A[0,]         # down out of bounds
    B[:,0] += A[:,-1]       # left out of bounds
    B[:,-1] += A[:,0]       # right out of bounds

    return B

# center grid on given x,y coord (dim 0 = y)
def centeredGrid(A,x,y):
    sz = A.shape
    dx = int(sz[1]/2-x)     # floor but returns int type
    dy = int(sz[0]/2-y)
    return np.roll(A,(dy,dx),axis=(0,1))

# return center of matrix based on radius
def middleMatrix(A,r):
    sz = A.shape
    cx = int(sz[1]/2)     # floor but returns int type
    cy = int(sz[0]/2)
    return A[cy-r:cy+r+1,cx-r:cx+r+1]
    

class CrimeWorld():
    # TODO: accept parameter input
    def __init__(self):
        self.set_params(-1,-1)
        self.new_episode()
    
    def set_params(self,x,y):
        # default parameters from Mohler
        # world parameters
        self.M = 128                    # world size (odd to allow centering)
        self.dt = 1.0/100.0             # time step
        
        # burglar parameters
        self.w = 1.0/15.0               # repeat victimization time scale
        self.gamma = 0.019              # burglar spawn rate
        self.eta = 0.03                 # neighbor influence (diffusion rate)
        self.theta = 0.56               # crime effect on attractiveness
        self.A0par = 1.0/30.0           # baseline attractiveness
        
        # police parameters
        self.psi = self.theta           # police presence effect on attractiveness
        self.w2 = self.w                # time decay
        self.eta2 = self.eta            # neighbor influence (diffusion rate)
        self.policeX0 = x               # police starting location (when reset)
        self.policeY0 = y
       
    def reset(self):
        data = np.load("resultFile.npz")
        self.B = data['B']
        self.n = data['n']
        self.P = data['P']
        self.C = data['C']
        self.D = data['D']
        self.totalC = data['totalC']
        self.C_buf = data['C_buf']
        self.window = data['window']
        
    def save_checkpoint(self, file):
        np.savez(file, B=self.B, C=self.C, n=self.n, P=self.P, D=self.D, totalC=self.totalC, C_buf=self.C_buf, window=self.window)  
        
    def new_episode(self):
        sz = (self.M,self.M)        
        # reset world
        self.B = np.zeros(sz)           # dynamic attractiveness
        self.A0 = self.A0par*np.ones(sz)# baseline attractiveness
        self.n = np.zeros(sz)           # burglar count
        self.C = np.zeros(sz)           # crime count from last step
        self.totalC = np.zeros(sz)      # running total of crime (windowed)
        
        # running sum of crime
        self.crime_window = 100
        self.window = 0
        self.C_buf = np.zeros((sz[0],sz[1],self.crime_window))
        
        # police
        self.policeX = self.policeX0    # police current location
        self.policeY = self.policeY0
        self.D = np.zeros(sz)           # deterrence by police presence (like B)
        self.P = np.zeros(sz)           # police count (like n)
        if self.policeX > -1:
            self.P[self.policeY,self.policeX] += 1
            
    
    def add_agent(self,x,y):
        # TODO: keep vector instead of replacing
        # set home location
        self.policeX0 = x % self.M
        self.policeY0 = y % self.M
        # update current location
        self.policeX = self.policeX0
        self.policeY = self.policeY0
        self.P[self.policeY,self.policeX] += 1
    
    def remove_agents(self):
        # clear all agents
        self.policeX0 = -1
        self.policeY0 = -1
        self.P = np.zeros((self.M,self.M))
    
    # info for agent
    def get_state(self):
        # return state (# crimes and position of police, MxMx2)
#         return np.stack((self.C,self.P),2)

        # return state centered around agent
#         return centeredGrid(self.C,self.policeX,self.policeY)
        return centeredGrid(self.totalC,self.policeX,self.policeY)
        
    
    def step(self,a):
        # returns reward, next state, done)
        return (self.make_action(a),self.get_state(),False)
    
    # perform action for agent
    # given agent and action?
    def make_action(self,a):
        # remove from location
        self.P[self.policeY,self.policeX] -= 1
        
        # TODO: add agent index?
        # up
        if a == 0:
            self.policeY = (self.policeY-1)%self.M
        # right
        elif a == 1:
            self.policeX = (self.policeX+1)%self.M
        # down
        elif a == 2:
            self.policeY = (self.policeY+1)%self.M
        # left
        elif a == 3:
            self.policeX = (self.policeX-1)%self.M
        # do nothing
#         else:
        
        # add in new location
        self.P[self.policeY,self.policeX] += 1
        
        # return reward (currently negative total crime in last iteration)
        # TODO: for multiple agents, wait until all specified
        prevC = self.C
        self.update()
#         return -self.C.sum()
        
        # change in C for whole grid
        deltaC = self.C-prevC
        # sum up only local region around agent (current/new location)
        # radius of local region
        r = 5
#         return -deltaC[self.policeY-r:self.policeY+r,
#                        self.policeX-r:self.policeX+r].sum() -1
        reward = -middleMatrix(deltaC,r).sum()-1
        return reward
    
    def actions(self,agent):
        # TODO: return list of actions?
        return 0
        
    # simulate one time step
    def update(self):
        # update effect of police
        self.D = self.psi*self.P + (1.0-self.w2*self.dt)*( (1-self.eta2)*self.D
                                   + (self.eta2/4)*neighborMat(self.D) )
        
        n_new = np.zeros(self.n.shape)               # temp burglar count
        self.C = np.zeros(self.n.shape)              # crime count (E in paper)
        A = np.maximum(self.A0 + self.B - self.D, 0) # total attractiveness (non-negative)

        # precalculate total neighbor attractiveness for grid
        tA = neighborMat(A)

        # determine burglar action
        nz = np.nonzero(self.n)
        # p_crime Poisson process 1-exp(-A*dt): depends on location only
        p_no_crime = np.exp(-A*self.dt)
        # row,col loop only spots where burlgars are present (n[i,j] > 0)
        for i,j in zip(nz[0],nz[1]):
            # calculate neighbor index once (periodic edge)
            i1 = (i+1)%self.M          # down
            i2 = (i-1)%self.M          # up
            j3 = (j+1)%self.M          # right
            j4 = (j-1)%self.M          # left
            totalA = tA[i,j]

            # probabilities for moving to neighbors (same order as above)
            p1 = A[i1,j] / totalA 
            p2 = p1 + A[i2,j] / totalA 
            p3 = p2 + A[i,j3] / totalA 
#             p4 = p3+A[i,j4] / totalA  # should always be 1

            # loop through burglars to commit crime or move (faster for small count)
            for criminal in range(np.int(self.n[i,j])):
                if(np.random.rand() < p_no_crime[i,j]):
                    # move to neighboring cell
                    u = np.random.rand()
                    if(u < p1):
                        n_new[i1,j] += 1
                    elif(u < p2):
                        n_new[i2,j] += 1
                    elif(u < p3):
                        n_new[i,j3] += 1
                    else:
                        n_new[i,j4] += 1
                else:
                    # increment crime
                    self.C[i,j] += 1
                    

        # criminal count after moving
        self.n = n_new
        # update total crime count (minus count dropping out of window)
        self.totalC += self.C - self.C_buf[:,:,self.window]
        # update crime window buffer
        self.C_buf[:,:,self.window] = self.C
        self.window = (self.window+1)%self.crime_window
        
        # also add criminals to system
        # 1-D version of n: updates n
        flat_n = self.n.ravel()
        # flip coin for all grids
        flat_n[np.random.rand(self.M**2) > np.exp(-self.gamma*self.dt)] += 1

        # update attractiveness based upon recent crimes
        self.B = self.theta*self.C + (1.0-self.w*self.dt)*( (1-self.eta)*self.B
                                   + (self.eta/4)*neighborMat(self.B) )

        # return # crimes, attractiveness, # burglars
        return self.C, self.B, self.n, self.P
    
    
    # for A3C
    def is_episode_finished(self):
        return False