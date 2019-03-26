import numpy as np

# helper functions
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

class CrimeWorld():
    # TODO: accept parameter input
    def __init__(self):
        self.new_episode()
        
    def new_episode(self):
        # default parameters from Mohler
        self.M = 128                    # world size
        self.dt = 1.0/100.0             # time step

        self.w = 1.0/15.0               # repeat victimization time scale
        self.gamma = 0.019              # burglar spawn rate
        self.eta = 0.03                 # neighbor influence (diffusion rate)
        self.theta = 0.56               # crime effect on attractiveness
        A0par = 1.0/30.0                # baseline attractiveness
        
        sz = (self.M,self.M)
        self.B = np.zeros(sz)           # dynamic attractiveness
        self.A0 = A0par*np.ones(sz)     # baseline attractiveness
        self.n = np.zeros(sz)           # burglar count
        self.C = np.zeros(sz)           # crime count from last step
        self.totalC = np.zeros(sz)      # running total of crime
    
    # info for agent
    def get_state(self):
        # return crime stats (also police position(s)?)
        return self.C
    
    # perform action for agent
    # given agent and action?
    def make_action(a):
        # return reward
        return 0
    
    def actions(agent):
        # return list of actions?
        return 0
        
    # simulate one time step
    def update(self):
        n_new = np.zeros(self.n.shape)  # temp burglar count
        self.C = np.zeros(self.n.shape) # crime count (E in paper)
        A = self.A0 + self.B            # total attractiveness

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
                if(p_no_crime[i,j] < np.random.rand()):
                    # increment crime
                    self.C[i,j] += 1 

                else:
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

        # criminal count after moving
        self.n = n_new
        # update total crime count
        self.totalC += self.C

        # also add criminals to system
        # 1-D version of n: updates n
        flat_n = self.n.ravel()
        # flip coin for all grids
        flat_n[np.random.rand(self.M**2) > np.exp(-self.gamma*self.dt)] += 1

        # update attractiveness based upon recent crimes
        self.B = self.theta*self.C + (1.0-self.w*self.dt)*( (1-self.eta)*self.B
                                   + (self.eta/4)*neighborMat(self.B) )

        # return # crimes, attractiveness, # burglars
        return self.C, self.B, self.n