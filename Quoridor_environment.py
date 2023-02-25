import gym
from gym import spaces
import pygame
import numpy as np

def ij2k(i,j,n):
    """From matrix position to index"""
    return int(i*n+j)

def k2ij(k,n):
    """From index to matrix position"""
    return [k//n, k%n]

class QuoridorWorld(gym.Env):
    """
    AI Gym Environment for the Quoridor game. Create the environment, manage reward and step functions, 
    display the current board and the pawns positions
    
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, grid_size = 5, n_fences = 2):
        self.window_size = 512                                              # for display
        self.grid_size = grid_size                                          # board size
        self.n_fences = n_fences                                            # initial number of fences
        self.player_1 = np.array([0, int(self.grid_size/2)])                # initial positions of the players
        self.player_2 = np.array([self.grid_size-1, int(self.grid_size/2)])
        self.r_fences = [self.n_fences,self.n_fences]                       # remainin fences for each player
        self.fences = 2*grid_size*grid_size * [0]                           # manage the position of each fence on the board
        self.neigbours = []                                                 # adjacency matrix of the board
        for i in range(grid_size):
            for j in range(grid_size):
                nb = []
                if i>0:
                    nb.append(ij2k(i-1,j, grid_size))
                if i<grid_size-1:
                    nb.append(ij2k(i+1,j, grid_size))
                if j>0:
                    nb.append(ij2k(i,j-1, grid_size))
                if j<grid_size-1:
                    nb.append(ij2k(i,j+1, grid_size))
                self.neigbours.append(nb)

        for i in range(grid_size):
            k = self.ij2f(grid_size-1, i, grid_size, i)
            self.fences[k] = 1
            k = self.ij2f(i, grid_size-1, i, grid_size)
            self.fences[k] = 1


        self.observation_space = spaces.Dict(                                       # Observation space.
            {
                "player_1": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),    # first player position
                "player_2": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
                "player_1_fences": spaces.Box(0, n_fences, shape=(1,), dtype=int),  # remaining fences of the first player
                "player_2_fences": spaces.Box(0, n_fences, shape=(1,), dtype=int),
                "fences": spaces.Box(0, 3, shape=(n_fences, n_fences), dtype=int)   # fences on the board
            }
        )

        # Each possible action: 1st argument describes the player (1 for the first player, 2 for the second one)
        # and the second argument describes the possible actions: 4 to move and the other one to put fences on the board
        self.action_space = spaces.Box(low = np.array([1,0]), high=np.array([2, 3+2*grid_size*grid_size]), dtype=int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """ returns the state of the game """
        return {"player_1": self.player_1, "player_2": self.player_2, "fences_player_1": self.r_fences[0], "fences_player_2": self.r_fences[1], "fences": self.fences}
    
    def _get_info(self):
        """ returns the minimal distance for each player"""
        return {"player_1": self.dist(1), "player_2": self.dist(2)}
    
    def reset(self, seed=None, options=None):
        """ Reset the game environment after a player won"""

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.player_1 = np.array([0, int(self.grid_size/2)])
        self.player_2 = np.array([self.grid_size-1, int(self.grid_size/2)])
        self.r_fences = [self.n_fences,self.n_fences]
        self.fences = 2*self.grid_size*self.grid_size * [0]
        self.neigbours = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                nb = []
                if i>0:
                    nb.append(ij2k(i-1,j, self.grid_size))
                if i<self.grid_size-1:
                    nb.append(ij2k(i+1,j, self.grid_size))
                if j>0:
                    nb.append(ij2k(i,j-1, self.grid_size))
                if j<self.grid_size-1:
                    nb.append(ij2k(i,j+1, self.grid_size))
                self.neigbours.append(nb)

        for i in range(self.grid_size):
            k = self.ij2f(self.grid_size-1, i, self.grid_size, i)
            self.fences[k] = 1
            k = self.ij2f(i, self.grid_size-1, i, self.grid_size)
            self.fences[k] = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def dist(self, player):
        """return: [dist, path for each ending square] for the given player, with dist = list(int) indicates the distance
        to each objective case, while path = list(tuples) indicates the shortest path to each objective case
        Compute according to Dijkstra's algorithm
        Also checks if each position is reachable"""
        if player == 1:
            init = ij2k(self.player_1[0], self.player_1[1], self.grid_size)
            end = self.grid_size-1
        else:
            init = ij2k(self.player_2[0], self.player_2[1], self.grid_size)
            end = 0
        
        spt = [False] * ((self.grid_size)**2)
        dist = [float('inf')] * ((self.grid_size)**2)
        dist[init] = 0
        path = [[]]*((self.grid_size)**2)
        path[init] = [init]

        for _ in range((self.grid_size)**2):
            
            m = float('inf')
            idx = 0
            for i,k in enumerate(dist):
                if k<m and not spt[i]:
                    m=k
                    idx = i
            
            spt[idx] = True

            for v in self.neigbours[idx]:
                if m+1<dist[v] and not spt[v]:
                    dist[v] = m+1
                    path[v] = path[idx]+[v]

        flag = not np.isfinite(dist).all()
        
        if flag:
            return self.grid_size*[float('inf')], self.grid_size*[0]

        int_dist = []
        int_path = []
        for i in range(self.grid_size):
            int_dist.append(dist[ij2k(end,i, self.grid_size)])
            int_path.append(path[ij2k(end,i, self.grid_size)])

        return int_dist, int_path
    
    def ij2f(self, i1, j1, i2, j2):
        """Convert matrix indices to fence index"""
        if i1 == i2:
            m = min(j1,j2)
            return 2*ij2k(i1, m, self.grid_size)
        else: 
            m = min(i1,i2)
            return 2*ij2k(m, j1, self.grid_size)+1
    
    def f2ij(self, k):
        """ Convert fence index to matrix indices"""
        if k % 2 == 0:
            k = k/2
            r = (k2ij(k, self.grid_size), k2ij(k, self.grid_size))
            r[1][1]+=1
            return r
        else :
            k-=1
            k/=2
            r = (k2ij(k, self.grid_size), k2ij(k, self.grid_size))
            r[1][0]+=1
            return r
    
    def reward(self, p):
        """ Reward function for the player p: only depends on the position of the player and the minimal distance to the objective"""
        if p == 1:
            manh = self.grid_size - 1 - self.player_1[0]   #Manhattan distance
            d = min(self.dist(1)[0])
            manha = self.player_2[0]
            da = min(self.dist(2)[0])
        else:
            manh = self.player_2[0]
            d = min(self.dist(2)[0])
            manha = self.grid_size - 1 - self.player_1[0]
            da = min(self.dist(1)[0])
        return (10 -manh-2*d + 0.5*manha + da)
    
    def step(self, action):
        """ Compute the effect of the chosen action"""

        terminated = False                                                              # indicates if a player won the game
        move = [np.array([1,0]), np.array([0,1]), np.array([-1,0]), np.array([0,-1])]
        if action[1] > 3:

############ The player uses a fence
            if self.r_fences[action[0]-1] > 0:                                          # if the player still have fences
                k = action[1]-4
                if self.fences[k] != 0 :
                    return (self._get_obs(), -20, terminated, True, None)
                else:   
                        self.fences[k] = 1
                        F = self.f2ij(k)
                        self.neigbours[ij2k(F[0][0], F[0][1], self.grid_size)].remove(ij2k(F[1][0], F[1][1], self.grid_size))
                        self.neigbours[ij2k(F[1][0], F[1][1], self.grid_size)].remove(ij2k(F[0][0], F[0][1], self.grid_size))
                        self.r_fences[action[0]-1] -= 1
                        d1 = self.dist(1)[0]
                        if not np.isfinite(d1).all():                                      # check if the fence is not blocking any player
                            self.fences[k] = -1
                            self.neigbours[ij2k(F[0][0], F[0][1], self.grid_size)].append(ij2k(F[1][0], F[1][1], self.grid_size))
                            self.neigbours[ij2k(F[1][0], F[1][1], self.grid_size)].append(ij2k(F[0][0], F[0][1], self.grid_size))
                            self.r_fences[action[0]-1] += 1
                            return (self._get_obs(), -20, terminated, True, None)
            else:
                return (self._get_obs(), -20, terminated, True, None)


############ the player moves his pawn

        else:
            if action[0] == 1:
                n_pos = self.player_1+move[action[1]]
                same_place = n_pos[0] == self.player_2[0] and n_pos[1] == self.player_2[1]
                if not same_place and ij2k(n_pos[0], n_pos[1], self.grid_size) in self.neigbours[ij2k(self.player_1[0], self.player_1[1], self.grid_size)]:

                    self.player_1 = np.clip(self.player_1 + move[action[1]], 0, self.grid_size-1)
                    terminated = self.player_1[0] == self.grid_size-1
                else:
                    return (self._get_obs(), -20, terminated, True, None)

            else:
                n_pos = self.player_2+move[action[1]]
                same_place = n_pos[0] == self.player_1[0] and n_pos[1] == self.player_1[1]

                if not same_place and ij2k(n_pos[0], n_pos[1], self.grid_size) in self.neigbours[ij2k(self.player_2[0], self.player_2[1], self.grid_size)]:

                    self.player_2 = np.clip(self.player_2 + move[action[1]], 0, self.grid_size-1)
                    terminated = self.player_2[0] == 0
                else:
                    return (self._get_obs(), -20, terminated, True, None)
       
        
        reward = self.reward(action[0])
        observation = self._get_obs()
        info = None #self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.player_1 + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.player_2 + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (0,0,0),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                (0,0,0),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )
        
        for k,f in enumerate(self.fences):
            if f == 1:
                F = self.f2ij(k)
                pygame.draw.line(
                    canvas,
                    (0,0,0),
                    (pix_square_size * F[1][0], pix_square_size * F[1][1]),
                    (pix_square_size * (F[0][0]+1), pix_square_size * (F[0][1]+1)),
                    width=10
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()