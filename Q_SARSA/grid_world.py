import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

UP = 0
DOWN = 1
LEFT = 3
RIGHT = 4

class GridWorldEnv(object):
    '''
        start_states: list of coordinates for starting state
        goal_states: list of coordinates for goal state
        max_steps: maximum steps before episode terminates
        action_fail_prob: probability with which chosen action fails and a random action is instead taken
    '''
    def __init__(self, grid_file, start_states = [(0,0)], goal_states = [(10,10)], goal_reward = 10, max_steps = 100,
                    action_fail_prob = 0.1, seed = None):
        # Load the grid from txt file
        self.grid = np.loadtxt(grid_file, delimiter=' ').astype('int')
        self.random_generator = np.random.RandomState(seed)

        self.start_states = start_states
        self.goal_states = goal_states
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.action_fail_prob = action_fail_prob
        self.action_space = [UP, DOWN, LEFT, RIGHT]
    
    '''
    is_in_grid: Check to see if given coord_x,coord_y coordinates is inside the grid
    '''
    def is_in_grid(self, coord_x, coord_y):
        if coord_x < 0 or coord_y<0:
            return False
        if coord_x >= self.grid.shape[0] or coord_y >= self.grid.shape[1]:
            return False
        
        return True
    '''
    choose_state: choose a state from the list
    list_state: list of tupules of two integers

    '''
    def choose_state(self, list_states):
        choice = self.random_generator.randint(len(list_states))
        return list_states[choice]


    '''
    reset: resets the environment by randomly choosing a start and goal state
    '''
    def reset(self, start_state=None, goal_state=None):
        self.start_state = self.state = self.choose_state(self.start_states) if start_state is None else start_state
        self.goal = self.choose_state(self.goal_states) if goal_state is None else goal_state
        self.done = False
        self.steps = 0

    '''
    step: change the state after taking action
    Returns: new state, environment reward, done (whether episode is completed)
    '''
    def step(self, action):
        assert action in self.action_space, "Wrong action %d chosen, Possible actions: %s"%(action, str(self.action_space))
        if self.done:
            print("Warning: Episode done")
        
        self.steps += 1

        # With prob = self.action_fail_prob choose a random action
        if self.random_generator.rand() < self.action_fail_prob:
            action = self.action_space[self.random_generator.randint(len(self.action_space))]

        if action == UP:
            new_state = (self.state[0]-1, self.state[1])
        elif action == DOWN:
            new_state = (self.state[0]+1, self.state[1])
        elif action == LEFT:
            new_state = (self.state[0], self.state[1]-1)
        elif action == RIGHT:
            new_state = (self.state[0], self.state[1]+1)
        
        if self.is_in_grid(new_state[0], new_state[1]):
            self.state = new_state
        
        if self.state == self.goal:
            self.reward = self.goal_reward
            self.done = True
            return self.state, self.reward, self.done
        
        if self.steps >= self.max_steps:
            self.done = True
        
        self.reward = self.grid[self.state[0], self.state[1]]

        return self.state, self.reward, self.done

    '''
    render: render a plot of the environment
    '''
    def render(self):
        grid = self.grid.copy()
        grid[self.start_state[0], self.start_state] = 3
        grid[self.goal[0], self.goal[1]] = 4
        grid[self.state[0], self.state[1]] = 5

        plt.clf()
        cmap = colors.ListedColormap(['#F5E5E1', '#F2A494', '#FF2D00', '#0004FF', '#00FF23', '#F0FF00'])
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap=cmap)
    
    '''
    render_policy: render a learnt policy
    '''
    def render_policy(self, policy):
        pass

'''
Grid World environment with leftward wind: extra left action with 0.5 probability
'''
class GridWorldWindyEnv(GridWorldEnv):
    def __init__(self, grid_file, start_states = [(0,0)], goal_states = [(10,10)], goal_reward = 10, max_steps = 100,
                    action_fail_prob = 0.1, seed = None, windy_probab = 0.5):
        super(GridWorldWindyEnv, self).__init__(grid_file, start_states, goal_states , goal_reward , max_steps,
                    action_fail_prob , seed)
        self.windy_probab = windy_probab
    
    def step(self, action):
        ans = super().step(action)
        if not self.done and self.random_generator.rand() < self.windy_probab:
            ans = super().step(LEFT)
        return ans