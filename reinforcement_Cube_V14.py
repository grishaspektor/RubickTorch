# Imports 
import argparse
from matplotlib.colors import LinearSegmentedColormap
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import os.path
import pickle
from datetime import datetime
import time

#get_ipython().run_line_magic('matplotlib', 'inline')

# global parameters
MEM_SIZE = 1
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
TEST_RATE = 1000
ADDED_ROT_TRIALS_TEST = 10
ADDED_ROT_TRIALS_REGULAR = 15
MAX_ROTS = 6
SAVE_DIR = './save/'
VERSION = '13'

''' CHANGE LOG:
Version 13: Changed cube to unique identification per cell.
Version 14: Increase network.
''' 
# prameters

parser = argparse.ArgumentParser(description='DeepCube')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
					
parser.add_argument('--curr_thresh', type=float, default=0.8, metavar='TH',
                    help='curriculum threshold (default: 0.8)')
					
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
					
parser.add_argument('--do_save', type=int, default=1, metavar='do_save',
                    help='save network and graphs (default: No)')
					
parser.add_argument('--do_regular',    type=int, default=1, metavar='do_regular',
                    help='run regular (default: Yes)')
parser.add_argument('--do_reverse',    type=int, default=1, metavar='do_reverse',
                    help='run reverse (default: Yes)')
parser.add_argument('--do_reverse_TD', type=int, default=1, metavar='do_reverse_TD',
                    help='run reverse TD (default: Yes)')
                    
parser.add_argument('--use_sparse', type=int, default=1,
                    help='use sparse reward (default: Yes)')
					
parser.add_argument('--do_rand_roll', type=int, default=0,
                    help='do random roll (default: No)')
					                    
args = parser.parse_args()
run_alg = [args.do_regular,args.do_reverse,args.do_reverse_TD]

# Settings

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Using device: {}'.format(device))


# ## General purpose functions

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Cube(object):
    def __init__(self,use_sparse=0):
        self.C, self.T, self.B, self.L, self.R, self.F = [np.zeros((3*4,3*3),dtype=bool) for _ in range(6)]
        self.T = (slice(0,3,1),slice(3,6,1))
        self.C = (slice(3,6,1),slice(3,6,1))
        self.B = (slice(6,9,1),slice(3,6,1))
        self.F = (slice(9,12,1),slice(3,6,1))
        self.L = (slice(3,6,1),slice(0,3,1))
        self.R = (slice(3,6,1),slice(6,9,1))
        self.face_dict = ['T','L','C','R','B','F']
        self.faces = [self.T,self.L,self.C,self.R,self.B,self.F]
        self.centers = np.array(((1,4,4,4,7,10),(4,1,4,7,4,4)),dtype=int)

                
        self.state = np.zeros((3*4,3*3))
        self.resetCube()

        
        self.solved_state = self.state.copy()
        
        # matplot_lib
        #            B       G       Y       R       W       B            Orange 
        colors = [(0,0,0),(0,1,0),(1,1,0),(1,0,0),(1,1,1),(0,0,1),(255.0/255,140.0/255,0)]
        cmap_name = 'cube'
        self.cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=7)
        
        self.mem_size = MEM_SIZE
        self.state_mem = np.ones((self.mem_size,3*4,3*3))*-1
        self.mem_pointer = 0
        
        self.use_sparse_reward = use_sparse
        
    def resetCube(self):
        #M = np.arange(9).reshape(3,3)
        #self.state[self.T],self.state[self.L],self.state[self.C]=10+M,20+M,30+M
        #self.state[self.R],self.state[self.B],self.state[self.F]=40+M,50+M,60+M
        
        self.state[self.T],self.state[self.L],self.state[self.C]=1,2,3
        self.state[self.R],self.state[self.B],self.state[self.F]=4,5,6
        
        
    def rotateFace(self, face, direction):
        state = self.state.copy()
        state = self._roll(state, face)
        state = self._rotate(state, direction)
        state = self._roll(state, face,forward=-1)
        return state
    
    @staticmethod
    def _rotate(state, direction):        
        # 1 = 90, -1 = -90
        state[2:7,2:7] = np.rot90(state[2:7,2:7],-direction)
        return state 
    
    def _roll(self,state, face, forward=1):
        
        faces =  self.faces
        
        if face == 'C':
            pass
        elif face == 'T' or face == 'B':
            D = 1 if face=='T' else -1
            state[:,3:6] = np.roll(state[:,3:6], D*3*forward,axis=0)
            state[faces[3]] = np.rot90(state[faces[3]], D*1*forward)
            state[faces[1]] = np.rot90(state[faces[1]], D*-1*forward)
        elif face == 'F':
            state[:,3:6] = np.roll(state[:,3:6],-2*3*forward,axis=0)
            state[faces[3]] = np.rot90(state[faces[3]],-2*forward)
            state[faces[1]] = np.rot90(state[faces[1]], 2*forward)
        elif face == 'L' or face == 'R':
            D = 1 if face=='L' else -1
            use_sl = faces[1] if face=='L' else faces[3]
            
            if forward==1:
                state[3:6,:] = np.roll(state[3:6,:],D*3*forward,axis=1)
            tmp = state[use_sl].copy()
            state[use_sl] = np.rot90(state[faces[5]],2)
            state[faces[5]] = np.rot90(tmp,2)
            if forward==-1:
                state[3:6,:] = np.roll(state[3:6,:],D*3*forward,axis=1)
            
            state[faces[0]] = np.rot90(state[faces[0]], D*1*forward)
            state[faces[4]] = np.rot90(state[faces[4]], D*-1*forward)   
        return state
            
    def showCube(self):
        plt.imshow(self.state, interpolation='nearest',cmap=self.cm)
        plt.axis('off')
        
    def printCube(self):
        print('--------------------------------------')
        print(self.state)
    
    #def getState(self):
    #    ret_idx = (np.array((0,1)) + self.mem_pointer) % self.mem_size
    #    return self.state_mem[ret_idx].astype(np.float32)
    def getState(self):
        ret_idx = (np.array((0,1)) + self.mem_pointer) % self.mem_size
        return self.state_mem[ret_idx[0]:ret_idx[1]+1].astype(np.float32)
    
    def setState(self,state):
        self.state_mem = state
        self.state = state[-1]
        return 

    def getNextState(self, action,do_rand_roll=0):
        states = np.ones((self.mem_size,3*4,3*3))*-1
        
        direction = 1 if action<6 else -1
        face = self.face_dict[action % 6]        
        state = self.rotateFace(face, direction)
        
        if do_rand_roll:
            rand_roll =  np.random.randint(6, size=1)
            state = self._roll(state, self.face_dict[int(rand_roll)])

        states[self.mem_size-1] = state
        
        if do_rand_roll==0:
            inv_act = (action +6)% 12
        else:
            if rand_roll==0:
                inv_act = 4+6 if direction==1 else 4
            if rand_roll==1:
                inv_act = 3+6 if direction==1 else 3
            if rand_roll==2:
                inv_act = 2+6 if direction==1 else 2
            if rand_roll==3:
                inv_act = 1+6 if direction==1 else 1
            if rand_roll==4:
                inv_act = 0+6 if direction==1 else 0
            if rand_roll==5:
                inv_act = 5+6 if direction==1 else 5
                
        return states.astype(np.float32),inv_act
            
    def step(self, action,do_rand_roll=0):
        direction = 1 if action<6 else -1
        face  = self.face_dict[action % 6]
        self.state = self.rotateFace(face, direction)
        
        if do_rand_roll:
            rand_roll =  np.random.randint(6, size=1)
            self.state = self._roll(self.state, self.face_dict[int(rand_roll)]) 
        
        done = self.isDone()
        reward = self.calcReward()
        # update memory 
        self.updateMem()
        
        return np.array(reward, dtype=np.float32), done
    
    def calcReward(self):
        if self.use_sparse_reward:
            if self.isDone():
                return 10
            else:
                return -1
        else:
            cor_faces=0
            for face in self.faces:
                cor_faces += np.count_nonzero(
                    np.equal(self.state[face],self.solved_state[face])
                )
            if cor_faces == 9*6:
                return 10
            else:
                return 1.0*cor_faces/(9*6)-1
                
    def isDone(self, state=None):
        if state is None:
            state = self.state
        #return (np.all(np.floor((state[self.T]/10))==np.floor(state[self.T][1,1]/10)) and
        #        np.all(np.floor((state[self.L]/10))==np.floor(state[self.L][1,1]/10)) and
        #        np.all(np.floor((state[self.C]/10))==np.floor(state[self.C][1,1]/10)) and
        #        np.all(np.floor((state[self.R]/10))==np.floor(state[self.R][1,1]/10)) and
        #        np.all(np.floor((state[self.B]/10))==np.floor(state[self.B][1,1]/10)))
                
        return (np.all(state[self.T]==state[self.T][1,1]) and
                np.all(state[self.L]==state[self.L][1,1]) and
                np.all(state[self.C]==state[self.C][1,1]) and
                np.all(state[self.R]==state[self.R][1,1]) and
                np.all(state[self.B]==state[self.B][1,1]))
    
    def reset(self,rand_steps,exclude=[],do_rand_roll=0):
        self.resetCube()
        self.state_mem = np.ones((self.mem_size,3*4,3*3))*-1
        
        for i in range(rand_steps):
            while 1:
                rand_action =  np.random.randint(12)
                if rand_action not in exclude:
                    break
            self.step(rand_action)
        
        # even reset(0) does a random roll
        if do_rand_roll:
            rand_roll =  np.random.randint(6, size=1)
            state = self._roll(self.state, self.face_dict[int(rand_roll)])
            self.state = state 
        
        self.mem_pointer = 0
        self.updateMem()
        if rand_steps> 0:
            return rand_action
        else:
            return -1
        
    def updateMem(self):    
        self.state_mem[self.mem_pointer] = self.state
        self.mem_pointer =(self.mem_pointer+1) % self.mem_size
		
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(MEM_SIZE, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.head = nn.Linear(6*3*64, 12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

def saveMetadata(filename):
    with open(filename+'.txt', 'w') as outfile:
        outfile.write( '-----GLOBAL PARAMETERS-----\n')
        outfile.write('MEM_SIZE        ={}\n'.format(MEM_SIZE         ))
        outfile.write('BATCH_SIZE      ={}\n'.format(BATCH_SIZE       ))
        outfile.write('GAMMA           ={}\n'.format(GAMMA            ))
        outfile.write('EPS_START       ={}\n'.format(EPS_START        ))
        outfile.write('EPS_END         ={}\n'.format(EPS_END          ))
        outfile.write('EPS_DECAY       ={}\n'.format(EPS_DECAY        ))
        outfile.write('TARGET_UPDATE   ={}\n'.format(TARGET_UPDATE    ))
        outfile.write('TEST_RATE       ={}\n'.format(TEST_RATE        ))
        outfile.write('ADDED_ROT_TRIALS_TEST={}\n'.format(ADDED_ROT_TRIALS_TEST ))
        outfile.write('ADDED_ROT_TRIALS_REGULAR={}\n'.format(ADDED_ROT_TRIALS_REGULAR ))
        outfile.write('MAX_ROTS        ={}\n'.format(MAX_ROTS         ))
        
        outfile.write( '\n-----Sim PARAMETERS-----\n')
        
        global args
        for arg in vars(args):
            outfile.write('{}={}\n'.format(arg,getattr(args, arg)))
            
def printMetadata():
    
    print( '-----GLOBAL PARAMETERS-----')
    print('MEM_SIZE        ={}'.format(MEM_SIZE         ))
    print('BATCH_SIZE      ={}'.format(BATCH_SIZE       ))
    print('GAMMA           ={}'.format(GAMMA            ))
    print('EPS_START       ={}'.format(EPS_START        ))
    print('EPS_END         ={}'.format(EPS_END          ))
    print('EPS_DECAY       ={}'.format(EPS_DECAY        ))
    print('TARGET_UPDATE   ={}'.format(TARGET_UPDATE    ))
    print('TEST_RATE       ={}'.format(TEST_RATE        ))
    print('ADDED_ROT_TRIALS_TEST={}'.format(ADDED_ROT_TRIALS_TEST ))
    print('ADDED_ROT_TRIALS_REGULAR={}'.format(ADDED_ROT_TRIALS_REGULAR ))
    print('MAX_ROTS        ={}'.format(MAX_ROTS         ))
    print( '-----Sim PARAMETERS-----')
    
    global args
    for arg in vars(args):
        print('{}={}'.format(arg,getattr(args, arg)))
    
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.subplot(1,2,1)
    plt.plot(durations_t.numpy())
    plt.subplot(1,2,2)
    plt.imshow(next_state_plt[1], interpolation='nearest',cmap=cube.cm)
    plt.axis('off')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.subplot(1,2,1)
        plt.plot(means.numpy())

    plt.pause(0.0001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
def init_network():
    global policy_net, target_net, optimizer, memory, steps_done
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=args.learning_rate)
    memory = ReplayMemory(50000)
    steps_done = 0

def save_network(filename):
    state = {
        'policy_net_state' : policy_net.state_dict(),
        'target_net_state' : target_net.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'memory' : memory,
        'steps_done' : steps_done
    }

    torch.save(state, filename)
    
def load_network(filename):
    if os.path.isfile(filename):
        print("=> loading network '{}'".format(filename))
        init_network()
        checkpoint = torch.load(filename)
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        target_net.load_state_dict(checkpoint['target_net_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        memory = checkpoint['memory']
        steps_done = checkpoint['steps_done']
        print("=> done!")
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        
def test_network(cube, n_rots, max_moves):
    n_tests = 500
    success = 0.0
    for i in range(n_tests):
        cube.reset(n_rots, do_rand_roll=0)
        for j in range(max_moves):
            state = cube.getState()
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            action = select_action(state, use_model=1)
            _, isDone = cube.step(action[0,0])
            if isDone:
                success += 1
                break
    success_rate = success / n_tests
    return success_rate

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    if non_final_next_states_list:
        non_final_next_states = torch.cat(non_final_next_states_list)
        
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states_list:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()

def select_action(state, use_model=0):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if use_model==0:
        steps_done += 1
    if sample > eps_threshold or use_model==1:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(12)]], device=device, dtype=torch.long)
    
def select_action_reverse_par(cube, TD_exploration):
    
    #global steps_done
    #steps_done += 1
    
    # current state and current state reward
    state = cube.getState()
    reward = cube.calcReward()

    next_state_vec = np.empty((12,12,9),dtype='float32')
    reverse_action_vec = np.empty((12,),dtype=int)
    loss = np.empty((12,))
    loss_out = np.empty((12,))
    
    # Compute value V(s_{t}) for current state.
    if cube.isDone():
        state_values = 0
    else:
        state_values = target_net(torch.from_numpy(state).unsqueeze(0).to(device)).max(1)[0].cpu().detach().numpy()
    
    # Compute the expected Q values of the current state 
    expected_state_action_values = (state_values * GAMMA) + reward
        
    # Check all 12 possible actions
    for action in range(12):  
        n_state, inv_act = cube.getNextState(action, do_rand_roll=0)
        #print(n_state.shape)
        # save for later
        next_state_vec[action] = n_state
        reverse_action_vec[action] = inv_act
    Q = policy_net(torch.from_numpy(next_state_vec).unsqueeze(1).to(device))

    # Compute TD loss: delta = Q - (r + gamma * max_a {Q})
    loss_out = Q.cpu().detach().numpy()[np.arange(12), reverse_action_vec] -  expected_state_action_values
    loss = np.abs(loss_out)
    #print(expected_state_action_values)
    
    if TD_exploration:
        # Sample selected state acording to TD-Loss
        loss = loss / np.sum(loss)
        selected_action = np.random.choice(range(12), p=loss)
    else:
        # Sample selected state uniformly
        selected_action = np.random.choice(range(12))
    cube.setState(np.expand_dims(next_state_vec[selected_action],0))
    with torch.no_grad():
        return cube, reverse_action_vec[selected_action], np.abs(loss_out[selected_action])

def learn_REGULAR(num_episodes, MAX_ROTS, curriculum=False, current_learning_level=1):

    exclude=[]
    success_rate_graph = {}
    success_rate_graph['current_level']=[]
    #success_rate_graph['first_level']=[]
    
    global cube
    
    if not curriculum:
        current_learning_level = MAX_ROTS
        
    max_moves_learn = current_learning_level + ADDED_ROT_TRIALS_REGULAR
    max_moves = current_learning_level + ADDED_ROT_TRIALS_TEST
    
    for i_episode in range(num_episodes):
        
        # Initialize the environment and state
        cube.reset(current_learning_level)
        state = cube.getState()
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        move_cnt=0

        for t in range(max_moves_learn):
            # Select and perform an action
            action = select_action(state)
            reward, done = cube.step(action.item())
            reward = torch.from_numpy(reward).unsqueeze(0).to(device)

            # Observe new state
            next_state_plt = cube.getState()
            if not done:
                next_state = torch.from_numpy(next_state_plt).unsqueeze(0).to(device)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                break
            move_cnt+=1

        if i_episode % TEST_RATE == 0:
            #success_rate = test_network(cube, n_rots=1,max_moves=max_moves)
            #success_rate_graph['first_level'].append(success_rate)
            
            success_rate = test_network(cube, n_rots=current_learning_level,max_moves=max_moves)
            success_rate_graph['current_level'].append(success_rate)
            
            # Prints and updates with and without curriculum learning
            if curriculum:
                print('\r','Current Level: {} rotations, {} episodes done, success rate = {}'.format(current_learning_level, i_episode, success_rate), end='')
                if success_rate>curriculum:
                    #if current_learning_level==MAX_ROTS:
                    print('\nCompleted level {} at {} iterations\n'.format(current_learning_level, i_episode))
                    #else:
                    current_learning_level += 1
                    max_moves += 1
                    max_moves_learn += 1
                        
            else:
                print('\r','{} episodes done: success rate = {}'.format(i_episode, success_rate), end='')

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    success_rate = test_network(cube, n_rots=current_learning_level,max_moves=max_moves)      
    print('\n','Trainging finished. {} episodes done: success rate = {} for level {}'.format(i_episode+1, success_rate,current_learning_level), end='')
    
    return success_rate_graph

def learn_REVERSE(num_episodes, MAX_ROTS, TD_exploration, curriculum=False, current_learning_level=1,use_par_reverse=1):

    success_rate_graph = {}
    success_rate_graph['current_level']=[]
    #success_rate_graph['first_level']=[]
    global cube
    
    if not curriculum:
        current_learning_level = MAX_ROTS
        
    max_moves = current_learning_level + ADDED_ROT_TRIALS_TEST        

    for i_episode in range(num_episodes):
        cube.reset(0)
        #move_cnt=0
        
        # init:
        TD_error_depth = np.empty(current_learning_level+1,)
        next_states, reverse_actions, rewards, states = [], [], [], []
        
        for t in range(current_learning_level+1):
            reward = cube.calcReward(); reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(0).to(device)
            if cube.isDone():
                next_state = None
            else:
                next_state = cube.getState(); next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)

            # Find the reverse state and action
            if use_par_reverse:
                cube, reverse_action, TD_error_depth[t] = select_action_reverse_par(cube, TD_exploration)
            else:
                cube, reverse_action, TD_error_depth[t] = select_action_reverse(cube, TD_exploration)                    
            reverse_action = torch.tensor([[reverse_action]], device=device, dtype=torch.long)
            state = torch.from_numpy(cube.getState()).unsqueeze(0).to(device)
            
            # Save states, reverse_actions, next_states, rewards
            states.append(state), reverse_actions.append(reverse_action)
            next_states.append(next_state), rewards.append(reward)
        
        final_state = 0
        if TD_exploration:
            # Sample selected state acording to TD-Loss
            TD_error_depth = TD_error_depth / np.sum(TD_error_depth)
            final_state = np.random.choice(range(current_learning_level+1), p=TD_error_depth)
            
        # Store the transitions in memory
        for s in range(final_state, current_learning_level+1):
            memory.push(states[s], reverse_actions[s], next_states[s], rewards[s])

        # Perform one step of the optimization (on the target network)
        optimize_model()
        #move_cnt += 1
            
        if i_episode % TEST_RATE == 0:
            #success_rate = test_network(cube, n_rots=1,max_moves=max_moves)
            #success_rate_graph['first_level'].append(success_rate)
            
            success_rate = test_network(cube, n_rots=current_learning_level,max_moves=max_moves)
            success_rate_graph['current_level'].append(success_rate)

            # Prints and updates with and without curriculum learning
            if curriculum:
                print('\r','Current Level: {} rotations, {} episodes done, success rate = {}'.format(current_learning_level, i_episode, success_rate), end='')
                if success_rate>curriculum:
                    print('\nCompleted level {} at {} iterations\n'.format(current_learning_level, i_episode))
                    #if current_learning_level+1==MAX_ROTS:
                    #    print('\n','LEARNING COMPLETED!!! Current Level: {} rotations, {} episodes done, success rate = {}'.format(current_learning_level, i_episode, success_rate), end='')
                    #else:
                    current_learning_level += 1
                    max_moves += 1
            else:
                print('\r','{} episodes done: success rate = {}'.format(i_episode, success_rate), end='')
                
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    success_rate = test_network(cube, n_rots=current_learning_level,max_moves=max_moves)      
    print('\n','Trainging finished. {} episodes done: success rate = {} for level {}'.format(i_episode+1, success_rate,current_learning_level), end='')

    return success_rate_graph

cube = Cube(args.use_sparse)

def main():
    printMetadata()
    sim_timestamp = datetime.now().strftime("@_%d_%m_%y_Time_%H_%M")
    saveMetadata(SAVE_DIR + 'simulation_' + sim_timestamp)
    success_rate_graph       = np.empty((3, int(args.num_episodes/TEST_RATE)))
    #success_rate_graph_first = np.empty_like(success_rate_graph)

    print('-------- Starting Curriculum Learning. Required success rate = {}% --------'.format(args.curr_thresh * 100))
    if run_alg[0]:
        save_str = 'V{}_Regular_{}'.format(VERSION,args.num_episodes)
        print('\n-------- REGULAR network:  MAX_ROTS = {}, aditional_rot_trials = {},  num_episodes = {} --------'.format(MAX_ROTS, ADDED_ROT_TRIALS_TEST, args.num_episodes))
        t = time.time()
        init_network()
        graph = learn_REGULAR(args.num_episodes, MAX_ROTS, curriculum=args.curr_thresh)
        success_rate_graph[0,:] =  np.array(graph['current_level'])
        #success_rate_graph_first[0,:] =  np.array(graph['first_level'])
        if args.do_save:
            # Save network params and graph
            filename = save_str + '_' + sim_timestamp
            save_network(SAVE_DIR+'net_' + filename)
            with open(SAVE_DIR+'graph_' + filename, "wb") as fp: 
                pickle.dump(success_rate_graph[0,:], fp)
        print('\n')        
        print(np.round_(time.time() - t, 3), 'sec elapsed')

    if run_alg[1]:
        save_str = 'V{}_Reverse_uniform_{}'.format(VERSION,args.num_episodes)    
        print('\n-------- REVERSE network (uniform-exploration):  MAX_ROTS = {}, aditional_rot_trials = {},  num_episodes = {} --------'.
            format(MAX_ROTS, ADDED_ROT_TRIALS_TEST, args.num_episodes))
        t = time.time()
        init_network()
        graph = learn_REVERSE(
            args.num_episodes, MAX_ROTS, TD_exploration=False, curriculum=args.curr_thresh)
        success_rate_graph[1,:] =  np.array(graph['current_level'])
        #success_rate_graph_first[1,:] =  np.array(graph['first_level'])
        # Save network params and graph
        if args.do_save:
            filename = save_str + '_' + sim_timestamp
            save_network(SAVE_DIR+'net_' + filename)
            with open(SAVE_DIR+'graph_' + filename, "wb") as fp:
                pickle.dump(success_rate_graph[1, :], fp)
        print('\n')        
        print(np.round_(time.time() - t, 3), 'sec elapsed')

    if run_alg[2]:  
        save_str = 'V{}_Reverse_TD_{}'.format(VERSION,args.num_episodes)   	
        print('\n -------- REVERSE network (TD-exploration):  MAX_ROTS = {}, aditional_rot_trials = {},  num_episodes = {} --------'.
            format(MAX_ROTS, ADDED_ROT_TRIALS_TEST, args.num_episodes))
        t = time.time()
        init_network()
        graph = learn_REVERSE(
            args.num_episodes, MAX_ROTS, TD_exploration=True, curriculum=args.curr_thresh)
        success_rate_graph[2,:] =  np.array(graph['current_level'])
        #success_rate_graph_first[2,:] =  np.array(graph['first_level'])
        # Save network params and graph
        if args.do_save:
            filename = save_str + '_' + sim_timestamp
            save_network(SAVE_DIR+'net_' + filename)
            with open(SAVE_DIR+'graph_' + filename, "wb") as fp:
                pickle.dump(success_rate_graph[2, :], fp)
        print('\n')        
        print(np.round_(time.time() - t, 3), 'sec elapsed')

    """ --------------------- Plot Results --------------------- """       
    plt.figure(figsize=(15,5))
    plt.ylim((0.0, 1.0))
    n_graphs = success_rate_graph.shape[0]
    x_axis = np.arange(len(success_rate_graph[0,:]))*TEST_RATE
    plots = [None]*n_graphs
    labels = ['Regular', 'Uniform Reverse Exploration', 'Prioritized Reverse Exploration']

    for p in range(n_graphs):
        plots[p] = plt.plot(x_axis, success_rate_graph[p,:], label=labels[p])
        advanced = np.where(success_rate_graph[p,:]>args.curr_thresh)[0]
        for i, a in enumerate(advanced):
            plt.gca().annotate(str(i+2),
                    xy=(a*TEST_RATE, args.curr_thresh), xycoords='data',
                    xytext=(-5, 25), textcoords='offset points',
                    size=14,
                    color=plots[p][0].get_color(),
                    arrowprops=dict(facecolor=plots[p][0].get_color()))
            
    plt.legend()
    plt.xlabel('Number of episodes')
    plt.gca().axhline(y=args.curr_thresh, color='red', linestyle='--')
    plt.title('Success rate [%] for {} cube rotations with Curriculum Learning'.format(MAX_ROTS))
    plt.savefig(SAVE_DIR+VERSION+'all_plots'+sim_timestamp+'.svg',dpi=1000)
    plt.savefig(SAVE_DIR+VERSION+'all_plots'+sim_timestamp,dpi=1000)


if __name__ == '__main__':
    main()