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

# prameters

parser = argparse.ArgumentParser(description='DeepCubeTest')
parser.add_argument('--filename', type=str, default='none',
                    help='network_filename')
										
parser.add_argument('--n_test', type=int, default=500,
                    help='number of cube scrambles to test (default: 500)')
                    
parser.add_argument('--max_moves', type=int, default=20,
                    help='maximum moves (default: 20)')

parser.add_argument('--test_level', type=int, default=5,
                    help='test up to level (default: 5)')

                    
args = parser.parse_args()

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

        
def init_network():
    global policy_net, target_net, optimizer, memory, steps_done
    
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    #optimizer = optim.RMSprop(policy_net.parameters(), lr=args.learning_rate)
    #memory = ReplayMemory(50000)
    #steps_done = 0
    
def load_network(filename):
    if os.path.isfile(filename):
        print("=> loading network '{}'".format(filename))
        
        checkpoint = torch.load(filename)
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        target_net.load_state_dict(checkpoint['target_net_state'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #memory = checkpoint['memory']
        #steps_done = checkpoint['steps_done']
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


def select_action(state, use_model=0):

    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

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
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        
cube = Cube()

def main():
    sim_timestamp = datetime.now().strftime("@_%d_%m_%y_Time_%H_%M")
    init_network()
    load_network(args.filename)
    print('Loading network {}'.format(args.filename))
    success_rate = np.zeros((args.test_level,))
    
    for i in range(args.test_level):   
        success_rate[i] = test_network(cube, n_rots=i+1,max_moves=args.max_moves)
        print('Level {}, Success rate {}'.format(i+1,success_rate[i]))
    
    plt.figure(figsize=(10,5))    
    plt.ylim((0.0, 1.0))
    plt.plot(success_rate)
            
    plt.xlabel('Cube level')
    plt.title('Success rate [%] vs. cube scramble level')
    plt.savefig('cube_test'+sim_timestamp+'.svg',dpi=1000)
    plt.savefig('cube_test'+sim_timestamp,dpi=1000)

if __name__ == '__main__':
    main()