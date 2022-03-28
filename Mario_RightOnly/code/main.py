import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import gym
import datetime
from pathlib import Path
from collections import deque
from gym_super_mario_bros.actions import RIGHT_ONLY as right_only_action_space
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
import gym_super_mario_bros as  smb
from skimage import transform
from gym.spaces import Box
import collections
import copy
gamma = 0.9

class saving_and_plotting():
    def __init__(self,current_directory):
        self.current_directory= current_directory
        #saves rewards,length,qverge q loss and average loss of every episode
        self.episode_rewards = []
        self.ep_qs = []
        #saves rewards,episodic q value,loss length of every step
        self.step_reward = 0.0
        self.step_q = 0.0
        self.step_losslength= 0.0
        # Moving averages to plot the plots base don episodic rewards, qloss
        self.serial_avg_episode_rewards = []
        self.serial_avg_episode_avg_qs = []
        
        
        
    def save_step( self,reward_gained, lost, q_val):
        
        """
        Parameters:
        reward : reward earned in that step
        loss : total loss in the step
        q : q value in that step
        
        Description:
        Stores the metrics recorded in every step of a particular episode.

        """
        self.step_reward =self.step_reward+ reward_gained
        if lost:
            self.step_q = self.step_q+q_val
            self.step_losslength = self.step_losslength+1          
    def save_episode(self):
        """
        Description:
        Stores the metrics recorded in every episode and resets curr_ep_reward,curr_ep_q,curr_ep_loss_length
        """
        self.episode_rewards.append(self.step_reward)
        if self.step_losslength == 0:
            self.episode_average_q = 0
        else:
            self.episode_average_q = np.round(self.step_q / self.step_losslength, 5)
        #since the episode ended, making ds that stores step metrics to zero
        self.ep_qs.append(self.episode_average_q)
        self.step_reward= 0
        self.step_q= 0
        self.step_losslength =0
            
    def add_to_plot(self):
        """
        makes plot for every 10 episodes
    
        """
        avg_episode_reward = np.round(np.mean(self.episode_rewards[-100:]), 3)
        avg_episode_q = np.round(np.mean(self.ep_qs[-100:]), 3)
        self.serial_avg_episode_rewards.append(avg_episode_reward)
        self.serial_avg_episode_avg_qs.append(avg_episode_q)
        #plotting the graphs
        
        plt.plot(getattr(self, 'serial_avg_episode_rewards'))
        plt.savefig(self.current_directory/ 'rewards_plot.jpg' )
        plt.clf()
        plt.plot(getattr(self,'serial_avg_episode_avg_qs'))
        plt.savefig(self.current_directory/ 'q_plot.jpg' )
        plt.clf()
        
#Preprocessing   
#WE HAVE DONE Preprocessing REFERRING TO THE REFERENCE WE ADDED IN THE PAPER   
class resize_observation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs

class skipframes(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip


    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info
    
#WE HAVE BUILT NN REFERRING TO THE REFERENCE WE ADDED IN THE PAPER
#convolutional neural network
class MarioNet(nn.Module):
    #creating neural network
    def __init__(self, input_dimensions, output_dimensoins):
        super().__init__()
        c, height, width = input_dimensions
        self.online = nn.Sequential(
            nn.Conv2d(kernel_size=8,in_channels=c,stride=4,out_channels=32),
            nn.ReLU(),
            nn.Conv2d(kernel_size=4,in_channels=32,  stride=2,out_channels=64),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3,in_channels=64, out_channels=64, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dimensoins)
        )
        #start off with same nets
        self.target = copy.deepcopy(self.online)
        # Q_target parameters are saved
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model.lower() == 'online':
            return self.online(input)
        elif model.lower() == 'target':
            return self.target(input)
#Agent 
class mario_c:
    def __init__(self, state_dimensions, action_dimensions, save_dir, checkpoint=None):
        
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimensions
        self.memory_queue = deque(maxlen=100000)
        self.batch_size = 32
        self.copy=5000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9
        self.use_cuda = torch.cuda.is_available()
        self.curr_step = 0
        self.save_dir = save_dir
        self.local_net = MarioNet(self.state_dimensions, self.action_dimensions).float()
        self.target_net = MarioNet(self.state_dimensions, self.action_dimensions).float()   
        if checkpoint:
            self.load(checkpoint)
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
    def pick_action(self,state):
        """
        Parameters
        
        state: state observed by agent

        Returns
        
        action_index : action to be picked
        
        Description:
        
        uses epsilon greedy to select th action and exploration factor is decayed periodically

        """
        # exploration
        if self.exploration_rate > np.random.rand() :
        
            action_picked = np.random.randint(self.action_dimensions)
        # exploitation
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_val = self.local_net(state,'online')
            action_picked = torch.argmax(action_val, axis=1).item()
        # decay of exploitation
        self.exploration_rate =self.exploration_rate*self. exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate,self.exploration_rate_min)
        self.curr_step =self.curr_step+ 1
        # actions picked is being returned
        return action_picked  
    def memory(self,state,next_state,action,reward,done):
        """
        Parameters
        
        state: state observed by agent
        next_state: next state on taking the action
        action: action picked when in state 
        done : if the episode has been terminated it has True else False
        
        Description:
        
        appends the experience to memory_queue
        """
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.FloatTensor([done])
        self.memory_queue.append( (state, next_state, action, reward, done,) )

    def learn(self):
        """
        Description: samples experiences from memory and does the learning by DDQN,
    
        """
        #randomly sampling fom memory
        if len(self.memory_queue)>=32:
              batch = random.sample(self.memory_queue, self.batch_size)
        else:
              batch = random.sample(self.memory_queue, len(self.memory_queue))
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        #Updates local net after every 5000 steps 
        if  self.curr_step % self.copy == 0:
            self.target_net.load_state_dict(self.local_net.state_dict())

        #update target netwok based on DDQN
        target = reward + torch.mul((self.gamma * self.target_net(next_state,'target').max(1).values.unsqueeze(1)), 1-done)
        #local net approximation of Q-value
        current = self.local_net(state,'online').gather(1, action.long()) 
        loss = self.loss_fn(current, target)
        self.optimizer.zero_grad()
        #backprogating
        loss.backward()
        self.optimizer.step()
        loss= loss.item()
        #returns td estimate mean and loss
        return (current.mean().item(), loss)
    
    def save(self):
        """
        Description: saves the agent by creating a checkpoint
        """
        save_path = self.save_dir / "mario.chkpt"
        torch.save(dict(model=self.target_net.state_dict(),exploration_rate=self.exploration_rate),save_path)
        print("MarioNet saved")
        
    def load(self, load_path):
        #print(load_path)
        if not load_path.exists():
            raise ValueError('path does not exist')
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        self.target_net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
def main():
    # Initialize Super Mario env
    env = smb.make('SuperMarioBros-1-1-v0')
    #defining joypad movements
    env = JoypadSpace(env,right_only_action_space)
    #Apply Wrappers to env
    env = skipframes(env, skip = 4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = resize_observation(env, shape = 84)
    env = TransformObservation(env, f=lambda pixel: pixel / 255.)
    env = FrameStack(env, num_stack=4)
    env.reset()
    #directory where the checkpoints and plots are going to be saved
    current_datetime=str(datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
    save_directory = 'checkpoints_train'+'/'+current_datetime
    save_directory=Path(save_directory)
    save_directory.mkdir(parents=True)
    #starting learning from previously created checkpoints, if there is none can be replaced by None
    checkpoint = Path('checkpoints_train/2021-11-26T08-14-10/mario.chkpt')
    #checkpoint = None
    #creatting object mario based on Mario class
    mario_agent = mario_c(state_dimensions=(4, 84, 84), action_dimensions=env.action_space.n,
                  save_dir=save_directory, checkpoint=checkpoint)
    #creating object of saving-and_plotting class
    plotter= saving_and_plotting(save_directory)
    #number of episodes we wanna train
    episodes= 21
    # looping in number of episodes to train the agent
    for e in range(episodes):
        #fetching the state from env
        state = env.reset()
        #continue learning inevery episode till the agent dies
        while True:
            #fetch action based on epsilon greedy
            action = mario_agent.pick_action(state)
            #gets next state reward from env based on the action mario has taken
            next_state, reward, terminal, info = env.step(action)
            reward_float= 0+reward
            #stores if end of the episode or no
            done_info= terminal
            reward = torch.tensor([reward]).unsqueeze(0)  
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)   
            #storing it in buffer queue
            mario_agent.memory(state, next_state, action, reward, terminal)
            #learning of the agent
            q, loss =mario_agent.learn()
            #save the metric of current step
            #print(q,loss,'q and loss')
            plotter.save_step(reward_float,loss,q)
            state = next_state
            #checks if died or level is ended
            if done_info or info['flag_get']:
              break
        #save episode metrics once episode ends
        plotter.save_episode()
        #plot graph and save the agent every 10 episodess
        if e % 10 == 0 and e>=10:
                plotter.add_to_plot()
                mario_agent.save()
if __name__=='__main__':
    main()