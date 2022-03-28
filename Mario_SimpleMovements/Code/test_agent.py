import datetime
from pathlib import Path
import gym_super_mario_bros as smb
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from main import saving_and_plotting
from main import mario_c
from main import resize_observation, skipframes
from torch import tensor as tensor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT as simple_movement_action_space


env = smb.make('SuperMarioBros-1-1-v0')
#defining joypad movements
env = JoypadSpace(env,simple_movement_action_space)
#preprocessing of environmeny by using Wrappers 
env = skipframes(env, skip = 4)
env = GrayScaleObservation(env, keep_dim=False)
env = resize_observation(env, shape = 84)
env = TransformObservation(env, f=lambda pixel: pixel / 255.)
env = FrameStack(env, num_stack=4)
env.reset()
current_time= datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
checkpoint_dir = Path('checkpoints_play') / current_time
checkpoint_dir.mkdir(parents=True)
#path of saved model
checkpoint_trained_agent = Path('checkpoints_train/2021-11-29T03-20-36/mario.chkpt')
mario_agent = mario_c(action_dimensions=env.action_space.n,state_dimensions=(4, 84, 84) ,save_dir=checkpoint_dir, checkpoint=checkpoint_trained_agent)
mario_agent.exploration_rate = mario_agent.exploration_rate_min
plotter = saving_and_plotting(checkpoint_dir)
total_episodes = 5
for ep in range(total_episodes):
    current_state = env.reset()
    running =True
    while running is True:
        env.render()
        action = mario_agent.pick_action(current_state)
        next_state, reward_gained, status_info, flag_status = env.step(action)
        reward_float= 0+reward_gained
        #stores if end of the episode or no
        done_info= status_info
        reward = tensor([reward_gained]).unsqueeze(0)  
        status_info = tensor([int(status_info)]).unsqueeze(0)
        mario_agent.memory(current_state, next_state, action, reward_gained, status_info)
        plotter.save_step(reward, None, None)
        state = next_state
        if done_info or flag_status['flag_get']:
            running=False
            break
    plotter.save_episode()
    if ep % 10 == 0:
        plotter.add_to_plot()
