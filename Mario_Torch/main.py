import datetime
import random
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import ResizeObservation, SkipFrame
from agent import Mario
from metrics import MetricLogger
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
import gym_super_mario_bros
import gym
import torch
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Initialize Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    SIMPLE_MOVEMENT
)

print(torch.cuda.is_available())
# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints_train') / \
    datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints_train/2021-11-24T10-57-00/mario.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n,
              save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 21

# for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )

    if e % 100 == 0:
        mario.save()
