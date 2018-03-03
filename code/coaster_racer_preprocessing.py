import gym
import universe # register the universe environments

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import gmtime, strftime


# crop video frame so NN is smaller and set range between 1 and 0; and
# stack-a-bitch!
def processFrame(observation_n):
    
    now = strftime("%Y%m%d-%H%M%S", gmtime())

    obs = observation_n

    if observation_n is not None and observation_n[0] is not None:
        obs = observation_n[0]['vision']

        # crop
        obs = cropFrame(obs)
        cv2.imwrite('../images/cropped/cropped-vision-' + now + '.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        # downscale resolution
        obs = cv2.resize(obs, (80, 80))
        cv2.imwrite('../images/resized/resized-vision-' + now + '.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        # grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../images/gray/gray-vision-' + now + '.png', obs)

        # Convert to float
        obs = obs.astype(np.float32)

        # scale from 1 to 255
        obs *= (1.0 / 255.0)

        # re-shape a bitch
        obs = np.reshape(obs, [80, 80])

    return obs

# crop frame to only flash portion:


def cropFrame(obs):
    # adds top = 84 and left = 18 to height and width:
    return obs[284:564, 18:658, :]


env = gym.make('flashgames.CoasterRacer-v0')  # You can run many environment in parallel
# env.configure(remotes=1)  # automatically creates a local docker container
env.configure(remotes='vnc://localhost:5900+15901')
observation_n = env.reset()  # Initiate the environment and get list of observations of its initial state

while True:
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)  # Reinforcement Learning action by agent
    
    observation_t = processFrame(observation_n)

    print ("observation: ", observation_n)  # Observation of the environment
    print ("reward: ", reward_n)  # If the action had any postive impact +1/-1
    env.render()  # Run the agent on the environment
