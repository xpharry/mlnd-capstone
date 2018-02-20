## OpenAI-Gym

Basic Agent Loop

```python
import gym
env = gym.make("Taxi-v2")
observation = env.reset()
for _ in range(1000):
env.render()
# your agent here (this takes random actions)
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
if done:
env.render()
break
```

Creating an Instance

- Each gym environment has a unique name of the form
([A-Za-z0-9]+-)v([0-9]+)
- To create an environment from the name use the
env = gym.make(env_name)
- For example, to create a Taxi environment:

```
env = gym.make('Taxi-v2')
```

Reset Function

- Used to reinitialize a new episode
- Returns the initial state

```bash
init_state = env.reset()
```


Step Function

```bash
step(action) -> (next_state,
                 reward,
                 is_terminal,
                 debug_info)
```

- Performs the specified action and returns the resulting state
- The main method your agent interacts with

Observations

If we ever want to do better than take random actions at each step, it'd probably be good to actually know what our actions are doing to the environment.

The environment's step function returns exactly what we need. In fact, step returns four values. These are:

- observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
- reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
- done (boolean): whether it's time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
- info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). However, official evaluations of your agent are not allowed to use this for learning.

![](./images/simple-flow.svg)

Render

- Optional method
- Used to display the state of your environment
- Useful for debugging and qualitatively comparing different
agent policies

Environment Space Attributes

- Most environments have two special attributes:
    - action space
    - observation space
- These contain instances of gym.spaces classes
- Makes it easy to find out what are valid states and actions
- There is a convenient sample method to generate uniform
random samples in the space

Spaces

The Discrete space allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1. The Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers.

gym.spaces

- Action spaces and State spaces are defined by instances of
classes of the gym.spaces modules
- Included types are:
    - gym.spaces.Discrete
    - gym.spaces.MultiDiscrete
    - gym.spaces.Box
    - gym.spaces.Tuple
- All instances have a sample method which will sample
random instances within the space

gym.spaces.Discrete

- Specifies a space containing n discrete points
- Each point is mapped to an integer from [0, n − 1]
- Discrete(10)
    - A space containing 10 items mapped to integers in [0, 9]
    - sample will return integers such as 0, 3, and 9.
    
gym.Env Class

- All environments should inherit from gym.Env
- Override a handful of methods:
    - step
    - reset
- Provide the following attributes:
    - action space
    - observation space

Attributes

- observation space represents the state space
- action space represents the action space
- Both are instances of gym.spaces classes
- You can also provide a reward range, but this defaults to
(−∞, ∞)

Registration

- How do you get your environment to work with gym.make()?
    - You must register it!
- Example
```python
from gym.envs.registration import register
register(
id='Deterministic-4x4-FrozenLake-v0',
entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
kwargs={'map_name': '4x4',
'is_slippery': False})
```
- id : the environment name used with gym.make
- entry point : module path and class name of environment
- kwargs: dictionary of keyword arguments to environment
constructor

Discrete Environment Class

- A subclass of the gym.Env which provides the following
attributes
- nS : number of states
- nA : number of actions
- P : model of environment
- isd : initial state distribution

OpenAI Gym Scoreboard
- The gym also includes an online scoreboard
- Gym provides an API to automatically record:
- learning curves of cumulative reward vs episode number
- Videos of the agent executing its policy
- You can see other people’s solutions and compete for the best
scoreboard

Monitor Wrapper

```python
import gym
from gym import wrappers
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
for i_episode in range(20):
observation = env.reset()
for t in range(100):
env.render()
print(observation)
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
if done:
print("Episode finished after {} timesteps".format(t+1))
break
env.close()
gym.upload('/tmp/cartpole-experiment-1', api_key='blah')
```