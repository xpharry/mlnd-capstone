import gym
import universe # register the universe environments
import random
import csv

env = gym.make('flashgames.CoasterRacer-v0')  # You can run many environment in parallel
env.configure(remotes=1)  # automatically creates a local docker container
# env.configure(remotes='vnc://localhost:5900+15901')

# define our turns or keyboard actions
left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]

fname = '../result/random_reward_history.csv'
wfile = open(fname, 'w')
writer = csv.writer(wfile, delimiter=' ')

observation_n = env.reset()  # Initiate the environment and get list of observations of its initial state
t = 0
while t <= 100000:
    t += 1
    action = random.choice([left, right, forward])
    action_n = [action for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)  # Reinforcement Learning action by agent
    if reward_n[0] >= 1000:
    	reward_n[0] -= 1000
    print("TIME STEP:", t, "    \tREWARD:", reward_n[0])
    writer.writerow([t, reward_n[0]])
    #  env.render()  # Run the agent on the environment
