import cv2
import csv
import numpy as np
import random
from collections import deque
import gym
import universe
import tensorflow as tf
from tensorflow.contrib.layers import flatten


ACTIONS = ['LeftTurnAccel', 'ForwardAccel', 'RightTurnAccel']
KEYS = [[('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)],
        [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)],   
        [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]]


# hyper params:
NUM_ACTIONS = 3
GAMMA = 0.995
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.005
EXPLORE = 100000
OBSERVE = 10000
REPLAY_MEMORY = 1000000
BATCH = 100
ENV_ID = 'flashgames.CoasterRacer-v0'
# ['flashgames.DuskDrive-v0', 'flashgames.CoasterRacer-v0', 'flashgames.CoasterRacer3-v0', 'flashgames.NeonRace-v0']


# crop video frame so NN is smaller and set range between 1 and 0; and
# stack-a-bitch!
def processFrame(observation_n):
    if observation_n is not None:
        obs = observation_n[0]['vision']
        # crop
        obs = cropFrame(obs)
        # downscale resolution
        obs = cv2.resize(obs, (32, 32))
        # greyscale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # Convert to float
        obs = obs.astype(np.float32)
        # scale from 1 to 255
        obs *= (1.0 / 255.0)
        # re-shape a bitch
        obs = np.reshape(obs, [32, 32])
    return obs

# crop frame to only flash portion:


def cropFrame(obs):
    # adds top = 84 and left = 18 to height and width:
    return obs[284:564, 18:658, :]


# Add appropiate actions to system
def appendActions(observation_n, argmax_t, previous_argmax):
    action = KEYS[np.argmax(argmax_t)]
    actions_n = ([action for obs in observation_n])
    return actions_n, argmax_t


def createGraph():

    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]), name='W_conv1')
    b_conv1 = tf.Variable(tf.zeros([32]), name='b_conv1')

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]), name='W_conv2')
    b_conv2 = tf.Variable(tf.zeros([64]), name='b_conv2')

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]), name='W_conv3')
    b_conv3 = tf.Variable(tf.zeros([64]), name='b_conv3')

    W_fc4 = tf.Variable(tf.zeros([2304, 512]), name='W_fc4')
    b_fc4 = tf.Variable(tf.zeros([512]), name='b_fc4')

    W_fc5 = tf.Variable(tf.zeros([512, NUM_ACTIONS]), name='W_fc5')
    b_fc5 = tf.Variable(tf.zeros([NUM_ACTIONS]), name='b_fc5')

    # input for pixel data
    inp = tf.placeholder("float", [None, 80, 80, 4], name='input')

    # Computes rectified linear unit activation fucntion on  a 2-D convolution
    # given 4-D input and filter tensors. and
    conv1 = tf.nn.relu(tf.nn.conv2d(inp, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    # flatten conv3:
    conv3_flat = tf.reshape(conv3, [-1, 2304])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    out = tf.matmul(fc4, W_fc5) + b_fc5

    return inp, out


def createLeNet():    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    # input for pixel data
    inp = tf.placeholder("float", [None, 32, 32, 4], name='input')
    
    # Layer 1: Convolutional. Input = 80x80x4. Output = 28x28x6.
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,4,6], mean=mu, stddev=sigma), name='W_conv1')
    b_conv1 = tf.Variable(tf.zeros(6), name='b_conv1')
    conv1 = tf.nn.relu(tf.nn.conv2d(inp, W_conv1, strides=[1,1,1,1], padding='VALID') + b_conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean=mu, stddev=sigma), name='W_conv2')
    b_conv2 = tf.Variable(tf.zeros(16),  name='b_conv2')
    conv2 = tf.nn.relu(tf.nn.conv2d(pool_1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') 
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    W_fc1 = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma), name='W_fc1')
    b_fc1 = tf.Variable(tf.zeros(120), name='W_fc1')
    fc1 = tf.nn.relu(tf.matmul(fc1, W_fc1) + b_fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    W_fc2 = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma), name='W_fc2')
    b_fc2 = tf.Variable(tf.zeros(84), name='b_fc2')
    fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    W_fc3 = tf.Variable(tf.truncated_normal(shape = (84,NUM_ACTIONS), mean = mu , stddev = sigma), name='W_fc3')
    b_fc3 = tf.Variable(tf.zeros(NUM_ACTIONS), name='b_fc3')
    out = tf.matmul(fc2, W_fc3) + b_fc3

    return inp, out


# loading saved models
def restoreSession(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("../saved_models")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    return sess


# deep q network. feed in pixel data to graph session
def trainGraph(inp, out, sess):

    # to calculate the argmax, we multiply the predicted output with a vector
    # with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, NUM_ACTIONS])
    gt = tf.placeholder("float", [None])  # ground truth

    # action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices=1)
    # cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(action - gt))
    # optimization fucntion to reduce our minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialise universe/gym kak:
    env = gym.make(ENV_ID)
    # env.configure(fps=5.0, remotes=1, start_timeout=15 * 60)
    env.configure(fps=5.0, remotes='vnc://localhost:5900+15901', start_timeout=15*60)

    # create a queue for experience replay to store policies
    D = deque()

    # intial frame
    env.reset()

    observation_n, reward_t, done_t, info = env.step([[('KeyValue', 'ArrowUp', True)]])
    while info['n'][0]['env_status.env_state'] is None:
        observation_n, reward_t, done_t, info = env.step([[('KeyValue', 'ArrowUp', True)]])
        env.render()

    observation_t = processFrame(observation_n)

    # stack frames, that is our input tensor
    inp_t = np.stack((observation_t, observation_t, observation_t, observation_t), axis=2)

    # saver
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    t = 0
    epsilon = INITIAL_EPSILON
    previous_argmax = 0

    print(done_t)
    print(info)

    fname = '../result/dqn_reward_history.csv'
    wfile = open(fname, 'w')
    writer = csv.writer(wfile, delimiter=' ')

    # training time
    while t <= 100000:

        # output tensor
        out_t = out.eval(feed_dict={inp: [inp_t]})
        # argmax function
        argmax_t = np.zeros([NUM_ACTIONS])

        #
        if random.random() <= epsilon:
            maxIndex = random.randrange(NUM_ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        action_t, previous_argmax = appendActions(observation_n, argmax_t, previous_argmax)
        observation_n, reward_t, done_t, info = env.step(action_t)
        env.render()

        while observation_n[0] is None:
            observation_n, reward_t, done_t, info = env.step([[('KeyValue', 'ArrowUp', True)]])

        # print('reward: ', type(reward_t[0]))
        if reward_t[0] >= 1000:
            reward_t[0] -= 1000

        observation_t = processFrame(observation_n)

        inp_t1 = np.append(np.reshape(observation_t, [32, 32, 1]), inp_t[:, :, 0:3], axis=2)

        # add our input tensor, argmax tensor, reward and updated input tensor
        # to stack of experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # training iteration
        if t > OBSERVE:

            # get values from our replay memory
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))  # target = reward + gamma * max(Q)
            gt_batch = np.mean(gt_batch, axis=1)

            # train on that
            train_step.run(feed_dict={gt: gt_batch, argmax: argmax_batch, inp: inp_batch})

        # update our input tensor the the next frame
        inp_t = inp_t1
        t = t + 1

        # print our where wer are after saving where we are
        if t % 1000 == 0:
            saver.save(sess, '../saved_models/' + 'CoasterRacer-dqn', global_step=t)

        print("TIMESTEP: ", t,  "\tEPSILON: ", epsilon, "\tACTION: ", ACTIONS[maxIndex], "   \tREWARD: ", reward_t[0], "  \tQ_MAX: %e" % np.max(out_t))
        writer.writerow([t, reward_t[0]])

def main():
    # create session
    sess = tf.InteractiveSession()

    # input layer and output layer by creating graph
    # inp, out = createGraph()
    inp, out = createLeNet()

    # restore sess
    sess = restoreSession(sess)

    # train our graph on input and output with session variables
    trainGraph(inp, out, sess)


if __name__ == "__main__":
    main()
