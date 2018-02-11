# Machine Learning Engineer Nanodegree
## Capstone Proposal

Peng Xu

Feb 9th, 2018

### 1. Domain Background

Reinforcement Learning

Deep Learning

Deep Reinforcement Learning,  Deep Q-Learning

motivation: 1. Deep Mind paper on video games 2. autonomous driving

Reinforcement learning addresses the problem of how agents should learn to take actions to maximize cumula- tive reward through interactions with the environment. The traditional approach for reinforcement learning algorithms requires carefully chosen feature representations, which are usually hand-engineered. Recently, significant progress has been made by combining advances in deep learning for learning feature representations (Krizhevsky et al., 2012; Hinton et al., 2012) with reinforcement learning, tracing back to much earlier work of Tesauro (1995) and Bert- sekas & Tsitsiklis (1995). Notable examples are training agents to play Atari games based on raw pixels (Guo et al., 2014; Mnih et al., 2015; Schulman et al., 2015a) and to acquire advanced manipulation skills using raw sensory in- puts (Levine et al., 2015; Lillicrap et al., 2015; Watter et al., 2015). Impressive results have also been obtained in train- ing deep neural network policies for 3D locomotion and manipulation tasks (Schulman et al., 2015a;b; Heess et al., 2015b).

Deep Learning was involved to solve the disadvantage of RL in high dimensional problem. One topical example of DRL is the Deep Q Network (DQN), which, after learning to play Atari 2600 games over 38 days, was able to match human performance when playing the game [Mnih et al., 2013; Mnih et al., 2015].

1.1 Motivation

Deep Reinforcement Learning has proven its ability in a varity of classic problems among video games and robotics. In some extent, a video game is equivelent to a simulation of real case, such as racing cars, drones etc. By solving a similar problem in a video game or a simulation setting, the final successful could be applied on the countpart problem in real case using transfer learning, which saves a lot of time and money or even reverse mission impossible. Among those challenging problems how to train a vehicle to be a self-learner interests me most. 

This paper takes the first steps towards enabling DQNs to be used for learning robotic manipulation. We focus on learning these skills from visual observation of the manipulator, without any prior knowledge of config- uration or joint state. Towards this end, as first steps, we assess the feasibility of using DQNs to perform a simple target reaching task, an important component of general manipulation tasks such as object picking. In particular, we make the following contributions:

### Problem Statement

In openAI-Universe, a racing car game, "", is provided, in which a vehicle is simply controlled by 3 inputs, left, right, forward. It is easily to be redesigned in a Reinoforcement Learning setting. The vehicle interacts with its environment in the way as shown in below. It is expected that the racing car can grasp a smart driving behavior after a long-term training leading to a maximal reward or namely score here.

one relevant potential solution: Deep Q-Learning to train an autonomous vehicle in a video game.

mathematical or logical terms

### Datasets and Inputs

real time image pixel level input

### Solution Statement

Deep Q-Learning

A positive reward is given only when the robot reaches the goal region. Reversely, a negative reward is given whenever it collides with other cars or the obstacles.

The scoring system plays a role as q-value in the environment setting.

### Benchmark Model

Deep Mind atari game

Google DeepMind released its remarkable paper on Deep Q-Learning leveling up the video game computer player. Deep Reinforcement learning has been proven its ability dealing with human-leveled control and learning problem for the first time. Alphago from GoogleMind even beten the best human players in Go game.



### Evaluation Metrics

Scores in the game

Human player performance in the game.

### Project Design

a theoretical workflow
