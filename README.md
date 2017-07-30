# Learning to Balance with Double Q-Learning
![Demo](/imgs/doubleq.gif)

## Dependencies
````
keras version 2.0.6
gym version 0.9.2
numpy 1.13.1
matplotlib 2.0.0
````

## Double Q-Learning
[Deep RL with Double Q-Learning](https://arxiv.org/abs/1509.06461):
Double Q-Learning (not to be confused with the DDQN algorithm), is an extension of Q-Learning with the goal of detangling action-selection from value prediction in the Q-Learning update rule. Instead of using a single Q-function to represent state-action values, two separate Q-functions are trained simulatenously. At each step of learning it is randomly chosen which network will be trained, and which network will act as the "target network" (a DQN term). The network chosen for training makes a prediction of the values taking each action at the given state. The value for the actual action chosen is updated as the reward for taking that action plus what the target network predicts is the value for the next state. This "seperation of powers" prevents a single network from over-estimating state-action values (a problem seen in DQN). It also leads to more stable learning in my experience. 

Double Q-Learning itself is a tabular algorithm, but just like Q-Learning, it can be expanded to continuous state spaces by the use of function approximators. In this case we use two artificial neural networks to represent our Q functions. 

## CartPole-v1
[OpenAI](https://gym.openai.com/envs/CartPole-v1)

This is an environment from OpenAI gym. The goal is to balance the pole for 500 steps by moving the cart left and right. The state contains information about the location/speed of the cart and pole.

This is fairly simple problem to solve, and Double Q-Learning is not the only (or best) solution for this environment. 

## Double Q-Learning w/experience replay
Double-Q Learning seems to be more robust to hyperparameter choices than many of the other Q-Learning variants. For this problem, I use an Adam optimizer with default learning rate, two small hidden layers with relu activation, and a linear output layer so that we don't restrict the actual values. These may not be the optimal choices, as I didn't do any parameter tuning (but they work). 

One important trick to getting good results with neural networks in RL is that of experience replay. Essentially, we save each experience the agent sees during its interaction with the environment. To learn we randomly sample several of these "memories" to use for training. This helps avoid "forgetting" parts of the state space experienced in the past.

## Results
Double Q-Learning solves the CartPole problem with ease!
![](/imgs/Learning.png)
