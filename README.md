# Implementation of Soft Actor-Critic (SAC) for Reinforcement Learning

![Demo File](https://github.com/thibautvalour/RL_Soft-Actor-Critic/blob/main/results/video/Start.gif)
![Demo File](https://github.com/thibautvalour/RL_Soft-Actor-Critic/blob/main/results/video/Middle.gif)
![Demo File](https://github.com/thibautvalour/RL_Soft-Actor-Critic/blob/main/results/video/presentation_hopper.gif)
![Demo File](https://github.com/thibautvalour/RL_Soft-Actor-Critic/blob/main/results/video/bipede.gif)


This is a reimplementation of the Soft Actor-Critic (SAC) algorithm as described in the original article "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" by Haarnoja et al, which is a state-of-the-art method in the field of RL.

In SAC, the agent learns a stochastic policy that maximizes the expected sum of rewards, while also maximizing its entropy. This encourages the agent to explore a wide range of actions and avoid getting stuck in local optima. The algorithm also uses a critic network to estimate the state-action value function, and updates both the policy and the critic network based on the sampled experience stored in the replay buffer. The resulting algorithm achieves state-of-the-art performance on a variety of RL tasks.

The implementation includes the SAC algorithm itself, as well as the necessary components such as the neural network architectures and the replay buffer. The code is designed to be modular and easy to use, and can be adapted to various RL environments.

#### Add a requirement file from Augustin env

## Usage
To train the SAC algorithm, create an environment compatible with the requirement.txt file, and simply run :
```
python3 main.py
```
Default env is HopperBulletEnv-v0, but you can change it as well as other training parameters such as the number of episode, batch size and size of neural networks involved in approximation functions. See main.py for a list of available command line arguments.


## Acknowledgements
This implementation is based on the original paper and the following reference implementation: https://github.com/rail-berkeley/softlearning
