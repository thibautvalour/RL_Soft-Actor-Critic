# Reinforcement_Learning : SAC Article Reimplementation

This is a reimplementation of the Soft Actor-Critic (SAC) algorithm as described in the original article "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" by Haarnoja et al.

#### Add a requirement file from Augustin env

## Usage
To train the SAC algorithm, create an environment compatible with the requirement.txt file, and simply run :
```
python3 main.py
```
Default env is HopperBulletEnv-v0, but you can change it as well as other training parameters such as the number of episodes and batch size and size of neural networks involved in the approximation functions. See main.py for a list of available command line arguments.


## Acknowledgements
This implementation is based on the original paper and the following reference implementation: https://github.com/rail-berkeley/softlearning
