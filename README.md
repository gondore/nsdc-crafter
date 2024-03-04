# nsdc-crafter
- This repo contains our work as a National Student Data Corps (NSDC) team in the Winter 2024 quarter 
- as beginners in reinforcement learning, we implemented the [classical deep reinforcement learning paper]([url](https://arxiv.org/abs/1312.5602)), _Playing Atari with Deep Reinforcement Learning_, onto [crafter]([url](https://github.com/danijar/crafter/tree/main))
- 
## Environment
- crafter is a custom research environment that was originally developed based off of the discontinued openAI gym, so some standard environment setup protocols are different since we are using gymnasium over gym
- we used default environment configs

## Reward signal

Agents are assessed by their geometric mean score of 22 achievements 

Reward score interval is `[-0.9, 22]`.

<ins>Best condition:</ins> all achievements unlocked + keeping or restoring all health until time limit is reached 

<ins>Worst condition:</ins> 0 health + no achievements unlocked

- score ≥ 21.1 indicates all achievements have been unlocked
  reward score

## Action space

 - the agent has 17 possible discrete actions to help it achieve all 22 achievements
 - some actions include placing a sapling, crafting a wooden pickaxe, etc.
 - there is no stochasticity in action execution: as long that you have the materials to craft a certain item, the action successfully executes

## DQN structure

The DQN implementation uses a convolutional neural network model defined in `ConvDQN` class formed through Pytorch modules

The first layer is a convolutional layer with 32 filters, each of size 8x8 and a stride of 4, that takes in 3 channel RGB input. The second layer has 64 filters of 4x4 each and a stride of 2. The third layer contains 64 filters each of size 3x3 and stride 1. The output from the previous layers is flattened. THe fully connected first layer is a linear layer of input size 64 * 4 * 4 and 512 output nodes, and the second layer has 512 input nodes and an output size equal to the number of actions possible, 16. We use the ReLU activation function.

# Replay Buffer
`ReplayBuffer` class is used to initialize a replay buffer that stores and samples from experiences to facilitate experience sampling 

# Target Network
The primary netwrok is updated continuously during training whyle the target network's weights are synchronized periodically with the intent to stabilize learning process and mitigate 'catastrophic forgetting'.

# Action Selection ('act' function) 
We use an off policy algorithm by selecting random actions under epsilon probability and otherwise use epsilon greedy by selecting the highest current q-value action

# Training ('train' function)
Samples a batch of experiences from the replay buffer, calculates the Q-values for the current and following states, computes the loss throuhg temporal difference error, performs backpropogation to update network weights, then decays the epsilon value.
