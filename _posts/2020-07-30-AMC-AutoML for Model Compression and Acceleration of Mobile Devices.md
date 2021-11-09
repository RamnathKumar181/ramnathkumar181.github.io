---
layout: post
title:  AMC-AutoML for Model Compression and Acceleration of Mobile Devices
published: true
---

An overview of the paper “[AMC: AutoML for Model Compression and Acceleration of Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf)”.
<!--break-->
This paper proposes a novel compression algorithm which allows us to efficiently deploy neural network models on mobile devices, etc. All images and tables in this post are from their paper.

Previously, models were compressed using hand-crafted heuristics and rule-based policies that require domain expert knowledge. Their algorithm is automated and less time consuming than others of the same kind. They create a reinforcement learning model called DDPG which penalizes accuracy loss while encouraging model shrinking and speedup. This agent works in a layerwise manner.

## AMC

The AMC algorithm can be broadly broken down into 3 different parts as follows.

<p align="center">
<b>Overview of AutoML for Model Compression (AMC) engine.</b>
</p>
<p align="center">
<img src="/assets/Papers/31/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

### Compression strategy

Two types of pruning strategy are experimented with in this work. The fine-grained pruning strategy prunes unimportant values in the weights or the weights with least magnitude. Coarse-grained pruning refers to dropping entire channels,rows or columns.

### Policy gradient RL

For this section, the deep deterministic policy gradient(DDPG) is employed. The state space <img src="https://latex.codecogs.com/svg.latex?s_t" title="s_t" /> is defined by 11 different features for layer <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, where <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> is layer index. Before being passed to the agent, they are scaled with <img src="https://latex.codecogs.com/svg.latex?[0,1]" title="[0,1]" />. The agent receives an embedding state <img src="https://latex.codecogs.com/svg.latex?s_t" title="s_t" /> of layer <img src="https://latex.codecogs.com/svg.latex?L_t" title="L_t" /> from the environment and then outputs a sparsity ratio as action <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" /> . The underlying layer is compressed with <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" /> using a specified compression algorithm. Then the agent moves to the next layer <img src="https://latex.codecogs.com/svg.latex?L_{t&plus;1}" title="L_{t+1}" /> , and receives state <img src="https://latex.codecogs.com/svg.latex?s_{t&plus;1}" title="s_{t+1}" />. After finishing the final layer <img src="https://latex.codecogs.com/svg.latex?L_{T}" title="L_{T}" /> , the reward accuracy is evaluated on the validation set and
returned to the agent.

### Reward function

Based on the constraint, the reward function is given as follows:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?R_{\mathit{metric}}&space;=&space;-\mathit{Error}&space;*&space;\log(\mathit{metric})" title="R_{\mathit{metric}} = -\mathit{Error} * \log(\mathit{metric})" />
</p>

In resource constrained optimization, we do not use a constraint. i.e. <img src="https://latex.codecogs.com/svg.latex?R_{\mathit{error}}&space;=&space;-\mathit{Error}" title="R_{\mathit{error}} = -\mathit{Error}" />. For instance, in fine-grained pruning, a is allowed to take any value for the first few layers, and later limits a, allowing an agent to achieve a target compression ratio.
In accuracy constrained optimization, we use FLOPs, #params and latency in the reward function as defined earlier.This also shows that any type of metric: proxy(FLOPs, #params) and real-time(latency) can be plugged in.
The paper goes on to provide clear experimental results on CIFAR-10 and a subset of ImageNet which clearly explain the performance boost over all previous research works.
