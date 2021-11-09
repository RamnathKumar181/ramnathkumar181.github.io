---
layout: post
title: Learning an Embedding Space for Transferable Robot Skills
published: true
---

An overview of the paper “[Learning an Embedding Space for Transferable Robot Skills](https://openreview.net/pdf?id=rk07ZXZRb)”.
<!--break-->
The author presents a method for reinforcement learning of closely related skills that are parameterized via a skill embedding space. All images and tables in this post are from their paper.

## Introduction

The idea is to learn such skills by taking advantage of latent variables and exploiting a connection between reinforcement learning and variational inference. The main contribution of their work is an entropy regularized policy gradient formulation for hierarchical policies, and an associated, data-efficient and robust off-policy gradient algorithm based on stochastic value gradients. They also show that, the proposed technique can interpolate and/or sequence previously learned skills in order to accomplish more complex tasks, even in the presence of sparse rewards.
Their method learns manipulation skills that are continuously parameterized in an embedding space. The authors show how to take advantage of these skills for rapidly solving new tasks, effectively by solving the control problem in the embedding space rather than the action space.

## Preliminaries

The authors perform reinforcement learning in Markov Decision processes (MDP). We denote state <img src="https://latex.codecogs.com/svg.latex?s&space;\in&space;\mathbb{R}^S" title="s \in \mathbb{R}^S" /> the continuous state of the agent; <img src="https://latex.codecogs.com/svg.latex?a&space;\in&space;\mathbb{R}^A" title="a \in \mathbb{R}^A" /> denotes the action <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" /> in <img src="https://latex.codecogs.com/svg.latex?s_t" title="s_t" />. Actions are drawn from a policy distribution <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}(a|s)" title="\pi_{\theta}(a|s)" />, with parameters <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />; in this case, a Gaussian distribution whose mean and diagonal covariance are parameterized via a neural network. At every step the agent receives a scalar reward <img src="https://latex.codecogs.com/svg.latex?r(s_t,a_t)" title="r(s_t,a_t)" /> and we consider the problem of maximizing the sum of discounted rewards <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{\tau_{\pi}}[\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t)]" title="\mathbb{E}_{\tau_{\pi}}[\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t)]" />.

## Learning Versatile Skills

The general goal of the method is to re-use skills learned for an initial set of tasks to speed up - or in some cases even enable - learning difficult target tasks in a transfer learning setting.

The authors train a multi-task setup, where the task id is given as one-hot input to the embedding network. The embedding network generates an embeedding distribution that is sampled and concatenated the current observation to serve as input to the policy. After interacting with the environment, a segment of states is collected and fed into the inference network. The inference network is trained to classify what embedding vector the segment of states was generated from.

<p align="center">
<b>Summary of proposed approach.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/1/Figure-4.png?raw=true" alt="Figure 4"/>
</p>

### Policy Learning via a Variational Bound on Entropy Regularized RL

To learn the skill-embedding, we assume to have access to a set of initial tasks <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}&space;=&space;[1,...,T]" title="\mathcal{T} = [1,...,T]" /> with accompanying, per-task reward functions <img src="https://latex.codecogs.com/svg.latex?r(s,a)" title="r(s,a)" />, which could be comprised of different environments, variable robot dynamics, reward functions, etc. During training time, we provide access to the task id <img src="https://latex.codecogs.com/svg.latex?t&space;\in&space;\mathcal{T}" title="t \in \mathcal{T}" /> (indicaating which task the agent is operating in) to our RL agent. To obtain data from from all training tasks for learning - we draw a task and its id randomly from the set of tasks <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}" title="\mathcal{T}" /> at the beginning of each episode and execute the agents current policy <img src="https://latex.codecogs.com/svg.latex?\pi(a|s,t)" title="\pi(a|s,t)" /> it it.

For our policy to learn a diverse set of skills instead of just <img src="https://latex.codecogs.com/svg.latex?T" title="T" /> seperate solutions (one per task), we endow it with a task-conditional latent variable <img src="https://latex.codecogs.com/svg.latex?z" title="z" />. The idea is that, with latent variable could also be called "skill embeddings", where the policy is able to represent a distribution over skilss for each task and to share these across tasks. In the simplest case, this latent variable could be resampled at every timestep and the state-task conditional policy would be defined as <img src="https://latex.codecogs.com/svg.latex?\inline&space;\pi(a|s,t)&space;=&space;\int&space;\pi(a|z,s,t)p(z|t)dz" title="\pi(a|s,t) = \int \pi(a|z,s,t)p(z|t)dz" />. One option would be to let <img src="https://latex.codecogs.com/svg.latex?\inline&space;z&space;\in&space;1,...,K" title="z \in 1,...,K" />, in which case the policy would correspond to a mixture of <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> subpolicies.

Introducing a latent variable facilitates the representation of several alternative solutions but it does not mean that several alternative solutions will be learned. To achieve this, the authors formulate the objective as an entropy regularized RL problem:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\max_{\pi}&space;\mathbb{E}_{\pi,&space;p_0,&space;t&space;\in&space;\mathcal{T}}&space;[\sum_{i=0}^{\infty}\gamma^i(r_t(s_i,a_i)&space;&plus;&space;\alpha\mathcal{H}[\pi(a_i|s_i,t)])|&space;a_i&space;\sim&space;\pi(.|s,t),s_{i&plus;1}\sim&space;p(s_{i&plus;1}|a_i,s_i)]" title="\max_{\pi} \mathbb{E}_{\pi, p_0, t \in \mathcal{T}} [\sum_{i=0}^{\infty}\gamma^i(r_t(s_i,a_i) + \alpha\mathcal{H}[\pi(a_i|s_i,t)])| a_i \sim \pi(.|s,t),s_{i+1}\sim p(s_{i+1}|a_i,s_i)]" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?p_0(s_0)" title="p_0(s_0)" /> is the initial state distribution, <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> is the weighting term - trading the arbitrarily scaled reward against the entropy - and we can define <img src="https://latex.codecogs.com/svg.latex?R(a,s,t)&space;=&space;\mathbb{E}_{\pi}[\sum_{i=0}^{\infty}\gamma^ir_t(s_i,a_i)|s_0=s,&space;a_i&space;\sim&space;\pi(.|s,t)]" title="R(a,s,t) = \mathbb{E}_{\pi}[\sum_{i=0}^{\infty}\gamma^ir_t(s_i,a_i)|s_0=s, a_i \sim \pi(.|s,t)]" /> to denote the expected return for task <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> (under policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />) when starting from state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> and taking action <img src="https://latex.codecogs.com/svg.latex?a" title="a" />. The entropy regularization term is defined as: <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathcal{H}[\pi(a|s,t)]&space;=&space;\mathbb{E}_{\pi}[-\log&space;\pi(a|s,t)]" title="\mathcal{H}[\pi(a|s,t)] = \mathbb{E}_{\pi}[-\log \pi(a|s,t)]" />. With some mathematical rigor, the authors adapt the above regularization term to work in the setting of latent variables.

The resulting equation maximizes the entropy of the embedding given the task <img src="https://latex.codecogs.com/svg.latex?\mathcal{H}[p(z|t)]" title="\mathcal{H}[p(z|t)]" /> and the entropy of the policy conditioned on the embedding <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{p(z|t)}\mathcal{H}(\pi(a|s,z))" title="\mathbb{E}_{p(z|t)}\mathcal{H}(\pi(a|s,z))" /> (thus, aiming to cover the embedding space with different skill clusters). The negative CE encourages different embedding vectors <img src="https://latex.codecogs.com/svg.latex?z" title="z" /> to have different effects in terms of executed actions and visited states: Intuitively, it will be high when we can predict <img src="https://latex.codecogs.com/svg.latex?z" title="z" /> from resulting action <img src="https://latex.codecogs.com/svg.latex?a" title="a" />, <img src="https://latex.codecogs.com/svg.latex?s^H" title="s^H" />.

## Learning an Embedding for Versatile Skills in an Off-Policy Setting

The objective presented above could be optimized directily only in an on-policy setting. To optimize in the off-policy setting, some minor approximations are required.
The authors assume that the availability of a replay buffer <img src="https://latex.codecogs.com/svg.latex?\mathcal{B}" title="\mathcal{B}" /> (containing full trajectory execution traces including states, actions, task id and reward), that is incrementally filled during training. Additional to the trajectory traces, we also store the probabilities of each selected action and denote them with the behavior policy probability <img src="https://latex.codecogs.com/svg.latex?b(a|z,s,t)" title="b(a|z,s,t)" /> as well as the behaviour probabilities of the embedding <img src="https://latex.codecogs.com/svg.latex?b(z|t)" title="b(z|t)" />.
Given this replay data, we formulate the off-policy perspective of our algorithm as follows:
* We start with the notion of a *lower-bound Q-function* that depends on both state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> and action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> and is conditioned on both, the embedding <img src="https://latex.codecogs.com/svg.latex?z" title="z" /> and the task id <img src="https://latex.codecogs.com/svg.latex?t" title="t" />.
* This encapsulates all time dependent terms from previous equation:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q^{\pi}(s_i,a_i;z,t)&space;=&space;\widehat{r}(s_i,&space;a_i,z,t)&space;&plus;&space;\gamma&space;\mathbb{E}_{p(s_{i&plus;1}|a_i,s_i)}[Q^{\pi}(s_{i&plus;1},a_{i&plus;1};z,t)]" title="Q^{\pi}(s_i,a_i;z,t) = \widehat{r}(s_i, a_i,z,t) + \gamma \mathbb{E}_{p(s_{i+1}|a_i,s_i)}[Q^{\pi}(s_{i+1},a_{i+1};z,t)]" />
</p>

The exact math to train the embedding network, and inference network is best described in the paper.

## Learning to Control the Previously-Learned Embedding

Once the skill-embedding is learned using the described multi-task setup, we utilize it to learn a new skill. There are multiple possibilities to employ the skill-embedding in such a scenario including fine-tuning the entire policy or learning only a new mapping to the embedding space.In this work, the authors focus on the latter: To adapt to a new task we freeze the policy network and only learn a new state embedding mapping <img src="https://latex.codecogs.com/svg.latex?z&space;=&space;f_{\upsilon}(x)" title="z = f_{\upsilon}(x)" /> via a neural network <img src="https://latex.codecogs.com/svg.latex?f_{\upsilon}" title="f_{\upsilon}" /> (parameterized by parameters <img src="https://latex.codecogs.com/svg.latex?\upsilon" title="\upsilon" />). On other words, we only allow the network to learn how to modulate and interpolate between the already-learned skills, but we do not allow to change the underlying policies.
The authors support the efficiacy of their model by training on multiple tasks. 
