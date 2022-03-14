---
layout: post
title: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
published: true
---

An overview of the paper “[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)”.
<!--break-->
The authors propose a general-purpose multi-agent learning algorithm that : (1) leads to learned policies that only use local information (i.e., their own observations) at execution time, (2) does not asssume a differentiable model of the environment dynamic and (3) is applicable to cooperative, competitive and mixed/social-dilemma environments. All images and tables in this post are from their paper.

## Introduction

In the normal RL environment, the transition dynamics  depend on the action and state pair of the given agent. However, in MARL, the transition dynamics depend on the state-action pair of all agents. This creates a natural soure of non-stationarity in RL problems. So we can imagine that if all agents but one have their policy fixed, then the environment would be stationary in terms of learning agent, since the behavior of all other agents are fixed. However, in a more realistic setting, this becomes tricky. Common approaches such as Q-learning or policy gradient will not converge due to this non-stationarity issue.


## Background

In MARL, the environments can be broadly divided into three categories:
* **Co-operative games:** In co-operative games, all the agents share the same reward function, such that they are incentivized to achieve the same behavior in the end.
* **Competitive games:** In these games, the model might share same reward but work in adversarial fashion.
* **Mixed-incentive/social dilemma games:** In these games, there may be some kind of alignment among agents where the rewards might differ and do something as well.

In this paper, the authors advocate the approach of centralized learning followed by decentralized execution.

## Methodology

### Mutli-agent Actor-Critic

The approach proposed by the authors is a centralized version of the policy gradient approach.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\bigtriangledown J(\theta_i) = \mathbb{E}[\bigtriangledown_{\theta_i}\log \pi_i(a_i|o_i)Q_i^{\pi}(x,a_1,a_2,...,a_N)]" title="\bigtriangledown J(\theta_i) = \mathbb{E}[\bigtriangledown_{\theta_i}\log \pi_i(a_i|o_i)Q_i^{\pi}(x,a_1,a_2,...,a_N)]" />
</p>

Here <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" /> is a *centralized action-value function* that takes as input the actions of all agents, in addition to some state informaion (concatenation of all states of agents). Since each <img src="https://latex.codecogs.com/svg.latex?Q_i^{\pi}" title="Q_i^{\pi}" /> is learned separately, agents can have arbitrary reward structures, including conflicting rewards in a competitive setting.

### Inferring Policies of Other Agents

To remove the assumption of knowing other agents' policies, an approximate policy is learned by maximizing the log probability of j's actions with an entropy regularizer.

### Agents with Policy Ensembles

To aid exploration process, they use a ensemble of policies to acoid overfitting and non-stationarity due to the agents' changing policies.

## Conclusion

The authors propose an approach for MARL, and experiment on different environments such as different settings of *multi-agent particle environments*. Another important reference to share for a baseline for MARL includes "[pettingzoo.ml](https://www.pettingzoo.ml/envs)".
