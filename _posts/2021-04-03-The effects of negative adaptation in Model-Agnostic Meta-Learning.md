---
layout: post
title:  The effects of negative adaptation in Model-Agnostic Meta-Learning
published: true
---

An overview of the paper “[The effects of negative adaptation in Model-Agnostic Meta-Learning](https://arxiv.org/pdf/1812.02159.pdf)”.
<!--break-->
The author points out that the adaptation algorithm like MAML can significantly recrease the performance of an agent in a meta-reinforcement learning setting, even on a range of meta-learning tasks. All images and tables in this post are from their paper.
The capacity of meta-learning algorithms to quickly adapt to a variety of tasks, including ones they did not experience during meta-training, has been a key factor in the recent success of these methods on few-shot learning problems. This particular advantage of using meta-learning over standard supervised or reinforcement learning is only well founded under the assumption that the adaptation phase does improve the performance of our model on the task of interest. However, in the classical framework of meta-learning, this constraint is only mildly enforced, if not at all, and we only see an improvement on average over a distribution of tasks.

## Introduction

Humans are capable of learning new skills and quickly adapting to new tasks they have never experienced before, from only a handful of interactions. Likewise, meta-learning benefits from the same capacity of fast learning in the low-data regime. The main advantage of using meta-learning over standard supervised or reinforcement learning relies on the premise that this adaptation phase actually increases the performance of our model. Howecer, there is no guarantee that the adaptation phase shows some improvement at the scale of an individual task.

## MAML in RL

### Reinforcement Learning

In this paper, the authors only consider the meta-reinforcement learning setting in this paper. In the context of meta-RL, a task <img src="https://latex.codecogs.com/svg.latex?T&space;=&space;<S,A,p(s'|s,a),r(s,a)>" title="T = <S,A,p(s'|s,a),r(s,a)>" /> is defined as a Markov Decision Process. For some discount factor <img src="https://latex.codecogs.com/svg.latex?\gamma&space;\in&space;[0,1]" title="\gamma \in [0,1]" />, they return <img src="https://latex.codecogs.com/svg.latex?G_t(\pi)" title="G_t(\pi)" /> at time <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> is a random variable corresponding to the discounted sum of rewards observed after <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> following a policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?G_t(\pi)&space;=&space;R_{t&plus;1}&plus;\gamma&space;R_{t&plus;2}&space;&plus;&space;...&space;=&space;\sum&space;_{k=0}^{&space;\infty}&space;\gamma&space;^k&space;R_{t&plus;k&plus;1}" title="G_t(\pi) = R_{t+1}+\gamma R_{t+2} + ... = \sum _{k=0}^{ \infty} \gamma ^k R_{t+k+1}" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?R_{t&plus;1}" title="R_{t+1}" /> is the random reward received after taking action <img src="https://latex.codecogs.com/svg.latex?A_t&space;=&space;\pi(S_t)" title="A_t = \pi(S_t)" /> in state <img src="https://latex.codecogs.com/svg.latex?S_t" title="S_t" />.
In meta-RL, low data regime corresponds to having limited amount of interactions with the task of interest.

### MAML

In this paper, the authors are interested in a meta-learning method based on parameter adaptation and inspired by fine-tuning called MAML (Model Agnostic Meta-Learning). The idea of MAML is to find a set of initial parameters <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> of our policy <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}" title="\pi_{\theta}" />, such that only a single step of gradient descent is necessary to get new parameters <img src="https://latex.codecogs.com/svg.latex?\theta'_{T}" title="\theta'_{T}" />, where the corresponding policy <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta'_{T}}" title="\pi_{\theta'_{T}}" /> is adapted on the task <img src="https://latex.codecogs.com/svg.latex?T" title="T" />. More precisely, given a dataset <img src="https://latex.codecogs.com/svg.latex?D_T" title="D_T" /> of trajectories samples from task <img src="https://latex.codecogs.com/svg.latex?T" title="T" />, following the policy <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}" title="\pi_{\theta}" />, and a corresponding loss function <img src="https://latex.codecogs.com/svg.latex?L" title="L" />, MAML returns new parameters <img src="https://latex.codecogs.com/svg.latex?\theta&space;'_{T}" title="\theta '_{T}" /> defined as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\theta&space;'_{T}&space;=&space;\theta&space;-&space;\alpha\triangledown&space;_{\theta}L(\theta;D_t)" title="\theta '_{T} = \theta - \alpha\triangledown _{\theta}L(\theta;D_t)" />
</p>

## Negative Adaptation

The minimization of the meta-objective loss only encourages the adaptive policy to have a high expected return, without any consideration of the policy we started with. There is no incentive for MAML to produce adapted parameters that improve performance on the task of interest <img src="https://latex.codecogs.com/svg.latex?T" title="T" /> over the initial policy.
The regime on which MAML shows negative adaptation seems to correspond to velocities where the return of the initial policy is already at its maximum. Intuitively, the meta-learning algorithm is unable to produce a better policy because it was already performing well on those tasks, leading to this decrease in performance. The authors believe that this overspecialization on some tasks could explain the negative adaptation.
<p align="center">
<b>Negative Adaptation example.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/23/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Discussion

In order to mitigate the effects of negative adaptation, we need to include a constraint on the improvement in our definition of the meta-objective. To characterize the improvement more precisely for a fixed task <img src="https://latex.codecogs.com/svg.latex?T" title="T" />, the authors introduce a random variable <img src="https://latex.codecogs.com/svg.latex?\Gamma&space;_T(\theta)" title="\Gamma _T(\theta)" />, which is the difference between the returns of the policies before and after the parameter update:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Gamma&space;_T(\theta)&space;=&space;G_0(\pi_{\theta})&space;-&space;G_0(\pi_{\theta'_T})" title="\Gamma _T(\theta) = G_0(\pi_{\theta}) - G_0(\pi_{\theta'_T})" />
</p>
Avoiding the negative adaptation translate to havin <img src="https://latex.codecogs.com/svg.latex?\Gamma&space;_T(\theta)&space;\leq&space;0" title="\Gamma _T(\theta) \leq 0" />, with high probability. Ideally, we would like to enforce this constraint over all tasks. However, in early experiments on MAML with this modified meta-objective, the authors were not able to significantly reduce the effect of negative adaptation, and more research is necessary.   
