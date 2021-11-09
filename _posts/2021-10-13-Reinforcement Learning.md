---
layout: post
title: Reinforcement Learning
published: true
---

An overview of the topic “[A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)”.
<!--break-->
Say, we have an agent in an unknown environment and this agent can obtain some rewards by interacting with the environment. The agent ought to take actions so as to maximize cumulative rewards. The goal of Reinforcement Learning (RL) is to learn a good strategy for the agent from experimental trials and relatively simple feedback received. With the optimal strategy, the agent is capable to actively adapt to the environment to maximize future rewards.

<p align="center">
<b>An agent interacts with the environment, trying to take smart actions to maximize cumulative rewards.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Background

The agent is acting in an **environment**. How the environment reacts to certain actions is defined by a **model** which we may or may not know. The agent can stay in one of the many **states** (<img src="https://latex.codecogs.com/svg.latex?s&space;\in&space;S" title="s \in S" />) of the environment, and choose to take one of the many **actions** (<img src="https://latex.codecogs.com/svg.latex?a&space;\in&space;A" title="a \in A" />) to switch from one state to another. Which state the agent will arrive in is decided by transition probabilities between states (<img src="https://latex.codecogs.com/svg.latex?P" title="P" />). Once an action is taken, the environment delivers a reward (<img src="https://latex.codecogs.com/svg.latex?r&space;\in&space;R" title="r \in R" />) as feedback.

The model defines the reward functions and transition probabilities. We may or may not know how the model works and this differentiate two circumstances:
* **Know the model**: planning with perfect information; do model-based RL. When we fully know the environment, we can find the optimal solution by Dynamic Programming.
* **Do not know the model**: learning with incomplete information; do model-free RL or try to learn the model explicitly as part of the algorithm. Most of the following content serves the scenarios when the model is unknown.

The agent's **policy** (<img src="https://latex.codecogs.com/svg.latex?\pi(s)" title="\pi(s)" />) provides the guidelines on what is the optimal action to take in a certain state with the goal to maximize total rewards. Each state is associated with a **value** function <img src="https://latex.codecogs.com/svg.latex?V(s)" title="V(s)" /> predicting the expected amount of future rewards we are able to receive in this state by acting the corresponding policy. In other words, the value function quantifies how good a state is. Both policy and value functions are what we try to learn in reinforcement learning.

<p align="center">
<b>Summary of approaches in RL based on whether we want to model the value, policy, or the environment.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

The interaction between the agent and the environment involves a sequence of actions and observed rewards in time, <img src="https://latex.codecogs.com/svg.latex?t&space;=&space;1,2,...,T" title="t = 1,2,...,T" />. During the process, the agent accumulates the knowledge about the environment, learns the optimal policy, and makes decisions on which action to take next so as to efficiently learn the best policy. Let's label the state, action, and reward at time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> as <img src="https://latex.codecogs.com/svg.latex?S_t" title="S_t" />, <img src="https://latex.codecogs.com/svg.latex?A_t" title="A_t" />, and <img src="https://latex.codecogs.com/svg.latex?R_t" title="R_t" />, respectively. Thus, the interaction sequence is fully described by one **episode** (also known as "trial" or "trajectory") and the sequence ends at the terminal state <img src="https://latex.codecogs.com/svg.latex?S_T" title="S_T" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?S_1,A_1,R_2,S_2,A_2,....,S_T" title="S_1,A_1,R_2,S_2,A_2,....,S_T" />
</p>

Some important terms in RL algorithms involve:
* **Model-based**: Rely on the model of the environment; either the model is known or the algorithm learns it explicitly.
* **Model-free**: No dependency on the model during learning.
* **On-policy**: Use the deterministic outcomes or samples from the target policy to train the algorithm.
* **Off-policy**: Training on a distribution of transitions or episodes produced by a different behavior policy rather than that produced by the target policy.

### Model: Transition and Reward

The model is a descriptor of the environment. With the model, we can learn or infer how the environment would interact with and provide feedback to the agent. The model has two major parts, transition prbability function <img src="https://latex.codecogs.com/svg.latex?P" title="P" />, and reward function <img src="https://latex.codecogs.com/svg.latex?R" title="R" />.

Suppose, we are in state <img src="https://latex.codecogs.com/svg.latex?s" title="s" />, we decide to take action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> to arrive in the next state <img src="https://latex.codecogs.com/svg.latex?s'" title="s'" /> and obtain reward <img src="https://latex.codecogs.com/svg.latex?r" title="r" />. This is known as one **transition step** represented by a tuple <img src="https://latex.codecogs.com/svg.latex?(s,a,s',r)" title="(s,a,s',r)" />.

The transition function <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> records the probability of transitioning from state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> to <img src="https://latex.codecogs.com/svg.latex?s'" title="s'" /> after taking action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> while obtaining reward <img src="https://latex.codecogs.com/svg.latex?r" title="r" />. We use <img src="https://latex.codecogs.com/svg.latex?\mathbb{P}" title="\mathbb{P}" /> as a symbol of probability.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(s',r|s,a)&space;=&space;\mathbb{P}[S_{t&plus;1}=s',&space;R_{t&plus;1}=r|S_t=s,&space;A_t=a]" title="P(s',r|s,a) = \mathbb{P}[S_{t+1}=s', R_{t+1}=r|S_t=s, A_t=a]" />
</p>

Thus, the state-transition function can be defined as a function of <img src="https://latex.codecogs.com/svg.latex?P(s',r|s,a)" title="P(s',r|s,a)" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P_{ss'}^a&space;=&space;P(s'|s,a)&space;=&space;\sum&space;_{r\in&space;\mathcal{R}}&space;P(s',r|s,a)" title="P_{ss'}^a = P(s'|s,a) = \sum _{r\in \mathcal{R}} P(s',r|s,a)" />
</p>

Similarly, the reward function <img src="https://latex.codecogs.com/svg.latex?R" title="R" /> predicts the next reward triggered by a given action:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?R(s,a)&space;=&space;\mathbb{E}[R_{t&plus;1}|S_t=s,A_t=a]&space;=&space;\sum&space;_{r&space;\in&space;\mathcal{R}}&space;r&space;\sum_{s'&space;\in&space;S}&space;P(s',r|s,a)" title="R(s,a) = \mathbb{E}[R_{t+1}|S_t=s,A_t=a] = \sum _{r \in \mathcal{R}} r \sum_{s' \in S} P(s',r|s,a)" />
</p>

### Policy

Policy is defined as the agent's behavior function <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />, and tells us which action to take in state <img src="https://latex.codecogs.com/svg.latex?s" title="s" />. It is a mapping from state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> to action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> and can either be deterministic or stochastic.

* Deterministic: <img src="https://latex.codecogs.com/svg.latex?\pi(s)&space;=&space;a" title="\pi(s) = a" />
* Stochastic: <img src="https://latex.codecogs.com/svg.latex?\pi(a|s)&space;=&space;\mathbb{P}_{\pi}[A=a|S=s]" title="\pi(a|s) = \mathbb{P}_{\pi}[A=a|S=s]" />

### Value function

Value function measures the goodness of a state, or how rewarding a state or action is by predicting future reward. The future reward is also known as **return**, and is the total sum of discounted rewards going forward.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?G_t&space;=&space;R_{t&plus;2}&space;&plus;&space;\gamma&space;R_{t&plus;2}&space;&plus;&space;..&space;=&space;\sum_{k=0}^\infty&space;\gamma&space;^{k}&space;R_{t&plus;k&plus;1}" title="G_t = R_{t+2} + \gamma R_{t+2} + .. = \sum_{k=0}^\infty \gamma ^{k} R_{t+k+1}" />
</p>

The discounting factor <img src="https://latex.codecogs.com/svg.latex?\gamma&space;\in&space;[0,1]" title="\gamma \in [0,1]" /> penalize the rewards in the future for few reasons:
* The future rewards may have higher uncertainty.
* The future rewards do not provide immediate benefits.
* Discounting provides mathematical convenieve; we can make approximations .
* We do not need to worry about the infinite loops in state transition graph.

The **state-value** of a state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> is the expected return if we are in this state at time <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, <img src="https://latex.codecogs.com/svg.latex?S_t&space;=&space;s" title="S_t = s" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V_{\pi}(s)&space;=&space;\mathbb{E}[G_t|S_t=s]" title="V_{\pi}(s) = \mathbb{E}[G_t|S_t=s]" />
</p>

Similarly, we define the **action-value** ("Q-value") of a state-action pair as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q_{\pi}(s,a)&space;=&space;\mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]" title="Q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]" />
</p>

Additionally, since we follow the target policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />, we can make use of the probility distribution over possible actions and the Q-values to recover the state-value:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V_{\pi}(s)&space;=&space;\sum_{a&space;\in&space;\mathcal{A}}&space;Q_{\pi}(s,a)\pi(a|s)" title="V_{\pi}(s) = \sum_{a \in \mathcal{A}} Q_{\pi}(s,a)\pi(a|s)" />
</p>

The difference between action-value and state-value is the action **advantage** function ("A-value"):
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A_{\pi}(s,a)&space;=&space;Q_{\pi}(s,a)&space;-&space;V_{\pi}(s)" title="A_{\pi}(s,a) = Q_{\pi}(s,a) - V_{\pi}(s)" />
</p>

This can be thought of as the advantage of selecting an action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> in a given state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> compared to all other actions available to you.

### Optimal Value and Policy

The optimal value function produces the maximum return:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V_{*}(s)&space;=&space;\max_{\pi}&space;V_{\pi}(s),&space;Q_{*}(s,a)&space;=&space;\max_{\pi}&space;Q_{\pi}(s,a)" title="V_{*}(s) = \max_{\pi} V_{\pi}(s), Q_{*}(s,a) = \max_{\pi} Q_{\pi}(s,a)" />
</p>

The optimal policy achieves optimal value functions:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pi_{*}&space;=&space;\arg&space;\max_{\pi}&space;V_{\pi}(s),&space;\pi_{*}&space;=&space;\arg&space;\max_{\pi}&space;Q_{\pi}(s,a)" title="\pi_{*} = \arg \max_{\pi} V_{\pi}(s), \pi_{*} = \arg \max_{\pi} Q_{\pi}(s,a)" />
</p>

## Markov Decision Processes

In formal terms, almost all RL problems can be framed as **Markov Decision Processses** (MDPs). All states in MDP have "Markov" property, referring to the fact that the future only depends on the current state, not the history:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbb{P}[S_{t&plus;1}|S_t]&space;=&space;\mathbb{P}[S_{t&plus;1}|S_1,...,S_t]" title="\mathbb{P}[S_{t+1}|S_t] = \mathbb{P}[S_{t+1}|S_1,...,S_t]" />
</p>

In other words, the future and the past are **conditionally independent** given the present, as the current state encapsulates all the statistics we need to decide the future.

<p align="center">
<b>The agent-environment interaction in a Markov decision process.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-3.png?raw=true" alt="Figure 3"/>
</p>

A Markov decision process consists of five elements <img src="https://latex.codecogs.com/svg.latex?\mathcal{M}&space;=&space;<&space;\mathcal{S},&space;\mathcal{A},\mathcal{P},\mathcal{R},\gamma>" title="\mathcal{M} = < \mathcal{S}, \mathcal{A},\mathcal{P},\mathcal{R},\gamma>" />, where the symbols have the same meanings as discussed in previous sections. Note that, in an unknown environment, we do not have perfect knowledge about <img src="https://latex.codecogs.com/svg.latex?\mathcal{P}" title="\mathcal{P}" /> and <img src="https://latex.codecogs.com/svg.latex?\mathcal{R}" title="\mathcal{R}" />.

## Bellman Equations

Bellman equations refer to the set of equations that decompose the value function into the immediate reward plus the discounted future values.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V(s)&space;=&space;\mathbb{E}[G_t|S_t=s]" title="V(s) = \mathbb{E}[G_t|S_t=s]" />
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;R_{t&plus;2}&plus;\gamma^2&space;R_{t&plus;3}&plus;...|S_t=s]" title="= \mathbb{E}[R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3}+...|S_t=s]" />
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;(R_{t&plus;2}&plus;\gamma&space;R_{t&plus;3}&plus;...)|S_t=s]" title="= \mathbb{E}[R_{t+1} + \gamma (R_{t+2}+\gamma R_{t+3}+...)|S_t=s]" />
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;G_{t&plus;1}|S_t=s]" title="= \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t=s]" />
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;V(S_{t&plus;1})|S_t=s]" title="= \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})|S_t=s]" />
</p>

Similarly, for Q-value,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q(s,a)&space;=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;V(S_{t&plus;1})|S_t=s,&space;A_t=a]" title="Q(s,a) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})|S_t=s, A_t=a]" />
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}[R_{t&plus;1}&space;&plus;&space;\gamma&space;\mathbb{E}_{a'&space;\sim&space;\pi}Q(S_{t&plus;1},a')|S_t=s,&space;A_t=a]" title="= \mathbb{E}[R_{t+1} + \gamma \mathbb{E}_{a' \sim \pi}Q(S_{t+1},a')|S_t=s, A_t=a]" />
</p>

### Bellman Expectation Equations

The recursive update process can be further decomposed to be equations built on both state-value functions. As we go further in future action steps, we extend <img src="https://latex.codecogs.com/svg.latex?V" title="V" /> and <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" /> alternatively by following the policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V_{\pi}(s)&space;=&space;\sum_{a&space;\in&space;\mathcal{A}}&space;\pi(a|s)&space;Q_{\pi}(s,a)" title="V_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q_{\pi}(s,a)" />
<img src="https://latex.codecogs.com/svg.latex?Q_{\pi}(s,a)&space;=&space;R(s,a)&space;&plus;&space;\gamma&space;\sum_{s'&space;\in&space;\mathcal{S}}&space;P^a_{ss'}&space;V_{\pi}(s')" title="Q_{\pi}(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P^a_{ss'} V_{\pi}(s')" />
<img src="https://latex.codecogs.com/svg.latex?V_{\pi}(s)&space;=&space;\sum_{a\in&space;\mathcal{A}}&space;\pi(a|s)(R(s,a)&space;&plus;&space;\gamma&space;\sum_{s'\in&space;\mathcal{s}}&space;P^a_{ss'}V_{\pi}(s'))" title="V_{\pi}(s) = \sum_{a\in \mathcal{A}} \pi(a|s)(R(s,a) + \gamma \sum_{s'\in \mathcal{s}} P^a_{ss'}V_{\pi}(s'))" />
<img src="https://latex.codecogs.com/svg.latex?Q_{\pi}(s,a)&space;=&space;R(s,a)&space;&plus;&space;\gamma&space;\sum_{s'\in&space;\mathcal{S}}&space;P^a_{ss'}&space;\sum_{a'&space;\in&space;\mathcal{A}}&space;\pi(a'|s')Q_{\pi}(s',a')" title="Q_{\pi}(s,a) = R(s,a) + \gamma \sum_{s'\in \mathcal{S}} P^a_{ss'} \sum_{a' \in \mathcal{A}} \pi(a'|s')Q_{\pi}(s',a')" />
</p>

<p align="center">
<b>Illustration of how Bellman expectation equations update state-value and action-value functions.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-4.png?raw=true" alt="Figure 4"/>
</p>

If we are interested in the optimal values rather than computing the expectation following a policy, we could jump right into the maximum returns during the alternative updates without using a policy. If we have complete information  of the environment, this turns into a planning problem, solvable by DP. Unfortunately, in most scenarios, we do not know <img src="https://latex.codecogs.com/svg.latex?P^a_{ss'}" title="P^a_{ss'}" /> or <img src="https://latex.codecogs.com/svg.latex?R(s,a)" title="R(s,a)" />, so we cannot solve MDPs by directly applying Bellmen equations, but it lays the theoretical foundation for many RL algorithms.

## Common Approaches

In this section, we discuss some of the common approaches and classical algorithms used for solving RL problems.

### Dynamic Programming

When the model is fully known, following Bellman equations, we can use DP to iteratively evaluate the value functions and improve policy.

#### Policy Evaluation

Policy evaluation is to compute the state-value <img src="https://latex.codecogs.com/svg.latex?V_{\pi}" title="V_{\pi}" /> for a given policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V_{t&plus;1}(s)&space;=&space;\sum_{a}&space;\pi(a|s)&space;\sum_{s',r}&space;P(s',r|s,a)(r&plus;\gamma&space;V_t(s'))" title="V_{t+1}(s) = \sum_{a} \pi(a|s) \sum_{s',r} P(s',r|s,a)(r+\gamma V_t(s'))" />
</p>

#### Policy Improvement

Based on the value functions, Policy improvement generates a better policy by acting greedily.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q_{\pi}(s,a)&space;=&space;\sum_{s',r}&space;P(s',r|s,a)(r&plus;\gamma&space;V_t(s'))" title="Q_{\pi}(s,a) = \sum_{s',r} P(s',r|s,a)(r+\gamma V_t(s'))" />
</p>

#### Policy iteration

The *Generalized Policy Iteration* (GPI) algorithm refers to an iterative procedure to improve the policy when combining policy evaluation and improvement.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pi_0&space;\xrightarrow[]{\textup{evaluation}}&space;V_{\pi_0}&space;\xrightarrow[]{\textup{improve}}&space;\pi_1&space;\xrightarrow[]{\textup{evaluation}}&space;V_{\pi_1}&space;\xrightarrow[]{\textup{improve}}&space;\pi_2&space;\xrightarrow[]{\textup{evaluation}}&space;...&space;\xrightarrow[]{\textup{improve}}&space;\pi_{*}&space;\xrightarrow[]{\textup{evaluation}}&space;V_{*}" title="\pi_0 \xrightarrow[]{\textup{evaluation}} V_{\pi_0} \xrightarrow[]{\textup{improve}} \pi_1 \xrightarrow[]{\textup{evaluation}} V_{\pi_1} \xrightarrow[]{\textup{improve}} \pi_2 \xrightarrow[]{\textup{evaluation}} ... \xrightarrow[]{\textup{improve}} \pi_{*} \xrightarrow[]{\textup{evaluation}} V_{*}" />
</p>


In GPI, the value function is approximated repeatedly to be closer to the true value of the current policy, and in meantime, the policy is improved repeatedly to approach optimality. Say, we have a policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" /> and then generate an improved version <img src="https://latex.codecogs.com/svg.latex?\pi&space;'" title="\pi '" /> by greedily taking actions, <img src="https://latex.codecogs.com/svg.latex?\pi&space;'(s)&space;=&space;\arg&space;\max_{a\in&space;\mathcal{A}}&space;Q_{\pi}(s,a)" title="\pi '(s) = \arg \max_{a\in \mathcal{A}} Q_{\pi}(s,a)" />.

### Monte-Carlo Methods

Monte-Carlo (MC methods use a simple idea: It learns from episodes of raw experience without modeling the environmental dynamics and computes the observed mean return as an approximation of the expected return. To compute the empirical return <img src="https://latex.codecogs.com/svg.latex?G_t" title="G_t" />, MC methods need to learn from complete episodes: <img src="https://latex.codecogs.com/svg.latex?S_1,&space;A_1,&space;R_2,&space;...,&space;S_T" title="S_1, A_1, R_2, ..., S_T" /> to compute  <img src="https://latex.codecogs.com/svg.latex?G_t&space;=&space;\sum_{k=0}^{T-t-1}&space;\gamma^k&space;R_{t&plus;k&plus;1}" title="G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}" /> and all episodes must eventually terminate.
THe empirical mean return for state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V(s)&space;=&space;\frac{\sum_{t=1}^T&space;\mathbb{I}[S_t=s]G_t}{\sum_{t=1}^T&space;\mathbb{I}[S_t=s]}" title="V(s) = \frac{\sum_{t=1}^T \mathbb{I}[S_t=s]G_t}{\sum_{t=1}^T \mathbb{I}[S_t=s]}" />
</p>
where, <img src="https://latex.codecogs.com/svg.latex?\mathbb{I}[S_t=s]" title="\mathbb{I}[S_t=s]" /> is a binary indicator function. We may count the visit of state <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> every time so that there could exist multiple visits of one state in one episode, or only count it the first time we encounter a state in one episode. This way of approximation can easily be extended to action-value functions by counting <img src="https://latex.codecogs.com/svg.latex?(s,a)" title="(s,a)" /> pair.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q(s,a)&space;=&space;\frac{\sum&space;_{t=1}^T&space;\mathbb{I}[S_t=s,&space;A_t=a]G_t}{\sum&space;_{t=1}^T&space;\mathbb{I}[S_t=s,&space;A_t=a]}" title="Q(s,a) = \frac{\sum _{t=1}^T \mathbb{I}[S_t=s, A_t=a]G_t}{\sum _{t=1}^T \mathbb{I}[S_t=s, A_t=a]}" />
</p>


<p align="center">
<b>Illustration of MC approach.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-5.png?raw=true" alt="Figure 5"/>
</p>


To learn the optimal policy by MC, we iterate it by following a similar idea to GPI.

* We improve the policy greedily with respect to the current value function: <img src="https://latex.codecogs.com/svg.latex?\pi(s)&space;=&space;\arg&space;\max_{a\in&space;\mathcal{A}}&space;Q(s,a)" title="\pi(s) = \arg \max_{a\in \mathcal{A}} Q(s,a)" />.
* Generate a new episode with the new policy <img src="https://latex.codecogs.com/svg.latex?\pi" title="\pi" />
* Estimate <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" /> using the new episode: <img src="https://latex.codecogs.com/svg.latex?q_{\pi}(s,a)&space;=&space;\frac{\sum_{t=1}^T&space;(\mathbb{I}[S_t=s,&space;A_t=a]&space;\sum_{k=0}^{T-t-1}\gamma^kR_{t&plus;k&plus;1})}{\sum_{t=1}^T&space;\mathbb{I}[S_t=s,&space;A_t=a]&space;}" title="q_{\pi}(s,a) = \frac{\sum_{t=1}^T (\mathbb{I}[S_t=s, A_t=a] \sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1})}{\sum_{t=1}^T \mathbb{I}[S_t=s, A_t=a] }" />


### Temporal-Difference Learning

Similar to Monte-Carlo methods, Temporal Difference (TD) learning is model-free and learns from episodes of experience. However, TD learning can learn from incomplete episodes and hence we don't need to track the episode up to termination.

#### Bootstrapping

TD learning methods update targets with regard to existing estimates rather than exclusively relying on actual rewards and complete returns as in MC methods. This approach is known as **bootstrapping**.

#### Value Estimation

The key idea in TD learning is to update the value function <img src="https://latex.codecogs.com/svg.latex?V(S_t)" title="V(S_t)" /> towards an estimated return <img src="https://latex.codecogs.com/svg.latex?R_{t&plus;1}&plus;\gamma&space;V(S_{t&plus;1})" title="R_{t+1}+\gamma V(S_{t+1})" /> (known as "**TD target**"). To what extent we want to update the value function is controlled by the learning rate hyperparameter <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" />:

<p align="center">

<img src="https://latex.codecogs.com/svg.latex?V(S_t)&space;\leftarrow&space;(1-\alpha)V(S_t)&space;&plus;&space;\alpha&space;G_t" title="V(S_t) \leftarrow (1-\alpha)V(S_t) + \alpha G_t" />\

<img src="https://latex.codecogs.com/svg.latex?V(S_t)&space;\leftarrow&space;V(S_t)&space;&plus;&space;\alpha&space;(G_t&space;-&space;V(S_t))" title="V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))" />\

<img src="https://latex.codecogs.com/svg.latex?V(S_t)&space;\leftarrow&space;V(S_t)&space;&plus;&space;\alpha&space;(R_{t&plus;1}&space;&plus;&space;\gamma&space;V(S_{t&plus;1})&space;-&space;V(S_t))" title="V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))" />
</p>

Similarly, for the action-value estimation:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,&space;A_t)&space;&plus;&space;\alpha&space;(R_{t&plus;1}&space;&plus;&space;\gamma&space;Q(S_{t&plus;1},&space;A_{t&plus;1})&space;-&space;Q(S_t,&space;A_t))" title="Q(S_t,A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))" />
</p>

#### SARSA: On-Policy TD control

"SARSA" refers to the procedure of updating Q-value by following a sequence of <img src="https://latex.codecogs.com/svg.latex?...,&space;S_t,&space;A_t,&space;R_{t&plus;1},S_{t&plus;1},&space;A_{t&plus;1},&space;..." title="..., S_t, A_t, R_{t+1},S_{t+1}, A_{t+1}, ..." />. The idea follows the same route of GPI. A brief explanation of the algorithm is as follows:
* Initialize <img src="https://latex.codecogs.com/svg.latex?t=0" title="t=0" />.
* Start with <img src="https://latex.codecogs.com/svg.latex?S_0" title="S_0" /> and choose action <img src="https://latex.codecogs.com/svg.latex?A_0&space;=&space;\arg&space;\max_{a&space;\in&space;\mathcal{A}}&space;Q(S_0,a)" title="A_0 = \arg \max_{a \in \mathcal{A}} Q(S_0,a)" />, where <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" />-greedy is commonly applied.
* At time <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, after applying action <img src="https://latex.codecogs.com/svg.latex?A_t" title="A_t" />, we observe reward <img src="https://latex.codecogs.com/svg.latex?R_{t&plus;1}" title="R_{t+1}" /> and get into the next state <img src="https://latex.codecogs.com/svg.latex?S_{t&plus;1}" title="S_{t+1}" />.
* Then pict the next action in the same way.
* Update the Q-value function:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,&space;A_t)&space;&plus;&space;\alpha&space;(R_{t&plus;1}&space;&plus;&space;\gamma&space;Q(S_{t&plus;1},&space;A_{t&plus;1})&space;-&space;Q(S_t,&space;A_t))" title="Q(S_t,A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))" />
</p>
* Set <img src="https://latex.codecogs.com/svg.latex?t&space;=&space;t&plus;1" title="t = t+1" /> and repeat from step 3.

In each step of SARSA, we need to choose the *next action* according to the *current policy*.

#### Q-Learning: Off-policy TD control

The development of Q-learning is a big breakout in the early days of Reinforcement Learning. Within one episode, it works as follows:
* Initialize <img src="https://latex.codecogs.com/svg.latex?t=0" title="t=0" />.
* Start with <img src="https://latex.codecogs.com/svg.latex?S_0" title="S_0" />
* At time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, we pick the action according to Q values, <img src="https://latex.codecogs.com/svg.latex?A_t&space;=&space;\arg&space;\max_{a\in&space;\mathcal{A}}&space;Q(S_t,&space;a)" title="A_t = \arg \max_{a\in \mathcal{A}} Q(S_t, a)" /> and <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" />-greedy is commonly applied.
* After applying action <img src="https://latex.codecogs.com/svg.latex?A_t" title="A_t" />, we observe reward <img src="https://latex.codecogs.com/svg.latex?R_{t&plus;1}" title="R_{t+1}" /> and get into the next state <img src="https://latex.codecogs.com/svg.latex?S_{t&plus;1}" title="S_{t+1}" />.
* Update the Q-value function:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q(S_t,&space;A_t)&space;\leftarrow&space;Q(S_t,&space;A_t)&space;&plus;&space;\alpha(R_{t&plus;1}&space;&plus;&space;\gamma&space;Q(S_{t&plus;1},A_{t&plus;1})&space;-&space;Q(S_t,&space;A_t))" title="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t, A_t))" />
</p>
* Set <img src="https://latex.codecogs.com/svg.latex?t&space;=&space;t&plus;1" title="t = t+1" /> and repeat from step 3.

The key difference from SARSA is that Q-learning does not follow the current policy to pick the second action <img src="https://latex.codecogs.com/svg.latex?A_{t&plus;1}" title="A_{t+1}" />. It estimates <img src="https://latex.codecogs.com/svg.latex?Q^{*}" title="Q^{*}" /> out of the best Q values, but which action (denoted as <img src="https://latex.codecogs.com/svg.latex?a^{*}" title="a^{*}" />) leads to this maximal Q does not matter. Instead, in the next step Q-learning may not follow <img src="https://latex.codecogs.com/svg.latex?a^{*}" title="a^{*}" />.

<p align="center">
<b>The backup diagrams for Q-learning and SARSA.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-6.png?raw=true" alt="Figure 6"/>
</p>

#### Deep Q-Network

### Combining TD and MC Learning

In the previous section on value estimation in TD learning, we only trace one step further down the action chain when calculating the TD target. One can easily extend it to take multiple steps to estimate the return.

Let's label the estimated return followin <img src="https://latex.codecogs.com/svg.latex?n" title="n" /> steps as <img src="https://latex.codecogs.com/svg.latex?G_t^{(n)}" title="G_t^{(n)}" />, <img src="https://latex.codecogs.com/svg.latex?n&space;=&space;1,...,\infty" title="n = 1,...,\infty" />, then:

<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-10.png?raw=true" alt="Figure 10"/>
</p>

The generalized n-step TD learning still has the same  form of value function:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V(S_t)&space;\leftarrow&space;V(S_t)&space;&plus;&space;\alpha(G_T^{(n)}&space;-&space;V(S_t))" title="V(S_t) \leftarrow V(S_t) + \alpha(G_T^{(n)} - V(S_t))" />
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-11.png?raw=true" alt="Figure 11"/>
</p>

We are free to pick any <img src="https://latex.codecogs.com/svg.latex?n" title="n" /> in TD learning as we like. Now, the question becomes what is the best <img src="https://latex.codecogs.com/svg.latex?n" title="n" />? Which <img src="https://latex.codecogs.com/svg.latex?G_t^{(n)}" title="G_t^{(n)}" /> gives us the best return approximation? A common yet smart solution is to apply a weighted sum of all possible n-step TD targets rather than pick the best <img src="https://latex.codecogs.com/svg.latex?n" title="n" />. The weight decay by a factor <img src="https://latex.codecogs.com/svg.latex?\lambda" title="\lambda" /> with n, <img src="https://latex.codecogs.com/svg.latex?\lambda^{n-1}" title="\lambda^{n-1}" />; the intuition is similar to why we want to discount future rewards when computing tje return: the more future we look into, the less confident we would be. To make all the weight (<img src="https://latex.codecogs.com/svg.latex?n&space;\rightarrow&space;\infty" title="n \rightarrow \infty" />) sum up to 1, we multiply every weight by (<img src="https://latex.codecogs.com/svg.latex?1-\lambda" title="1-\lambda" />).

The weighted sum of many n-step returns is called <img src="https://latex.codecogs.com/svg.latex?\lambda" title="\lambda" />-return <img src="https://latex.codecogs.com/svg.latex?G_t^{\lambda}&space;=&space;(1-\lambda)\sum_{n=1}^{\infty}&space;\lambda^{n-1}G_t^{(n)}" title="G_t^{\lambda} = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1}G_t^{(n)}" />. TD learning that adopts <img src="https://latex.codecogs.com/svg.latex?\lambda" title="\lambda" />-return for value updating is labeled as TD(<img src="https://latex.codecogs.com/svg.latex?\lambda" title="\lambda" />). The original version introduced above is equivalent to TD(0).

<p align="center">
<b>Comparison of backup diagrams of Monte-Carlo, Temporal-Difference learning, and Dynamic Programming for state value functions.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-12.png?raw=true" alt="Figure 12"/>
</p>

### Policy Gradient

All the methods we have introduced above aim to learn the state/action function and then to select actions accordingly. Policy gradient methods instead learn the policy directly with a parameterized function with respect to <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, <img src="https://latex.codecogs.com/svg.latex?\pi(a|s;\theta)" title="\pi(a|s;\theta)" />. Let's define the reward function (opposite of loss function) as the expected return and train the algorithm with the goal to maximize the reward function. In discrete space:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{J}(\theta)&space;=&space;V_{\pi_{\theta}}(S_1)&space;=&space;\mathbb{E}_{\pi_{\theta}}[V_1]" title="\mathcal{J}(\theta) = V_{\pi_{\theta}}(S_1) = \mathbb{E}_{\pi_{\theta}}[V_1]" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?S_1" title="S_1" /> is the initial starting state.

Or in continuous space:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{J}(\theta)&space;=&space;\sum_{s\in&space;\mathcal{S}}d_{\pi_{\theta}}(s)&space;\sum_{a\in&space;\mathcal{A}}\pi(a|s,\theta)Q_{\pi}(s,a)" title="\mathcal{J}(\theta) = \sum_{s\in \mathcal{S}}d_{\pi_{\theta}}(s) \sum_{a\in \mathcal{A}}\pi(a|s,\theta)Q_{\pi}(s,a)" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?d_{\pi_{\theta}}(s)" title="d_{\pi_{\theta}}(s)" /> is stationary distribution of Markov chain for <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}" title="\pi_{\theta}" />. Using gradient ascent, we can find the best <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> that produces the highest return. It is natural to expect policy-based methods are more useful in continuous space, because there is an infinite number of actions and/or states to estimate the values for in continuous space and hence value-based approaches are computationally much more expensive.

#### Policy Gradient Theorem

Computing the gradient numerically can be done perturbing <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> by a small amount <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" /> in the k-th dimension. It works wven when <img src="https://latex.codecogs.com/svg.latex?J(\theta)" title="J(\theta)" /> is not differentiable, but unsurprisingly very slow.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\mathcal{J}(\theta)}{\partial&space;\theta_k}&space;\approx&space;\frac{\mathcal{J}(\theta&space;&plus;&space;\varepsilon&space;u_k)&space;-&space;\mathcal{J}(\theta)}{\varepsilon}" title="\frac{\partial \mathcal{J}(\theta)}{\partial \theta_k} \approx \frac{\mathcal{J}(\theta + \varepsilon u_k) - \mathcal{J}(\theta)}{\varepsilon}" />
</p>

Or analytically,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{J}(\theta)&space;=&space;\mathbb{E}_{\pi_{\theta}}[r]&space;=&space;\sum_{s\in&space;\mathcal{S}}d_{\pi_{\theta}}(s)&space;\sum_{a\in&space;\mathcal{A}}&space;\pi&space;(a|s;\theta)R(s,a)" title="\mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta}}[r] = \sum_{s\in \mathcal{S}}d_{\pi_{\theta}}(s) \sum_{a\in \mathcal{A}} \pi (a|s;\theta)R(s,a)" />
</p>

Actually we have nice theoretical support for replacing <img src="https://latex.codecogs.com/svg.latex?d(.)" title="d(.)" /> with <img src="https://latex.codecogs.com/svg.latex?d_{\pi}(.)" title="d_{\pi}(.)" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{J}(\theta)&space;=&space;\sum_{s\in&space;\mathcal{S}}d_{\pi_{\theta}}(s)&space;\sum_{a\in&space;\mathcal{A}}&space;\pi&space;(a|s;\theta)Q_{\pi}(s,a)&space;\propto&space;\sum_{s\in&space;\mathcal{S}}d(s)&space;\sum_{a\in&space;\mathcal{A}}&space;\pi&space;(a|s;\theta)Q_{\pi}(s,a)" title="\mathcal{J}(\theta) = \sum_{s\in \mathcal{S}}d_{\pi_{\theta}}(s) \sum_{a\in \mathcal{A}} \pi (a|s;\theta)Q_{\pi}(s,a) \propto \sum_{s\in \mathcal{S}}d(s) \sum_{a\in \mathcal{A}} \pi (a|s;\theta)Q_{\pi}(s,a)" />
</p>

This result is named "Policy Gradient Theorem" which lays the theoretical foundation for various policy gradient algorithms:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\bigtriangledown&space;\mathcal{J}(\theta)&space;=&space;\mathbb{E}_{\pi_{\theta}}[\bigtriangledown&space;\ln&space;\pi(a|s;\theta)Q_{\pi}(s,a)]" title="\bigtriangledown \mathcal{J}(\theta) = \mathbb{E}_{\pi_{\theta}}[\bigtriangledown \ln \pi(a|s;\theta)Q_{\pi}(s,a)]" />
</p>

#### Reinforce

REINFORCE, also known as Monte-Carlo policy gradient, relies on <img src="https://latex.codecogs.com/svg.latex?Q_{\pi}(s,a)" title="Q_{\pi}(s,a)" />, an estimated return by MC methods using episode samples, to update the policy parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />.

A commonly used variation of REINFORCE is to subtract a baseline value from the return <img src="https://latex.codecogs.com/svg.latex?G_t" title="G_t" /> to reduce the variance of gradient estimation while keeping the bias unchanged. For example, a common baseline is state-value, and if applied, we would use <img src="https://latex.codecogs.com/svg.latex?A(s,a)=Q(s,a)-V(s)" title="A(s,a)=Q(s,a)-V(s)" /> in the gradient ascent update.

* Initialize <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> at random.
* Generate one episode: <img src="https://latex.codecogs.com/svg.latex?S_1,&space;A_1,&space;R_2,&space;S_2,&space;A_2,&space;...,&space;S_T" title="S_1, A_1, R_2, S_2, A_2, ..., S_T" />
* For <img src="https://latex.codecogs.com/svg.latex?t=1,2,...,T" title="t=1,2,...,T" />:
    * Estimate the return <img src="https://latex.codecogs.com/svg.latex?G_t" title="G_t" /> since the time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />.
    * <img src="https://latex.codecogs.com/svg.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha&space;\gamma^t&space;\bigtriangledown&space;\ln&space;\pi(A_t|S_t,&space;\theta)" title="\theta \leftarrow \theta + \alpha \gamma^t \bigtriangledown \ln \pi(A_t|S_t, \theta)" />.

#### Actor-Critic

If the value function is learned in addition to the policy, we would get Actor-Critic algorithm.

* **Critic**: updates value function parameters <img src="https://latex.codecogs.com/svg.latex?w" title="w" /> and depending on the algorithm it could be action-value <img src="https://latex.codecogs.com/svg.latex?Q(a|s;w)" title="Q(a|s;w)" /> or state value <img src="https://latex.codecogs.com/svg.latex?V(s;w)" title="V(s;w)" />.
* **Actor**: updates policy parameters <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, in the direction suggested by the critic, <img src="https://latex.codecogs.com/svg.latex?\pi(a|s,\theta)" title="\pi(a|s,\theta)" />.

* Initialize <img src="https://latex.codecogs.com/svg.latex?s" title="s" />, <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, <img src="https://latex.codecogs.com/svg.latex?w" title="w" /> at random; sample <img src="https://latex.codecogs.com/svg.latex?a&space;\sim&space;\pi(a|s;\theta)" title="a \sim \pi(a|s;\theta)" />.
* For <img src="https://latex.codecogs.com/svg.latex?t=1,2,...,T" title="t=1,2,...,T" />:
    * Sample reward <img src="https://latex.codecogs.com/svg.latex?r_t&space;\sim&space;R(s,a)" title="r_t \sim R(s,a)" /> and next state <img src="https://latex.codecogs.com/svg.latex?s'&space;\sim&space;P(s'|s,a)" title="s' \sim P(s'|s,a)" />.
    * Then sample the next action <img src="https://latex.codecogs.com/svg.latex?a'&space;\sim&space;\pi(s',a';\theta)" title="a' \sim \pi(s',a';\theta)" />.
    * Update policy parameters: <img src="https://latex.codecogs.com/svg.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha&plus;{\theta}Q(s,a;w)\bigtriangledown&space;_{\theta}\ln&space;\pi(a|s;\theta)" title="\theta \leftarrow \theta + \alpha+{\theta}Q(s,a;w)\bigtriangledown _{\theta}\ln \pi(a|s;\theta)" />.
    * Compute the correction for action-value at time t: <img src="https://latex.codecogs.com/svg.latex?G_{t:t&plus;1}&space;=&space;r_t&space;&plus;&space;\gamma&space;Q(s',a';w)&space;-&space;Q(s,a;w)" title="G_{t:t+1} = r_t + \gamma Q(s',a';w) - Q(s,a;w)" />, and use it to update value function parameters: <img src="https://latex.codecogs.com/svg.latex?w&space;\leftarrow&space;w&space;&plus;&space;\alpha_w&space;G_{t:t&plus;1}&space;\bigtriangledown&space;_w&space;Q(s,a;w)" title="w \leftarrow w + \alpha_w G_{t:t+1} \bigtriangledown _w Q(s,a;w)" />.
    * Update <img src="https://latex.codecogs.com/svg.latex?a&space;\leftarrow&space;a'" title="a \leftarrow a'" /> and <img src="https://latex.codecogs.com/svg.latex?s&space;\leftarrow&space;s'" title="s \leftarrow s'" />.

<img src="https://latex.codecogs.com/svg.latex?a_{\theta}" title="a_{\theta}" /> and <img src="https://latex.codecogs.com/svg.latex?a_{w}" title="a_{w}" /> are two learning rates for policy and value function parameter updates, respectively.

#### A3C

Asynchronous Advantage Actor Critic, short for A3C is a classic policy gradient method with the special focus on parallel training.
In A3C, the critics learn the state-value function, <img src="https://latex.codecogs.com/svg.latex?V(s;w)" title="V(s;w)" />, while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is good for parallel training by default.

The loss function for state-value is to minimize the mean squared error, <img src="https://latex.codecogs.com/svg.latex?\mathcal{J}_v&space;(w)&space;=&space;(G_t-V(s;w))^2" title="\mathcal{J}_v (w) = (G_t-V(s;w))^2" /> and we use gradient descent to find the optimal <img src="https://latex.codecogs.com/svg.latex?w" title="w" />. This state-value function is used as the baseline in the policy gradient update.

<p align="center">
<b>Outline of the A3C Algorithm.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-13.png?raw=true" alt="Figure 9"/>
</p>


A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.

### Evolution Strategies

Evolution Strategies(ES) is a type of model-agnostic optimization approach. It learns the optimal solution by imitating Darwin's theory of the evolution of species by natural selection. Two prerequisites for applying ES: (1) our solutions can freely interact with the encironment and see whether they can sove the problem; (2) we are able to compute a fitness score of how good each solution is. We don't have to know the environment configuration to solve the problem.

Say, we start with a population of random solutions. All of them are capable of interacting with the environment and only candidates with high fitness scores can survive (only the fittest can survive in a competition for limited resources). A new generation is then created by recombining the settings (gene mutation) of high-fitness survivors. This process is repeated until the new solutions are good enough.

Very different from the popular MDP-based approaches as what we have introduced above, ES claims to learn the policy parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> without value approximation. Let's assume the distribution over the parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> is an isotropic multivariate Gaussian with mean <img src="https://latex.codecogs.com/svg.latex?\mu" title="\mu" /> and fixed covariance <img src="https://latex.codecogs.com/svg.latex?\sigma^2&space;I" title="\sigma^2 I" />. The gradient of <img src="https://latex.codecogs.com/svg.latex?F(\theta)" title="F(\theta)" /> is calculated:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\bigtriangledown_{\theta}&space;\mathbb{E}_{\theta&space;\sim&space;N(\mu,&space;\sigma&space;^2)}&space;F(\theta)&space;=&space;\bigtriangledown_{\theta}&space;\int&space;_{\theta}&space;F(\theta)\textup{Pr}(\theta)" title="\bigtriangledown_{\theta} \mathbb{E}_{\theta \sim N(\mu, \sigma ^2)} F(\theta) = \bigtriangledown_{\theta} \int _{\theta} F(\theta)\textup{Pr}(\theta)" /><br/>
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}_{\theta&space;\sim&space;N(\mu,&space;\sigma&space;^2)}&space;[F(\theta)\frac{\theta&space;-&space;\mu}{\sigma^2}]" title="= \mathbb{E}_{\theta \sim N(\mu, \sigma ^2)} [F(\theta)\frac{\theta - \mu}{\sigma^2}]" />
</p>

We can rewrite this formula in terms of a "mean" parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> (different from the <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> above; this <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> is the base gene for durther information), <img src="https://latex.codecogs.com/svg.latex?\varepsilon&space;\sim&space;N(0,I)" title="\varepsilon \sim N(0,I)" /> and therefore <img src="https://latex.codecogs.com/svg.latex?\theta&space;&plus;&space;\varepsilon&space;\sigma&space;\sim&space;N(\mu,&space;\sigma^2)" title="\theta + \varepsilon \sigma \sim N(\mu, \sigma^2)" />. <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" /> controls how much Gaussian noises should be added to create mutation:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\bigtriangledown&space;_{\theta}&space;\mathbb{E}_{\varepsilon&space;\sim&space;N(0,I)}&space;F(\theta&space;&plus;&space;\sigma&space;\varepsilon)&space;=&space;\frac{1}{\sigma}&space;\mathbb{E}_{\varepsilon&space;\sim&space;N(0,I)}&space;[F(\theta&space;&plus;&space;\sigma&space;\varepsilon)\varepsilon]" title="\bigtriangledown _{\theta} \mathbb{E}_{\varepsilon \sim N(0,I)} F(\theta + \sigma \varepsilon) = \frac{1}{\sigma} \mathbb{E}_{\varepsilon \sim N(0,I)} [F(\theta + \sigma \varepsilon)\varepsilon]" />
</p>

ES, as a black-box optimization algorithm, is another approach to RL problems. It has few good characteristics:
* ES is fast and easy to train;
* ES does not need value function approximation;
* ES does not perform gradient back-propagation;
* ES is invariant to delayed or long-term rewards;
* ES is highly parallelizable with very little data communication.

## Known Problems

### Exploration-Exploitation Dilemma

When the RL problem faces an unknown environment, this issue is especially key to finding a good solution: without enough exploration, we cannot learn the environment well enough; without enough exploitation, we cannot complete our reward optimization task.

Different RL algorithms balance between exploration and exploitation in different ways. In MC methods, Q-learning or many on-policy algorithms, the exploration is commonly implemented by <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" />-greedy; In ES, the exploration is captured by the policy parameter perturbation.

### Deadly Triad Issue

We do seek the efficiency and flexibility of TD methods that involve bootstrapping. However, when off-policy, nonlinear function approximation, and bootstrapping are combined to one RL algorithm, the training could be unstable and hard to converge. This issue is known as the **deadly triad**. Many architectures using deep learning models were proposed to resolve the problem, including DQN to stabilize the training with experience replay and occasionally frozen target network.

### Case Study: AlphaGO Zero

The game of Go has been an extremely hard problem in the field of AI. AlphaGo and AlphaGo Zero are two programs developed by a team at DeepMind. Both involve deep CNN and Monte Carlo Tree Search (MCTS), and both have been approved to achieve the level of professional human Go players. Different from AlphaGo that relied on supervised learning from expert human moves, AlphaGo Zero used only reinforcement learning and self-play without human knowledge beyond basic rules.

<p align="center">
<b>The board of Go. Two players play black and white stones alternatively on the vacant intersections of a board with 19 x 19 lines. A group of stones must have at least one open point (an intersection, called a “liberty”) to remain on the board and must have at least two or more enclosed liberties (called “eyes”) to stay “alive”. No stone shall repeat a previous position.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-8.png?raw=true" alt="Figure 8"/>
</p>

The main component is a deep CNN over the game board configuration (precisely, a ResNet with batch normalization and ReLU). This network outputs two values:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?(p,v)&space;=&space;f_{\theta}(s)" title="(p,v) = f_{\theta}(s)" />
</p>

* <img src="https://latex.codecogs.com/svg.latex?s" title="s" />: the game board configuration, 19 x 19 x 17 stacked  feature plans; 17 features for each position, 8 past configurations for the current player + 8 past configurations for the opponent + 1 feature indication the color (1=black, 0=white). We need to code the color specifically because the network is playing with itself and the colors of current players and opponents are switching between steps.
* <img src="https://latex.codecogs.com/svg.latex?p" title="p" />: the probability of selecting a move over 19^2 _1 candidates.
* <img src="https://latex.codecogs.com/svg.latex?v" title="v" />: the winning probability given the current setting.

<p align="center">
<b>AlphaGo Zero is trained by self-play while MCTS improves the output policy further in every step.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/34/Figure-9.png?raw=true" alt="Figure 9"/>
</p>

During self-play, MCTS further improves the action probability distribution <img src="https://latex.codecogs.com/svg.latex?\pi&space;\sim&space;p(.)" title="\pi \sim p(.)" /> and then the action <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" /> is sampled from this improved policy. The reward <img src="https://latex.codecogs.com/svg.latex?z_t" title="z_t" /> is a binary value indicating whether the current player eventually wins the game.  Each move generates an episode tuple <img src="https://latex.codecogs.com/svg.latex?(s_t,&space;\pi_t,&space;z_t)" title="(s_t, \pi_t, z_t)" /> and it is saved into the replay memory.

The network is trained with the samples in the replay memory to minimize the loss:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}&space;=&space;(z-v)^2&space;-&space;\pi^T&space;\log&space;p&space;&plus;&space;c||\theta||^2" title="\mathcal{L} = (z-v)^2 - \pi^T \log p + c||\theta||^2" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?c" title="c" /> is a hyperparameter controlling the intensity of L2 penalty to avoid overfitting.

AlphaGo Zero simplified AlphaGo by removing supervised learning and merging seperated policy and value networks into one.
