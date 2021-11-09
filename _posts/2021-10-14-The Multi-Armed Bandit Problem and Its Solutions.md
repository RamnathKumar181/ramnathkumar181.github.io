---
layout: post
title: The Multi-Armed Bandit Problem and Its Solutions
published: true
---

An overview of the topic “[The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)”.
<!--break-->

The exploration vs exploitation dilemma exists in many aspects of our life. Say, your favorite restaurant is right around the corner. If you go there every day, you would be confident of what you will get, but miss the chances of discovering an even better option. If you try new places all the time, very likely you are gonna have to eat unpleasant food from time to time.

If we have learned all the information about the environment, we are able to find the best strategy by even just simulating brute-force, let alone many other smart approaches. The dilemma comes from the incomplete information: we need to gather enough information to make best overall decisions while keeping the risk under control. With exploitation, we take advantage of the best option we know. With exploration, we take some risk to collect information about unknown options. The best long-term strategy may involve short-term sacrifices. For example, one exploration trial could be a total failure, but it warns us of not taking that action too often in the future.

## Background

The multi-armed bandit problem is a clssic problem that demonstrates the exploration vs exploitation dilemma well. Imagine you are in a casino facing multiple slot machines and each is configured with an unknown probability of how likely you can get a reward. The question is: *What is the best strategy to achieve the highest long-term rewards?*

In the blog, the author only covered the case of finite number of trials, and mentioned that this scenario offers a new type of exploration problem. For instance, if the number of trials is smaller than the number of slot machines, we cannot even try every machine to estimate the reward probability and hence we have to behave smartly w.r.t. a limited set of knowledge and resources (i.e. time).

<p align="center">
<b>An illustration of how a Bernoulli multi-armed bandit works. The reward probabilities are unknown to the player.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/35/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

A naive approach can be to continue playing on the machine for many rounds so as to eventually estimate the "true" reward probability. However, this is quite wasteful and does not guarantee the best long-term rewards regardless.

## Definitions

A Bernoulli multi-armed bandit can be described as a typle of <img src="https://latex.codecogs.com/svg.latex?<\mathcal{A},\mathcal{R}>" title="<\mathcal{A},\mathcal{R}>" />, where:
* We have <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> machines with reward probabilities, <img src="https://latex.codecogs.com/svg.latex?\{\theta_1,&space;...,&space;\theta_K\}" title="\{\theta_1, ..., \theta_K\}" />.
* At each time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, we take an action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> on one slot machine and receive a reward <img src="https://latex.codecogs.com/svg.latex?r" title="r" />.
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{A}" title="\mathcal{A}" /> is a set of actions, each referring to the interaction with one slot machine. The value of action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> is the expected reward, <img src="https://latex.codecogs.com/svg.latex?Q(a)&space;=&space;\mathbb{E}[r|a]&space;=&space;\theta" title="Q(a) = \mathbb{E}[r|a] = \theta" />. If action <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" /> at the time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" /> is on the i-th machine, then <img src="https://latex.codecogs.com/svg.latex?Q(a_t)=\theta_i" title="Q(a_t)=\theta_i" />.
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{R}" title="\mathcal{R}" /> is a reward function. In case of Bernoulli bandit, we observe reward <img src="https://latex.codecogs.com/svg.latex?r" title="r" /> in a stochastic fashion. At time step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, <img src="https://latex.codecogs.com/svg.latex?r_t&space;=&space;\mathcal{R}(a_t)" title="r_t = \mathcal{R}(a_t)" /> may return reward 1 with a probability <img src="https://latex.codecogs.com/svg.latex?Q(a_t)" title="Q(a_t)" /> or 0 otherwise.

It is a simplified version of Markov decision, as there is no state <img src="https://latex.codecogs.com/svg.latex?\mathcal{S}" title="\mathcal{S}" />.
The goal is to maximize the cumulative reward <img src="https://latex.codecogs.com/svg.latex?\sum_{t=1}^T&space;r_t" title="\sum_{t=1}^T r_t" />. If we know the optimal action with the best reward, then the goal is same as to minimize the potential regret or loss by not picking the optimal action.
The optimal reward probability <img src="https://latex.codecogs.com/svg.latex?\theta^{*}" title="\theta^{*}" /> of the optimal action <img src="https://latex.codecogs.com/svg.latex?a^{*}" title="a^{*}" /> is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\theta^{*}&space;=&space;Q(a^*)&space;=&space;\max_{a\in&space;\mathcal{A}}&space;Q(a)&space;=&space;\max_{1\leq&space;i&space;\leq&space;K}&space;\theta_i" title="\theta^{*} = Q(a^*) = \max_{a\in \mathcal{A}} Q(a) = \max_{1\leq i \leq K} \theta_i" />
</p>

Our loss function is the total regret we might have by not selecting the optimal selecting the optimal action up to the time step <img src="https://latex.codecogs.com/svg.latex?T" title="T" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_T&space;=&space;\mathbb{E}&space;[\sum_{t=1}^T&space;(\theta^*&space;-&space;Q(a_t))]" title="\mathcal{L}_T = \mathbb{E} [\sum_{t=1}^T (\theta^* - Q(a_t))]" />
</p>

Based on how we do exploration, there are several ways to solve the multi-armed bandit problem:
* No exploration: the most naive approach and a bad one.
* Exploration at random
* Exploration smartly with a preference to uncertainty.

## <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" />-Greedy Algorithm

The <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" />-greedy algorithm takes the best action most of the time, but does random exploration occasionally. The action value estimated according to the past experience by averaging the rewards associated with the target action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> that we have observed so far:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{Q}_t(a)&space;=&space;\frac{1}{N_t(a)}&space;\sum_{\tau=1}^t&space;r_{\tau}\mathbb{I}[a_{\tau}=a]" title="\widehat{Q}_t(a) = \frac{1}{N_t(a)} \sum_{\tau=1}^t r_{\tau}\mathbb{I}[a_{\tau}=a]" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?\mathbb{I}" title="\mathbb{I}" /> is a binary indicator function and <img src="https://latex.codecogs.com/svg.latex?N_t(a)" title="N_t(a)" /> is how many times the action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> has been selected so far, <img src="https://latex.codecogs.com/svg.latex?N_t(a)&space;=&space;\sum_{\tau=1}^t&space;\mathbb{I}[a_{\tau}&space;=&space;a]" title="N_t(a) = \sum_{\tau=1}^t \mathbb{I}[a_{\tau} = a]" />.

According to the <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" />-greedy algorithm, with a small probability <img src="https://latex.codecogs.com/svg.latex?\varepsilon" title="\varepsilon" /> we take a random action, but otherwise (which should be most of the time), we pick the best action that we have learnt so far.

## Upper Confidence Bounds

Random exploration gives us an opportunity to try out options that we have not known much about. However, due to the randomness, it is possible we end up exploring a bad action which we have confirmed in the past (bad luck!). To avoid such inefficient exploration, one approach is to decrease the parameter ε in time and the other is to be optimistic about options with *high uncertainty* and thus to prefer actions for which we haven’t had a confident value estimation yet. Or in other words, we favor exploration of actions with a strong potential to have a optimal value.

The Upper Confidence Bounds (UCB) algorithm measures this potential by an upper confidence bound of the reward value, <img src="https://latex.codecogs.com/svg.latex?\widehat{U}_t(a)" title="\widehat{U}_t(a)" />, so that the true value is below with bound <img src="https://latex.codecogs.com/svg.latex?Q(a)&space;\leq&space;\widehat{Q}_t(a)&space;&plus;&space;\widehat{U}_t(a)" title="Q(a) \leq \widehat{Q}_t(a) + \widehat{U}_t(a)" /> with high probability. The upper bound <img src="https://latex.codecogs.com/svg.latex?\widehat{U}_t(a)" title="\widehat{U}_t(a)" /> is a function of <img src="https://latex.codecogs.com/svg.latex?N_t(a)" title="N_t(a)" />; a larger number of trials <img src="https://latex.codecogs.com/svg.latex?N_t(a)" title="N_t(a)" /> should give us a smaller bound <img src="https://latex.codecogs.com/svg.latex?\widehat{U}_t(a)" title="\widehat{U}_t(a)" />.

In UCB algorithm, we always select the greediest action to maximize the upper confidence bound:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?a_t^{\textup{UCB}}&space;=&space;\arg&space;\max_{a\in&space;\mathcal{A}}&space;\widehat{Q}_t(a)&space;&plus;&space;\widehat{U}_t(a)" title="a_t^{\textup{UCB}} = \arg \max_{a\in \mathcal{A}} \widehat{Q}_t(a) + \widehat{U}_t(a)" />
</p>

### Hoeffding's Inequality

If we do not want to assign any prior knowledge on how the distribution looks like, we can get help from "Hoeffding's Inequality" - a theorem applicable to any bounded distribution.

Let <img src="https://latex.codecogs.com/svg.latex?X_1,&space;...,&space;X_t" title="X_1, ..., X_t" /> be the i.i.d. random random variables and they are all bounded by the interval <img src="https://latex.codecogs.com/svg.latex?[0,1]" title="[0,1]" />. The sample mean is <img src="https://latex.codecogs.com/svg.latex?\overline{X}_t&space;=&space;\frac{1}{t}&space;\sum_{\tau=1}^t&space;X_{\tau}" title="\overline{X}_t = \frac{1}{t} \sum_{\tau=1}^t X_{\tau}" />. Then for <img src="https://latex.codecogs.com/svg.latex?u&space;>&space;0" title="u > 0" />, we have:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbb{P}[\mathbb{E}[X]&space;>&space;\overline{X}_t&space;&plus;&space;u]&space;\leq&space;e^{-2tu^2}" title="\mathbb{P}[\mathbb{E}[X] > \overline{X}_t + u] \leq e^{-2tu^2}" />
</p>

Given one target action <img src="https://latex.codecogs.com/svg.latex?a" title="a" />, let us consider:
* <img src="https://latex.codecogs.com/svg.latex?r_t(a)" title="r_t(a)" /> as the random variables,
* <img src="https://latex.codecogs.com/svg.latex?Q(a)" title="Q(a)" /> as the true mean,
* <img src="https://latex.codecogs.com/svg.latex?\widehat{Q}_t(a)" title="\widehat{Q}_t(a)" /> as the sample mean,
* And <img src="https://latex.codecogs.com/svg.latex?u" title="u" /> as the upper confidence bound, <img src="https://latex.codecogs.com/svg.latex?u&space;=&space;U_t(a)" title="u = U_t(a)" />

Then we have, <img src="https://latex.codecogs.com/svg.latex?\mathbb{P}[Q(a)>\widehat{Q}_t(a)&plus;U_t(a)]&space;\leq&space;e^{-2tU_t(a)^2}" title="\mathbb{P}[Q(a)>\widehat{Q}_t(a)+U_t(a)] \leq e^{-2tU_t(a)^2}" />.

We want to pick a bound so that with high chances the true mean is below the sample mean + the upper confidence bound. Thus, <img src="https://latex.codecogs.com/svg.latex?e^{-2tU_t(a)^2}" title="e^{-2tU_t(a)^2}" /> should be a small probability. Thus, we set:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?U_t(a)&space;=&space;\sqrt{\frac{-\log&space;p}{2N_t(a)}}" title="U_t(a) = \sqrt{\frac{-\log p}{2N_t(a)}}" />
</p>

### UCB1

One heuristic is to reduce the threshold <img src="https://latex.codecogs.com/svg.latex?p" title="p" /> in time, as we want to make more confident bound estimation with more rewards observed. Setting <img src="https://latex.codecogs.com/svg.latex?p&space;=&space;t^{-4}" title="p = t^{-4}" />, we get **UCB1** algorithm:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?U_t(a)&space;=&space;\sqrt{\frac{2log&space;t}{N_t(a)}}" title="U_t(a) = \sqrt{\frac{2log t}{N_t(a)}}" />
</p>

and,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?a_t^{UCB1}&space;=&space;\arg&space;\max_{a\in&space;\mathcal{A}}&space;Q(a)&space;&plus;&space;\sqrt{\frac{2log&space;t}{N_t(a)}}" title="a_t^{UCB1} = \arg \max_{a\in \mathcal{A}} Q(a) + \sqrt{\frac{2log t}{N_t(a)}}" /></p>

### Bayesian UCB

In UCB or UCB1 algorithm, we do not make any prior on the reward distribution and therefore we have to rely on the Hoeffding's Inequality for a very generalized estimation. If we are able to know the distribution upfront, we would be able to make a better bound estimation.

For instance, if we expect the mean reward of every slot machine to be a Gaussian, we can set the upper bound as 95\% confidence interval by setting <img src="https://latex.codecogs.com/svg.latex?\widehat{U}_t(a)" title="\widehat{U}_t(a)" /> to be twice the standard deviation.

## Thompson Sampling

At each time step, we want to select action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> according to the probability that <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> is optimal:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\pi(a|h_t)&space;=&space;\mathbb{P}[Q(a)>Q(a'),&space;\forall&space;a'&space;\neq&space;a&space;|h_t]" title="\pi(a|h_t) = \mathbb{P}[Q(a)>Q(a'), \forall a' \neq a |h_t]" />
<img src="https://latex.codecogs.com/svg.latex?=&space;\mathbb{E}_{\mathcal{R}|h_t}[\mathbb{I}(a=\arg&space;\max_{a\in&space;\mathcal{A}}Q(a))]" title="= \mathbb{E}_{\mathcal{R}|h_t}[\mathbb{I}(a=\arg \max_{a\in \mathcal{A}}Q(a))]" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?\pi(a|h_t)" title="\pi(a|h_t)" /> is the probability of taking action <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> given history <img src="https://latex.codecogs.com/svg.latex?h_t" title="h_t" />.

For the Bernoulli bandit, it is natural to assume that <img src="https://latex.codecogs.com/svg.latex?Q(a)" title="Q(a)" /> follows a Beta distribution, as <img src="https://latex.codecogs.com/svg.latex?Q(a)" title="Q(a)" /> is essentially the success probability <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> in Bernoulli distribution. The value of Beta(<img src="https://latex.codecogs.com/svg.latex?\alpha,&space;\beta" title="\alpha, \beta" />) is within the interval <img src="https://latex.codecogs.com/svg.latex?[0,1]" title="[0,1]" />; <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> and <img src="https://latex.codecogs.com/svg.latex?\beta" title="\beta" /> correspond to the counts when we succeeded or failed to get a reward respectively.

Initially, we set the Beta parameters based on some prior knowledge or belief. For instance,
* <img src="https://latex.codecogs.com/svg.latex?\alpha=1" title="\alpha=1" /> and <img src="https://latex.codecogs.com/svg.latex?\beta=1" title="\beta=1" />; we expect the reward probability to be around 50\% but are not very confident.
* <img src="https://latex.codecogs.com/svg.latex?\alpha=1000" title="\alpha=1000" /> and <img src="https://latex.codecogs.com/svg.latex?\beta=9000" title="\beta=9000" />; we strongly believe that the reward probability is 10\%.

At each time <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, we sample an expected reward, <img src="https://latex.codecogs.com/svg.latex?\widetilde{Q}(a)" title="\widetilde{Q}(a)" /> from the prior distribution <img src="https://latex.codecogs.com/svg.latex?\textup{Beta}(\alpha_i,&space;\beta_i)" title="\textup{Beta}(\alpha_i, \beta_i)" /> for every action. The best action is selected among samplers. After the true reward is observed, we can update the Beta distribution accordingly, which is essentially doing Bayesian inference to compute posterior with the known prior and the likelihood of getting the sampled data.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\alpha_i&space;\leftarrow&space;\alpha_i&space;&plus;&space;r_t\mathbb{I}[a_t^{\textup{TS}}=a_i]" title="\alpha_i \leftarrow \alpha_i + r_t\mathbb{I}[a_t^{\textup{TS}}=a_i]" />\
<img src="https://latex.codecogs.com/svg.latex?\beta_i&space;\leftarrow&space;\beta_i&space;&plus;&space;(1-r_t)\mathbb{I}[a_t^{\textup{TS}}=a_i]" title="\beta_i \leftarrow \beta_i + (1-r_t)\mathbb{I}[a_t^{\textup{TS}}=a_i]" />
</p>

Thompson sampling implements the idea of probability matching. Because its reward estimations <img src="https://latex.codecogs.com/svg.latex?\widetilde{Q}" title="\widetilde{Q}" /> are sampled from posterior distributions, each of these probabilities is equivalent to the probability that the corresponding action is optimal, conditioned on observed history.

## Conclusion

We need exploration because information is valuable. In terms of the exploration strategies, we cannot do exploration at all, focusing on the short-term returns. Or we occasionally explore at random. Or even further, we explore and we are picky about which options to explore — actions with higher uncertainty are favored because they can provide higher information gain.

<p align="center">
<b>Summary of algorithms discussed.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/35/Figure-2.png?raw=true" alt="Figure 2"/>
</p>
