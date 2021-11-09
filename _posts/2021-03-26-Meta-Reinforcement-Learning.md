---
layout: post
title: Meta-Reinforcement-Learning
published: true
---

An overview of the topic Meta Reinforcement Learning from the "[Meta-Learning: Learning to Learn Fast ICML tutorial](https://sites.google.com/view/icml19metalearning)".
<!--break-->
All images and tables in this post are from their respective paper.
The reinforcement learning model is quite interesting, but these methods take thousands or millions of iterations to learn a given task. We do not notice this, since we emulate the scenario, but the algorithms by itself are quite poor. However, people can learn new skills extremely quickly. This is probably because, we never learn from scratch and incorporating from previous experiences. The field of Meta-RL provides a hope for algorithms that are much more efficient and more human-like.

## Reinforcement Learning Problem

In reinforcement learning, we deal with a markov decision process. This is defined as a quadruple <img src="https://latex.codecogs.com/svg.latex?M&space;=&space;\begin{Bmatrix}&space;S,A,P,r&space;\end{Bmatrix}" title="M = \begin{Bmatrix} S,A,P,r \end{Bmatrix}" /> where, <img src="https://latex.codecogs.com/svg.latex?S" title="S" /> is the state space (the current state of the agent), <img src="https://latex.codecogs.com/svg.latex?A" title="A" /> is the action space (decision that the agent gets to choose), the transition function <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> (<img src="https://latex.codecogs.com/svg.latex?(s_t,a_t)\rightarrow&space;s_{t&plus;1}" title="(s_t,a_t)\rightarrow s_{t+1}" />), and the reward function is a scalar function <img src="https://latex.codecogs.com/svg.latex?r" title="r" />. The goal of the model is to increase this reward function, but not in a greedy way. It needs to maximize the reward over the entire episode/execution of the agent. The goal is to learn a policy <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}(a|s))" title="\pi_{\theta}(a|s))" />, which describes how the actions are chosen. The entire workflow of any RL algorithm can be quickly summarized in one figure as shown below.

<p align="center">
<b>Every RL algorithm in a nutshell.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/24/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Meta Reinforcement Learning

Similar to the generic "meta-learning" framework, the optimum value of <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> is one that minimizus the loss over the sum of all tasks/MDP. The meta training set is now a set <img src="https://latex.codecogs.com/svg.latex?\begin{Bmatrix}&space;M_1,...,M_n&space;\end{Bmatrix}" title="\begin{Bmatrix} M_1,...,M_n \end{Bmatrix}" /> which is a set of meta-training MDPs drawn/sampled over some distribution of tasks <img src="https://latex.codecogs.com/svg.latex?p(M)" title="p(M)" />. Once, we have trained, we sample another set for test samples, and we want to optimize our model such that all the tasks in the test set are performing fairly well.

### Meta-RL with recurrent policies

The goal of a meta RL is to optimize <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> such that <img src="https://latex.codecogs.com/svg.latex?\theta^*&space;=&space;\arg&space;\max_{\theta}&space;\sum&space;_{i=1}^n&space;\mathbb{E}_{\pi_{\phi_i}(\tau)}[R(\tau)]" title="\theta^* = \arg \max_{\theta} \sum _{i=1}^n \mathbb{E}_{\pi_{\phi_i}(\tau)}[R(\tau)]" /> where, <img src="https://latex.codecogs.com/svg.latex?\phi_i&space;=&space;f_{\theta}(M_i)" title="\phi_i = f_{\theta}(M_i)" />. The main question is how to implement <img src="https://latex.codecogs.com/svg.latex?f_{\theta}(M_i)" title="f_{\theta}(M_i)" />?
We need to implement a <img src="https://latex.codecogs.com/svg.latex?f_{\theta}(M_i)" title="f_{\theta}(M_i)" /> such that:
* Improve policy with experience <img src="https://latex.codecogs.com/svg.latex?M" title="M" />
* Choose how to interact, i.e. choose <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" />. Meta-RL must also choose how to explore.

## Meta-RL as a black-box inference

In our black box model, it will read in the transitions one at a time, and will produce some type of a hidden state. The policy will then use the hidden state and the current state to choose an action. As before, <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^*" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta^*" title="\theta^*" /></a> will be the weights of the RNN. The last bit is the <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_{\phi}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\pi_{\phi}" title="\pi_{\phi}" /></a>. Just as before, the <a href="https://www.codecogs.com/eqnedit.php?latex=\phi_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi_i" title="\phi_i" /></a> consists of <a href="https://www.codecogs.com/eqnedit.php?latex=[h_i,\theta_{\pi}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?[h_i,\theta_{\pi}]" title="[h_i,\theta_{\pi}]" /></a>. Only <a href="https://www.codecogs.com/eqnedit.php?latex=h" target="_blank"><img src="https://latex.codecogs.com/svg.latex?h" title="h" /></a> is adapted, whereas, <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_{\pi}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta_{\pi}" title="\theta_{\pi}" /></a> is meta-trained but not adapted. Furthermore, <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_{\pi}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\theta_{\pi}" title="\theta_{\pi}" /></a> are the meta-learned weights.
The second condition is satisfied by default, if we do not reset RNN hidden states between episodes. <b>Crucially, RNN hidden states are not reset between episodes!</b>

## Meta-RL as an optimization problem

For meta-RL we have <img src="https://latex.codecogs.com/svg.latex?\theta^*&space;=&space;\arg&space;\max_{\theta}&space;\sum_{i=1}^n&space;E_{\pi_{\phi_i}(\tau)}[R(\tau)]" title="\theta^* = \arg \max_{\theta} \sum_{i=1}^n E_{\pi_{\phi_i}(\tau)}[R(\tau)]" />, where <img src="https://latex.codecogs.com/svg.latex?\phi_i&space;=&space;f_{\theta}(M_i)" title="\phi_i = f_{\theta}(M_i)" />. If we consider <img src="https://latex.codecogs.com/svg.latex?f_{\theta}(M_i)" title="f_{\theta}(M_i)" /> itself as an RL algorithm, we will denote <img src="https://latex.codecogs.com/svg.latex?f_{\theta}(M_i)&space;=&space;\theta&space;&plus;&space;\alpha\triangledown&space;_{\theta}J_i(\theta)" title="f_{\theta}(M_i) = \theta + \alpha\triangledown _{\theta}J_i(\theta)" />. This is very similar to the regular MAML algorithm.

## Meta-RL as partially observed RL
Extending the regular RL, the MDP's also contain the observation space <img src="https://latex.codecogs.com/svg.latex?O" title="O" /> and the emission probability <img src="https://latex.codecogs.com/svg.latex?p(o_t|s_t)" title="p(o_t|s_t)" /> such that <img src="https://latex.codecogs.com/svg.latex?M&space;=&space;\begin{Bmatrix}&space;S,A,O,P,E,r&space;\end{Bmatrix}" title="M = \begin{Bmatrix} S,A,O,P,E,r \end{Bmatrix}" />.
Here, the policy must act on the observations <img src="https://latex.codecogs.com/svg.latex?o_t" title="o_t" />, i.e. <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}(a|o_t)" title="\pi_{\theta}(a|o_t)" /> which typically requires either:
* Explicit state estimation, i.e. to estimate <img src="https://latex.codecogs.com/svg.latex?p(s_t|o_{1:t})" title="p(s_t|o_{1:t})" />
* Policies with memory
We can define the policy as <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}(a|s,z)" title="\pi_{\theta}(a|s,z)" /> where <img src="https://latex.codecogs.com/svg.latex?z" title="z" /> encapsulates information policy needed to solve the current task. Furthermore, learning a task is equivalent to inferring <img src="https://latex.codecogs.com/svg.latex?z" title="z" />. <img src="https://latex.codecogs.com/svg.latex?z" title="z" /> is inferred from context/transitions upto now.
Note that the initial MDP is defined as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?M&space;=&space;\begin{Bmatrix}&space;S,A,P,r&space;\end{Bmatrix}" title="M = \begin{Bmatrix} S,A,P,r \end{Bmatrix}" />
</p>
and the new POMDP (Partially observed MDP) as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?M&space;=&space;\begin{Bmatrix}&space;\widetilde{S},A,\widetilde{O},\widetilde{P},\varepsilon&space;,r&space;\end{Bmatrix}" title="M = \begin{Bmatrix} \widetilde{S},A,\widetilde{O},\widetilde{P},\varepsilon ,r \end{Bmatrix}" />
</p>
Here, <img src="https://latex.codecogs.com/svg.latex?\widetilde{S}&space;=&space;S\times&space;Z" title="\widetilde{S} = S\times Z" /> and <img src="https://latex.codecogs.com/svg.latex?\widetilde{O}&space;=&space;S" title="\widetilde{O} = S" />. We could re-interpret our policy as <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}(a|\widetilde{s})" title="\pi_{\theta}(a|\widetilde{s})" />. hence, solving the POMDP is equivalent to meta-learning.

### Posterior Sampling Strategy
To solve the above problem, we would need to estimate <img src="https://latex.codecogs.com/svg.latex?z" title="z" />, which can be written as <img src="https://latex.codecogs.com/svg.latex?p(z_t|s_{1:t},a_{1:t},r_{1:t})" title="p(z_t|s_{1:t},a_{1:t},r_{1:t})" />. In the method of posterior sampling with latent context, we follow a 2 step process:
* Sample <img src="https://latex.codecogs.com/svg.latex?z\sim&space;\widehat{p}(z_t|s_{1:t},a_{1:t},r_{1:t})" title="z\sim \widehat{p}(z_t|s_{1:t},a_{1:t},r_{1:t})" /> which is some approximate posterior, which we assume to be the correct value.
* Act according to policy <img src="https://latex.codecogs.com/svg.latex?\pi_{\theta}(a|s,z)" title="\pi_{\theta}(a|s,z)" /> to collect more data. Repeat Step 1 again.

Although this algorithm is not optimal, it is pretty good, both in theory and practice.
In the case, where <img src="https://latex.codecogs.com/svg.latex?\widehat{p}" title="\widehat{p}" /> is a variational encoder, we would reach the following equation:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?(\theta,&space;\phi)&space;=&space;\arg&space;\max_{\theta,&space;\phi}&space;\frac{1}{N}\sum&space;_{i=1}^N&space;\mathbb{E}_{z\sim&space;q_{\phi},&space;\tau&space;\sim&space;\pi_{\theta}}[R_i(\tau)-D_{KL}(q(z|...)||p(z))]" title="(\theta, \phi) = \arg \max_{\theta, \phi} \frac{1}{N}\sum _{i=1}^N \mathbb{E}_{z\sim q_{\phi}, \tau \sim \pi_{\theta}}[R_i(\tau)-D_{KL}(q(z|...)||p(z))]" />
</p>
In this equation, the first term maximizes post-update reward (same as meta-RL in RNN), and the second term makes sure that our context is very close to our prior context. Having a stochastic <img src="https://latex.codecogs.com/svg.latex?z" title="z" /> enables exploration via posterior sampling.

<p align="center">
<b>Posterior Sampling.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/24/Figure-2.gif?raw=true" alt="Figure 2"/>
</p>

## Three perspectives on meta-RL

So far, we have discussed three perspectives of meta-RL:
* <b>Black-box inference:</b> In this perspective, we use a RNN without flushing out hidden states after each episode. These models are conceptually simple, relatively easy to apply, but are vulnerable to meta-overfitting andd challenging to optimize in practice.
* <b>Gradient based optimization:</b> In this perspective, we use an optimization similar to using MAML. These models are good in extrapolation (and are consistent), conceptually elegant but might complex to implement and may require many samples to optimize correctly.
* <b>POMDP:</b> In this perspective, we use hidden states/context using posterior sampling to learn our model. These models are simple, effective exploration models, and are elegant reduction to solving a special POMDP. However, similar to the first perspective, these would be vulnerable to meta-overfitting.

In all these cases, the algorithm must do the same two essential steps, and are closely related to each other:
* Improve policy with experience <img src="https://latex.codecogs.com/svg.latex?M" title="M" />
* Choose how to interact, i.e. choose <img src="https://latex.codecogs.com/svg.latex?a_t" title="a_t" />. Meta-RL must also choose how to explore.

## Challenges and Frontiers

### Meta-Overfitting

Meta learning requires to define the task distributions in order to meta-train. The idea is that, if we have a wide variety of tasks, we would learn a policy that is very easily adaptable to our meta-test. However, these tasks have to be designed by hand, and if there are too few tasks at hand, we could meta-overfit a model. By definition, the model would be very good at the meta-train tasks, but would fail to adapt to meta-test tasks, even though they are structurally similar. Specifying theese tasks distributions are quite hard, and require further research. However, some algorithms meta-overfit less than other algorithms. For example,
* In the case of black-box adaptation, the performance of the model replies entirely on extrapolation of learned adaptation procedure. If the meta-train dataset is very different from the meta-test, the model would fail.
* In optimization based models, the issue is a bit less. Pure gradient descent is not efficient without the benefit of a good initialization, and the problem would persist.
* For non-parametric models, the model might not adapt all parameters of metric on new data, i.e. the nearest neighbor prediction might be derived from a bad metric space.

A consistent meta-learner will converge to a (locally) optimal solution on any new task, regardless of meta-learning. By this definition, the optimization based approach would be consistent even in the case of o.o.d. distributions. However, teh black-box adaptation approach would be non-consistent, since, there is no training in the test domain. The non-parametric approaches are somehwat in the middle, and are consistent, but might be so in the wrong metric space.

<p align="center">
<b>Learning performance on o.o.d tasks. (SNAIL and Meta-Net are Meta-learners, and MAML is optimization based)</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/24/Figure-3.png?raw=true" alt="Figure 3"/>
</p>

Besides using methods that are not prone to overfitting, we could propose new tasks automatically. This would lead to an <b>unsupervised meta-learning algorithm</b>. These algorithms learn to solve tasks efficiently, without using hand-specified labels during meta-training. For example, the data could be clustered to form tasks which are then fed to the meta-learning algorithm.
The same idea could be used for RL, where we generate random task proposals which are then fed to our meta-learning algorithm.

### Memorization

This problem is related to meta-overfitting, but is subtly different. In this case, we have a problem where the task data isn't strictly needed to learn the task. For example, our meta-train set consists of the task to distinguish a dog from a cat, and our meta-test consists of a task to recognize new breeds of a cat. In this scenario, the meta-train set might not capture any features which vary among the breeds of cats, and might focus only on the differences between the two animals.
This is essentially a zero-shot task. The task needs to be mutually exclusive, i.e. not possible to learn a single function to learn all the tasks. What you want the learner to learn from the data must be not present in training set. Hence, there comes a tradeoff between which information should be in the input/train, and which should be in the data/test.

In the scenario where <b>Input contains no information about the task</b> (or for broad meta-RL task distributions), we notice that exploration becomes exceedingly challenging. In these scenarios, there could be additional information provided along with the task, such as a demonstration+trials, language instructions, etc.

Ultimately, the goal of meta-learning was to given i.i.d. task distribution, learning a new task becomes more efficient. However, more realistically, we might learn tasks sequentially, but becoming more efficient in learning new tasks over time (slow learning -> rapid learning). This becomes problem of online learning or continual learning.
