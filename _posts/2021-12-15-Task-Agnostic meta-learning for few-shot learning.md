---
layout: post
title: Task-Agnostic Meta-Learning for few-shot learning
published: true
---

An overview of the paper “[Task-Agnostic Meta-Learning for few-shot learning](https://arxiv.org/pdf/1805.07722.pdf)”.
<!--break-->
The author presents a method for a task-agnostic meta-learning algorithm built on top of MAML. The formulation could be extended to other algorithms with little effort. All images and tables in this post are from their paper.

## Introduction

Typically, a meta-learner is trained on a variety of tasks in the hopes of being generalizable to new tasks. However, the generalizability on new tasks of a meta-learner could be fragile when it is over-trained on existing tasks during meta-training phase. In other words, the initial model of a meta-learner could be too biased towards existing tasks to adapt to new task, especially when only very few examples are available to update the model.
The authors propose TAML to avoid a biased meta-learner. Specifically, they present an entropy-based approach that meta-learns an unbiased initial model with the largest uncertainty over the output labels by preventing it from over-performing in classification tasks. However, this approach is limited to discrete outputs from a model making it more amenable to classification tasks. Alternatively, a more general inequality-minimization TAML is presented for more ubiquitous scenarios by directly minimizing the inequality of initial losses beyond the classification tasks wherever a suitable loss can be defined. This makes the paradigm more ubiquitous and can be extended to other domains of regression and reinforcement learning.

## Task Agnostic Meta-Learning

The problem with the current meta-learning approach is that the initial model or learner can be biased towards some tasks during the meta-training phase, particularly when future tasks in the test phase may have discrepancy from those in the training tasks. In this case, the authors wish to avoid the initial model over-performing on some tasks. Moreover, an over-performed initial model could also prevent the meta-learner to learn a better update rule with consistent performance across tasks.


### Entropy-Maximization/Reduction TAML

To prevent the initial model <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" /> from over-performing on a task, we prefer it makes a random guess over predicted labels with an equal probability so that it is not biased towards the task. This can be expressed as a maximum-entropy prior over <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> so that the initial model should have a large entropy over the predicted labels. The entropy for task <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}_i" title="\mathcal{T}_i" /> is computed by sampling <img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" /> from <img src="https://latex.codecogs.com/svg.latex?P_{\mathcal{T}_i}(x)" title="P_{\mathcal{T}_i}(x)" /> over its output probabilities <img src="https://latex.codecogs.com/svg.latex?y_{i,n}" title="y_{i,n}" /> over <img src="https://latex.codecogs.com/svg.latex?N" title="N" /> predicted labels:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{H}_{\mathcal{T}_i}(f_{\theta})&space;=&space;-\mathbb{E}_{x_i&space;\sim&space;P_{\mathcal{T}_i}(x)}&space;\sum_{n=1}^{N}&space;\widehat{y}_{i,n}&space;\log(\widehat{y}_{i,n})" title="\mathcal{H}_{\mathcal{T}_i}(f_{\theta}) = -\mathbb{E}_{x_i \sim P_{\mathcal{T}_i}(x)} \sum_{n=1}^{N} \widehat{y}_{i,n} \log(\widehat{y}_{i,n})" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?\left&space;[&space;y_{i,1},...,y_{i,N}&space;\right&space;]&space;=&space;f_{\theta}(x_i)" title="\left [ y_{i,1},...,y_{i,N} \right ] = f_{\theta}(x_i)" /> is the prediction by <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" />, which are often an output from a softmax layer in a classification task.

Alternatively, one can not only maximize the entropy before the update of initial model's parameter, but also minimize the entropy after the update. So overall, we maximize the entropy reduction for each task <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}_i" title="\mathcal{T}_i" /> as
<img src="https://latex.codecogs.com/svg.latex?\mathcal{H}_{\mathcal{T}_i}(f_{\theta})-\mathcal{H}_{\mathcal{T}_i}(f_{\theta_i})" title="\mathcal{H}_{\mathcal{T}_i}(f_{\theta})-\mathcal{H}_{\mathcal{T}_i}(f_{\theta_i})" />. The minimization of <img src="https://latex.codecogs.com/svg.latex?\mathcal{H}_{\mathcal{T}_i}(f_{\theta_i})" title="\mathcal{H}_{\mathcal{T}_i}(f_{\theta_i})" /> means that the model can become more certain about the labels with a higher confidence after upding the parameter from <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> to <img src="https://latex.codecogs.com/svg.latex?\theta_i" title="\theta_i" />. This entropy term can be combined with the typical meta-training objective term as a regularizer to find the optimal <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, which is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\min_{\theta}\mathbb{E}_{\mathcal{T}_i\sim&space;P(\mathcal{T})}&space;\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i})&space;&plus;&space;\lambda&space;\left&space;[&space;-\mathcal{H}_{\mathcal{T}_i}(f_{\theta})&space;&plus;&space;\mathcal{H}_{\mathcal{T}_i}(f_{\theta_i})&space;\right&space;]" title="\min_{\theta}\mathbb{E}_{\mathcal{T}_i\sim P(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i}) + \lambda \left [ -\mathcal{H}_{\mathcal{T}_i}(f_{\theta}) + \mathcal{H}_{\mathcal{T}_i}(f_{\theta_i}) \right ]" />
</p>

Unfortunately, the entropy-based TAML is subject to a critical limitation - it is only amenable to discrete labels in classification tasks to compute the entropy.

### Inequality-Minimization TAML

We wish to train a task-agnostic model in meta-learning such that its initial performance is unbiased towards any particular task <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}_i" title="\mathcal{T}_i" />. Such a task-agnostic meta-learner would do so by minimizing the inequality of its performances over different tasks.

Specifically, the bias of the initial model towards any particular tasks is minimized during meta-learning by minimizing the inequality over the losses of sampled tasks in a batch. So, given an unseen task during testing phase, a better generalization performance is expected on the new task by updating from an unbiased initial model with few examples. The key difference between both TAMLs lies that for entropy, we only consider one task at a time by computing the entropy of its output labels. Moreover, entropy depends on a particular form or explanation of output function. On the contrary, the inequality only depends on the loss, thus it is more ubiquitous. The algorithm learns to update the model parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> by minimizing the objective:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\min_{\theta}\mathbb{E}_{\mathcal{T}_i\sim&space;P(\mathcal{T})}&space;\left&space;[&space;\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i})&space;\right&space;]&space;&plus;&space;\lambda&space;\mathcal{I}_{\mathcal{E}}(\{\mathcal{L}_{\mathcal{T}_i}(f_{\theta})\})" title="\min_{\theta}\mathbb{E}_{\mathcal{T}_i\sim P(\mathcal{T})} \left [ \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i}) \right ] + \lambda \mathcal{I}_{\mathcal{E}}(\{\mathcal{L}_{\mathcal{T}_i}(f_{\theta})\})" />
</p>

It is worth noting that the inequality measure is computed over a set of losses from sampled tasks. The first term is the expected loss by the model <img src="https://latex.codecogs.com/svg.latex?f_{\theta_i}" title="f_{\theta_i}" /> after the update, while the second is the inequality of losses by the initial model <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" /> before the update.

## Inequality measures

Inequality measures are instrumental towards calculating the economic inequalities in the outcomes that can be wealth, incomes, or health related metrics. In meta-learning context, we use <img src="https://latex.codecogs.com/svg.latex?\ell_i&space;=&space;\mathcal{L}_{\mathcal{T}_i}(f_{\theta})" title="\ell_i = \mathcal{L}_{\mathcal{T}_i}(f_{\theta})" /> to represent the loss of a task <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}_i" title="\mathcal{T}_i" />, <img src="https://latex.codecogs.com/svg.latex?\overline{\ell}" title="\overline{\ell}" /> represents the mean of the losses over sampled tasks, and <img src="https://latex.codecogs.com/svg.latex?M" title="M" /> is the number of tasks in a single batch. There are few options of inequality measure that can be employed in our formulation:

* **Theil Index:** This inequality measure has been derived from redundancy in information theory, which is defined as the difference between the maximum entropy of the data and an observed entropy. Suppose that we have <img src="https://latex.codecogs.com/svg.latex?M" title="M" /> losses <img src="https://latex.codecogs.com/svg.latex?\{\ell_i|i=1,...,M\}" title="\{\ell_i|i=1,...,M\}" />, then Thiel Index is defined as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?T_T&space;=&space;\frac{1}{M}\sum&space;_{i=1}^M&space;\frac{\ell_i}{\overline{\ell}}&space;\ln&space;\frac{\ell_i}{\overline{\ell}}" title="T_T = \frac{1}{M}\sum _{i=1}^M \frac{\ell_i}{\overline{\ell}} \ln \frac{\ell_i}{\overline{\ell}}" />
</p>

* **Generalized Entropy Index:** Generalized entropy index has been proposed to measure the income inequality. It is not a single inequality measure, but is a family that includes many inequalitiy measures like Thiel Index, Thiel L, etc. For some real value <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" />, it is defined as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathit{GE}(\alpha)&space;=&space;\left\{\begin{matrix}&space;\frac{1}{M\alpha(\alpha-1)}\sum_{i=1}^M&space;\left&space;[&space;\left&space;(&space;\frac{\ell_i}{\overline{\ell}}&space;\right)^{\alpha}&space;-1&space;\right&space;],&space;&&space;\alpha&space;\neq&space;0,1,\\&space;\frac{1}{M}&space;\sum&space;_{i=1}^M&space;\frac{\ell_i}{\overline{\ell}}&space;\ln&space;\frac{\ell_i}{\overline{\ell}},&space;&&space;\alpha=1\\&space;-\frac{1}{M}&space;\sum&space;_{i=1}^M&space;\ln&space;\frac{\ell_i}{\overline{\ell}},&space;&&space;\alpha=0\\&space;\end{matrix}\right." title="\mathit{GE}(\alpha) = \left\{\begin{matrix} \frac{1}{M\alpha(\alpha-1)}\sum_{i=1}^M \left [ \left ( \frac{\ell_i}{\overline{\ell}} \right)^{\alpha} -1 \right ], & \alpha \neq 0,1,\\ \frac{1}{M} \sum _{i=1}^M \frac{\ell_i}{\overline{\ell}} \ln \frac{\ell_i}{\overline{\ell}}, & \alpha=1\\ -\frac{1}{M} \sum _{i=1}^M \ln \frac{\ell_i}{\overline{\ell}}, & \alpha=0\\ \end{matrix}\right." />
</p>

When <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> is zero, it is called a mean log deviation of Thiel L, and when <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> is one, it is actually Thiel Index. A larfer GE <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> value makes this index more sensitive to differences at the upper part of the distribution, and a smaller <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> value makes it more sensitive to difference at the bottom of the distribution.

* **Atkinson Index:** It is another measure for income inequality which is useful in determining which end of the distribution contributed the most to the observed inequality. It is defined as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?A_{\epsilon}&space;=&space;\left\{\begin{matrix}&space;1-\frac{1}{\mu}\left&space;(&space;\frac{1}{M}\sum&space;_{i=1}^M&space;\ell_{i}^{1-\epsilon}\right&space;)^{\frac{1}{1-\epsilon}},&space;&&space;\text{for&space;}&space;0\leq&space;\epsilon&space;\neq&space;1\\&space;1-\frac{1}{\overline{\ell}}\left&space;(&space;\frac{1}{M}&space;\prod&space;_{i=1}^M&space;\ell_i\right&space;)^{\frac{1}{M}},&space;&&space;\text{for&space;}&space;\epsilon&space;=&space;1\\&space;\end{matrix}\right." title="A_{\epsilon} = \left\{\begin{matrix} 1-\frac{1}{\mu}\left ( \frac{1}{M}\sum _{i=1}^M \ell_{i}^{1-\epsilon}\right )^{\frac{1}{1-\epsilon}}, & \text{for } 0\leq \epsilon \neq 1\\ 1-\frac{1}{\overline{\ell}}\left ( \frac{1}{M} \prod _{i=1}^M \ell_i\right )^{\frac{1}{M}}, & \text{for } \epsilon = 1\\ \end{matrix}\right." />
</p>

Where <img src="https://latex.codecogs.com/svg.latex?\epsilon" title="\epsilon" /> is called "inequality aversion parameter". When <img src="https://latex.codecogs.com/svg.latex?\epsilon&space;=&space;0" title="\epsilon = 0" />, the index becomes more sensitive to the changes in upper end of the distribution, and when it approaches to 1, the index becomes more sensitiveto the changes in lower end of the distribution.

* **Gini-Coefficient:** It is usually defined as the half of relative absolute mean difference. Gini-coefficient is more sensitive to deviation around the middle of the distribution than at the upper or lower part of the distribution.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?G&space;=&space;\frac{\sum_{i=1}^M&space;\sum_{j=1}^M&space;\left&space;|&space;\ell_i&space;-&space;\ell_j&space;\right&space;|}{2n&space;\sum_{i=1}^M&space;\ell_i}" title="G = \frac{\sum_{i=1}^M \sum_{j=1}^M \left | \ell_i - \ell_j \right |}{2n \sum_{i=1}^M \ell_i}" />
</p>

* **Variance of Logarithms:** This metric has a greater emphasis on the lower losses of the distribution.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?V_{L}(\ell)&space;=&space;\frac{1}{M}&space;\sum_{i=1}^M&space;\left&space;[&space;\ln&space;\ell_i&space;-&space;\ln&space;g(\ell)&space;\right&space;]^2" title="V_{L}(\ell) = \frac{1}{M} \sum_{i=1}^M \left [ \ln \ell_i - \ln g(\ell) \right ]^2" />
</p>

## Conclusion

The authors show the performance of their proposed TAML model on several experiments from classification to reinforcement-learning problems. 
