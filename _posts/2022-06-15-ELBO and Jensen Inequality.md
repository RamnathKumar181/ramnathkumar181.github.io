---
layout: post
title: ELBO and Jensen Inequality
published: true
---

An overview of the concept  “[ELBO](https://mbernste.github.io/posts/elbo/)” and Jensen Inequality.
<!--break-->
In this work, we discuss two important concepts to theoretical machine learning, often used in proofs/derivation of energy functions, convergence, etc.

## ELBO

The **evidence lower bound (ELBO)** is an important quantity that lies at the core of number of important probabilistic inferences such as expectation-maximization, and variational inference.

In a latent variable model, we posit that our observed data <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> is a realization from some random variable <img src="https://latex.codecogs.com/svg.latex?X" title="X" />. Moreover, we posit the existence of another random variable <img src="https://latex.codecogs.com/svg.latex?Z" title="Z" /> where <img src="https://latex.codecogs.com/svg.latex?X" title="X" /> and <img src="https://latex.codecogs.com/svg.latex?Z" title="Z" /> are distributed according to a joint distribution <img src="https://latex.codecogs.com/svg.latex?p(X,Z;\theta)" title="p(X,Z;\theta)" /> where <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> parameterizes the distribution. Note that <img src="https://latex.codecogs.com/svg.latex?X" title="X" /> is observed an not <img src="https://latex.codecogs.com/svg.latex?Z" title="Z" /> (latent).

There are two predominant tasks that are of interest here:
* Given some fixed value of <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, we would like to compute the posterior distribution of the latent, i.e., <img src="https://latex.codecogs.com/svg.latex?P(Z|X;\theta)" title="p(Z|X;\theta)" />.
* Given that <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> is unknown, we would like to find the maximum likelihood estimate of <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />: <img src="https://latex.codecogs.com/svg.latex?\arg \max_{\theta}l(\theta)" title="\arg \max_{\theta}l(\theta)" />. Here, <img src="https://latex.codecogs.com/svg.latex?l" title="l" /> is the log-likelihood function:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?l(\theta) := \log p(x;\theta) = \log \int _z p(x,z;\theta)dz" title="l(\theta) := \log p(x;\theta) = \log \int _z p(x,z;\theta)dz" />
</p>

Variational inference is used for Task 1, and expectation maximization is used for Task 2.

### What is ELBO?

**Evidence** is a name given to the likelihood function evaluated at a fixed parameter <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{evidence} :=  \log p(x;\theta)" title="\text{evidence} :=  \log p(x;\theta)" />
</p>
Intuitively, if we have chosen a right <img src="https://latex.codecogs.com/svg.latex?p" title="p" /> and <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, then we would expect that the marginal probability of our observed data <img src="https://latex.codecogs.com/svg.latex?x" title="x" />, would be high. Thus a higher value of evidence would suggest we are on the right track.



<p align="center">
<b> The student model is trained via minimizing the dynamic loss functions taught by the teacher model (yellow curve). The bottom black plane represents the parameter space of student model, and the four colored mesh surfaces denote different loss functions outputted via teacher model at different phases of student model training.</b>
</p>
<p align="center">
<img src="/assets/Papers/5/Figure-4.png?raw=true" alt="Figure 1"/>
</p>

## Model

From a technical point of view, the paper offers two distinctive concepts accourding to the authors:
* They leverage gradient based optimization method rather than reinforce-
ment learning. This would be ideal since RL approaches would be unstable and require millions of samples to learn an optimal policy.
* It is difficult when the error information cannot be
directly back propagated from the loss function, since they aim at discovering the best loss function for the machine learning models. They design an algorithm based on Reverse-Mode Differentiation (RMD) to tackle such a difficulty.

Their overall model called L2T-DLF includes two parts, a student model and teacher model.

### Student Model

The student model hopes to learn an optimal <img src="https://latex.codecogs.com/svg.latex?w^{*}" title="w^{*}" /> by minimizing the loss function provided by the teacher network. The learnt student model is evaluated on the test dataset to obtain a score, which measures the similarity between the true output and predicted output. The loss function used by the model acts as the surrogate of <img src="https://latex.codecogs.com/svg.latex?m" title="m" /> to evaluate the student model during it's training process. This loss is given by <img src="https://latex.codecogs.com/svg.latex?l_{\phi}(\widehat{y}, y)" title="l_{\phi}(\widehat{y}, y)" />. It could be a simple linear model, or a deep learning network which learns this.


### Teacher Model

A teacher model is responsible for setting the proper loss function <img src="https://latex.codecogs.com/svg.latex?l" title="l" /> to the student model by outputting appropriate loss function coefficients <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" />. To cater for different status of student model training, we ask the teacher model to output different loss functions <img src="https://latex.codecogs.com/svg.latex?l_t" title="l_t" /> at each training step <img src="https://latex.codecogs.com/svg.latex?t" title="t" />. To achieve that, the
status of a student model is represented by a state vector <img src="https://latex.codecogs.com/svg.latex?s_t" title="s_t" /> at timestep <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, which contains for example the current training/dev accuracy and iteration number. The teacher model, denoted as <img src="https://latex.codecogs.com/svg.latex?\mu" title="\mu" />, then takes
<img src="https://latex.codecogs.com/svg.latex?s_t" title="s_t" /> as inputs to compute the coefficients of loss function <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" />t at <img src="https://latex.codecogs.com/svg.latex?t" title="t" />-th timestep as <img src="https://latex.codecogs.com/svg.latex?t" title="t" />, where <img src="https://latex.codecogs.com/svg.latex?\phi_t = \mu_{\theta}(s_t)" title="\phi_t = \mu_{\theta}(s_t)" />,<img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" /> is the parameters of the teacher model. The goal of the teacher model is to maximize the performance of the induced student model on the stand-alone test/development dataset. The training process of the teacher model is described in more detail in the upcoming section.

### Training Process of Teacher Model

<p align="center">
<b> Left: the bilinear neural network specifying the loss function. Right: the teacher model outputting <img src="https://latex.codecogs.com/svg.latex?\phi_t" title="\phi_t" />.</b>
</p>
<p align="center">
<img src="/assets/Papers/5/Figure-5.png?raw=true" alt="Figure 2"/>
</p>

We update the teacher parameter to decrease the similarity/loss on the test dataset.

## Conclusion

In contrast to expert designed and fixed loss functions in conventional machine learning systems, this paper studies how to learn dynamic loss functions so as to better teach a student machine learning model. Since loss functions provided by the teacher model dynamically change with respect
to the growth of the student model and the teacher model is trained through end-to-end optimization, the quality of the student model gets improved significantly, as shown in their experiments from the paper.
