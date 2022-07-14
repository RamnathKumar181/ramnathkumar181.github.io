---
layout: post
title: Evidence Lower Bound
published: true
book_title: Random Math Concepts
---

An overview of the concept  “[ELBO](https://mbernste.github.io/posts/elbo/)”.
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

Intuitively, if we have chosen a right <img src="https://latex.codecogs.com/svg.latex?p" title="p" /> and <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, then we would expect that the marginal probability of our observed data <img src="https://latex.codecogs.com/svg.latex?x" title="x" />, would be high. Thus a higher value of evidence would suggest we are on the right track. The derivation goes something as follows:


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\log p(x;\theta) = \log \int _z p(x,z;\theta)dz" title="\log p(x;\theta) = \log \int _z p(x,z;\theta)dz" />
<img src="https://latex.codecogs.com/svg.latex?= \log \int _z p(x,z;\theta)\frac{q(z)}{q(z)}dz" title="= \log \int _z p(x,z;\theta)\frac{q(z)}{q(z)}dz" />
<img src="https://latex.codecogs.com/svg.latex?= \log \mathbb{E}_{Z \sim q}\begin{bmatrix} \frac{p(x,Z; \theta)}{q(z)} \end{bmatrix}" title="= \log \mathbb{E}_{Z \sim q}\begin{bmatrix} \frac{p(x,Z; \theta)}{q(z)} \end{bmatrix}" />
<img src="https://latex.codecogs.com/svg.latex?\geq \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{p(x,Z; \theta)}{q(z)}\end{bmatrix}" title="\geq \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{p(x,Z; \theta)}{q(z)}\end{bmatrix}" />
</p>

The final line makes use of Jensen's Inequality of concave function (log, since its double derivative is less than 0).

The gap between the evidence and ELBO can also be computed as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{KL}(q(z)||p(z|x;\theta)) := \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{q(Z)}{p(Z|x;\theta)} \end{bmatrix}" title="\text{KL}(q(z)||p(z|x;\theta)) := \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{q(Z)}{p(Z|x;\theta)} \end{bmatrix}" />
<img src="https://latex.codecogs.com/svg.latex?=\mathbb{E}_{Z \sim q} \begin{bmatrix} \log q(Z) \end{bmatrix} - \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{p(x,Z;\theta)}{p(x;\theta)} \end{bmatrix}" title="=\mathbb{E}_{Z \sim q} \begin{bmatrix} \log q(Z) \end{bmatrix} - \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{p(x,Z;\theta)}{p(x;\theta)} \end{bmatrix}" />
<img src="https://latex.codecogs.com/svg.latex?=\mathbb{E}_{Z \sim q} \begin{bmatrix} \log q(Z) \end{bmatrix} - \mathbb{E}_{Z \sim q} \begin{bmatrix} \log p(x,Z;\theta) \end{bmatrix} + \mathbb{E}_{Z \sim q} \begin{bmatrix} \log p(x;\theta) \end{bmatrix}" title="=\mathbb{E}_{Z \sim q} \begin{bmatrix} \log q(Z) \end{bmatrix} - \mathbb{E}_{Z \sim q} \begin{bmatrix} \log p(x,Z;\theta) \end{bmatrix} + \mathbb{E}_{Z \sim q} \begin{bmatrix} \log p(x;\theta) \end{bmatrix}" />
<img src="https://latex.codecogs.com/svg.latex?=\log p(x;\theta) - \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{p(x,Z;\theta)}{q(Z)} \end{bmatrix}" title="=\log p(x;\theta) - \mathbb{E}_{Z \sim q} \begin{bmatrix} \log \frac{p(x,Z;\theta)}{q(Z)} \end{bmatrix}" />
<img src="https://latex.codecogs.com/svg.latex?= \text{evidence} - \text{ELBO}" title="= \text{evidence} - \text{ELBO}" />
</p>
