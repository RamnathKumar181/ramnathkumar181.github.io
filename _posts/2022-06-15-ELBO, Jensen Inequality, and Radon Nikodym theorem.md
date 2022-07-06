---
layout: post
title: ELBO, Jensen Inequality, and Radon Nikodym theorem
published: true
---

An overview of the concept  “[ELBO](https://mbernste.github.io/posts/elbo/)”, “[Jensen Inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)”, and “[Radon-Nikodym theorem](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem)”.
<!--break-->
In this work, we discuss two important concepts to theoretical machine learning, often used in proofs/derivation of energy functions, convergence, etc.

## Jensen Inequality

If <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> is a convex function, then
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f(\mathbb{E}[x]) \leq \mathbb{E}[f(x)]" title="f(\mathbb{E}[x]) \leq \mathbb{E}[f(x)]" />
</p>

Similarly, if <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> is a concave function, then:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)]" title="f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)]" />
</p>

The difference between the two sides of the inequality, \mathbb{E}[f(x)] - f(\mathbb{E}[x]), is called Jensen gap.

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

## Radon-Nikodym theorem

Before we get into the Radon-Nikodym theorem, it would be worthwhile to recall the two types of Integration: Riemann Integration, and Labesgue Integration.

### Riemann Integration

Let us consider a function <img src="https://latex.codecogs.com/svg.latex?f" title="f" />, and we would like to compute the area under this graph. To this, we create small rectangles below this graph, and sum up all rectangles as the number of rectangles approaches infinity. This is our regular integration. However, this type of integration has its own sets of limitations:
* It is very difficult to extend to **higher dimensions**. To approximate the volume of a surface, we could use cuboids instead of rectangles, but for 4D, visualization of this integration becomes increasingly difficult.
* The second shortcoming is some assumption of continuity. It is extremely ill suited for step functions. A function is Riemann Integrable if it's discontinuity set has a measure 0. In the scenario when we need to integrate over all real numbers where each rational number has a score 0 and each irrational number has a score 1. Since the discontinuity set although less, is still infinity, the measure is not 0, and is not Riemann Integrable.

To overcome these limitations, the French mathematician Henri Lebesgue introduced the concept of Lebesgue integrations.

### Lebesgue Integration

Instead of splitting the integral along the interval or the x axis, he split it across the range or the y axis. This intrinsically solves the above example introduced. We need to consider two cases, (i) for rational, when <img src="https://latex.codecogs.com/svg.latex?f(x)=0" title="f(x)=0" />, and (ii) for irrational, when <img src="https://latex.codecogs.com/svg.latex?f(x)=1" title="f(x)=1" />. Here,

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\int_0^1 f(x)d\mu = 0.\mu(A_0) + 1.\mu(A_1)" title="\int_0^1 f(x)d\mu = 0.\mu(A_0) + 1.\mu(A_1)" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\mu" title="\mu" /> is the size of set, or the Labesgue measure. In general, the Labesgue integral can be defined as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\int_a^b f(x)d\mu = \sum _{i=1}^n y_i.\mu(A_{y_i})" title="\int_a^b f(x)d\mu = \sum _{i=1}^n y_i.\mu(A_{y_i})" />
</p>
For the continuous case, we need to imagine an infinite number of step functions which roughly map the function we wish to find the integrate for, i.e.:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\int_a^b f(x)d\mu = \lim_{n\rightarrow \infty}\int _{a}^b f_nd\mu" title="\int_a^b f(x)d\mu = \lim_{n\rightarrow \infty}\int _{a}^b f_nd\mu" />
</p>


The Radon-Nikodym theorem essentially states that, under certain conditions, any measure <img src="https://latex.codecogs.com/svg.latex?\nu" title="\nu" /> can be expressed in this way with respect to another measure <img src="https://latex.codecogs.com/svg.latex?\mu" title="\mu" /> on the same space.
One way to derive a new measure from one already given is to assign a density to each point of the space, and integrate over the measurable subset of interest. This can be expressed as follows:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\nu(A) = \int_A fd\mu" title="\nu(A) = \int_A fd\mu" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\nu" title="\nu" /> is a new measure defined for any measurable subset <img src="https://latex.codecogs.com/svg.latex?A" title="A" />, and the function <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> is the density at a given point.
The function <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> is then called the **Radon-Nikodym derivative** and is denoted by <img src="https://latex.codecogs.com/svg.latex? \frac{\partial \nu}{\partial \mu}" title=" \frac{\partial \nu}{\partial \mu}" />.
