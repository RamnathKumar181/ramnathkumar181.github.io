---
layout: post
title: Deep Learning- Probability and Information Theory
published: true
---

An overview of the chapter “[Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)” from the famous book “[Deep Learning](https://www.deeplearningbook.org/)” written by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
<!--break-->
The authors create a brief introduction of the important concepts of probability and information theory that help guide machine learning. All images and tables in this post are from their book.

## Introduction

Probability can be seen as the extension of logic to deal with unveratinty. Probability can be viewed in two ways:
* <b>Bayesian Probability:</b> This indicates the certainty or the degree of belief that the event occurs. A degree of belief of 1 indicates absolute certainty of the event. This probability is qualitative in nature.
* <b>Frequentist Probability:</b> This indicates the rate at which the event occurs. This probability is quantitative in nature.

## Random Variables

A random variable is a variable that can take on different values randomly. These may be discrete or continuous. A discrete random variable is one that has a finity or countably infinite number of states. These nese states are not necessarily integers, and can also just be named states that are not considered to have any numberical value.
A continuous random value is associated with a real value.

## Probability Distributions

A probability distribtuion is a description of how likely a random variable or set of random variables is to take on each of its possible states. THe way we describe probability distributions depend on whether the variables are discrete or continuous.

### Discrete Variables and Probability Mass Functions

A probability over discrete variables may be described using a probability mass function. The probability of the event <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> taking place is <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x)" title="P(x)" />.
If there are more than one variable, we define a <b>joint probability distribution</b> as <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x,y)" title="P(x,y)" />.
To be a probability mass function on a random variable <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />, a function <img src="https://latex.codecogs.com/svg.latex?\inline&space;P" title="P" /> must satisfy the following properties:
* The domain of <img src="https://latex.codecogs.com/svg.latex?\inline&space;P" title="P" /> must be the set of all possible states of <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />.
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;\forall&space;x,&space;0\leq&space;P(x)\leq&space;1" title="\forall x, 0\leq P(x)\leq 1" />
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;\sum_{x}P(x)&space;=1" title="\sum_{x}P(x) =1" />

For example, consider uniform distribution where all the states are equally likely. Suppose there are <img src="https://latex.codecogs.com/svg.latex?\inline&space;k" title="k" /> states, the probability mass function is:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x=x_i)=\frac{1}{k}" title="P(x=x_i)=\frac{1}{k}" />
</p>

### Continuous Variables and Probability Density Functions

While working with continuous random variables, we describe probability distributions using a probability density function rather than a prabability mass function. To be a probability density function, a function <img src="https://latex.codecogs.com/svg.latex?\inline&space;p" title="p" /> must satisfy the following properties:
* The domain of <img src="https://latex.codecogs.com/svg.latex?\inline&space;p" title="p" /> must be the set of all possible states of <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />.
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;\forall&space;x,&space;p(x)\geq&space;0" title="\forall x, p(x)\geq 0" />
* <img src="https://latex.codecogs.com/svg.latex?\inline&space;\int&space;p(x)dx&space;=&space;1" title="\int p(x)dx = 1" />

The probability of event between <img src="https://latex.codecogs.com/svg.latex?\inline&space;a" title="a" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;b" title="b" />, can be obtained by computing <img src="https://latex.codecogs.com/svg.latex?\inline&space;\int_{a}^{b}&space;p(x)dx" title="\int_{a}^{b} p(x)dx" />. Also the probability of an event <img src="https://latex.codecogs.com/svg.latex?\inline&space;a" title="a" /> occuring is 0, since <img src="https://latex.codecogs.com/svg.latex?\inline&space;\int_{a}^{a}&space;p(x)dx&space;=&space;0" title="\int_{a}^{a} p(x)dx = 0" />.

## Marginal Probability

The probability over a subset is known as the marginal probability distribution. For discrete variables, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\forall&space;x,&space;P(x)&space;=&space;\sum&space;_y&space;P(x,y)" title="\forall x, P(x) = \sum _y P(x,y)" />. For continuous variables, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\forall&space;x,&space;p(x)&space;=&space;\int&space;p(x,y)dy" title="\forall x, p(x) = \int p(x,y)dy" />.

## Conditional Probability

Conditional probability is the probability of some event, given that some other event has happened.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(y|x)&space;=&space;\frac{P(x,y)}{P(x)}" title="P(y|x) = \frac{P(x,y)}{P(x)}" />
</p>
The conditional probability is only defined when <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x)>0" title="P(x)>0" />, we cannot compute the conditional probability conditioned on an event that never happens.
It is important not to confuse conditional probability with computing what would happen if some action were undertaken. Computing the conequences of an action is called making an <b>intervention queries</b>. Intervention queries are the domain of <b>causal modeling</b>.

## The Chain Rule of Conditional Probabilities

Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(a,b,c)=P(a|b,c)P(b,c)&space;=&space;P(a|b,c)P(b|c)P(c)" title="P(a,b,c)=P(a|b,c)P(b,c) = P(a|b,c)P(b|c)P(c)" />
</p>

## Independence and Conditional Independence

Two variables <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> are <b>independent</b> if their probability distribution can be expressed as a product of two factors, one involving only <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and one involving only <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" />:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x,y)&space;=&space;P(x)P(y)&space;\Leftrightarrow&space;P(x|y)&space;=&space;P(x)" title="P(x,y) = P(x)P(y) \Leftrightarrow P(x|y) = P(x)" />
</p>
Two random variables <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> are <b>conditionally independent</b> given a random variable <img src="https://latex.codecogs.com/svg.latex?\inline&space;z" title="z" /> if the conditional probability distribution over <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> factorizes in this way for every value of <img src="https://latex.codecogs.com/svg.latex?\inline&space;z" title="z" />:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x,y|z)=P(x|z)P(y|z)&space;\Leftrightarrow&space;P(x|y,z)&space;=&space;P(x|z)" title="P(x,y|z)=P(x|z)P(y|z) \Leftrightarrow P(x|y,z) = P(x|z)" />
</p>
We can denote the independence with compact notation: <img src="https://latex.codecogs.com/svg.latex?\inline&space;x\perp&space;y" title="x\perp y" /> means that <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> are independent, while <img src="https://latex.codecogs.com/svg.latex?\inline&space;x\perp&space;y&space;|z" title="x\perp y |z" /> means that <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" /> are conditionally independent given <img src="https://latex.codecogs.com/svg.latex?\inline&space;z" title="z" />.

## Expectation, Variance and Covariance

### Expectation

The expectation or expected value of some function <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x)" title="f(x)" /> with respect to a probability distribution <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x)" title="P(x)" /> is the average or mean value that <img src="https://latex.codecogs.com/svg.latex?\inline&space;f" title="f" /> takes on when <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> is drawn from <img src="https://latex.codecogs.com/svg.latex?\inline&space;P" title="P" />.
For discrete variables:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{x\sim&space;P}[f(x)]&space;=&space;\sum_x&space;P(x)f(x)" title="\mathbb{E}_{x\sim P}[f(x)] = \sum_x P(x)f(x)" />
</p>
For continuous variables:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{x\sim&space;p}[f(x)]&space;=&space;\int_x&space;p(x)f(x)dx" title="\mathbb{E}_{x\sim p}[f(x)] = \int_x p(x)f(x)dx" />
</p>
Moreover, expectations are linear, for example,
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{x}[\alpha&space;f(x)&space;&plus;&space;\beta&space;g(x)]&space;=&space;\alpha&space;\mathbb{E}_{x}[f(x)]&space;&plus;&space;\beta&space;\mathbb{E}_{x}[g(x)]" title="\mathbb{E}_{x}[\alpha f(x) + \beta g(x)] = \alpha \mathbb{E}_{x}[f(x)] + \beta \mathbb{E}_{x}[g(x)]" />
</p>
when <img src="https://latex.codecogs.com/svg.latex?\inline&space;\alpha" title="\alpha" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\beta" title="\beta" /> are not dependent on <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />.

### Variance

The variance gives a measure of how much the values of a function of a random variable <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> vary as we sample different values of <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> from its probability distribution:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Var(f(x))&space;=&space;\mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2]" title="Var(f(x)) = \mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2]" />
</p>
When the variance is low, the values of <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x)" title="f(x)" /> cluster near their expected value. The square root of the variance is known as the <b>standard deviation</b>.

### Covariance

The covariance gives some sense of how much to values are linearly related to each other, as wlel as the scale of these variables:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Cov(f(x),g(y))&space;=&space;\mathbb{E}[(f(x)-\mathbb{E}[f(x)])(g(y)-\mathbb{E}[g(y)])]" title="Cov(f(x),g(y)) = \mathbb{E}[(f(x)-\mathbb{E}[f(x)])(g(y)-\mathbb{E}[g(y)])]" />
</p>
High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time. If the sign of the covariance is positive, then both variables tend to take on relatively high values simultaneously. If the sign of the covariance is negative, then one variable tends to take on a relatively high value at the times that the other takes on a relatively low value and vice versa. Other measures such as <b>correlation</b> normalize the contribution of each variable in order to measure only how much the variables are related, rather than also veing affected by the scale of the seperate variables.
The notions of covariance and dependence are related but still distinct. If the covariance is zero, the two variables are linearly dependent. However, there could also be cases where the variables show non-linear dependency and covariance is not zero. However, there cannot be dependency if the covariance is not zero.
The <b>covariance matrix</b> of a random vector <img src="https://latex.codecogs.com/svg.latex?\inline&space;x&space;\in&space;\mathbb{R}^n" title="x \in \mathbb{R}^n" /> is an <img src="https://latex.codecogs.com/svg.latex?\inline&space;n*n" title="n*n" /> matrix, such that
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Cov(x)_{i,j}&space;=&space;\left\{\begin{matrix}&space;Cov(x_i,x_j)&space;&&space;,i\neq&space;j\\&space;Var(x_i)&space;&&space;,i=j&space;\end{matrix}\right." title="Cov(x)_{i,j} = \left\{\begin{matrix} Cov(x_i,x_j) & ,i\neq j\\ Var(x_i) & ,i=j \end{matrix}\right." />
</p>

## Common Probability Distribution

### Bernoulli Distribution

The bernoulli distribution is a distribution over a single binary random variable.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x=1)&space;=&space;\Phi" title="P(x=1) = \Phi" /> <br>
<img src="https://latex.codecogs.com/svg.latex?P(x=0)&space;=&space;1-\Phi" title="P(x=0) = 1-\Phi" /> <br>
<img src="https://latex.codecogs.com/svg.latex?P(x)&space;=&space;\Phi^x(1-\Phi)^{1-x}" title="P(x) = \Phi^x(1-\Phi)^{1-x}" /> <br>
<img src="https://latex.codecogs.com/svg.latex?E_x[x]&space;=&space;\Phi" title="E_x[x] = \Phi" /> <br>
<img src="https://latex.codecogs.com/svg.latex?Var_x(x)&space;=&space;\Phi(1-\Phi)" title="Var_x(x) = \Phi(1-\Phi)" /> <br>
</p>

### Multinomial Distribution

THe mulitnomial distribution is a distribution over a single discrete variable with <img src="https://latex.codecogs.com/svg.latex?\inline&space;k" title="k" /> different states, where <img src="https://latex.codecogs.com/svg.latex?\inline&space;k" title="k" /> is finite. Multinoulli distributions are often used to refer to distributions over categories of objects, so we do not usually assume that state 1 has numerical value 1, etc. For this reason, we do not usually need to compute the expectation or variance of multinoulli-distributed random variables.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x)&space;=&space;\Phi^x(1-\Phi)^{n-x}" title="P(x) = \Phi^x(1-\Phi)^{n-x}" />
</p>

### Gaussian Distribution

The most commonly used distribution over real numbers is the normal distribution, also known as the gaussian distribution.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?N(x;\mu&space;,\sigma&space;^2)&space;=&space;\frac{1}{\sqrt{2\Pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}" title="N(x;\mu ,\sigma ^2) = \frac{1}{\sqrt{2\Pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}" /> <br>
<img src="https://latex.codecogs.com/svg.latex?E_x[x]&space;=&space;\mu" title="E_x[x] = \mu" /> <br>
<img src="https://latex.codecogs.com/svg.latex?Var_x(x)&space;=&space;\sigma^2" title="Var_x(x) = \sigma^2" /> <br>
</p>
When we need to frequently evaluate the PDF with different parameter values, a more efficient way parametrizing the distribution is to use a parameter <img src="https://latex.codecogs.com/svg.latex?\inline&space;\beta" title="\beta" /> to control the <b>precision</b> or inverse variance of the distribution, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\beta&space;=&space;\sigma^{-2}" title="\beta = \sigma^{-2}" />

In the absence of prior knowledge about what form a distribution over the real numbers should take, the normal distribution is a good default choice for two major reasons:
* Many distributions we wish to model are truly close to being normal distributions. The <b>central limit theorem</b> shows that the sum of many independent random variables is approximately normally distributed. This means that in practice, many complicated systems can be modeled successfully as normally distributed noise, even if the system can be decomposed into parts with more structured behavior.
* Out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty over the
real numbers. We can thus think of the normal distribution as being the one that inserts the least amount of prior knowledge into a model.
The normal distribution generalizes to <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbb{R}^{n}" title="\mathbb{R}^{n}" />, in which case it is known as the <b>multivariate normal distribution</b>. It may be parametrized with a positive definite symmetric matrix <img src="https://latex.codecogs.com/svg.latex?\inline&space;\sum" title="\sum" />:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?N(x;\mu&space;,\sum&space;)&space;=&space;\frac{1}{(2\Pi)^{n/2}\left&space;|&space;\sum&space;\right&space;|^{0.5}}&space;e^{-\frac{1}{2}(x-\mu)^T\sum&space;^{-1}(x-\mu)}" title="N(x;\mu ,\sum ) = \frac{1}{(2\Pi)^{n/2}\left | \sum \right |^{0.5}} e^{-\frac{1}{2}(x-\mu)^T\sum ^{-1}(x-\mu)}" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mu" title="\mu" /> is mean, and <img src="https://latex.codecogs.com/svg.latex?\inline&space;\sum" title="\sum" /> is covariance matrix.
We often fix the covariance matrix to be a diagonal matrix. An even simpler version is the <b>isotropic</b> Gaussian distribution, whose covariance matrix is a scalar times the identity matrix.

### Exponential and Laplace Distributions

In the context of deep learning, we often want to have a probability distribution with a sharp point at <img src="https://latex.codecogs.com/svg.latex?\inline&space;x=0" title="x=0" />. To accomplish this, we can use the exponential distribution:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x;\lambda)&space;=&space;\lambda&space;e^{-\lambda&space;x}" title="P(x;\lambda) = \lambda e^{-\lambda x}" />
</p>
A closely related probability distribution that allows us to place a sharp peak of probability mass at an arbitrary point <img src="https://latex.codecogs.com/svg.latex?\inline&space;\mu" title="\mu" /> is the Laplace distribution:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Laplace(x;\mu,,\gamma&space;)&space;=&space;\frac{1}{2\gamma&space;}e^{-\frac{\left&space;|&space;x-\mu&space;\right&space;|}{\gamma&space;}}" title="Laplace(x;\mu,,\gamma ) = \frac{1}{2\gamma }e^{-\frac{\left | x-\mu \right |}{\gamma }}" />
</p>

### The Dirac Distribution and Empirical Distribution

In some cases, we wish to specify that all of the mass in a probability mass in a probability distribution clusters around a single point. This can be accomplished by defining a PDF using the Dirac delta function, <img src="https://latex.codecogs.com/svg.latex?\inline&space;\delta&space;(x)" title="\delta (x)" />. The Dirac delta function is defined such that is is zero-valued everywhere except 0, yet integrates to 1. Dirac function is not an ordinary fucntion and falls under the category of <b>generalized function</b>.

A common use of the Dirac delta distribution is as a component of an empirical distribution,
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\hat{p}(x)&space;=&space;\frac{1}{m}\sum&space;_{i=1}^{m}\delta&space;(x-x^{(i)})" title="\hat{p}(x) = \frac{1}{m}\sum _{i=1}^{m}\delta (x-x^{(i)})" />
</p>
which puts a probability mass <img src="https://latex.codecogs.com/svg.latex?\inline&space;\frac{1}{m}" title="\frac{1}{m}" /> on each of the <img src="https://latex.codecogs.com/svg.latex?\inline&space;m" title="m" /> points <img src="https://latex.codecogs.com/svg.latex?\inline&space;x^{(1)},...,x^{(m)}" title="x^{(1)},...,x^{(m)}" /> forming a given dataset or collection of samples.

### Mixtures of Distributions

A mixture distribution is made up of several component distributions. On each trial, the choice of whihc component distribution generates the sample is determined by sampling a component identity from a multinomial distribution:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x)&space;=&space;\sum&space;_i&space;P(c=i)P(x|c=i)" title="P(x) = \sum _i P(c=i)P(x|c=i)" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(c)" title="P(c)" /> is the multinomial distribution over component identities.

A <b>latent variable</b> is a random variable that we cannot observe directly. The component identity variable <img src="https://latex.codecogs.com/svg.latex?\inline&space;c" title="c" /> of the mixture model provides an example. Latent variables may be related to <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> through the joint distribution, in this case, <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x,c)&space;=&space;P(x|c)P(c)" title="P(x,c) = P(x|c)P(c)" />. The distribution <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(c)" title="P(c)" /> over the latent variable and the distribution <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x|c)" title="P(x|c)" /> relating the latent variables to the visible variables determines the shape of the distribution <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x)" title="P(x)" /> even though it is possible to describe <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x)" title="P(x)" /> without reference to the latent variable.

A very powerful and common type of mixture model is the <b>Gaussian mixture</b> model, in which the components <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x|c=i)" title="P(x|c=i)" /> are Gaussians. We would also need to specify the <b>prior probability</b> <img src="https://latex.codecogs.com/svg.latex?\inline&space;\alpha&space;_i&space;=&space;P(c=i)" title="\alpha _i = P(c=i)" /> given to each component <img src="https://latex.codecogs.com/svg.latex?\inline&space;i" title="i" />. The word "prior" is used since it expresses the model's beliefs about <img src="https://latex.codecogs.com/svg.latex?\inline&space;c" title="c" /> before it observes <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />.
By comparison, <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(c|x)" title="P(c|x)" /> is a <b>posterior probability</b>, because it is computed after observation of <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />. A Gaussian mixture model is a <b>universal approximator</b> of densities, in the sense that any smooth density can be approximated with any specific, non-zero amount of error by a Gaussian mixture model with enough components.

## Useful Properties of Common Functions
* <b>Logistic Sigmoid:</b>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sigma&space;(x)&space;=&space;\frac{1}{1&plus;\exp(-x)}" title="\sigma (x) = \frac{1}{1+\exp(-x)}" />
</p>
The sigmoid function saturates when its argument is very positive or very negative, meaning that the function becomes very flat and insensitive to small changes in its input.
<p align="center">
<b>The logistic sigmoid function.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/3/Figure-1.png?raw=true" alt="Figure 1"/>
</p>
* <b>Softplus function:</b>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\zeta&space;(x)&space;=&space;\log(1&plus;\exp(x))" title="\zeta (x) = \log(1+\exp(x))" />
</p>
The name of softplus function comes from the fact that it is a smoothed or "softened" version of ReLU.
<p align="center">
<b>The softplus function.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/3/Figure-2.png?raw=true" alt="Figure 1"/>
</p>
The following properties are all useful enough to memorize:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\sigma&space;(x)&space;=&space;\frac{\exp(x)}{\exp(x)&plus;\exp(0)}" title="\sigma (x) = \frac{\exp(x)}{\exp(x)+\exp(0)}" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\frac{\mathrm{d}}{\mathrm{d}&space;x}&space;\sigma(x)&space;=&space;\sigma(x)(1-\sigma(x))" title="\frac{\mathrm{d}}{\mathrm{d} x} \sigma(x) = \sigma(x)(1-\sigma(x))" /> <br>
<img src="https://latex.codecogs.com/svg.latex?1-\sigma(x)&space;=&space;\sigma(-x)" title="1-\sigma(x) = \sigma(-x)" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}\zeta&space;(x)&space;=&space;\sigma(x)" title="\frac{\mathrm{d} }{\mathrm{d} x}\zeta (x) = \sigma(x)" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\sigma^{-1}(x)&space;=&space;\log(\frac{x}{1-x})" title="\sigma^{-1}(x) = \log(\frac{x}{1-x})" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\zeta&space;^{-1}(x)&space;=&space;\log(\exp(x)-1)" title="\zeta ^{-1}(x) = \log(\exp(x)-1)" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\zeta(x)&space;=&space;\int_{-\infty}^{x}&space;\sigma(y)dy" title="\zeta(x) = \int_{-\infty}^{x} \sigma(y)dy" /> <br>
<img src="https://latex.codecogs.com/svg.latex?\zeta(x)&space;-&space;\zeta(-x)&space;=&space;x" title="\zeta(x) - \zeta(-x) = x" /> <br>
</p>

The function <img src="https://latex.codecogs.com/svg.latex?\inline&space;\sigma^{-1}(x)" title="\sigma^{-1}(x)" /> is called the <b>logit</b> in statistics.

## Bayes' Rule
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(x|y)&space;=&space;\frac{P(x)P(y|x)}{P(y)}" title="P(x|y) = \frac{P(x)P(y|x)}{P(y)}" />
</p>
Although <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(y)" title="P(y)" /> appears to be in the formula, it is usually feasible to compute <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(y)&space;=&space;\sum&space;_x&space;P(y|x)P(x)" title="P(y) = \sum _x P(y|x)P(x)" /> so we do not need to begin with knowledge of <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(y)" title="P(y)" />.

## Technical Details of Continuous Variables

<b>Measure zero</b> is a convept which is used to define a very small set of points. Another useful term is <b>almost everywhere</b> where a property holds throughout all of space except for on a set of measure zero.
Another technical detail of continuous variables relates to handling continuous random variables that are deterministic functions of one another. Suppose we have two random variables, <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;y" title="y" />, such that <img src="https://latex.codecogs.com/svg.latex?\inline&space;y&space;=&space;g(x)" title="y = g(x)" />. One might expect that <img src="https://latex.codecogs.com/svg.latex?\inline&space;p_y(y)&space;=&space;p_x(g^{-1}(y))" title="p_y(y) = p_x(g^{-1}(y))" /> which is not the case.

For higher dimensions, the derivative generalizes to the determinant of the <b>Jacobian matrix</b>, the matrix with <img src="https://latex.codecogs.com/svg.latex?\inline&space;J_{i,j}=&space;\frac{\partial&space;x_i}{\partial&space;y_j}" title="J_{i,j}= \frac{\partial x_i}{\partial y_j}" />.

## Information Theory

The basic intuition behind information theory is that learning that an unlikely event has ovvured is more informative than learning that a likely event has occured.
A common function used is the <b>self-information</b> of an event <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" /> to be
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?I(x)&space;=&space;-\log&space;P(x)" title="I(x) = -\log P(x)" />
</p>
Here, we mean to use natural logarithm, with base <img src="https://latex.codecogs.com/svg.latex?\inline&space;e" title="e" /> and wriiten in units of <b>nats</b>. Others use a base-2 logarithms and units called <b>bits</b> or <b>shannons</b>. However, the self-information deals with only a single outcome.

We can quantify the amount of uncerainty in an entire probability distribution using the <b>Shannon entropy</b>:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?H(P)&space;=&space;\mathbb{E}_{x\sim&space;P}[I(x)]&space;=&space;-\mathbb{E}_{x\sim&space;P}[\log&space;P(x)]" title="H(P) = \mathbb{E}_{x\sim P}[I(x)] = -\mathbb{E}_{x\sim P}[\log P(x)]" />
</p>
When x is continuous, the Shannon entropy is known as the <b>differential entropy</b>.
If we have two seperate probability distributions <img src="https://latex.codecogs.com/svg.latex?\inline&space;P(x)" title="P(x)" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q(x)" title="Q(x)" /> over the same random variable <img src="https://latex.codecogs.com/svg.latex?\inline&space;x" title="x" />, we can measure how different these two distributions are using the <b>Kullback-Leibler (KL) divergence</b>:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D_KL(P||Q)&space;=&space;\mathbb{E}_{x\sim&space;P}\begin{bmatrix}&space;\log&space;\frac{P(x)}{Q(x)}&space;\end{bmatrix}" title="D_KL(P||Q) = \mathbb{E}_{x\sim P}\begin{bmatrix} \log \frac{P(x)}{Q(x)} \end{bmatrix}" />
</p>
In the case of discrete variables, it is the extra amount of information needed to send a message containing symbols drawn from probability distribution <img src="https://latex.codecogs.com/svg.latex?\inline&space;\P" title="P" />, when we use a code that was designed to minimize the length of messages drawn from probability distribution <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q" title="Q" />.

The KL divergence has many useful properties, most notably that it is nonnegative.The KL divergence is 0 if and only if <img src="https://latex.codecogs.com/svg.latex?\inline&space;\P" title="P" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q" title="Q" /> are the same distribution in the case of discrete variables, or equal “almost everywhere” in the case of continuous variables. Because the KL divergence is non-negative and measures the difference between two distributions, it is often conceptualized as measuring some sort of distance between these distributions. However, it is not a true distance measure because it is not symmetric: <img src="https://latex.codecogs.com/svg.latex?\inline&space;D_{KL}(P||Q)&space;\neq&space;D_{KL}(Q||P)" title="D_{KL}(P||Q) \neq D_{KL}(Q||P)" /> for some <img src="https://latex.codecogs.com/svg.latex?\inline&space;\P" title="P" /> and <img src="https://latex.codecogs.com/svg.latex?\inline&space;Q" title="Q" />. This asymmetry means that there are important consequences to the choice of whether to use <img src="https://latex.codecogs.com/svg.latex?\inline&space;D_{KL}(P||Q)" title="D_{KL}(P||Q)" /> or <img src="https://latex.codecogs.com/svg.latex?\inline&space;D_{KL}(Q||P)" title="D_{KL}(Q||P)" />.

<p align="center">
<b>KL divergence is assymetric.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/3/Figure-3.png?raw=true" alt="Figure 1"/>
</p>

A quantity that is closely related to the KL divergence is the <b>cross-entropy</b> <img src="https://latex.codecogs.com/svg.latex?\inline&space;H(P,Q)&space;=&space;H(P)&space;&plus;&space;D_{KL}(P||Q)" title="H(P,Q) = H(P) + D_{KL}(P||Q)" />

## Structured Probabilistic Models

Machine learning algorithms often involve probability distributions over a very large number of random variables. Using a single function to describe the entire joint probability distribution can be very inefficient (both computationally and statistically). Instead of using a single function to represent a probability distribution into many factors that we multiply together. These factorizations can be describes using graphs and call them <b>structured probabilistic model</b> or <b>graphical model</b>.
There are two main kinds of structured probabilistic models: directed and undirected

### Directed Models

Graphs with directed edges, and they represent factorizations into conditional probability distributions.
<p align="center">
<b>Directed model.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/3/Figure-4.png?raw=true" alt="Figure 1"/>
</p>
Here, <img src="https://latex.codecogs.com/svg.latex?\inline&space;p(a,b,c,d,e)&space;=&space;p(a)p(b|a)p(c|a,b)p(d|b)p(e|c)" title="p(a,b,c,d,e) = p(a)p(b|a)p(c|a,b)p(d|b)p(e|c)" />

### Undirected Models

Graphs with undirected edges, and represent factorizations into a set of functions. Usually, these functions are not probability distributions of any kind. Any set of nodes that are all connected to each other in <img src="https://latex.codecogs.com/svg.latex?\inline&space;G" title="G" /> is called a clique. Each clique <img src="https://latex.codecogs.com/svg.latex?\inline&space;C^{(i)}" title="C^{(i)}" /> in an undirected model is associated with a factor <img src="https://latex.codecogs.com/svg.latex?\inline&space;\Phi&space;^{(i)}(C^{(i)})" title="\Phi ^{(i)}(C^{(i)})" />. The output of each factor must be non-negative, but there is no constraint that the factor must sum or integrate to 1 like a probability distribution.
<p align="center">
<b>Undirected model.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/3/Figure-5.png?raw=true" alt="Figure 1"/>
</p>
Here, <img src="https://latex.codecogs.com/svg.latex?\inline&space;p(a,b,c,d,e)&space;=&space;\frac{1}{Z}\Phi&space;^{(1)}(a,b,c)\Phi&space;^{(2)}(b,d)\Phi&space;^{(3)}(c,e)" title="p(a,b,c,d,e) = \frac{1}{Z}\Phi ^{(1)}(a,b,c)\Phi ^{(2)}(b,d)\Phi ^{(3)}(c,e)" />.
