---
layout: post
title: Information Theory Basics
published: true
book_title: Random Math Concepts
---

An overview of the concept  “[Entropy](https://en.wikipedia.org/wiki/Entropy)”, “[Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)”, “[KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)”, and “[F-Divergence](https://en.wikipedia.org/wiki/F-divergence)”.
<!--break-->
In this work, we discuss important concepts to information theory, often useful for intuition, and proofs.


### Entropy

The entropy of a probability distribution (<img src="https://latex.codecogs.com/svg.latex?x" title="x" />) is denoted with the symbol <img src="https://latex.codecogs.com/svg.latex?\mathcal{H}(x)" title="\mathcal{H}(x)" />, such that <img src="https://latex.codecogs.com/svg.latex?\mathcal{H}(x) = -\sum p \log p" title="\mathcal{H}(x) = -\sum p \log p" />. Furthermore, we can prove that this entropy is bounded above using Jensen's Inequality as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}
H(X) &= \mathbb{E}\left(\log_2 \frac{1}{\mathbb{P}_X(X)}\right)\\
&\le \log_2 \mathbb{E} \left(\frac{1}{\mathbb{P}_X(X)}\right)\hspace{2cm}(\textrm{Jensen's inequality})\\
& = \log_2 |\mathcal{X}|.
\end{align*}" title="\begin{align*}
H(X) &= \mathbb{E}\left(\log_2 \frac{1}{\mathbb{P}_X(X)}\right)\\
&\le \log_2 \mathbb{E} \left(\frac{1}{\mathbb{P}_X(X)}\right)\hspace{2cm}(\textrm{Jensen's inequality})\\
& = \log_2 |\mathcal{X}|.
\end{align*}" />
</p>


### Mutual Information

In probability theory and information theory, the mutual information (MI) of two random variables is a measure of the mutual dependence between the two variables. More specifically, it quantifies the "amount of information" (in units such as shannons (bits), nats or hartleys) obtained about one random variable by observing the other random variable. The concept of mutual information is intimately linked to that of entropy of a random variable, a fundamental notion in information theory that quantifies the expected "amount of information" held in a random variable. The Mutual information is basically the KL divergence between the joint probability function <img src="https://latex.codecogs.com/svg.latex?P(X,Y)" title="P(X,Y)" />, and the marginal probability distributions <img src="https://latex.codecogs.com/svg.latex?P(X)" title="P(X)" />, and <img src="https://latex.codecogs.com/svg.latex?P(Y)" title="P(Y)" />.

### KL Divergence

In mathematical statistics, Kullback-Leibler divergence, denoted by <img src="https://latex.codecogs.com/svg.latex?D_{\text{KL}}(P||Q)" title="D_{\text{KL}}(P||Q)" />, is a type of statistical distance: a measure of how one probability distribution <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> is different from the second reference probability distribution <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" />. A simple interpretation of KL divergence of <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> from <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" /> is the expected excess surprise from using <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" /> as a model when the actual distribution is <img src="https://latex.codecogs.com/svg.latex?P" title="P" />. The exact definition of KL divergence is explained below.

<p align="center">
<img src="D_{\text{KL}}(P||Q) = \sum _{x\in\mathcal{X}} P(x)\log \begin{pmatrix}
\frac{P(x)}{Q(x)}
\end{pmatrix}" title="D_{\text{KL}}(P||Q) = \sum _{x\in\mathcal{X}} P(x)\log \begin{pmatrix}
\frac{P(x)}{Q(x)}
\end{pmatrix}" />
</p>


### F Divergence

In probability theory, an *f-divergence* is a function <img src="https://latex.codecogs.com/svg.latex?D_f(P||Q)" title="D_f(P||Q)" /> that measures the difference between two probability distributions <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> and <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" />. Many common divergences, such as KL-divergence, Hellinger distance, and total variational distance, are special cases of f-divergence.

<p align="center">
<img src="D_{f}(P||Q) = \int _{\Omega } f \begin{pmatrix}
\frac{dP}{dQ}
\end{pmatrix}dQ" title="D_{f}(P||Q) = \int _{\Omega } f \begin{pmatrix}
\frac{dP}{dQ}
\end{pmatrix}dQ" />
</p>

Here, *f* is called the generator of <img src="https://latex.codecogs.com/svg.latex?D_{f}" title="D_{f}" />. In concrete applications, there is usally a reference distribution <img src="https://latex.codecogs.com/svg.latex?\mu" title="\mu" /> on <img src="https://latex.codecogs.com/svg.latex?\Omega" title="\Omega" />, such that <img src="https://latex.codecogs.com/svg.latex?P, Q \ll \mu" title="P, Q \ll \mu" />, then we can use Radon-Nikodym theorem to take their probability densities *p* and *q* giving:

<p align="center">
<img src="D_{f}(P||Q) = \int _{\Omega } f \begin{pmatrix}
\frac{p(x)}{q(x)} 
\end{pmatrix}q(x) d\mu(x)" title="D_{f}(P||Q) = \int _{\Omega } f \begin{pmatrix}
\frac{p(x)}{q(x)} 
\end{pmatrix}q(x) d\mu(x)" />
</p>

When the generator function is x\log x, we get the KL-Divegence.