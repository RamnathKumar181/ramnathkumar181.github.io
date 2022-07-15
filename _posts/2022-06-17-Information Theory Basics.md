---
layout: post
title: Information Theory Basics
published: true
book_title: Random Math Concepts
---

An overview of the concept  “[Entropy](https://en.wikipedia.org/wiki/Entropy)”, “[Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)”, KL Divergence, and F-Divergence.
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
