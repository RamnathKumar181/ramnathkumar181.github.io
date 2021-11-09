---
layout: post
title: Generative Adversarial Networks
published: true
---

An overview of the paper “[Generative Adversarial Networks](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)”.
<!--break-->
The author presents a new framework for estimating generative models via an adversarial process. All images and tables in this post are from their paper.

## Introduction

The GAN usually consists of two networks - a Generator(<img src="https://latex.codecogs.com/svg.latex?G" title="G" />) and a Discriminator(<img src="https://latex.codecogs.com/svg.latex?D" title="D" />). The idea is that the two networks play a minimax two-player game where, the goal of <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> is to create data capable of fooling D, and the goal of <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> is to not get fooled by <img src="https://latex.codecogs.com/svg.latex?G" title="G" />. If allowed to run for millions of iterations, the entire network reaches a very good understanding of the distribution of the data, and can replicate it.  

## Adversarial Nets

The idea is that we try to train <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> to maximize the probability of assigning the correct label to both training examples and generated examples. Furthermore, we train <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> to minimize <img src="https://latex.codecogs.com/svg.latex?\log&space;(1-D(G(z)))" title="\log (1-D(G(z)))" />. The two players play a minimax game with the value function <img src="https://latex.codecogs.com/svg.latex?V(G,D)" title="V(G,D)" />.

The authors also suggest that training the two networks in an iterative way leads to overfitting on small datasets. Instead, we train <img src="https://latex.codecogs.com/svg.latex?k" title="k" /> steps of <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> and one step of optimizing <img src="https://latex.codecogs.com/svg.latex?G" title="G" />. Hence, <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> will remain around optimal as long as <img src="https://latex.codecogs.com/svg.latex?G" title="G" /> changes slowly enough.

There are also a few proofs to support their claims. A simple summary is given below.

### Theorem 1

For fixed <img src="https://latex.codecogs.com/svg.latex?G" title="G" />, the optimal <img src="https://latex.codecogs.com/svg.latex?D" title="D" /> is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D^*&space;=&space;\frac{p_{\textup{data}}(x)}{p_{\textup{data}}(x)&space;&plus;&space;p_g(x)}" title="D^* = \frac{p_{\textup{data}}(x)}{p_{\textup{data}}(x) + p_g(x)}" />
</p>

While training <img src="https://latex.codecogs.com/svg.latex?D" title="D" />, our goal is to maximize <img src="https://latex.codecogs.com/svg.latex?V(G,D)" title="V(G,D)" />. Just simplifying the "expected" terms to integral allows us to reach a very simple equation of <img src="https://latex.codecogs.com/svg.latex?a\log(y)&space;&plus;&space;b\log(1-y)" title="a\log(y) + b\log(1-y)" />. The value of <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> which maximizes this equation is <img src="https://latex.codecogs.com/svg.latex?\frac{a}{a&plus;b}" title="\frac{a}{a+b}" /> as long as <img src="https://latex.codecogs.com/svg.latex?a" title="a" /> and <img src="https://latex.codecogs.com/svg.latex?b" title="b" /> are both not equal to 0. Hence, we prove the above statement.

### Theorem 2

The global minimum of the virtual training criterion <img src="https://latex.codecogs.com/svg.latex?C(G)" title="C(G)" /> is achieved if and only if <img src="https://latex.codecogs.com/svg.latex?p_g&space;=&space;p_{\textup{data}}" title="p_g = p_{\textup{data}}" /> . At that point, <img src="https://latex.codecogs.com/svg.latex?C(G)" title="C(G)" /> achieves the value <img src="https://latex.codecogs.com/svg.latex?-\log(4)" title="-\log(4)" />.

With a simple addition and subtraction of <img src="https://latex.codecogs.com/svg.latex?\log(4)" title="\log(4)" />, we can explain the same equation in terms of KL divergence. This can further be simplified to Jensen Shannon Divergence. The final equation we have is now <img src="https://latex.codecogs.com/svg.latex?-\log(4)&space;&plus;&space;2.\textup{JSD}(p_{\textup{data}}||p_g)" title="-\log(4) + 2.\textup{JSD}(p_{\textup{data}}||p_g)" />. Since, JSD is a function in the range of 0-1. The minimum of this equation will occur when JSD gives a output of 0. This only happens when <img src="https://latex.codecogs.com/svg.latex?p_g&space;=&space;p_{\textup{data}}" title="p_g = p_{\textup{data}}" />. Hence, we have proved the above statement.

## Proving Convergence of the Algorithm

To the best of my knowledge, there is no proof which claims that neural networks converge at global optimum. However, its excellent performance in practice suggests that they are a reasonable model despite their lack of theoretical guarantees.
