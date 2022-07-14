---
layout: post
title: Jensen Inequality
published: true
book_title: Random Math Concepts
---

An overview of the concept “[Jensen Inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)”.
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
