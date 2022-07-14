---
layout: post
title: Riemann and Labesgue Integral
published: true
book_title: Random Math Concepts
---

An overview of the concept  “[Riemann and Lebesgue Integration](https://en.wikipedia.org/wiki/Lebesgue_integration)”.
<!--break-->
In this work, we discuss important concepts to theoretical machine learning, often used in proofs/derivation of energy functions, convergence, etc.


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
