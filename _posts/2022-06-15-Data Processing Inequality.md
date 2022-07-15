---
layout: post
title: Data Processing Inequality
published: false
book_title: Random Math Concepts
---

An overview of the concept  “[Data Processing Inequality](https://en.wikipedia.org/wiki/Data_processing_inequality)”.
<!--break-->
In this work, we discuss important concepts to information theory, often useful for intuition, and proofs. The data processing inequality is an information theoretic concept which states that the information content of a signal cannot be increased via a local physical operation. This can be expressed concisely as **'post-processing cannot increase information'**.


### Definition



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
