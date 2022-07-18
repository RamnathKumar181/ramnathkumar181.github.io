---
layout: post
title: Smooth Maximum
published: true
book_title: Random Math Concepts
---

An overview of the concept  “[Smooth Maximum](https://en.wikipedia.org/wiki/Smooth_maximum)”.
<!--break-->
In this work, we discuss important concepts of smooth max, a concept useful to make the maximum operator differentiable in deep learning.


### Smooth Maximum

For large positive values of parameter <img src="https://latex.codecogs.com/svg.latex?\alpha > 0" title="\alpha > 0" />, the following formulation is a smooth, differentiable approximation of the maximum function. For negative values of the parameter that are large in absolute value, it approximates the minimum. 



<p align="center">
<img src="https://latex.codecogs.com/svg.latex?S_{\alpha}(x_1,...,x_n) = \frac{\sum _{i=1}^n x_i e^{\alpha x_i}}{\sum _{i=1}^n e^{\alpha x_i}}" title="S_{\alpha}(x_1,...,x_n) = \frac{\sum _{i=1}^n x_i e^{\alpha x_i}}{\sum _{i=1}^n e^{\alpha x_i}}" />
</p>

Thus, <img src="https://latex.codecogs.com/svg.latex?S_{\alpha}" title="S_{\alpha}" /> has the following useful properties:
* <img src="https://latex.codecogs.com/svg.latex?S_{\alpha} \rightarrow \max" title="S_{\alpha} \rightarrow \max" /> as <img src="https://latex.codecogs.com/svg.latex?\alpha \rightarrow \infty" title="\alpha \rightarrow \infty" />.
* <img src="https://latex.codecogs.com/svg.latex?S_{\alpha} \rightarrow \text{mean}" title="S_{\alpha} \rightarrow \text{mean}" /> as <img src="https://latex.codecogs.com/svg.latex?\alpha \rightarrow 0" title="\alpha \rightarrow 0" />.
* <img src="https://latex.codecogs.com/svg.latex?S_{\alpha} \rightarrow \max" title="S_{\alpha} \rightarrow \min" /> as <img src="https://latex.codecogs.com/svg.latex?\alpha \rightarrow -\infty" title="\alpha \rightarrow -\infty" />.

### LogSumExp

Another option for a smooth maximum function is the LogSumExp.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\text{LSE}_{\alpha}(x_1,...,x_n) = (\frac{1}{\alpha})\log(\exp(\alpha x_1)+...+\exp(\alpha x_n))" title="\text{LSE}_{\alpha}(x_1,...,x_n) = (\frac{1}{\alpha})\log(\exp(\alpha x_1)+...+\exp(\alpha x_n))" />
</p>

The formulation shares derivation from entropic regularization process in reinforcement learning. 

### p-Norm

Another smooth maximum is the p-norm. As <img src="https://latex.codecogs.com/svg.latex?p \rightarrow \infty" title="p \rightarrow \infty" />, the p-Norm tends to the maximum funciton.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{Vmatrix}
(x_1,x_2,...,x_n)
\end{Vmatrix}_p = (|x_1|^p+...+|x_n|^p)^{1/p}" title="\begin{Vmatrix}
(x_1,x_2,...,x_n)
\end{Vmatrix}_p = (|x_1|^p+...+|x_n|^p)^{1/p}" />
</p>

An intrinsic advantage of the p-norm is that it is a norm. As such, it is "scale invariant" (homogeneous).