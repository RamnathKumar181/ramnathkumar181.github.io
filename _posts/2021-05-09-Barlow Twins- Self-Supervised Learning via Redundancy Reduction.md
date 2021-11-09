---
layout: post
title:  Barlow Twins- Self-Supervised Learning via Redundancy Reduction
published: true
---

An overview of the paper “[Barlow Twins- Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)”.
<!--break-->
Self-supervised learning is rapidly closing the gap with supervised learning methods on large vision benchmarks. A successful approach is to learn representations which are invariant to distortions of the input sample. However, a recurring issue with this approach is the existence of trivial constant solutions. Most current methods avoid these trivial solutions by careful implementation details. In this paper, the authors propose an objective function that naturally avoids such collapse by measuring the cross-correlation matrix between the outputs of two identical networks fed with distorted versions of a sample, and making them as close to identity as possible. This causes the representation vectors of distorted versions of a sample to be similar, while minimizing the redundancy between the components of these vectors. All images and tables in this post are from their paper.
Barlow in his paper, hypothesized that the goal of sensory processing is to recode highly redundant sensory inputs into a factorial code (a code with statistically independent components).

<p align="center">
<b>Brief Overview of Methodology</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/26/Figure-1.png?raw=true" alt="Figure 1"/>
</p>


## Methodology

Like other methods for Semi-Supervised Learning, Barlow Twins operates on a joint embedding of distorted images. More specifically, it produces two distorted views for all images of a batch <img src="https://latex.codecogs.com/svg.latex?X" title="X" /> sampled from a dataset. The distorted views are obtained via a distribution of data augmentations <img src="https://latex.codecogs.com/svg.latex?\tau" title="\tau" />. The two batches of distorted views <img src="https://latex.codecogs.com/svg.latex?Y^{A}" title="Y^{A}" /> and  <img src="https://latex.codecogs.com/svg.latex?Y^{B}" title="Y^{B}" /> are then fed to a function <img src="https://latex.codecogs.com/svg.latex?f_{\theta}" title="f_{\theta}" />, typically a deep network with trainable parameters <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />, producing batches of representations <img src="https://latex.codecogs.com/svg.latex?Z^{A}" title="Z^{A}" /> and <img src="https://latex.codecogs.com/svg.latex?Z^{B}" title="Z^{B}" /> respectively. To simplify notations, <img src="https://latex.codecogs.com/svg.latex?Z^{A}" title="Z^{A}" /> and <img src="https://latex.codecogs.com/svg.latex?Z^{B}" title="Z^{B}" /> are assumed to be mean-centered along the batch dimension, such that each unit has mean output 0 over the batch.

Barlow twins uses an innovative loss function <img src="https://latex.codecogs.com/svg.latex?L_{BT}" title="L_{BT}" />:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{BT}&space;=&space;\sum&space;_{i}(1-C_{ii})^2&space;&plus;&space;\lambda&space;\sum_{i}&space;\sum_{j\neq&space;i}C_{ij}^2" title="L_{BT} = \sum _{i}(1-C_{ii})^2 + \lambda \sum_{i} \sum_{j\neq i}C_{ij}^2" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?\lambda" title="\lambda" /> is a positive constant trading off the importance of the first and second terms of the loss, and where <img src="https://latex.codecogs.com/svg.latex?C" title="C" /> is the cross-correlation matrix computed between the outputs of the two identical networks along the batch dimension:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?C_{ij}&space;=&space;\frac{\sum_b&space;z_{b,i}^Az_{b,j}^B}{\sqrt{\sum_b&space;(z_{b,i}^A)^2}\sqrt{\sum_b&space;(z_{b,j}^B)^2}}" title="C_{ij} = \frac{\sum_b z_{b,i}^Az_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2}\sqrt{\sum_b (z_{b,j}^B)^2}}" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?b" title="b" /> indexes batch samples and <img src="https://latex.codecogs.com/svg.latex?i" title="i" />, <img src="https://latex.codecogs.com/svg.latex?j" title="j" /> index the vector dimension of the networks' outputs.

Intuitively, the <b>invariance term</b> of the objective, by trying to equate the diagonal elements of the cross-correlation matrix to 1, makes the representation invariant to distortions applied. The <b>redundancy reduction term</b>, by trying to equate the off-diagonal elements of the cross-correlation matrix to 0, decorrelates the different vector components of the representation. This decorrelation reduces the redundancy between output units so that the output units contatin non-redundant information about the sample.
