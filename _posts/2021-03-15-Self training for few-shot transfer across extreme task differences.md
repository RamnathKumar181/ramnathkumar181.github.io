---
layout: post
title: Self training for few-shot transfer across extreme task differences
published: true
---

An overview of the paper “[Self training for few-shot transfer across extreme task differences](https://arxiv.org/pdf/2010.07734.pdf)”.
<!--break-->
The authors propose a new approach to overcome large domain gap between base dataset and target dataset. All images and tables in this post are from their paper.

Usually, all few-shot leanring techniques must be pre-trained on a large, labeled "base dataset". In problem domains where such large labeled datasets are not available for pre-training (e.g., X-ray images), one must resort to pre-training in a different "source" problem domain (e.g., ImageNet), which can be very different from the desired target task.

In this paper, they propose a methodology where the model can be divided into two sections. The first model uses the labeled base dataset and the unlabeled target domain data for representation learning. This step is similar to pre-training. The second part of the model is used for evaluation and classifying samples.

## Problem Setup

The goal is to build learners for novel domains that can be quickly trained to recognize new classes when presented with very few labeled data points. Formally, the target domain is defined by as set of data points <img src="https://latex.codecogs.com/svg.latex?X_{N}" title="X_{N}" />, an unknown set of classes <img src="https://latex.codecogs.com/svg.latex?y_{N}" title="y_{N}" />, and a distribution <img src="https://latex.codecogs.com/svg.latex?D_{N}" title="D_{N}" /> over <img src="https://latex.codecogs.com/svg.latex?X_{N}&space;*&space;Y_{N}" title="X_{N} * Y_{N}" />. A "few-shot learning task" will consist of a set of classes <img src="https://latex.codecogs.com/svg.latex?Y&space;\subset&space;Y_{n}" title="Y \subset Y_{n}" /> , a very small training set ("support")
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?S&space;=&space;\begin{Bmatrix}&space;(x_i,y_i)&space;\end{Bmatrix}_{i=1}^n&space;\sim&space;D_N^n,&space;y_i&space;\in&space;Y" title="S = \begin{Bmatrix} (x_i,y_i) \end{Bmatrix}_{i=1}^n \sim D_N^n, y_i \in Y" />
</p>
and a small test set ("query")
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Q&space;=&space;\begin{Bmatrix}&space;x_i&space;\end{Bmatrix}_{i=1}^m&space;\sim&space;D_N^m" title="Q = \begin{Bmatrix} x_i \end{Bmatrix}_{i=1}^m \sim D_N^m" />
</p>
When presented with such a few-shot learning task, the learner must rapidly learn the classes presented and accurately classify the query images.
As with prior few-shot learning work, we will assume that before being presented with few-shot learning tasks in the target domain, the learner has access to a large annotated dataset <img src="https://latex.codecogs.com/svg.latex?D_B" title="D_B" /> known as the base dataset. However, unlike prior work on few-shot learning, we assume that this base dataset is drawn from a very different distribution.

<p align="center">
<b>Workflow of proposed framework.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/12/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Approach

During the representation learning stage, the model performs three steps:
* Learn a teacher model <img src="https://latex.codecogs.com/svg.latex?\theta_0" title="\theta_0" /> on the base dataset <img src="https://latex.codecogs.com/svg.latex?D_B" title="D_B" /> by minimizing cross entropy loss.
* Use the teacher model to construct a softly-labeled set <img src="https://latex.codecogs.com/svg.latex?D_u^{*}&space;=&space;\begin{Bmatrix}&space;(x_i,&space;\overline{y}_i)&space;\end{Bmatrix}_{i=1}^{N_u}" title="D_u^{*} = \begin{Bmatrix} (x_i, \overline{y}_i) \end{Bmatrix}_{i=1}^{N_u}" /> for all the unlabeled data available such that <img src="https://latex.codecogs.com/svg.latex?\overline{y}_i&space;=&space;f_{\theta_0}(x_i)" title="\overline{y}_i = f_{\theta_0}(x_i)" />.
* Learn a student model <img src="https://latex.codecogs.com/svg.latex?\theta^*" title="\theta^*" /> by optimizing the loss
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L&space;=&space;l_{CE}(f_{\theta}(x_i),y_i)&space;&plus;&space;l_{KL}(f_{\theta}(x_i),\overline{y}_i)&space;&plus;&space;l_{unlabeled}(D_u)" title="L = l_{CE}(f_{\theta}(x_i),y_i) + l_{KL}(f_{\theta}(x_i),\overline{y}_i) + l_{unlabeled}(D_u)" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?l_{unlabeled}" title="l_{unlabeled}" /> is a type of contrastive loss to help the learner extract additional useful knowledge specific to the target domain.

<p align="center">
<b>t-SNE plot of EuroSAT and CropDisease prior to and after model.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/12/Figure-3.png?raw=true" alt="Figure 3"/>
</p>

## Evaluation

Here, we freeze the representations <img src="https://latex.codecogs.com/svg.latex?\Phi" title="\Phi" /> after performing the representation learning and train a linear classifier on the support set and evaluate the classifier on the query set. Furthermore, here, the student is initialized to the teacher embedding with a randomly initialized classifier by default.

<p align="center">
<b>Performance of the model on different datasets.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/12/Figure-2.png?raw=true" alt="Figure 2"/>
</p>
