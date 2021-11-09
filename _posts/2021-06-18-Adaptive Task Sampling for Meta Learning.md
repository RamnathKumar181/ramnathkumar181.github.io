---
layout: post
title:  Adaptive Task Sampling for Meta Learning
published: true
---

An overview of the paper “[Adaptive Task Sampling for Meta Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630732.pdf)”.
<!--break-->
In this paper, the authors propose a method called adaptive task sampling to improve the generalization performance. Unlike instance based sampling, task based sampling is much more challenging due to the implicit definition of the task in each episode. All images and tables in this post are from their paper.

## Introduction

In order to compensate for the scarcity of training data in few-shot classification tasks, meta-learning approaches rely on an episodic training paradigm. A series of few-shot tasks are sampled from meta-training data for the extraction of transferable knowledge across tasks, which is then applied to new few-shot classification tasks consisting of unseen classes during the meta-testing phase. Despite their noticeable improvements, these meta-learning approaches leverage uniform sampling over classes to generate few-shot tasks, which ignores the intrinsic relationships between classes when forming episodes. Furthermore, the authors argue that exploiting class structures to construct more informative tasks is critical in meta-learning, which improves its ability to adapt to novel classes. The difficulty of a class, and even the semantics of a class, is dependent on each other. For instance, the characteristics to discriminate “dog” from “laptop” or “car” are relatively easier to uncover than those for discriminating “dog” from “cat” or “tiger”.



## Class Pair based Adaptive Task Sampling

The authors propose a method which helps determine the task
selection distribution by computing the difficulty of all class-pairs in it. As a result, it could capture the complex-structured relationships between classes in a multi-class few-shot classification problem. The authors propose a greedy class-pair based adaptive task sampling method which improves on the computational bottleneck.

### Adaptive Sampling

The most common paradigm is to calculate the importance of each instance based on the gradient norm [1], bound on the gradient
norm, loss, approximate loss or prediction probability. Researchers also consider improving the generalization performance rather
than speeding up training.
*  Hard example mining methods also prioritize challenging training examples.6
*  Some other researchers prioritize uncertain examples that are
close to the model’s decision boundary.

### Class based Sampling

Class-based sampling (c-sampling) approach updates the class selection probability <img src="https://latex.codecogs.com/svg.latex?p^{t&plus;1}_C(c)" title="p^{t+1}_C(c)" /> in each episode.
If the given target variable <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> is such that <img src="https://latex.codecogs.com/svg.latex?y\neq&space;c" title="y\neq c" />, then we would increase the score such that the model is more confident about its prediction. However, when <img src="https://latex.codecogs.com/svg.latex?y=&space;c" title="y= c" />, we will increase score based on max error. This would enforce the model to choose classes which not only improve performance of other classes, but also lead to most margin of improvement for the given class <img src="https://latex.codecogs.com/svg.latex?c" title="c" />.
It implicitly assumes that the difficulty of each class is independent. Therefore, it updates the class selection probability in a decoupled way. However, the limitation of such a model is that, assembling the most difficult classes do not necessarily lead to a difficult task.

### Class Pair Based Sampling

To address the above issue, we further propose a class-pair based sampling (cp-sampling) approach that exploits the pair-wise relationships between classes. In this case, we will choose classes which are most often confused with each other. This is the intuition behind the method.
An extension of this work is the Greedy Class-Pair Based Sampling, which reduces the complexity of the current approach.
