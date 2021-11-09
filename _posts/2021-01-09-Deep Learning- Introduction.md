---
layout: post
title: Deep Learning- Introduction
published: true
---

An overview of the chapter “[Introduction](https://www.deeplearningbook.org/contents/intro.html)” from the famous book “[Deep Learning](https://www.deeplearningbook.org/)” written by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
<!--break-->
The authors create a brief introduction of the important concepts that guide machine learning. All images and tables in this post are from their book.

## Introduction

AI needs immense amount of knowledge about the world. This can be provided through:
* <b>Knowledge base (Cyc):</b> An inference engine and a database of statements in a language called CycL. These statements are entered by human supervisors. The main limitation is that people ourselves, struggle to devise formal rules with enough complexity to accurately describe the world.
* <b>Machine Learning:</b> The systems have the ability to acquire their own knowledge by extracting patterns from raw data. However, the performance of these models depend heavily on the representation of the data they are given. Each piece of information is known as a feature. These are systems which map representations to output.
* <b>Representation Learning:</b> These systems learn the representations as well. For example, autoencoders. While designing features or algorithms for learning features, our goal is usually to separate the factors of variation that explain the observed data. Such factors are often not directly observed. Most applications require us to disentangle the factors of variation and discard the ones we do not care about. When it is nearly as difficult to obtain a representation as to solve the original problem, representation learning does not, at first glance, seem to help us.
* <b>Deep Learning:</b> This solves the central problem of representation learning by introducing representations that are expressed in terms of other, simpler representations. For example, MLP

## Historical Trends in DL

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?Cybernetics&space;\rightarrow&space;Connectionism&space;\rightarrow&space;Deep\&space;Learning" title="Cybernetics \rightarrow Connectionism \rightarrow Deep\ Learning" />
</p>
### Cybernetics
<img src="https://latex.codecogs.com/svg.latex?\inline&space;(x_1,&space;x_2,...,&space;x_n)&space;\rightarrow&space;y" title="(x_1, x_2,..., x_n) \rightarrow y" />. These models would learn a set of weights <img src="https://latex.codecogs.com/svg.latex?\inline&space;(w_1,&space;w_2,...,&space;w_n)" title="(w_1, w_2,..., w_n)" /> and compute their output <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x,w)&space;=&space;x_1w_1&space;&plus;&space;x_2w_2&space;&plus;&space;...&space;&plus;&space;x_nw_n" title="f(x,w) = x_1w_1 + x_2w_2 + ... + x_nw_n" />.
  * <b>McCullochs-Pitts Neuron:</b> An early model of brain function. Recognize different categories by testing whether <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x,w)" title="f(x,w)" /> is positive or negative. Of course, for the model to correspond to the desired definition of the categories, the weights needed to be set correctly. These weights could be set by the human operator.
  * <b>Perceptron:</b> The first model that could learn the weights defining the categories given examples of inputs. Used Stochastic Gradient Descent.
  * <b>Adaptive Linear Elementt (Adaline):</b> Returns the value of <img src="https://latex.codecogs.com/svg.latex?\inline&space;f(x)" title="f(x)" /> itself to predict a real numbers from the data. Uses a slight modification of the Stochastic Gradient Descent.

Perceptron and Adaline are  linear models, and are not capable of learning XOR functions.

### Connectionism
This includes the introduction of neural networks. The central idea in connectionism is that a large number of simple computational units can achieve intelligent behavior when networked together. Key concept from this era includes the distributed representation. This is the idea that each input to a system should be represented by many features, and each feature should be involved in the representation of many possible inputs. This was also the era where backpropagation was popularized and successfully used with deep neural networks. Progess was also made in modeling sequences with neural networks and introduced the LSTM (long short-term memory) networks. Due to unrealistic expectations from investors, difficulties in training, followed by introduction of kernel machines and graphical models, the popularity of neural networks declined.
### Deep Learning
This era began with a breakthrough in hardware. They also introduced the idea of training Deep belief networks using greedy layerwise pre-training, and could also be applied with deep neural networks. There was a shift of focus towards unsupervised learning techniques and ability to generalize on smaller datasets. This was also the era of introduction of Reinforcement learning.
