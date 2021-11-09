---
layout: post
title: Imagenet classification with deep convolutional neural networks
published: true
---

An overview of the paper “[Imagenet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)”.
<!--break-->
The paper proposes a novel approach using deep CNN on the imagenet task. They also use the ReLU activation instead of tanh or softmax since, ReLU is a non-saturating linearity unlike tanh and softmax. Furthermore, ReLU is also faster to train in comparison. All images and tables in this post are from their paper.

## Local Response Normalization

ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron. However, the authors found out that the local normalization scheme aids generalization.

## Overlapping Pooling

Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally, the neighborhoods summarized by adjacent pooling units do not overlap. A pooling layer can be thought of as consisting of a grid of pooling units spaced <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> pixels apart, each summarizing a neighborhood of size <img src="https://latex.codecogs.com/svg.latex?z&space;\times&space;z" title="z \times z" /> centered at the location of the pooling unit. If we set <img src="https://latex.codecogs.com/svg.latex?s=z" title="s=z" />, we obtain traditional local pooling as commonly employed in CNNs.

## Reducing Overfitting

Due to the sheer number of parameters, there is a very high probability of overfitting. To combat this, the authors used various methods in parallel, including data augmentation and dropout.
