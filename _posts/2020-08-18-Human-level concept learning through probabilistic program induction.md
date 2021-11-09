---
layout: post
title: Human-level concept learning through probabilistic program induction
published: true
---

An overview of the paper “[Human-level concept learning through probabilistic program induction](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf)”.
<!--break-->
Despite recent advances in artificial intelligence and machine learning, two aspects of human conceptual learning have always eluded machine systems. For one, people can learn new concepts from just one or a handful of examples. Secondly, people learn richer representations than machines do, even for simple concepts unlike machines. The paper raises a few important concerns: How do people learn new concepts from just one or a few examples? And how do
people learn such abstract, rich, and flexible representations? An even greater challenge arises when putting them together. This paper also introduces the Bayesian program learning (BPL) framework, capable of learning a large class of visual concepts from just a single example and generalizing in ways that are mostly indistinguishable from people. All images and tables in this post are from their paper.

## Bayesian Program Learning

The BPL approach learns simple stochastic programs to represent concepts, building them compositionally from parts, subparts, and spatial relations. The joint distribution on types y, a set of M tokens of that type <img src="https://latex.codecogs.com/svg.latex?q(1)&space;,&space;.&space;.&space;.,&space;q(M)" title="q(1) , . . ., q(M)" />, and the corresponding binary images <img src="https://latex.codecogs.com/svg.latex?I(1)&space;,&space;.&space;.&space;.,&space;I(M)" title="I(1) , . . ., I(M)" /> factors as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P(\psi&space;,\theta&space;^{(1)},...,\theta&space;^{(M)},I^{(1)},...,I^{(M)})&space;=&space;p(\psi)&space;\prod&space;_{m=1}^M&space;p(I^{(m)}|\theta&space;^{(m)})p(\theta&space;^{(m)}|&space;\psi)" title="P(\psi ,\theta ^{(1)},...,\theta ^{(M)},I^{(1)},...,I^{(M)}) = p(\psi) \prod _{m=1}^M p(I^{(m)}|\theta ^{(m)})p(\theta ^{(m)}| \psi)" />
</p>

This is a simple bayes application which can be easily proved. This concept was particulary applied on handwritten characters. Note that these tokens are generated using another code and not readily availble in our data.

## Inference

Although successful on these tasks, BPL still sees less structure in visual concepts than people do. It lacks explicit knowledge of the general environment. Furthermore, capturing how people learn all these concepts at the level the authors reached with handwritten characters is a long-term goal.
