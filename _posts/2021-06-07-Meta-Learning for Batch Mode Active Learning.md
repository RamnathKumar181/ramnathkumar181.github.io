---
layout: post
title:  Meta-Learning for Batch Mode Active Learning
published: true
---

An overview of the paper “[Meta-Learning for Batch Mode Active Learning](https://openreview.net/pdf?id=r1PsGFJPz)”.
<!--break-->
In this paper, the authors propose a method to construct the best set of unlabeled items to label given a classifier trained on a small training set. All images and tables in this post are from their paper.


## Problem Statement

The majority of popular approaches are based on heuristics such as  choosing the item whose label the model is most uncertain about, choosing the item whose addition will cause the model to be least uncertain about other items, or choosing the item that is most “different” compared to other unlabeled items according to some similarity function . However, there are some limitations when extending these heuristics to batch setting:
* Suboptimal performance and produce sets with overly redundant items
* Complexity for selecting each new item that is atleast quadratic, making them prohibitive to use for large unlabeled datasets.
* It is assumed that unlabeled items belong to at least one of the classes we are interested in classifying; how this is not the case always. The data will often consist of distractor items, that do not belong to any one of the classes.

## Method

The method involves supplementing the support and query sets with an unlabeled set <img src="https://latex.codecogs.com/svg.latex?U&space;=&space;\{\widetilde{x}_1,...,\widetilde{x}_m\}" title="U = \{\widetilde{x}_1,...,\widetilde{x}_m\}" /> that consists of <img src="https://latex.codecogs.com/svg.latex?M" title="M" /> unlabeled examples. We consider <img src="https://latex.codecogs.com/svg.latex?K" title="K" />-shot, <img src="https://latex.codecogs.com/svg.latex?N" title="N" />-class, <img src="https://latex.codecogs.com/svg.latex?B" title="B" />-batch
episodes where we need to select a subset <img src="https://latex.codecogs.com/svg.latex?A\subseteq&space;U" title="A\subseteq U" /> of size <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> to be labeled and added to our support
set <img src="https://latex.codecogs.com/svg.latex?S" title="S" /> to get a new support set <img src="https://latex.codecogs.com/svg.latex?{S}'&space;=&space;S\cup&space;A" title="{S}' = S\cup A" />. The goal is to use the classifier formed from the original
support set <img src="https://latex.codecogs.com/svg.latex?S" title="S" /> to select the best subset of <img src="https://latex.codecogs.com/svg.latex?B" title="B" /> examples from <img src="https://latex.codecogs.com/svg.latex?U" title="U" /> to label to create the new support set <img src="https://latex.codecogs.com/svg.latex?{S}'" title="{S}'" /> and associated new classifier so as to most improve the performance on the query set <img src="https://latex.codecogs.com/svg.latex?Q" title="Q" />.

We can calculate a set of statistics relating each unlabeled item <img src="https://latex.codecogs.com/svg.latex?\widetilde{x}_i&space;\in&space;U" title="\widetilde{x}_i \in U" /> to the set of prototypes and we denote this set of item-classifier statistics by <img src="https://latex.codecogs.com/svg.latex?\prod&space;(\{c_k\}^K_{k=1},\widetilde{x}_i)" title="\prod (\{c_k\}^K_{k=1},\widetilde{x}_i)" />. These statistics are used to compute two distributions quality and diversity, which represent two distributions over which unlabeled item to add next to the existing subset <img src="https://latex.codecogs.com/svg.latex?A" title="A" />.

* <b>Quality Distribution:</b> The probability of selecting an unlabeled item according to its quality is defined as <img src="https://latex.codecogs.com/svg.latex?p_{quality}(\widetilde{x}_i)&space;\propto&space;\exp(q_i)" title="p_{quality}(\widetilde{x}_i) \propto \exp(q_i)" />, where:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?q_i&space;=&space;f_q(\prod&space;(\{c_k\}^K_{k=1},\widetilde{x}_i))" title="q_i = f_q(\prod (\{c_k\}^K_{k=1},\widetilde{x}_i))" />
</p>

<img src="https://latex.codecogs.com/svg.latex?f_q" title="f_q" /> is a MLP with parameters <img src="https://latex.codecogs.com/svg.latex?q" title="q" />. This distribution independently maps the probability of each unlabeled item being selected based on a prediction of how useful the item will be to the existing classifier according to a learned function of item-classifier statistics.

* <b>Diversity Distribution:</b> The same set of statistics can also be used to compute a feature vector describing the unlabeled item to classifier relationship as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\phi_i&space;=&space;f_{\phi}(\prod(\{c_k\}^K_{k=1},&space;\widetilde{x}_i))" title="\phi_i = f_{\phi}(\prod(\{c_k\}^K_{k=1}, \widetilde{x}_i))" />
</p>

<img src="https://latex.codecogs.com/svg.latex?f_{\phi}" title="f_{\phi}" /> is a MLP with parameters <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" />. The goal of the diversity distribution is to increase the probability of selecting unlabeled items which are dissimilar from the items that already make up the set <img src="https://latex.codecogs.com/svg.latex?A" title="A" /> where similarity between each item's corresponding feature vector. The probability of selecting an unlabeled item according to its diversity is then:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p_{diversity}&space;\propto&space;\exp(v(\phi_i)/\tau)" title="p_{diversity} \propto \exp(v(\phi_i)/\tau)" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?v(\phi_i)&space;=&space;\min_{\widetilde{x}_j&space;\in&space;A}\{\sin\theta_{ij}\}" title="v(\phi_i) = \min_{\widetilde{x}_j \in A}\{\sin\theta_{ij}\}" />. Here,  <img src="https://latex.codecogs.com/svg.latex?\theta_{ij}" title="\theta_{ij}" /> is the angle between feature vectors <img src="https://latex.codecogs.com/svg.latex?\phi_i" title="\phi_i" /> and <img src="https://latex.codecogs.com/svg.latex?\phi_j" title="\phi_j" /> and <img src="https://latex.codecogs.com/svg.latex?\tau" title="\tau" /> is a learned temperature parameter that allows us to control the flatness of this distribution. The probability of an item being picked increases as its feature vector is more orthogonal to feature vectors corresponding to items already having been added to the subset <img src="https://latex.codecogs.com/svg.latex?A" title="A" />.

* <b>Product of Experts:</b> The final probability distribution is attained as a product of experts model combining the distributions of quality and diversity.

## Training

The model is trained on a loss such that the final accuracy of the query set is improved, and all the parameters <img src="https://latex.codecogs.com/svg.latex?{\theta}'&space;=&space;\{\phi,q,\tau\}" title="{\theta}' = \{\phi,q,\tau\}" /> are learned. The model is trained in an episodic fashion and new batches are sampled based on previous probabilities.
