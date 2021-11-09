---
layout: post
title:  Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization
published: true
---

An overview of the paper “[Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://arxiv.org/pdf/2106.06607.pdf)”.
<!--break-->
In this paper, the authors propose an approach called the IB-IRM approach that would fare well even in OOD distribution cases. All images and tables in this post are from their paper.

## Introduction

The success of deep learning has been truly remarkable. However, recent years have witnessed an explosion of examples showing deep learning models are prone to exploiting shortcuts (spurious features) which fail to generalize out-of-distribution (OOD). The paper motivates this problem with the example of the problem of cow-camel-classification. For instance, a convolutional neural network trained to classify camels from cows; however, it was found that the model relied on the background color (e.g., green pastures for cows) and not on the properties of the animals (e.g., shape). This is a classic example of causation is not the same as correlation.


## Background

Intervention is defined as the model distribution shifts (images from different locations). This definition is consistent with the theory from causal learning. The cause is defined as a subset of features <img src="https://latex.codecogs.com/svg.latex?P(Y|X_s)" title="P(Y|X_s)" /> invariant across all interventions.
The training set is denoted by <img src="https://latex.codecogs.com/svg.latex?\varepsilon_{tr}" title="\varepsilon_{tr}" />, and the set of all examples including training is denoted by <img src="https://latex.codecogs.com/svg.latex?\varepsilon_{all}" title="\varepsilon_{all}" />.
The network is defined as <img src="https://latex.codecogs.com/svg.latex?f:X\rightarrow&space;y" title="f:X\rightarrow y" /> where the error is defined as <img src="https://latex.codecogs.com/svg.latex?\underset{f}{\min}&space;\underset{e\in&space;\varepsilon_{all}}{\max}&space;R^e(f)" title="\underset{f}{\min} \underset{e\in \varepsilon_{all}}{\max} R^e(f)" /> where <img src="https://latex.codecogs.com/svg.latex?R^e(f)&space;=&space;\mathbb{E}[l(f(x^e),y^e)]" title="R^e(f) = \mathbb{E}[l(f(x^e),y^e)]" />. The idea of this loss is to minimize the maximum loss obtained by OOD.

### Structural Equation Model (SEM)

In such cases, the input sample is defined as <img src="https://latex.codecogs.com/svg.latex?X^e&space;\leftarrow&space;g(z_{inv}^e,&space;z_{spu}^e)" title="X^e \leftarrow g(z_{inv}^e, z_{spu}^e)" />, where <img src="https://latex.codecogs.com/svg.latex?X^e" title="X^e" /> is the input sample. <img src="https://latex.codecogs.com/svg.latex?z_{inv}" title="z_{inv}" /> are the latent invariant features and <img src="https://latex.codecogs.com/svg.latex?z_{spu}" title="z_{spu}" /> are the latent spurious features. Furthermore, <img src="https://latex.codecogs.com/svg.latex?g" title="g" /> is the function that maps these latent features to the actual input to the model. Furthermore, this model also assumes that the output feature <img src="https://latex.codecogs.com/svg.latex?Y^e&space;\leftarrow&space;\gamma&space;(z_{inv}^e)&space;&plus;&space;\mathcal{N}^e" title="Y^e \leftarrow \gamma (z_{inv}^e) + \mathcal{N}^e" />, where the output feature is solely dependent on the invariant features with some added gaussian noise. Under this setting, we also assume the functions <img src="https://latex.codecogs.com/svg.latex?g" title="g" /> and <img src="https://latex.codecogs.com/svg.latex?\gamma" title="\gamma" /> to be constant, and <img src="https://latex.codecogs.com/svg.latex?\gamma&space;\circ&space;g_{inv}^{-1}" title="\gamma \circ g_{inv}^{-1}" /> is trained to minimize OOD.

### Invariant Risk Minimization (IRM)

Before we understand IRM, it is important to discuss what invariant predictors are. Suppose a network <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> is defined as <img src="https://latex.codecogs.com/svg.latex?f&space;=&space;w&space;\circ&space;\phi" title="f = w \circ \phi" />, where <img src="https://latex.codecogs.com/svg.latex?w" title="w" /> is the classifier and <img src="https://latex.codecogs.com/svg.latex?\phi" title="\phi" /> is the reprentator. If the representator is fixed, and we allow changes to the classifier depending on the environment, and the classifer converges to the same weights everytime, we call the model an invariant predictor.

In IRM, the goal is <img src="https://latex.codecogs.com/svg.latex?\min_{w,\phi}&space;\sum&space;R^e(w\circ\phi)" title="\min_{w,\phi} \sum R^e(w\circ\phi)" />, where <img src="https://latex.codecogs.com/svg.latex?w\circ\phi" title="w\circ\phi" /> is the invariant predictor. If the condition that <img src="https://latex.codecogs.com/svg.latex?w\circ\phi" title="w\circ\phi" />  is an invariant predictor is dropped, we reach the Empirical Risk Minimization setting.

If <img src="https://latex.codecogs.com/svg.latex?\gamma" title="\gamma" />, <img src="https://latex.codecogs.com/svg.latex?g" title="g" /> are linear, IRM would be optimal, but ERM would fail.
However, it is possible for both ERM and IRM to fail in cases of linear classification as shown in the example below:

<p align="center">
<b>Failures of ERM and IRM in linear classification</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/29/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

As shown in the figure, If latent invariant features in the training environments are separable, then there are multiple equally good candidates that could have generated the data, and the algorithm cannot distinguish between these. The models that use the spurious features for their prediction are bad classifiers, and would be easy to provide a counter-example in OOD setting where the model fails.

### 2-D fully informative invariant features (FIIF)

In this setting, the invariant features are assumed to have all the information for the models. The inputs are defined as <img src="https://latex.codecogs.com/svg.latex?X^e&space;\leftarrow&space;S(Z_{inv}^e,&space;Z_{spu}^e)" title="X^e \leftarrow S(Z_{inv}^e, Z_{spu}^e)" /> and the outputs are defined as <img src="https://latex.codecogs.com/svg.latex?Y^e&space;\leftarrow&space;I(w_{inv}^*,&space;Z_{inv}^e)\oplus&space;\mathcal{N}^e" title="Y^e \leftarrow I(w_{inv}^*, Z_{inv}^e)\oplus \mathcal{N}^e" /> where <img src="https://latex.codecogs.com/svg.latex?\mathcal{N}^e" title="\mathcal{N}^e" /> is the bernoulli distribution. The above setting comes with a few assumptions:
* Bounded invariant features: <img src="https://latex.codecogs.com/svg.latex?\cup&space;_{e\in\varepsilon&space;_{tr}}Z_{inv}^e" title="\cup _{e\in\varepsilon _{tr}}Z_{inv}^e" /> is a bounded set.
* Bounded spurious features: <img src="https://latex.codecogs.com/svg.latex?\cup&space;_{e\in\varepsilon&space;_{tr}}Z_{spu}^e" title="\cup _{e\in\varepsilon _{tr}}Z_{inv}^e" /> is a bounded set.
* Invariant feature support overlap: <img src="https://latex.codecogs.com/svg.latex?\forall&space;e&space;\in&space;\varepsilon_{all},&space;Z_{inv}^e&space;\subseteq&space;\cup_{e'\in\varepsilon_{tr}}Z_{inv}^{e'}" title="\forall e \in \varepsilon_{all}, Z_{inv}^e \subseteq \cup_{e'\in\varepsilon_{tr}}Z_{inv}^{e'}" />
* Spurious feature support overlap: <img src="https://latex.codecogs.com/svg.latex?\forall&space;e&space;\in&space;\varepsilon_{all},&space;Z_{spu}^e&space;\subseteq&space;\cup_{e'\in\varepsilon_{tr}}Z_{spu}^{e'}" title="\forall e \in \varepsilon_{all}, Z_{spu}^e \subseteq \cup_{e'\in\varepsilon_{tr}}Z_{spu}^{e'}" />

Note that, the last two assumptions state that the support set of the invariant (spurious) features for unseen environments is the same as the union of the support over the training environments. However, support overlap does not imply that the distribution over the invariant features does not change. Furtherore, to measure how much the training support of invariant features is seperated by the labelling hyperplane <img src="https://latex.codecogs.com/svg.latex?w_{inv}^*" title="w_{inv}^*" />, the authors define Inv-margin as <img src="https://latex.codecogs.com/svg.latex?\mathit{IM}&space;=&space;\min_{z\in\cup_{e\in\varepsilon_{tr}}}z_{inv}^e&space;sgn(w_{inv}^*.z)(w_{inv}^*.z)" title="\mathit{IM} = \min_{z\in\cup_{e\in\varepsilon_{tr}}}z_{inv}^e sgn(w_{inv}^*.z)(w_{inv}^*.z)" />. If the Inv-Margin is greater than 0, then the labelling hyperplane seperates the support set into two halves.
The authors further show that if the support of latent invariant features are strivtly separated by the labelling hyperplane <img src="https://latex.codecogs.com/svg.latex?w_{inv}^*" title="w_{inv}^*" />, then we can find another valid hyperplane <img src="https://latex.codecogs.com/svg.latex?w_{inv}^&plus;" title="w_{inv}^+" /> that is equally likely to have generated the same data. There is no algorithm that distinguish between the two hyperplanes. As a result, if we use data from the region where the hyperplanes disagree (yellow region), then the algorithm fails.

<p align="center">
<b>Impossibility result</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/29/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

Thus, even if invariant support overlap is guaranteed, the model could still fail if the model relies on spurious changes. However, if spurious features are kept constant, then both ERM and IRM could potentially succeed.

### Invariance + IB

The idea in this setting is to pick the classifer with minimum differential entropy.  This would allow the model to keep as less information about the spurious features. The proposed approach is able to succeed in all the above cases where IRM or ERM fail. The authors propose a new loss in order to train in this setting, which involves the ERM loss, IRM V1 penalty and the variance penalty.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?R^{\mathit{IB-IRM}}(\phi)&space;=&space;\sum_{e\in\varepsilon_{tr}}R^2(\phi)&space;&plus;&space;\lambda&space;\triangledown&space;_{w=1}&space;\left&space;\|&space;R^e(w\circ\phi)&space;\right&space;\|^2&space;&plus;&space;\nu&space;Var(\phi)" title="R^{\mathit{IB-IRM}}(\phi) = \sum_{e\in\varepsilon_{tr}}R^2(\phi) + \lambda \triangledown _{w=1} \left \| R^e(w\circ\phi) \right \|^2 + \nu Var(\phi)" />
</p>
