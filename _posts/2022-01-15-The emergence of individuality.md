---
layout: post
title: The Emergence of Individuality
published: true
---

An overview of the paper “[The Emergence of Individuality](https://arxiv.org/pdf/2006.05842.pdf)”.
<!--break-->
Individuality is key for multi-agent cooperation. The authors propose a simple yet efficient method for the emergence of individuality(EOI) in multi-agent reinforcemnet learning. All images and tables in this post are from their paper.

## Introduction

Although multi-agent reinforcement learning (MARL) has been applied to multi-agent cooperation, it is widely ovserved that agents usually learn similar behaviors, especially when the agents are homogeneous with shared global reward and co-trained.
For example, multi-camera multi-object tracking, where camera agents learn to cooperatively track multiple objects, the camera agents all tend to track the easy object. However, such similar behaviors can easily make the learned policies fall into local optimum. If the agents can respectively track different objects, they are more likely to solve the task optimally.
EOI learns a probabilistic classifier that predicts a probability distribution over agents given their observation and gives each agent an intrinsic reward of being correctly predicted probability by the classifier. Encouraged by the intrinsic reward, agents tend to visit their own familiar observations.

<p align="center">
<b> Multi-camera multi-target capturing.</b>
</p>
<p align="center">
<img src="/assets/Papers/2/Figure-5.png?raw=true" alt="Figure 1"/>
</p>

## Methodology

Individuality is of being an individual separate from others. Motivated by this, the authors propose EOI, where agents are intrinsically rewarded in terms of being correctly predicted by a probabilistic classifier that is learned based on agents' observations. If the classifier learns to accurately distinguish agents, agents should behave differently and thus individuality emerges.

### Intrinsic Reward

To enable agents to develop indivuality, EOI learns a probabilistic classifier <img src="https://latex.codecogs.com/svg.latex?P(I|O)" title="P(I|O)" /> to predict a probability distribution over agents given on their observation, and each agent takes the correctly predicted probability as the intrinsic reward at each timestep. Thus, the reward function for agent <img src="https://latex.codecogs.com/svg.latex?i" title="i" /> is modified as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?r+\alpha p(i|o_i)" title="r+\alpha p(i|o_i)" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?r" title="r" /> is the global environmental reward, <img src="https://latex.codecogs.com/svg.latex?p(i|o_i)" title="p(i|o_i)" /> is the predicted probability of agent <img src="https://latex.codecogs.com/svg.latex?i" title="i" /> given its observation <img src="https://latex.codecogs.com/svg.latex?o_i" title="o_i" />, and <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> is a tuning hyperparameter to weight the intrinsic reward.

### Regularizers of <img src="https://latex.codecogs.com/svg.latex?P_{\phi}(I|O)" title="P_{\phi}(I|O)" />

The previous section would not work if there is no difference between agents' policies. However, in general, the difference between initial policies is small (even no differences if agents' policies are initially by the same network weights), and policies will quickly learn similar behaviors as mentioned in Figure-1. To address this issue, the authors propose two regularizers: positive distance (PD) and mutual children (MI) for learning <img src="https://latex.codecogs.com/svg.latex?P_{\phi}(I|O)" title="P_{\phi}(I|O)" />.

**Positive Distance:** The positive distance is inspired from the triplet loss in contrastive learning, which is proposed to learn identifiable embeddings. The positive distance minimizes the intra-distance between the observations with the same "identity", which hence enlarges the margin between different "identities". As a result, the observations become more identifiable.

### Mutual Information

If the observation are more identifiable, it is easier to infer the agent that visits the given observation most identifiable, it is easier to infer the agent that visits the given observation most, which indicates the higher mutual information between agent index and observation.Therefore, to further increase the discriminability of the classifier, we maximize their mutual information.

## Conclusion

The authors propose a novel approach EOI for the emergence of individuality in MARL. EOI learns a probabilistic classifier that predicts a probability distribution over agents given their observation and gives each agent an intrinsic reward of being correctly predicted by the classifier.
