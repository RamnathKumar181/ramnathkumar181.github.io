---
layout: post
title: Explaining Black-Box Algorithms Using Probabilistic Contrastive Counterfactuals
published: false
---

An overview of the paper “[Explaining Black-Box Algorithms Using Probabilistic Contrastive Counterfactuals](https://arxiv.org/pdf/2103.11972)”.
<!--break-->
Imagine a bank that has customers applying for loans. Depending on their characteristics, the bank decides if it should grant them loan or not. Keeping up with the times, the bank will deploy state-of-the-art machine learning models to evaluate if a user can be relied to pay back the debt in the future. However, there is growing concern that these models are not transparent, non-intuitive, are hard to interpret. In order to increase the accpeptance of these models, it is essential that people understand the system's decision. To explain such decisions, the authors take inspiration from vast bodies of research such as philosophy, cognitive science, psychology, etc. All images and tables in this post are from their paper.

## What kind of explanations do humans seek?

The key insight is to recognize that one does not explain events per se, but that one explains why the puzzling event occurred in the target cases but not in some counterfactual contrast case. This is called contrastive explanations. Secondly, through causal explanations, it is important to understand if some cause is absent, its effects-some of them at least, would have been absent as well.

In this work, the authors present LEWIS which provides both contrastive and causal explanations. This work is based on the probabilistic contrastive counterfactuals.

For instance, for negative samples, LEWIS asks the question "To what extent is an attribute sufficient?" What if the interventional value was something else.
For positive samples however, the question is now "To what extent is an attribute necessary?". This would study the probability that changing the attribute value changes the outcome.
Necessasity and Sufficiency have been shown in the theory of causation to be a strong criteria for preferred explanations.

## Scores

### Necessity score

<p align="center">
<img src="/assets/Papers/5/Figure-7.png?raw=true" alt="Figure 1"/>

<img src="/assets/Papers/5/Figure-8.png?raw=true" alt="Figure 2"/>
</p>


## Explanations

Based on these notions, the authors define three probabilistic scores. By changing the context, you could obtain three separate reasons, which are highlighted in Figure 1.

### Global Explanations

When the context is not shown, we can study the necessity score of the features. So the decline of the feature with the highest necessity score would most likely lead to a flip in the positive decision.

### Contextual Explanations

Setting a context to a sub-population, the scores offers the following statements. The feature with a higher necessary and sufficient scores would indicate that the feature plays a decisive role in the outcome of the sample.

### Local Explanations

Finally, when the context is set to an individual, the scores explain the algorithm's output using necessity score for positive samples and sufficiency score for negative samples. This is shown through the contribution of their attributes towards their outcomes - in the sense, what is the gain or loss in staying at the current value and not changing.

Now computing these scores from data is known to be difficult; however, under certain assumptions that we have knowledge about underlying causal graph, the authors show that this is bounded and can be computed from historical data.
