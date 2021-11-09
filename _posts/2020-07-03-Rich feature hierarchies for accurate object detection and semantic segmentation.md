---
layout: post
title: Rich feature hierarchies for accurate object detection and semantic segmentation
published: true
---

An overview of the paper “[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)”.
<!--break-->
The paper proposes a novel approach where the image is broken down into regions before being classified by the model. All images and tables in this post are from their paper.

## Region proposals

There are various options available for this stage. The RCNN was initially built using selective search. The method here is agnostic, and any other method could also be used.

## Feature extraction

Regardless of the size or aspect ratio of the candidate region, the authors warp all pixels in a tight bounding box around it to the required size. Prior to warping, they dilate the tight bounding box so that at the warped size there are exactly <img src="https://latex.codecogs.com/svg.latex?p" title="p" /> pixels of warped image context around the original box.

## Testing

At test time, they run a selective search on the test image to extract around 2000 region proposals. Then, they warp each proposal and forward propagate it through the CNN in order to compute features. Then, for each class, the extracted feature vectors are scored using the SVM trained for that class. Given all scored regions in an image, we apply a greedy non-maximum suppression (for each class independently) that rejects a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region larger than a learned threshold.

<p align="center">
<b>Object Detection system overview.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/32/Figure-1.png?raw=true" alt="Figure 2"/>
</p>
