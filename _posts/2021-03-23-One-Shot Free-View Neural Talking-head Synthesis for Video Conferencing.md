---
layout: post
title: One-Shot Free-View Neural Talking-head Synthesis for Video Conferencing
published: true
---

An overview of the paper “[One-Shot Free-View Neural Talking-head Synthesis for Video Conferencing](https://arxiv.org/pdf/2011.15126.pdf)”.
<!--break-->
The authors propose a model that can re-create a talking-head video using only a single source image and a sequence of unsupervisedly-learned 4D keypoints, representing motions in the video. The model is 10x more efficient than the H.264 standard. All images and tables in this post are from their paper.

## Introduction

The authors propose a neural talking-head video synthesis model and demonstrate its applications to video conferencing. The model learns to synthesize a talking-head video using a source image containing the target person's appearance and a driving video that dictates the motion in the output. The motion is encoded based on a novel keypoint representation, where the identity-specific and motion related information is decomposed unsupervisedly. In this work, the authors use a graphical model which is local free view. This allows to synthesize the talking-head from other viewpoints and addresses the fixed viewpoint limitation and achieves local free-view synthesis. The contribution from this work are 3-fold:
* <b>Contribution 1:</b>A novel one-shot neural talking-head synthesis approach, which achieves better visual quality than state-of-the art methods on the benchmark datasets.
* <b>Contribution 2:</b>Local free-view control of the output video, without the need for a 3D graphics model. The model allows changing the viewpoint of the talking-head during synthesis.
* <b>Contribution 3:</b>Reduction in bandwidth for video-streaming by almost 10x.


<p align="center">
<b>Overview of proposed approach.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/18/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

People's faces have an inherent structure- from the shape to the relative arrangement of different parts such as eyes, nose, mouth, etc. This allows us to use keypoints and associated metadata for efficient compression, an order of magnitude better than traditional codecs. The model does not guarantee pixel aligned output videos; however, it faithfully models facial movements and emotions.
The proposed method can be divided into three major steps:
* Source image feature extraction
* Driving video feature extraction
* Video generation


### Source image feature extraction

<p align="center">
<b>Different features extracted from the source image.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/18/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

* <b>3D appearance feature extraction (<img src="https://latex.codecogs.com/svg.latex?F" title="F" />)</b>: Using a neural network <img src="https://latex.codecogs.com/svg.latex?F" title="F" />, the source image <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> is mapped to a 3D appearance feature volume <img src="https://latex.codecogs.com/svg.latex?f_s" title="f_s" />.

* <b>3D canonical keypoint extraction (<img src="https://latex.codecogs.com/svg.latex?L" title="L" />)</b>: Using a canonical 3D keypoint detection network <img src="https://latex.codecogs.com/svg.latex?L" title="L" />, a set of <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> canonical 3D keypoints <img src="https://latex.codecogs.com/svg.latex?x_{c,k}&space;\in&space;\mathbb{R}^3" title="x_{c,k} \in \mathbb{R}^3" /> and their Jacobians <img src="https://latex.codecogs.com/svg.latex?J_{c,k}&space;\in&space;\mathbb{R}^{3*3}" title="J_{c,k} \in \mathbb{R}^{3*3}" /> are extracted from <img src="https://latex.codecogs.com/svg.latex?s" title="s" />. The Jacobians represent how a local path around the keypoint can be transformed into a patch in another image via an affine transformation. The authors have used a U-Net style encoder-decoder to extract canonical keypoints.

* <b>Head Pose (<img src="https://latex.codecogs.com/svg.latex?H" title="H" />) and Expression Extraction (<img src="https://latex.codecogs.com/svg.latex?\Delta" title="\Delta" />)</b>: A pose estimation network <img src="https://latex.codecogs.com/svg.latex?H" title="H" /> is used to estimate the head pose of the person in <img src="https://latex.codecogs.com/svg.latex?s" title="s" />. It is parameterized by a rotation matrix <img src="https://latex.codecogs.com/svg.latex?R_s\in&space;\mathbb{R}^{3*3}" title="R_s\in \mathbb{R}^{3*3}" /> and a translation vector <img src="https://latex.codecogs.com/svg.latex?t_s\in&space;\mathbb{R}^{3}" title="t_s\in \mathbb{R}^{3}" />. The rotation matrix in practice is composed of three matrices. Expression deformation estimation network <img src="https://latex.codecogs.com/svg.latex?\Delta" title="\Delta" /> is used to estimate deformation of keypoints from the neutral expression. Thus, there are <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> 3D deformation <img src="https://latex.codecogs.com/svg.latex?\delta&space;_{s,k}" title="\delta _{s,k}" />. The same architecture is used to extract motion-related information from the driving video.

### Driving Video Feature Extraction

<p align="center">
<b>Different features extracted from the driving video.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/18/Figure-3.png?raw=true" alt="Figure 3"/>
</p>

The driving video is used to extract motion-related information. To this end, head pose estimation network <img src="https://latex.codecogs.com/svg.latex?H" title="H" /> and expression deformation estimator network <img src="https://latex.codecogs.com/svg.latex?\Delta" title="\Delta" /> is used. In video conferencing, we can change a person's head pose in the video stream freely despite the original view angle.

### Video Generation

<p align="center">
<b>Video synthesis pipeline.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/18/Figure-4.png?raw=true" alt="Figure 4"/>
</p>

To summarize so far, we have source image <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> and driving video <img src="https://latex.codecogs.com/svg.latex?d" title="d" />. The task is to generate output video <img src="https://latex.codecogs.com/svg.latex?y" title="y" /> such that it has the identity-specific information from <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> and motion-specific information from <img src="https://latex.codecogs.com/svg.latex?d" title="d" />. To obtain identity-specific information different neural networks are used and the same goes for motion-specific information. These pieces of information are used to obtain <img src="https://latex.codecogs.com/svg.latex?K" title="K" /> 3D keypoints and Jacobians for both <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> and <img src="https://latex.codecogs.com/svg.latex?d" title="d" />.
These keypoints and Jacobians are then used to warp the source appearance feature <img src="https://latex.codecogs.com/svg.latex?f_s" title="f_s" /> extracted from <img src="https://latex.codecogs.com/svg.latex?s" title="s" /> from which they generate the final output image using a generator network <img src="https://latex.codecogs.com/svg.latex?G" title="G" />.

## Training the models

For each video, two frames were sampled, one as source image, and other frame from the driving video.
The networks are trained together by minimizing the following loss:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L&space;=&space;L_p(d,y)&space;&plus;&space;L_G(d,y)&space;&plus;&space;L_E(\begin{Bmatrix}x_{d,k}\end{Bmatrix},\begin{Bmatrix}J_{d,k}\end{Bmatrix})&plus;L_L(\begin{Bmatrix}x_{d},k\end{Bmatrix})&plus;L_H(R_d,\overline{R}_d)&plus;L_{\Delta}(\begin{Bmatrix}\delta_{d,k}\end{Bmatrix})" title="L = L_p(d,y) + L_G(d,y) + L_E(\begin{Bmatrix}x_{d,k}\end{Bmatrix},\begin{Bmatrix}J_{d,k}\end{Bmatrix})+L_L(\begin{Bmatrix}x_{d},k\end{Bmatrix})+L_H(R_d,\overline{R}_d)+L_{\Delta}(\begin{Bmatrix}\delta_{d,k}\end{Bmatrix})" />
</p>
Here,
* <b>Perpetual Loss (<img src="https://latex.codecogs.com/svg.latex?L_p" title="L_p" />)</b>: Perpetual loss is used in image reconstruction tasks. Derived from the VGG network layer.
* <b>GAN Loss(<img src="https://latex.codecogs.com/svg.latex?L_G" title="L_G" />)</b>: Here, they used patch GAN implementation along with hinge loss.
* <b>Equivalence Loss (<img src="https://latex.codecogs.com/svg.latex?L_E" title="L_E" />)</b>: This loss ensures the consistency of the estimated keypoints.
* <b>Key prior Loss (<img src="https://latex.codecogs.com/svg.latex?L_L" title="L_L" />)</b>: This loss encourages the estimated image-specific keypoints to spread out across the face region, instead of crouding around a small neighborhood.
* <b>Head Pose Loss(<img src="https://latex.codecogs.com/svg.latex?L_H" title="L_H" />)</b>: <img src="https://latex.codecogs.com/svg.latex?L_1" title="L_1" /> distance is computed between the estimated head pose and the one predicted by a pre-trained estimator. This approximation is as good as the pre-trained model head pose estimator.
* <b>Deformation prior Loss(<img src="https://latex.codecogs.com/svg.latex?L_{\Delta&space;}" title="L_{\Delta }" />)</b>: This loss is simply given as <img src="https://latex.codecogs.com/svg.latex?L_1" title="L_1" /> norm of the deviation such that, <img src="https://latex.codecogs.com/svg.latex?L_{\Delta}&space;=&space;\left&space;\|&space;\delta&space;_{d,k}&space;\right&space;\|" title="L_{\Delta} = \left \| \delta _{d,k} \right \|" />.

## Conclusion

The results of this research are really quite promising. The techniques we talked about here resulted in 10X bandwidth reduction, and the video quality is high considering the reduction. Models like this could make it possible to demonetize access to viedo conferencing and reduce strain on networks, especially in residential and rural areas where bandwidth is already harder to come by.
