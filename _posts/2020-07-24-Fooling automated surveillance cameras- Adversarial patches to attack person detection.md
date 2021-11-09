---
layout: post
title: Fooling automated surveillance cameras- Adversarial patches to attack person detection
published: true
---

An overview of the paper “[Fooling automated surveillance cameras: Adversarial patches to attack person detection](https://arxiv.org/pdf/1904.08653.pdf)”.
<!--break-->
The author proposes an approach to generate adversarial patches to targets with lots of intra-class variety, namely persons. The goal is to generate a patch that is able to successfully hide a person from a person detector. All images and tables in this post are from their paper.
<p align="center">
<b>An adversarial patch that is successfully able to hide persons.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/4/Figure-1.png?raw=true" alt="Figure 1"/>
</p>

## Generating adversarial patches against person detectors

The authors suggest an optimisation process (on the image pixels) where they try to find a patch that, on a large dataset, effectively lowers the accuracy of person detection. The loss of this process can be broken down into three parts:
* Non-printability score:- This factor represents how well the colours in our patch can be represented by a common printer.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{nps}&space;=&space;\sum&space;_{p_{patch&space;\in&space;p}}\min_{c_{print}&space;\in&space;C}&space;\left&space;|&space;p_{patch}&space;-c_{print}\right&space;|" title="L_{nps} = \sum _{p_{patch \in p}}\min_{c_{print} \in C} \left | p_{patch} -c_{print}\right |" />
</p>
where <img src="https://latex.codecogs.com/svg.latex?p_{patch}" title="p_{patch}" /> is a pixel in of our patch <img src="https://latex.codecogs.com/svg.latex?P" title="P" /> and <img src="https://latex.codecogs.com/svg.latex?c_{print}" title="c_{print}" /> is a colour in a set of printable colours <img src="https://latex.codecogs.com/svg.latex?C" title="C" />. The loss favours colors in our image that lie closely to colours in our set of printable colours.
* Total variation score:- This loss makes sure that our optimiser favours an image with smooth colour transitions and prevents noisy images. The score is low if neighbouring pixels are similar, and high if neighbouring pixel are different.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{tv}&space;=&space;\sum&space;_{i,j}\sqrt{(p_{i,j}-p_{i&plus;1,j})^2)&plus;(p_{i,j}-p_{i,j&plus;1})^2}" title="L_{tv} = \sum _{i,j}\sqrt{(p_{i,j}-p_{i+1,j})^2)+(p_{i,j}-p_{i,j+1})^2}" />
</p>
* Objectness score:- This loss is to minimize the object or class score outputted by the detector. The goal is to fool the classifier. Hence, the loss is defined as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L_{obs}&space;=&space;\max&space;(p_{obj_1},&space;p_{obj_2},...,p_{obj_n})" title="L_{obs} = \max (p_{obj_1}, p_{obj_2},...,p_{obj_n})" />
</p>

We then use a simple weighted sum to calculate the total loss. These weights are trained using Adam Optimizer.
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L&space;=&space;\alpha&space;L_{nps}&space;&plus;&space;\beta&space;L_{tv}&space;&plus;&space;L_{obj}" title="L = \alpha L_{nps} + \beta L_{tv} + L_{obj}" />
</p>


## Methodology

We need to first run the target person detector over our dataset of images. This yields bounding boxes that show where people occur in the image according to the detector. On a fixed position relative to these bounding boxes, we then apply the current version of our patch to the image under different transformations (which are explained in Section 3.3). The resulting image is then fed (in a batch together with other images) into the detector. We measure the score of the persons that are still detected, which we use to calculate a loss function. Using back propagation over the entire network, the optimiser then changes the pixels in the patch further in order to fool the detector even more.

<p align="center">
<b> Overview of the pipeline to get the object loss.</b>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ramnathkumar181/ramnathkumar181.github.io/master/assets/Papers/4/Figure-2.png?raw=true" alt="Figure 2"/>
</p>

The authors show that, if we combine this technique with a sophisticated clothing simulation, we can design a T-shirt print that can make a person virtually invisible for automatic surveillance cameras (using the YOLO detector).
