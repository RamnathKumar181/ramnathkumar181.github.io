---
layout: post
title: Deep Learning- Numerical Computation
published: true
---

An overview of the chapter “[Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)” from the famous book “[Deep Learning](https://www.deeplearningbook.org/)” written by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
<!--break-->
The authors create a brief introduction of the important concepts from numerical computation that help guide machine learning. All images and tables in this post are from their book.

## Introduction

Machine Learning algorithms usually require a very high amount of numerical computation: usually referring to algorithms that solve mathematical problems by methods that update estimates of the solution via an iterative process, rather than analytically deriving a formula providing a symbolic expression for the correct solution. Common operations include optimization and solving linear equations.

## Overflow and Underflow

The fundamental difficulty in performing continuous math on a computer is that we need to represent infinitely many real numbers with a finite number of bit patterns. This means that for almost all real numbers, we incur some approximation error when we represent the number in the computer. In many cases, it would just be rounding error, but could cause problems when compunded across many operations.

* One form of rounding error is **Underflow**. Underflow occurs when numbers near zero are rounded to zero. Many functions behave qualitatively differently when their argument is zero rather than a small positive number. For instance, division by zero, or taking logarithm of zero.
* Another highly damaging form of numerical error is **Overflow**. Overflow occurs when very large numbers are rounded to <img src="https://latex.codecogs.com/svg.latex?\infty" title="\infty" /> or <img src="https://latex.codecogs.com/svg.latex?-\infty" title="-\infty" />.

## Poor Conditioning

Conditioning refers to how rapidly a function changes with respect to small changes in its inputs. Functions that change rapidly when their inputs are perturbed slightly can be problematic for scientific computation because rounding errors in the inputs can result in large changes in the output.
Suppose <img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;\mathbf{A}^{-1}x" title="f(x) = \mathbf{A}^{-1}x" /> where <img src="https://latex.codecogs.com/svg.latex?\mathbf{A}&space;\in&space;\mathbb{R}^{n&space;\times&space;n}" title="\mathbf{A} \in \mathbb{R}^{n \times n}" /> has an eigenvalue decomposition, its condition number is:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\max_{i,j}&space;\left&space;|&space;\frac{\lambda_i}{\lambda_j}&space;\right&space;|" title="\max_{i,j} \left | \frac{\lambda_i}{\lambda_j} \right |" />
</p>

This is the ratio of the magnitude of the largest and smallest eigenvalue. When this number is large, matrix inversion is particularly sensitive to error in the input.

This sensitivity is an intrinsic property of the matrix itself, not the result of rounding error during matrix inversion. Poorly conditioned matrices amplify pre-existing errors when we multiply by the true matrix inverse.

## Gradient-Based Optimization

Optimization refers to the task of either minimizing or maximizing some function <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> by altering <img src="https://latex.codecogs.com/svg.latex?x" title="x" />.The function we want to minimize or maximize is called the **objective function** or **criterion**. When we are minimizing it, we may also call it **loss function**, **cost function** or **error function**.

We denote the value that minimizes or maximizes a function with a superscript <img src="https://latex.codecogs.com/svg.latex?*" title="*" />. For example, we might say <img src="https://latex.codecogs.com/svg.latex?x^*&space;=&space;\arg&space;\min_{x}&space;f(x)" title="x^* = \arg \min_{x} f(x)" />.

<p align="center">
<b>An illustration of how the gradient descent algorithm uses the derivatives of a function can be used to follow the function downhill to a minimum.</b>
</p>
<p align="center">
<img src="/assets/Papers/4/Figure-4.png?raw=true" alt="Figure 1"/>
</p>

When <img src="https://latex.codecogs.com/svg.latex?f^{'}(x)&space;=&space;0" title="f^{'}(x) = 0" />, the derivative provides no information about which direction to move. Points where <img src="https://latex.codecogs.com/svg.latex?f^{'}(x)&space;=&space;0" title="f^{'}(x) = 0" /> are known as **critical points** or **stationary points**. A **local minima** is a point where <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> is lower than at all neighboring points, so it is no longer possible to decrease <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> by making infinitesimal steps. A **local maxima** is a point where <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> is higher than all neighboring points, so it is not possible to increase <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> by making infinitesimal steps. Some critical points are neither mina or maxima, and are known as **saddle points**.

<p align="center">
<b>Examples of each of the three types of critical points in 1-D.</b>
</p>
<p align="center">
<img src="/assets/Papers/4/Figure-5.png?raw=true" alt="Figure 2"/>
</p>

A point that obtains the absolute lowwest value of <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> is a global minimum. In the context of deep learning, we optimize functions that may have many local minima that are optimal, and many saddle points surrounded by very flat regions. All this makes optimization very difficult especially when the input to the function is multidimensional. We therefore settle for finding a value of <img src="https://latex.codecogs.com/svg.latex?f" title="f" /> that is very low but not necessarily the global minima in any formal sense.

For functions with multiple inputs, we must make use of the concept of **partial derivatives**. The partial derivative <img src="https://latex.coecogs.com/svg.latex?\frac{\partial }{\partial x_i} f(x)" title=""/> measures how <img src="https://latex.coecogs.com/svg.latex?f"/> changes as only the variable <img src="https://latex.coecogs.com/svg.latex?x_i"/> increases at point <img src="https://latex.coecogs.com/svg.latex?x"/>. The **gradient** generalizes the notion of derivative to the case where the derivative is with respect to a vector: The gradient of <img src="https://latex.coecogs.com/svg.latex?f"/> is the vector containing all of the partial derivatives, denoted <img src="https://latex.coecogs.com/svg.latex?\bigtriangledown _xf(x)"/>. In multiple dimensions, critical points are points where every element of the gradient is equal to zero.

To minimize <img src="https://latex.coecogs.com/svg.latex?f"/>, we would like to find the direction in which <img src="https://latex.coecogs.com/svg.latex?f"/> decreases the fastest. We can do this using the directional derivative as follows:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?\min_{u,u^Tu=1}u^T\bigtriangledown _x f(x)"/>
<img src="https://latex.codecogs.com/svg.latex?\min_{u,u^Tu=1} \left \| u \right \|_2 \left \| \bigtriangledown _x f(x) \right \|_2 \cos\theta"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?\theta"/> is the angle between <ig src"https://latex.codecogs.com/svg.latex?u"/> and the gradient. The gradient points directly uphill, and the negative gradient points directly downhill. We can decrease <img src="https://latex.codecogs.com/svg.latex?f"/> by moving in the direction of the negative gradient. This is known as the **method of steepest descent** or **gradient descent**.

Gradient descent proposes a new point - <img src="https://latex.coecogs.com/svg.latex?x' = x - \epsilon \bigtriangledown _x f(x)"/> where <img src="https://latex.coecogs.com/svg.latex?\epsilon"/> is the **learning rate**. Finding the right hyperparameter for the learning rate is quite important. To achieve this, it is quite common to perform a **line search** where we evaluate <img src="https://latex.coecogs.com/svg.latex?f(x')"/> for several values of <img src="https://latex.coecogs.com/svg.latex?\epsilon"/>. Although, gradient descent is limited to optimization in continuous spaces, the general concept of repeatedly making a small move towards better configurations can be generalized to discrete spaces. Ascending an objective function of discrete parameters is called hill climbing.

Note that optimization algorithms may fail to find a global minimum when there are multiple local minima or plateaus present. In the context of deep learning, we generally accept such solutions even though they are not truly minimal, so long as they correspond to significantly low values of the cost function as shown below.

<p align="center">
<img src="/assets/Papers/4/Figure-6.png?raw=true" alt="Figure 3"/>
</p>


### Jacobian and Hessian Matrices

Sometimes, we need to find all of the partial derivatives of a function whose input and output are both vectors. The matrix containing all such partial derivatives is known as the **Jacobian matrix**. Specifically, we would have a function <img src="https://latex.coecogs.com/svg.latex?f: \mathbb{R}^m \rightarrow \mathbb{R}^n"/>, then the Jacobian matrix <img src="https://latex.coecogs.com/svg.latex?J \in \mathbb{R}^{n \times m}"/> of <img src="https://latex.coecogs.com/svg.latex?f"/> is defined such that <img src="https://latex.coecogs.com/svg.latex?J_{i,j} = \frac{\partial }{\partial x_j} f(x)_i"/>.

We are sometimes also interested in the second order derivative. The second derivative tells us how the first derivative will change as we vary the input. This is important because it tells us whether a gradient step will cause as much of an improvement as we would expect based on the gradient alone. We can think of the second derivative as measuring the **curvature**, as shown in the figure below.

<p align="center">
<img src="/assets/Papers/4/Figure-7.png?raw=true" alt="Figure 4"/>
</p>

When our function has multiple input dimensions, there are many second derivatives. These derivatives can be collected together into a matrix called the **Hessian matrix**. The Hessian matrix <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{H}(f)(x)"/> is defined such that:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?\boldsymbol{H}(f)(x)_{i,j} = \frac{\partial^2 }{\partial x_i \partial x_j} f(x)"/>
</p>

Anywhere the second partial derivatives are continuous, the differential operators are commutative, i.e. their orders can be swapped:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?\frac{\partial^2 }{\partial x_i \partial x_j} f(x) = \frac{\partial^2 }{\partial x_j \partial x_i} f(x)"/>
</p>

This implies that <img src="https://latex.coecogs.com/svg.latex?H_{i,j} = H_{j,i}"/>, and the Hessian matrix is symmetric in nature. Since the Hessian matrix is real and symmetric everywhere in the context of deep learning, we can decompose it into a set of real eigenvalues and an orthogonal basis of eigenvectors. The second derivative is a specific direction represented by a unit vector <img src="https://latex.codecogs.com/svg.latex?d"/> is given by <img src="https://latex.coecogs.com/svg.latex?d^THd"/>. When <img src="https://latex.coecogs.com/svg.latex?d"/> is an eigenvector of <img src="https://latex.coecogs.com/svg.latex?H"/>, the second derivative in that direction is given by the corresponding eigenvalue. For other directions of <img src="https://latex.coecogs.com/svg.latex?d"/>, the directional second derivative is a weighted average of all of the eigenvalues, with weights between 0 and 1, and eigenvectors that have smaller angle with <img src="https://latex.coecogs.com/svg.latex?d"/> receiving more weight.

The second derivative tells us how well we can expect the gradient descent to perform. We can make a second-order Taylor-series approximation to the function <img src="https://latex.coecogs.com/svg.latex?f(x)"/> around the current point <img src="https://latex.coecogs.com/svg.latex?x^{(0)}"/>:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?f(x) \approx f(x^{(0)}) + (x-x^{(0)})^T \boldsymbol{g} + \frac{1}{2}(x-x^{(0)})^T\boldsymbol{H}(x-x^{(0)})"/>
</p>

where <img src="https://latex.coecogs.com/svg.latex?\boldsymbol{g}"/> is the gradient and <img src="https://latex.coecogs.com/svg.latex?\boldsymbol{H}"/> is the Hessian at <img src="https://latex.coecogs.com/svg.latex?x^{(0)}"/>. If we use a learning rate of <img src="https://latex.coecogs.com/svg.latex?\epsilon"/>, then the new point will have the value:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?f(x-\epsilon\boldsymbol{g}) \approx f(x^{(0)}) -\epsilon \boldsymbol{g}^T\boldsymbol{g} + \frac{1}{2}\epsilon^2 \boldsymbol{g}^T \boldsymbol{H}\boldsymbol{g}"/>
</p>

There are three terms here:
* the original value of the function
* The expected improvement due to the slope of the function
* The correction we must apply to account for the curvature of the function.

When the last term is too large, the gradient descent step can actually move uphill. When the third term is zero or negative, the approximation predicts that increasing the learning rate forever will decrease <img src="https://latex.coecogs.com/svg.latex?f"/> forever. However, in practice, the Taylor series is unlikely to remain accurate for large <img src="https://latex.coecogs.com/svg.latex?\epsilon"/>, so one must resort to more heuristic choices of <img src="https://latex.coecogs.com/svg.latex?\epsilon"/> in this case.  However, when the last term is positive, solving for the optimal step size that decreases the Taylor series approximation of the function the most yields:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?\epsilon^{*} = \frac{g^Tg}{g^THg}"/>
</p>

### Tests for finding nature of critical points

Furthermore, the second derivative can be used to determine whether a critical point is a local maxima, a local minima or saddle point. Note that on a critical point, <img src="https://latex.codecogs.com/svg.latex?f'(x)=0"/>. So,
* If <img src="https://latex.coecogs.com/svg.latex?f''(x)>0"/>, then it is local minima.
* If <img src="https://latex.coecogs.com/svg.latex?f''(x)<0"/>, then it is local maxima.
* If <img src="https://latex.coecogs.com/svg.latex?f''(x)=0"/>, then the test is inconclusive. The point could be a saddle point, or a part of a flat region.

In multiple dimensions, we need to examine all of the second derivatives of the function. At a critical point, where <img src="https://latex.codecogs.com/svg.latex?\bigtriangledown _x f(x) = 0"/>, we can examine the eigenvalues of the Hessian to determine whether the critical point is a local maxima, local minima or saddle point.
* When the Hessian is positive definite (all its eigenvectors are positive), the point is a local minima.
* When the Hessian is negative definite (all its eigenvectors are negative), the point is a local maxima.
* When atleast one positive and negative eigenvector exists of a given Hessian, there is a possibility of a saddle point, but the test remains inconclusive.

### Limitations of Gradient Descent, and Proposed Solution

In multiple dimensions, there is a different second derivative for each direction at a single point. The **condition number** of the Hessian at this point measures how much the second derivatives differ from each other. When the Hessian has a poor condition number, gradient descent performs poorly. This is because in one direction, the derivative increases rapidly, while in another direction, it increases slowly. Gradient descent is unaware of this change in derivative so it does not know that it needs to explore preferentially in the direction where the derivative remains negative for longer.

However, this issue can be resolved by using information from the Hessian matrix to guide the search. The simplest method for doing so is known as **Newton's method**. Newton's method is based on using a second-order Taylor series expansion to approximate <img src="https://latex.coecogs.com/svg.latex?f(x)"/> near some point <img src="https://latex.coecogs.com/svg.latex?x^{(0)}"/>. We update <img src="https://latex.coecogs.com/svg.latex?x"/> such that:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?x^* = x^{(0)} - \boldsymbol{H}(f)(x^{(0)})^{-1}\bigtriangledown _x f(x^{(0)})"/>
</p>

* When <img src="https://latex.coecogs.com/svg.latex?f"/> is a positive definite quadratic equation, Newton's method jumps directly to the minimum of the function.
* When <img src="https://latex.coecogs.com/svg.latex?f"/> is not truly quadratic but can be locally approximated as a postive definite quadratic, applying Newton's method multiple times helps reach the critical point much faster than gradient descent.

This is a very useful property near a local minima, but can become highly harmful when near a saddle point. Newton's method is only appropriate when the nearby critical point is a minimum, whereas gradient descent is not attracted to saddle points unless gradient points towards them.

Optimization algorithms that use only the gradient such as gradient descent are called **first-order optimization algorithms**. Optimization algorithms that also use the Hessian matrix, such as Newton's method are called **second-order optimization algorithms**.

### Lipschitz continuous functions

In the context of deep learning, we sometimes gain some guarantees by restricting ourselves to functions that are either **Lipschitz continuous** or have Lipschitz continuous derivatives. A Lipschitz continuous function is a function <img src="https://latex.coecogs.com/svg.latex?f"/> whose rate of change is bounded by a **Lipschitz constant** <img src="https://latex.coecogs.com/svg.latex?\mathcal{L}"/>:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?\forall x, \forall y, \left | f(x)-f(y) \right | \leq \mathcal{L}\left \| x-y \right \|_2"/>
</p>

This property is useful because it allows us to quantify our assumption that a small change in the input made by an algorithm such as gradient descent will have a small change in the output. Lipschitz continuity is also a fairly weak constraint, and many optimization problems in deep learning can be made Lipschitz continuous with relatively minor modifications.

### Convex Optimization

Perhaps the most successful field of specialized optimization is **convex optimization**. Convex optimization algorithms are able to provide many more guarantees by making stronger restrictions. Convex optimization algorithms are applicable only to convex functions - functions for which the Hessian is positive semidefinite everywhere. Such functions are well-behaved because they lack saddle points and all of their local minima are necessarily global minima. However, most problems in deep learning are difficult to express in terms of convex optimization.

## Constrained Optimization

Sometimes we wish not only to maximize or minimize a function <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" />  over all possible values of <img src="https://latex.codecogs.com/svg.latex?x" title="x" />. Instead, we may mish to find the maximal or minimal value of <img src="https://latex.codecogs.com/svg.latex?f(x)" title="f(x)" /> for values of <img src="https://latex.codecogs.com/svg.latex?x" title="x" /> in some set <img src="https://latex.codecogs.com/svg.latex?\mathcal{S}" title="\mathcal{S}" />. This is known as constrained optimization. Points <img src="https://latex.codecogs.com/svg.latex?x"> that lie within the set <img src="https://latex.codecogs.com/svg.latex?\mathcal{S}" title="\mathcal{S}" /> are called feasible points in constrained optimization terminology. We often wish to find a solution that is small in some sense. A common approach in such situations is to impose a norm constraint, such as  <img src="https://latex.codecogs.com/svg.latex?\left \| x \right \| \leq 1"/>.
There are few approaches to help solve this problem:

### Approach \#1:

A simple approach is to simply modify gradient descent taking the constraint into account. If we use a small constant step size <img src="https://latex.codecogs.com/svg.latex?\epsilon"/>, we can make gradient descent steps, then project the result back into <img src="https://latex.codecogs.com/svg.latex?\mathcal{S}"/>. If we use a line search, we can only search over step sizes <img src="https://latex.codecogs.com/svg.latex?\epsilon"/> that yield new <img src="https://latex.codecogs.com/svg.latex?x"/> <img src="https://latex.coecogs.com/svg.latex?x"/> points that are feasible, or we can project each point on the line back into the constraint region. When possible, this method can be made more efficient by projecting the gradient into the tangent space of the feasible region before taking the step.

### Approach \#2:

A more sophisticated approach is to design a different, unconstrained optimization problem whose solutions can be converted into a solution to the original constrained optimization problem. For example, if we want to minimize <img src="https://latex.coecogs.com/svg.latex?f(x)"/> for <img src="https://latex.coecogs.com/svg.latex?x\in\mathbb{R}^2"/> with <img src="https://latex.coecogs.com/svg.latex?x"/> constrained to have exactly unit <img src="https://latex.coecogs.com/svg.latex?L^2"/> norm, we can instead minimize <img src="https://latex.coecogs.com/svg.latex?g(\theta)=f([\cos\theta, \sin\theta]^T)"/> with respect to <img src="https://latex.coecogs.com/svg.latex?\theta"/>, then return <img src="https://latex.coecogs.com/svg.latex?[\cos\theta, \sin\theta]"/> as the solution to the original problem. This approach is less generic and requires creativity for each case we encounter.

### Approach \#3:

The **Karush-Kuhn-Tucker** (KKT) approach provides a more general solution to constrained optimization. This approach introduces the lagrange optimization procedure. Suppose we want a description of <img src="https://latex.coecogs.com/svg.latex?\mathcal{S}"/> in terms of <img src="https://latex.coecogs.com/svg.latex?m"/> equality constraints <img src="https://latex.coecogs.com/svg.latex?g^{(i)}"/> nd <img src="https://latex.coecogs.com/svg.latex?n"/> inequality constraints <img src="https://latex.coecogs.com/svg.latex?h^{(j)}"/>. To optimize this equation, we introduce new variables <img src="https://latex.coecogs.com/svg.latex?\lambda_i"/> and <img src="https://latex.coecogs.com/svg.latex?\alpha_j"/> for each constraint. These variables are called the KKT multipliers. The generalized Lagrangian is then defined as:

<p align="center">
<img src="https://latex.coecogs.com/svg.latex?L(x,\lambda,\alpha) = f(x) + \sum_{i}\lambda_ig^{(i)}(x) + \sum_j \alpha_jh^{(j)}(x)"/>
</p>

With this simple conversion, we can solve the constrained optimization problem similar to our unconstrained optimization problem. A simple set of properties describe the optimal points of constrained optimization problems. These properties are called the Karush-Kuhn-Tucker (KKT) conditions. They are necessary conditions, but not always sufficient conditions for a point to be optimal. The conditions are:
* The gradient of the generalized Lagrangian is zero.
* All constraints on both <img src="https://latex.coecogs.com/svg.latex?x"/. and the KKT multipliers are satisfied.
* The inequality constraints exhibit "complementary slackness", i.e. <img src="https://latex.coecogs.com/svg.latex?\alpha \odot h(x) = 0"/>.
