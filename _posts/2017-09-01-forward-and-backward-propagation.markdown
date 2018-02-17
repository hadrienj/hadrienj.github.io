---
bg: "tools.jpg"
layout: post
mathjax: true
title:  "Forward and backward propagation"
crawlertitle: "Linear algebra"
summary: "Deep learning: Forward and backward propagation."
date:   2017-09-01 20:09:47 +0700
categories: posts
tags: ['deep-learning']
author: hadrien
---

Derivative of $\mathcal{L}(a, y)$ and the chain rules.

$\frac{dJ}{dw}$ shapes

$\frac{dJ}{db}$ shapes

# Forward propagation

## Linear model

$z = w1x1 + w2x2 + b$

### Vectorization

$z = w^TX + b$

with $w^TX$ calculated with the following dot product:

$
\begin{bmatrix}
    w_{0} & w_{1} & \dots & w_{n}
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{00} & x_{01}  & \dots & x_{0m} \\\
x_{10} & x_{11} & \dots & x_{1m} \\\
\dots & \dots & \dots & \dots \\\
x_{n0} & x_{n1} & \dots & x_{nm}
\end{bmatrix}
= \begin{bmatrix}
    w_{0}x_{00} & w_{0}x_{01} & \dots & w_{0}x_{0m} \\\
    + & + & \dots & + \\\
    w_{1}x_{10} & w_{1}x_{11} & \dots & w_{1}x_{1m} \\\
    + & + & \dots & + \\\
    \dots & \dots & \dots & \dots \\\
    + & + & \dots & + \\\
    w_{n}x_{n0} & w_{n}x_{n1} & \dots & w_{n}x_{nm} \\\
\end{bmatrix}
$
## Logistic model

$a = \sigma(z) = \frac{1}{1+e^{-z}}$

## Loss function

$\mathcal{L}(a, y) = -(ylog(a) + (1-y)log(1-a))$


# Backward propagation

We can derive the loss function with the chain rule.

#### 1. Derivative of $\mathcal{L}(y, a)$ in respect of a

$$\frac{d\mathcal{L}(a, y)}{da} = -\frac{y}{a} + \frac{1-y}{1-a}$$

because

$$\frac{d}{da}ylog(a) = \frac{y}{a}$$

$$\frac{d}{da}1-a = -1$$

$$\frac{d}{da}log(1-a) = -1\times\frac{1}{1-a}$$ (chain rule)

#### 2. Derivative of $\sigma(z)$ in respect of z

The derivative of the sigmoid function can be expressed in function of the sigmoid function itself.

$$\frac{da}{dz} =a(1-a)$$

because

$$a = \sigma(z) = \frac{1}{1+e^{-z}}$$

$$\frac{d}{dz}(\frac{1}{z}) = -\frac{1}{z^2}$$

$$\frac{d}{dz}(1+e^{-z}) = -e^{-z}$$

$$\frac{da}{dz} = \frac{e^{-z}}{(1+e^{-z})^2}$$

This expression can be simplified by adding 1 and -1:

$$\frac{da}{dz} = \frac{1+e^{-z}-1}{(1+e^{-z})^2} = \frac{1+e^{-z}}{(1+e^{-z})^2} - \frac{1}{(1+e^{-z})^2} = a-a^2 = a(1-a)$$

#### 3. Derivative of $\mathcal{L}(y, a)$ in respect of z

By combining 1. and 2. with the chain rule we have

$$\begin{align*}
\frac{d\mathcal{L}(a, y)}{dz} &= \frac{d\mathcal{L}(a, y)}{da} \cdot \frac{da}{dz} \\\\
&= (-\frac{y}{a} + \frac{1-y}{1-a}) \cdot a(1-a) \\\\
&= (-\frac{ya(1-a)}{a} + \frac{(1-y)a(1-a)}{1-a}) \\\\
&= -y(1-a) + a(1-y) \\\\
&= -y+ya+a-ay \\\\
&= a-y
\end{align*}$$

#### 4. Derivative of $\mathcal{L}(y, a)$ in respect of w

$$\begin{align*}
\frac{d\mathcal{L}(a, y)}{dw1} &= x1\frac{d}{da}\mathcal{L}(a, y) \\\\
&= x1(a-y))
\end{align*}$$

$$\begin{align*}
\frac{d\mathcal{L}(a, y)}{dw2} &= x2\frac{d}{da}\mathcal{L}(a, y) \\\\
&= x2(a-y))
\end{align*}$$

and so on...

#### 5. Derivative of the cost function J in respect of w

The cost function is the average of losses over the m training example:

$$J(a, y) = \frac{1}{m}\sum\limits_{i=1}^m\mathcal{L}(a, y)$$

So the derivative of the cost function in respect of w is the average over the m training example of the loss function derivatives:

$$\frac{dJ(a, y)}{dw} = \frac{1}{m}\sum\limits_{i=1}^m\frac{d\mathcal{L}(a, y)}{dw}$$

This can be vectorized like that

$$\frac{d\mathcal{L}(a, y)}{dw} = \frac{1}{m}X(A-Y)^T$$

$$
\frac{1}{m}
\cdot
\begin{bmatrix}
    x_{00} & x_{01}  & \dots & x_{0m} \\\\
    x_{10} & x_{11} & \dots & x_{1m} \\\\
    \dots & \dots & \dots & \dots \\\\
    x_{n0} & x_{n1} & \dots & x_{nm}
\end{bmatrix}
\cdot
\begin{bmatrix}
    (A_0-Y_0) \\\\
    (A_1-Y_1) \\\\
    \dots  \\\\
    (A_m-Y_m) \\\\
\end{bmatrix}
= \begin{bmatrix}
    \frac{1}{m}x_{00}(A_0-Y_0) + \frac{1}{m}x_{01}(A_1-Y_1) + \dots + \frac{1}{m}x_{0m}(A_m-Y_m) \\\\
    \frac{1}{m}x_{10}(A_0-Y_0) + \frac{1}{m}x_{11}(A_1-Y_1) + \dots + \frac{1}{m}x_{1m}(A_m-Y_m) \\\\
    \dots + \dots + \dots + \dots \\\\
    \frac{1}{m}x_{n0}(A_0-Y_0) + \frac{1}{m}x_{n1}(A_1-Y_1) + \dots + \frac{1}{m}x_{nm}(A_m-Y_m) \\\\
\end{bmatrix}
$$

**Shape of the dot product:**

X.shape = **(n,m)**

A.shape = Y.shape = **(1,m)**

(A-Y)^T.shape = **(m,1)**

X.(A-Y)^T.shape = **(n,m)**.**(m,1)** = **(n,1)**

#### 6. Derivative of the cost function J in respect of b

The derivative of the cost function in respect of b is

$$\frac{dJ(a, y)}{db} = \frac{1}{m}\sum\limits_{i=1}^m(a^{(i)}-y^{(i)})$$

# Optimization


Use gradient descent to update the parameters and find their value that minimize the cost, that is to say the difference between the prediction A and Y.

$$w = w - \alpha\frac{dJ(a, y)}{dw}$$

$$b = b - \alpha\frac{dJ(a, y)}{db}$$

$\alpha$ is the learning rate.


# Questions and remarks

- week3, activation functions, 04:30

The use of the sigmoid or tanh function have small gradient when z is very small or very large and that slow down the gradient descent.

The gradient descent is done on the cost function, so why it could slow it down?

The cost function is composed of the activation function and one part of the chain rule in the gradient descent process is to find the gradient of the activation function.

- week3, neural network overview

In the case of logistic regression (top chart), does X contains 3 features (n=3)?

- week3, ReLU function

What is the max function?

- week3, gradient descent for neural network, 02:10

The cost function depends of the wanted classification. For binary classification, the logistic cost function can be used as in logistic regression.




