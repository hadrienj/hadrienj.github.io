---
bg: "tools.jpg"
layout: post
mathjax: true
title:  "Deep learning book - Linear algebra"
crawlertitle: "Linear algebra"
summary: "Linear algebra: notes on the Goodfellow book."
date:   2016-07-29 20:09:47 +0700
categories: posts
tags: ['deep-learning']
author: hadrien
---

Linear algebra: notes on the Goodfellow book.

# 2.1 Scalars, Vectors, Matrices and Tensors

- A scalar is a single number
- A vector is an array of numbers. Geometrically, each value can be thought of a different axis.

$$
\boldsymbol{x} =\begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    \cdots \\\\
    x_n
\end{bmatrix}
$$

- A matrix is a 2-D array

$$
\boldsymbol{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2} & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & A_{2,n} \\\\
    \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & A_{m,n}
\end{bmatrix}
$$

- A tensor is a n-D array with n>2

### Transpose

The transpose $A^{\text{T}}$ of the matrix A corresponds to the mirrored axes.

$$
\boldsymbol{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}
$$

$$
\boldsymbol{A}^{\text{T}}=
\begin{bmatrix}
    A_{1,1} & A_{2,1} & A_{3,1} \\\\
    A_{1,2} & A_{2,2} & A_{3,2} \\\\
\end{bmatrix}
$$

### Addition

Matrices can be added if they have the same shape

$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}+
\begin{bmatrix}
    B_{1,1} & B_{1,2} \\\\
    B_{2,1} & B_{2,2} \\\\
    B_{3,1} & B_{3,2}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} + B_{1,1} & A_{1,2} + B_{1,2} \\\\
    A_{2,1} + B_{2,1} & A_{2,2} + B_{2,2} \\\\
    A_{3,1} + B_{3,1} & A_{3,2} + B_{3,2}
\end{bmatrix}
$$

$$\boldsymbol{A}_{i,j} + \boldsymbol{B}_{i,j} = \boldsymbol{C}_{i,j}$$

It is also possible to add a scalar to a matrix

$$
\alpha+ \begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}=
\begin{bmatrix}
    \alpha + A_{1,1} & \alpha + A_{1,2} \\\\
    \alpha + A_{2,1} & \alpha + A_{2,2} \\\\
    \alpha + A_{3,1} & \alpha + A_{3,2}
\end{bmatrix}
$$

### Broadcasting

$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}+
\begin{bmatrix}
    B_{1,1} \\\\
    B_{2,1} \\\\
    B_{3,1}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} + B_{1,1} & A_{1,2} + B_{1,1} \\\\
    A_{2,1} + B_{2,1} & A_{2,2} + B_{2,1} \\\\
    A_{3,1} + B_{3,1} & A_{3,2} + B_{3,1}
\end{bmatrix}
$$

is equivalent to

$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}+
\begin{bmatrix}
    B_{1,1} & B_{1,1} \\\\
    B_{2,1} & B_{2,1} \\\\
    B_{3,1} & B_{3,1}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} + B_{1,1} & A_{1,2} + B_{1,1} \\\\
    A_{2,1} + B_{2,1} & A_{2,2} + B_{2,1} \\\\
    A_{3,1} + B_{3,1} & A_{3,2} + B_{3,1}
\end{bmatrix}
$$

where the ($3 \times 1$) matrix is converted to the right shape ($3 \times 2$) by copying the first column.

# 2.2 Multiplying Matrices and Vectors

The matrix product is the sum of the rows of the first matrix and the columns of the second one. Thus, if the shape of the first matrix is ($m \times n$) the second matrix need to be of shape ($n \times x$).

### Example 1. $\boldsymbol{A} _{3,2} \times \boldsymbol{B} _{2, 1} = \boldsymbol{C} _{3, 1}$

$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}\times
\begin{bmatrix}
    B_{1,1} \\\\
    B_{2,1}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} \times B_{1,1} + A_{1,2} \times B_{2,1} \\\\
    A_{2,1} \times B_{1,1} + A_{2,2} \times B_{2,1} \\\\
    A_{3,1} \times B_{1,1} + A_{3,2} \times B_{2,1}
\end{bmatrix}
$$

### Example 2. $\boldsymbol{A} _{4,3} \times \boldsymbol{B} _{3, 2} = \boldsymbol{C} _{4, 2}$

$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} & A_{1,3} \\\\
    A_{2,1} & A_{2,2} & A_{2,3} \\\\
    A_{3,1} & A_{3,2} & A_{3,3} \\\\
    A_{4,1} & A_{4,2} & A_{4,3}
\end{bmatrix}\times
\begin{bmatrix}
    B_{1,1} & B_{1,2} \\\\
    B_{2,1} & B_{2,2} \\\\
    B_{3,1} & B_{3,2}
\end{bmatrix}=
\begin{bmatrix}
    A_{1,1} \times B_{1,1} + A_{1,2} \times B_{2,1} + A_{1,3} \times B_{3,1} & A_{1,1} \times B_{1,2} + A_{1,2} \times B_{2,2} + A_{1,3} \times B_{3,2} \\\\
    A_{2,1} \times B_{1,1} + A_{2,2} \times B_{2,1} + A_{2,3} \times B_{3,1} & A_{2,1} \times B_{1,2} + A_{2,2} \times B_{2,2} + A_{2,3} \times B_{3,2} \\\\
    A_{3,1} \times B_{1,1} + A_{3,2} \times B_{2,1} + A_{3,3} \times B_{3,1} & A_{3,1} \times B_{1,2} + A_{3,2} \times B_{2,2} + A_{3,3} \times B_{3,2} \\\\
    A_{4,1} \times B_{1,1} + A_{4,2} \times B_{2,1} + A_{4,3} \times B_{3,1} & A_{4,1} \times B_{1,2} + A_{4,2} \times B_{2,2} + A_{4,3} \times B_{3,2} \\\\
\end{bmatrix}
$$

So the dot product can be formalized like that:

$$
C_{i,j} = A_{i,k}B_{k,j} = \sum\limits_{k}A_{i,k}B_{k,i}
$$

More detailed examples can be found [here](https://www.mathsisfun.com/algebra/matrix-multiplying.html).

## Matrices mutliplication is distributive

$\boldsymbol{A}(\boldsymbol{B}+\boldsymbol{C}) = \boldsymbol{AB}+\boldsymbol{AC}$

#### Example 1.

$
\boldsymbol{A}=\begin{bmatrix}
    2 & 3 \\\\
    1 & 4 \\\\
    7 & 6
\end{bmatrix},
\boldsymbol{B}=\begin{bmatrix}
    5 \\\\
    2
\end{bmatrix},
\boldsymbol{C}=\begin{bmatrix}
    4 \\\\
    3
\end{bmatrix}
$


$
\boldsymbol{A}(\boldsymbol{B}+\boldsymbol{C})=\begin{bmatrix}
    2 & 3 \\\\
    1 & 4 \\\\
    7 & 6
\end{bmatrix}\times
\left(\begin{bmatrix}
    5 \\\\
    2
\end{bmatrix}+
\begin{bmatrix}
    4 \\\\
    3
\end{bmatrix}\right)=
\begin{bmatrix}
    2 & 3 \\\\
    1 & 4 \\\\
    7 & 6
\end{bmatrix}\times
\begin{bmatrix}
    9 \\\\
    5
\end{bmatrix}=
\begin{bmatrix}
    2 \times 9 + 3 \times 5 \\\\
    1 \times 9 + 4 \times 5 \\\\
    7 \times 9 + 6 \times 5
\end{bmatrix}=
\begin{bmatrix}
    33 \\\\
    29 \\\\
    93
\end{bmatrix}
$

is equivalent to

$
\boldsymbol{A}\boldsymbol{B}+\boldsymbol{A}\boldsymbol{C} = \begin{bmatrix}
    2 & 3 \\\\
    1 & 4 \\\\
    7 & 6
\end{bmatrix}\times
\begin{bmatrix}
    5 \\\\
    2
\end{bmatrix}+
\begin{bmatrix}
    2 & 3 \\\\
    1 & 4 \\\\
    7 & 6
\end{bmatrix}\times
\begin{bmatrix}
    4 \\\\
    3
\end{bmatrix}=
\begin{bmatrix}
    2 \times 5 + 3 \times 2 \\\\
    1 \times 5 + 4 \times 2 \\\\
    7 \times 5 + 6 \times 2
\end{bmatrix}+
\begin{bmatrix}
    2 \times 4 + 3 \times 3 \\\\
    1 \times 4 + 4 \times 3 \\\\
    7 \times 4 + 6 \times 3
\end{bmatrix}=
\begin{bmatrix}
    16 \\\\
    13 \\\\
    47
\end{bmatrix}+
\begin{bmatrix}
    17 \\\\
    16 \\\\
    46
\end{bmatrix}=
\begin{bmatrix}
    33 \\\\
    29 \\\\
    93
\end{bmatrix}
$


### System of linear equations

It is possible to write a system of multiple linear equations under the form of a matrix. The follwing system:

$$
A_{1,1}x_1 + A_{1,2}x_2 + A_{1,n}x_n = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2 + A_{2,n}x_n = b_2 \\\\
\cdots \\\\
A_{m,1}x_1 + A_{m,2}x_2 + A_{m,n}x_n = b_n
$$

can simply be wrote $\boldsymbol{Ax} = \boldsymbol{b}$.

with $\boldsymbol{A}$ being a matrix, $\boldsymbol{b}$ and $\boldsymbol{x}$ vectors (with $\boldsymbol{x}$ unknown):

$
\boldsymbol{A} =
\begin{bmatrix}
    A_{1,1} & A_{1,2} & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & A_{2,n} \\\\
    \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & A_{m,n}
\end{bmatrix}
$

$
\boldsymbol{x} =
\begin{bmatrix}
    x_{1} \\\\
    x_{2} \\\\
    \cdots \\\\
    x_{n}
\end{bmatrix}
$

$
\boldsymbol{b} =
\begin{bmatrix}
    b_{1} \\\\
    b_{2} \\\\
    \cdots \\\\
    b_{n}
\end{bmatrix}
$

The equation is:

$
\begin{bmatrix}
    A_{1,1} & A_{1,2} & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & A_{2,n} \\\\
    \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & A_{m,n}
\end{bmatrix}
\times
\begin{bmatrix}
    x_{1} \\\\
    x_{2} \\\\
    \cdots \\\\
    x_{n}
\end{bmatrix}=
\begin{bmatrix}
    b_{1} \\\\
    b_{2} \\\\
    \cdots \\\\
    b_{n}
\end{bmatrix}
$

The dot product of ***A*** and ***x*** gives:

$
\begin{bmatrix}
    A_{1,1}x_1 + A_{1,2}x_2 + A_{1,n}x_n \\\\
    A_{2,1}x_1 + A_{2,2}x_2 + A_{2,n}x_n \\\\
    \cdots \\\\
    A_{m,1}x_1 + A_{m,2}x_2 + A_{m,n}x_n
\end{bmatrix}=
\begin{bmatrix}
    b_{1} \\\\
    b_{2} \\\\
    \cdots \\\\
    b_{n}
\end{bmatrix}
$

# 2.3 Identity and Inverse Matrices

### Identity matrices

The identity matrix $\boldsymbol{I}_n$ is a matrix of shape ($n \times n$) that has all 0 except the diagonal filled with 1.

$
\boldsymbol{I}_3=
\begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & 1 & 0 \\\\
    0 & 0 & 1 \\\\
\end{bmatrix}
$

When multiplied with a vector the result is this same vector:

$\boldsymbol{I}_n\boldsymbol{x} = \boldsymbol{x}$

### Example

$
\begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & 1 & 0 \\\\
    0 & 0 & 1 \\\\
\end{bmatrix}
\times
\begin{bmatrix}
    x_{1} \\\\
    x_{2} \\\\
    x_{3}
\end{bmatrix}=
\begin{bmatrix}
    1 \times x_1 + 0 \times x_2 + 0\times x_3 \\\\
    0 \times x_1 + 1 \times x_2 + 0\times x_3 \\\\
    0 \times x_1 + 0 \times x_2 + 1\times x_3 \\\\
\end{bmatrix}=
\begin{bmatrix}
    x_{1} \\\\
    x_{2} \\\\
    x_{3}
\end{bmatrix}
$

### Inverse Matrices

The matrix inverse of $\boldsymbol{A}$ is denoted $\boldsymbol{A}^{-1}$ and correspond to the matrix that result in the identity matrix when it is multiplied by $\boldsymbol{A}$:

$\boldsymbol{A}^{-1}\boldsymbol{A}=\boldsymbol{I}_n$

### Example

$
\begin{align*}
\boldsymbol{A}^{-1}\boldsymbol{A} &=
\begin{bmatrix}
    3 & 0 & 2 \\\\
    2 & 0 & -2 \\\\
    0 & 1 & 1 \\\\
\end{bmatrix} \times
\begin{bmatrix}
    0.2 & 0.2 & 0 \\\\
    -0.2 & 0.3 & 1 \\\\
    0.2 & -0.3 & 0 \\\\
\end{bmatrix}\\\\
&= \begin{bmatrix}
    3\times0.2 + 0\times-0.2 + 2\times0.2 & 3\times0.2 + 0\times0.3 + 2\times-0.3  & 3\times0 + 0\times1 + 2\times0  \\\\
    2\times0.2 + 0\times-0.2 + -2\times0.2 & 2\times0.2 + 0\times0.3 + -2\times-0.3  & 2\times0 + 0\times1 + -2\times0  \\\\
    0\times0.2 + 1\times-0.2 + 1\times0.2 & 0\times0.2 + 1\times0.3 + 1\times-0.3  & 0\times0 + 1\times1 + 1\times0  \\\\
\end{bmatrix}\\\\
&= \begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & 1 & 0 \\\\
    0 & 0 & 1 \\\\
\end{bmatrix}
\end{align*}
$

The inverse matrix can be used to solve the equation $\boldsymbol{Ax}=\boldsymbol{b}$ by adding it to each term:

$\boldsymbol{A}^{-1}\boldsymbol{Ax}=\boldsymbol{A}^{-1}\boldsymbol{b}$

$\boldsymbol{I}_n\boldsymbol{x}=\boldsymbol{A}^{-1}\boldsymbol{b}$

$\boldsymbol{x}=\boldsymbol{A}^{-1}\boldsymbol{b}$

# 2.4 Linear Dependence and Span

To find the solution of the equation $\boldsymbol{Ax}=\boldsymbol{b}$ when we know $A_{1,1}...A_{m,n}$ and $b_1...b_n$ corresponds to find the values of each variables $x_1...x_n$ in this linear system:

$$
A_{1,1}x_1 + A_{1,2}x_2 + A_{1,n}x_n = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2 + A_{2,n}x_n = b_2 \\\\
\cdots \\\\
A_{m,1}x_1 + A_{m,2}x_2 + A_{m,n}x_n = b_n
$$

### Graphical view

Graphically, a linear system of equations can be viewed as a set of *m* *n-1*-D hyperplanes in a *n*-D space. With this notation, we have *m* equations and *n* variables. Let's see some examples to clarify this.

#### Example 1. *m*=1, *n*=2

The graphical interpretation of *n*=2 is that we have a 2-D space, so we can represent it with 2 axes.

Since our hyperplanes are of *n-1* dimensions, we have 1-D hyperplanes. So our hyperplanes are lines.

As *m*=1, we have only one equation in our linear system.

Binding these information, our linear system can be graphically represented with one line in a 2-D space.












