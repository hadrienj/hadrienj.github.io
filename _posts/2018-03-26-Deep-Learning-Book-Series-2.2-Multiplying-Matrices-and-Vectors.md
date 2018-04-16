---
bg: "galoped.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.2 Multiplying Matrices and Vectors
crawlertitle: "Deep Learning Book Series · 2.2 Multiplying Matrices and Vectors"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.2%20Multiplying%20Matrices%20and%20Vectors/2.2%20Multiplying%20Matrices%20and%20Vectors.ipynb
date: 2018-03-26 11:00:00
excerpt: We will see some very important concepts in this chapter. The dot product is used in every equation explaining data science algorithms so it's worth the effort to understand it. Then we will see some properties of this operation. Finally, we will get some intuition on the link between matrices and systems of linear equations.
excerpt-image: <img src="../../assets/images/2.2/dot-product.png" width="400" alt="An example of how to calculate the dot product between a matrix and a vector" title="The dot product between a matrix and a vector">
  <em>The dot product between a matrix and a vector</em>
---

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

{% include deep-learning-book-toc.html %}

# Introduction

We will see some very important concepts in this chapter. The dot product is used in every equation explaining data science algorithms so it's worth the effort to understand it. Then we will see some properties of this operation. Finally, we will get some intuition on the link between matrices and systems of linear equations.

# 2.2 Multiplying Matrices and Vectors

The standard way to multiply matrices is not to multiply each element of one with each element of the other (this is the element-wise product) but to calculate the sum of the products between rows and columns. The matrix product, also called **dot product**, is calculated as following:

<img src="../../assets/images/2.2/dot-product.png" width="400" alt="An example of how to calculate the dot product between a matrix and a vector" title="The dot product between a matrix and a vector">
<em>The dot product between a matrix and a vector</em>

The number of columns of the first matrix must be equal to the number of rows of the second matrix. Thus, if the dimensions, or the shape of the first matrix, is ($m \times n$) the second matrix need to be of shape ($n \times x$). The resulting matrix will have the shape ($m \times x$).

### Example 1.

As a starter we will see the multiplication of a matrix and a vector.

<div>
$$\bs{A} \times \bs{b} = \bs{C}$$
</div>

with:

<div>
$$
\bs{A}=
\begin{bmatrix}
    1 & 2\\\\
    3 & 4\\\\
    5 & 6
\end{bmatrix}
$$
</div>

and:

<div>
$$
\bs{b}=\begin{bmatrix}
    2\\\\
    4
\end{bmatrix}
$$
</div>

We saw that the formula is the following:

<div>
$$
\begin{align*}
&\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\\
    A_{2,1} & A_{2,2} \\\\
    A_{3,1} & A_{3,2}
\end{bmatrix}\times
\begin{bmatrix}
    B_{1,1} \\\\
    B_{2,1}
\end{bmatrix}=\\\\
&\begin{bmatrix}
    A_{1,1}B_{1,1} + A_{1,2}B_{2,1} \\\\
    A_{2,1}B_{1,1} + A_{2,2}B_{2,1} \\\\
    A_{3,1}B_{1,1} + A_{3,2}B_{2,1}
\end{bmatrix}
\end{align*}
$$
</div>

So we will have:

<div>
$$
\begin{align*}
&\begin{bmatrix}
    1 & 2 \\\\
    3 & 4 \\\\
    5 & 6
\end{bmatrix}\times
\begin{bmatrix}
    2 \\\\
    4
\end{bmatrix}=\\\\
&\begin{bmatrix}
    1 \times 2 + 2 \times 4 \\\\
    3 \times 2 + 4 \times 4 \\\\
    5 \times 2 + 6 \times 4
\end{bmatrix}=
\begin{bmatrix}
    10 \\\\
    22 \\\\
    34
\end{bmatrix}
\end{align*}
$$
</div>

It is a good habit to check the dimensions of the matrix to see what is going on. We can see in this example that the shape of $\bs{A}$ is ($3 \times 2$) and the shape of $\bs{b}$ is ($2 \times 1$). So the dimensions of $\bs{C}$ are ($3 \times 1$).

### With Numpy

The Numpy function `dot()` can be used to compute the matrix product (or dot product). Let's try to reproduce the last exemple:


```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A
```

<pre class='output'>
array([[1, 2],
       [3, 4],
       [5, 6]])
</pre>



```python
B = np.array([[2], [4]])
B
```

<pre class='output'>
array([[2],
       [4]])
</pre>



```python
C = np.dot(A, B)
C
```

<pre class='output'>
array([[10],
       [22],
       [34]])
</pre>


It is equivalent to use the method `dot()` of Numpy arrays:


```python
C = A.dot(B)
C
```

<pre class='output'>
array([[10],
       [22],
       [34]])
</pre>


### Example 2.

Multiplication of two matrices.

<div>
$$\bs{A} \times \bs{B} = \bs{C}$$
</div>

with:

<div>
$$\bs{A}=\begin{bmatrix}
    1 & 2 & 3 \\\\
    4 & 5 & 6 \\\\
    7 & 8 & 9 \\\\
    10 & 11 & 12
\end{bmatrix}
$$
</div>

and:

<div>
$$\bs{B}=\begin{bmatrix}
    2 & 7 \\\\
    1 & 2 \\\\
    3 & 6
\end{bmatrix}
$$
</div>

So we have:

<div>
$$
\begin{align*}
&\begin{bmatrix}
    1 & 2 & 3 \\\\
    4 & 5 & 6 \\\\
    7 & 8 & 9 \\\\
    10 & 11 & 12
\end{bmatrix}\times
\begin{bmatrix}
    2 & 7 \\\\
    1 & 2 \\\\
    3 & 6
\end{bmatrix}=\\\\
&\begin{bmatrix}
    2 \times 1 + 1 \times 2 + 3 \times 3 & 7 \times 1 + 2 \times 2 + 6 \times 3 \\\\
    2 \times 4 + 1 \times 5 + 3 \times 6 & 7 \times 4 + 2 \times 5 + 6 \times 6 \\\\
    2 \times 7 + 1 \times 8 + 3 \times 9 & 7 \times 7 + 2 \times 8 + 6 \times 9 \\\\
    2 \times 10 + 1 \times 11 + 3 \times 12 & 7 \times 10 + 2 \times 11 + 6 \times 12 \\\\
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    13 & 29 \\\\
    31 & 74 \\\\
    49 & 119 \\\\
    67 & 164
\end{bmatrix}
\end{align*}
$$
</div>

Let's check the result with Numpy:


```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
A
```

<pre class='output'>
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [ 7,  8,  9],
       [10, 11, 12]])
</pre>



```python
B = np.array([[2, 7], [1, 2], [3, 6]])
B
```

<pre class='output'>
array([[2, 7],
       [1, 2],
       [3, 6]])
</pre>



```python
C = A.dot(B)
C
```

<pre class='output'>
array([[ 13,  29],
       [ 31,  74],
       [ 49, 119],
       [ 67, 164]])
</pre>


It works!

# Formalization of the dot product

<div>
$$
C_{i,j} = A_{i,k}B_{k,j} = \sum_{k}A_{i,k}B_{k,j}
$$
</div>

You can find more examples about the dot product [here](https://www.mathsisfun.com/algebra/matrix-multiplying.html).

# Properties of the dot product

We will now see some interesting properties of the matrix multiplication. It will become useful as we move forward in the chapters. Using simple examples for each property will provide a way to check them while we get used to the Numpy functions.

## Matrices mutliplication is distributive

<div>
$$\bs{A}(\bs{B}+\bs{C}) = \bs{AB}+\bs{AC}$$
</div>

### Example 3.

<div>
$$
\bs{A}=\begin{bmatrix}
    2 & 3 \\\\
    1 & 4 \\\\
    7 & 6
\end{bmatrix},
\bs{B}=\begin{bmatrix}
    5 \\\\
    2
\end{bmatrix},
\bs{C}=\begin{bmatrix}
    4 \\\\
    3
\end{bmatrix}
$$
</div>


<div>
$$
\begin{align*}
\bs{A}(\bs{B}+\bs{C})&=\begin{bmatrix}
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
\end{bmatrix}\\\\
&=
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
\end{align*}
$$
</div>

is equivalent to

<div>
$$
\begin{align*}
\bs{A}\bs{B}+\bs{A}\bs{C} &= \begin{bmatrix}
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
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    2 \times 5 + 3 \times 2 \\\\
    1 \times 5 + 4 \times 2 \\\\
    7 \times 5 + 6 \times 2
\end{bmatrix}+
\begin{bmatrix}
    2 \times 4 + 3 \times 3 \\\\
    1 \times 4 + 4 \times 3 \\\\
    7 \times 4 + 6 \times 3
\end{bmatrix}\\\\
&=
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
\end{align*}
$$
</div>


```python
A = np.array([[2, 3], [1, 4], [7, 6]])
A
```

<pre class='output'>
array([[2, 3],
       [1, 4],
       [7, 6]])
</pre>



```python
B = np.array([[5], [2]])
B
```

<pre class='output'>
array([[5],
       [2]])
</pre>



```python
C = np.array([[4], [3]])
C
```

<pre class='output'>
array([[4],
       [3]])
</pre>


$\bs{A}(\bs{B}+\bs{C})$:


```python
D = A.dot(B+C)
D
```

<pre class='output'>
array([[33],
       [29],
       [93]])
</pre>


is equivalent to $\bs{AB}+\bs{AC}$:


```python
D = A.dot(B) + A.dot(C)
D
```

<pre class='output'>
array([[33],
       [29],
       [93]])
</pre>


## Matrices mutliplication is associative

<div>
$$\bs{A}(\bs{BC}) = (\bs{AB})\bs{C}$$
</div>



```python
A = np.array([[2, 3], [1, 4], [7, 6]])
A
```

<pre class='output'>
array([[2, 3],
       [1, 4],
       [7, 6]])
</pre>



```python
B = np.array([[5, 3], [2, 2]])
B
```

<pre class='output'>
array([[5, 3],
       [2, 2]])
</pre>


$\bs{A}(\bs{BC})$:



```python
D = A.dot(B.dot(C))
D
```

<pre class='output'>
array([[100],
       [ 85],
       [287]])
</pre>


is equivalent to $(\bs{AB})\bs{C}$:


```python
D = (A.dot(B)).dot(C)
D
```

<pre class='output'>
array([[100],
       [ 85],
       [287]])
</pre>


## Matrix multiplication is not commutative

<div>
$$\bs{AB} \neq \bs{BA}$$
</div>


```python
A = np.array([[2, 3], [6, 5]])
A
```

<pre class='output'>
array([[2, 3],
       [6, 5]])
</pre>



```python
B = np.array([[5, 3], [2, 2]])
B
```

<pre class='output'>
array([[5, 3],
       [2, 2]])
</pre>


$\bs{AB}$:


```python
AB = np.dot(A, B)
AB
```

<pre class='output'>
array([[16, 12],
       [40, 28]])
</pre>


is different from $\bs{BA}$:


```python
BA = np.dot(B, A)
BA
```

<pre class='output'>
array([[28, 30],
       [16, 16]])
</pre>


## However vector multiplication is commutative

<div>
$$\bs{x^{ \text{T}}y} = \bs{y^{\text{T}}x} $$
</div>


```python
x = np.array([[2], [6]])
x
```

<pre class='output'>
array([[2],
       [6]])
</pre>



```python
y = np.array([[5], [2]])
y
```

<pre class='output'>
array([[5],
       [2]])
</pre>


$\bs{x^\text{T}y}$:


```python
x_ty = x.T.dot(y)
x_ty
```

<pre class='output'>
array([[22]])
</pre>


is equivalent to $\bs{y^\text{T}x}$:


```python
y_tx = y.T.dot(x)
y_tx
```

<pre class='output'>
array([[22]])
</pre>


## Simplification of the matrix product

<div>
$$(\bs{AB})^{\text{T}} = \bs{B}^\text{T}\bs{A}^\text{T}$$
</div>


```python
A = np.array([[2, 3], [1, 4], [7, 6]])
A
```

<pre class='output'>
array([[2, 3],
       [1, 4],
       [7, 6]])
</pre>



```python
B = np.array([[5, 3], [2, 2]])
B
```

<pre class='output'>
array([[5, 3],
       [2, 2]])
</pre>


$(\bs{AB})^{\text{T}}$:


```python
AB_t = A.dot(B).T
AB_t
```

<pre class='output'>
array([[16, 13, 47],
       [12, 11, 33]])
</pre>


is equivalent to $\bs{B}^\text{T}\bs{A}^\text{T}$:


```python
B_tA = B.T.dot(A.T)
B_tA
```

<pre class='output'>
array([[16, 13, 47],
       [12, 11, 33]])
</pre>


# System of linear equations

This is an important part of why linear algebra can be very useful to solve variety of problems. Here we will see that it can be use to represent system of equations.

A system of equations is a set of multiple equations (at least 1). For instance we could have:

<div>
$$
\begin{cases}
y = 2x + 1 \\\\
y = \frac{7}{2}x +3
\end{cases}
$$
</div>

A system of equations is defined by its number of equations and its number of unknowns. In our example above, the system has 2 equations and 2 unknowns ($x$ and $y$). In addition we call this a system of **linear** equations because each equations is linear. It is easy to see that in 2 dimensions: we will have one straight line per equation and the dimensions are the unknowns. Here is the plot of the first one:

<img src="../../assets/images/2.2/plot-linear-equation.png" width="300" alt="Representation of a line from an equation" title="Plot of a linear equation">
<em>Representation of a linear equation</em>

<span class='pquote'>
    In our system of equations, the unknowns are the dimensions and the number of equations is the number of lines (in 2D) or $n$-dimensional planes.
</span>

## Using matrices to describe the system

Matrices can be used to describe a system of linear equations of the form $\bs{Ax}=\bs{b}$. Here is such a system:

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 + A_{1,n}x_n = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2 + A_{2,n}x_n = b_2 \\\\
\cdots \\\\
A_{m,1}x_1 + A_{m,2}x_2 + A_{m,n}x_n = b_n
$$
</div>

The unknowns (what we want to find to solve the system) are the variables $x_1$ and $x_2$ corresponding to the previous variables $x$ and $y$. It is exactly the same form as with the last example but with all the variables on the same side. $y = 2x + 1$ becomes $-2x + y = 1$ with $x$ corresponding to $x_1$ and $y$ corresponding to $x_2$. We will have $n$ unknowns and $m$ equations.

The variables are named $x_1, x_2, \cdots, x_n$ by convention because we will see that it can be summarised in the vector $\bs{x}$.

### Left hand side

The left hand term can considered as the product of a matrix $\bs{A}$ containing weights for each variable ($n$ columns) and each equation ($m$ rows):

<div>
$$
\bs{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\\\
    \cdots & \cdots & \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
$$
</div>

with a vector $\bs{x}$ containing the $n$ unknowns

<div>
$$
\bs{x}=
\begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    \cdots \\\\
    x_n
\end{bmatrix}
$$
</div>

The dot product of $\bs{A}$ and $\bs{x}$ gives a set of equations. Here is a simple example:

<img src="../../assets/images/2.2/system-linear-equations-matrix-form.png" width="400" alt="Matrix form of a system of linear equation" title="Matrix form of a system of linear equation">
<em>Matrix form of a system of linear equations</em>

We have a set of two equations with two unknowns. So the number of rows of $\bs{A}$ gives the number of equations and the number of columns gives the number of unknowns.

### Both sides

The equation system can be wrote like that:

<div>
$$
\begin{bmatrix}
    A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\\\
    \cdots & \cdots & \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
\times
\begin{bmatrix}
    x_1 \\\\
    x_2 \\\\
    \cdots \\\\
    x_n
\end{bmatrix}
=
\begin{bmatrix}
    b_1 \\\\
    b_2 \\\\
    \cdots \\\\
    b_m
\end{bmatrix}
$$
</div>

Or simply:

<div>
$$\bs{Ax}=\bs{b}$$
</div>

### Example 4.

We will try to convert the common form of a linear equation $y=ax+b$ to the matrix form. If we want to keep the previous notation we will have instead:

<div>
$$x_2=ax_1+b$$
</div>

Don't confuse the variable $x_1$ and $x_2$ with the vector $\bs{x}$. This vector contains actually all the variables of our equations. Here we have:

<div>
$$
\bs{x} =
\begin{bmatrix}
    x_1 \\
    x_2
\end{bmatrix}
$$
</div>

In this example we will use the following equation:

<div>
$$
\begin{align*}
&x_2=2x_1+1\\\\
\Leftrightarrow& 2x_1-x_2=-1
\end{align*}
$$
</div>

In order to end up with this system when we multiply $\bs{A}$ and $\bs{x}$ we need $\bs{A}$ to be a matrix containing the weights of each variable. The weight of $x_1$ is $2$ and the weights of $x_2$ is $-1$:

<div>
$$
\bs{A}=
\begin{bmatrix}
    2 & -1
\end{bmatrix}
$$
</div>

So we have

<div>
$$
\begin{bmatrix}
    2 & -1
\end{bmatrix}
\begin{bmatrix}
    x_1 \\
    x_2
\end{bmatrix}
=
\begin{bmatrix}
2x_1-1x_2
\end{bmatrix}
$$
</div>

To complete the equation we have

<div>
$$
\bs{b}=
\begin{bmatrix}
    -1
\end{bmatrix}
$$
</div>

which gives

<div>
$$
\begin{bmatrix}
    2 & -1
\end{bmatrix}
\begin{bmatrix}
    x_1 \\
    x_2
\end{bmatrix}
=
\begin{bmatrix}
    -1
\end{bmatrix}
$$
</div>

This system of equations is thus very simple and contains only 1 equation ($\bs{A}$ has 1 row) and 2 variables ($\bs{A}$ has 2 columns).

To summarise, $\bs{A}$ will be the matrix of dimensions $m\times n$ containing scalars multiplying these variables (here $x_1$ is multiplied by 2 and $x_2$ by -1). The vector $\bs{x}$ contains the variables $x_1$ and $x_2$. And the right hand term is the constant $\bs{b}$:

<div>
$$
\bs{A}=
\begin{bmatrix}
    2 & -1
\end{bmatrix}
$$
</div>

<div>
$$
\bs{x}=
\begin{bmatrix}
    x_1\\\\
    x_2
\end{bmatrix}
$$
</div>

<div>
$$
\bs{b}=
\begin{bmatrix}
    -1
\end{bmatrix}
$$
</div>

We can write this system

<div>
$$
\bs{Ax}=\bs{b}
$$
</div>

We will see at the end of the [the next chapter](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) that this compact way of writing sets of linear equations can be very usefull. It provides actually a way to solve the equations.

# References

- [Math is fun - Multiplying matrices](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>