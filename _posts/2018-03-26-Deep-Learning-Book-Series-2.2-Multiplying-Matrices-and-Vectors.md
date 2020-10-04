---
bg: "galoped.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.2 Multiplying Matrices and Vectors
crawlertitle: "Introduction to Multiplying Matrices and Vectors using Python/Numpy examples and drawings"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.2%20Multiplying%20Matrices%20and%20Vectors/2.2%20Multiplying%20Matrices%20and%20Vectors.ipynb
date: 2018-03-26 11:00:00
excerpt: This short introduction will give you the intuition and Python/Numpy code behind matrices and vectors multiplication. Multiplying matrices and understanding the dot product is crucial to more advanced linear algebra needed for data science, machine learning and deep learning.
excerpt-image: <img src="../../assets/images/2.2/dot-product.png" width="400" alt="Multiplying matrices. An example of how to calculate the dot product between a matrix and a vector" title="The dot product between a matrix and a vector">
  <em>The dot product between a matrix and a vector</em>
deep-learning-book-toc: true
---

*Last update: May 2020*

# Introduction

The dot product is a major concept of linear algebra and thus machine learning and data science. We will see some properties of this operation. Then, we will get some intuition on the link between matrices and systems of linear equations.

{% include mailchimp.html %}

# 2.2 Multiplying Matrices and Vectors

The standard way to multiply matrices is not to multiply each element of one with each element of the other (called the *element-wise product*) but to calculate the sum of the products between rows and columns. The matrix product, also called **dot product**, is calculated as following:

<img src="../../assets/images/2.2/dot-product.png" width="400" alt="An example of how to calculate the dot product between a matrix and a vector" title="The dot product between a matrix and a vector">
<em>The dot product between a matrix and a vector</em>

The number of columns of the first matrix must be equal to the number of rows of the second matrix. If the dimensions of the first matrix is ($m \times n$), the second matrix needs to be of shape ($n \times x$). The resulting matrix will have the shape ($m \times x$).

### Example 1.

Let's start with the multiplication of a matrix and a vector.

<div>
$
\bs{A} \times \bs{b} = \bs{C}
$
</div>

with:

<div>
$
\bs{A}=
\begin{bmatrix}
    1 & 2 \\\
    3 & 4 \\\
    5 & 6
\end{bmatrix}
$
</div>

and:

<div>
$
\bs{b}=\begin{bmatrix}
    2\\\
    4
\end{bmatrix}
$
</div>

We saw that the formula is the following:

<div>
$
\begin{aligned}
&\begin{bmatrix}
    A_{1,1} & A_{1,2} \\\
    A_{2,1} & A_{2,2} \\\
    A_{3,1} & A_{3,2}
\end{bmatrix}\times
\begin{bmatrix}
    B_{1,1} \\\
    B_{2,1}
\end{bmatrix}=\\\
&\begin{bmatrix}
    A_{1,1}B_{1,1} + A_{1,2}B_{2,1} \\\
    A_{2,1}B_{1,1} + A_{2,2}B_{2,1} \\\
    A_{3,1}B_{1,1} + A_{3,2}B_{2,1}
\end{bmatrix}
\end{aligned}
$
</div>

So we will have:

<div>
$
\begin{aligned}
&\begin{bmatrix}
    1 & 2 \\\
    3 & 4 \\\
    5 & 6
\end{bmatrix}\times
\begin{bmatrix}
    2 \\\
    4
\end{bmatrix}=\\\
&\begin{bmatrix}
    1 \times 2 + 2 \times 4 \\\
    3 \times 2 + 4 \times 4 \\\
    5 \times 2 + 6 \times 4
\end{bmatrix}=
\begin{bmatrix}
    10 \\\
    22 \\\
    34
\end{bmatrix}
\end{aligned}
$
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
$\bs{A} \bs{B} = \bs{C}$
</div>

with:

<div>
$\bs{A}=\begin{bmatrix}
    1 & 2 & 3 \\\
    4 & 5 & 6 \\\
    7 & 8 & 9 \\\
    10 & 11 & 12
\end{bmatrix}
$
</div>

and:

<div>
$\bs{B}=\begin{bmatrix}
    2 & 7 \\\
    1 & 2 \\\
    3 & 6
\end{bmatrix}
$
</div>

So we have:

<div>
$
\begin{aligned}
&\begin{bmatrix}
    1 & 2 & 3 \\\
    4 & 5 & 6 \\\
    7 & 8 & 9 \\\
    10 & 11 & 12
\end{bmatrix}
\begin{bmatrix}
    2 & 7 \\\
    1 & 2 \\\
    3 & 6
\end{bmatrix}=\\\
&\begin{bmatrix}
    2 \times 1 + 1 \times 2 + 3 \times 3 & 7 \times 1 + 2 \times 2 + 6 \times 3 \\\
    2 \times 4 + 1 \times 5 + 3 \times 6 & 7 \times 4 + 2 \times 5 + 6 \times 6 \\\
    2 \times 7 + 1 \times 8 + 3 \times 9 & 7 \times 7 + 2 \times 8 + 6 \times 9 \\\
    2 \times 10 + 1 \times 11 + 3 \times 12 & 7 \times 10 + 2 \times 11 + 6 \times 12 \\\
\end{bmatrix}\\\
&=
\begin{bmatrix}
    13 & 29 \\\
    31 & 74 \\\
    49 & 119 \\\
    67 & 164
\end{bmatrix}
\end{aligned}
$
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

The dot product can be formalized through the following equation:

<div>
$
C_{i,j} = A_{i,k}B_{k,j} = \sum_{k}A_{i,k}B_{k,j}
$
</div>

You can find more examples about the dot product [here](https://www.mathsisfun.com/algebra/matrix-multiplying.html).

# Properties of the dot product

We will now see some interesting properties of the dot product. Using simple examples for each property, we'll get used to the Numpy functions.

## Simplification of the matrix product

<div>
$(\bs{AB})^{\text{T}} = \bs{B}^\text{T}\bs{A}^\text{T}$
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

## Matrix mutliplication is distributive

<div>
$\bs{A}(\bs{B}+\bs{C}) = \bs{AB}+\bs{AC}$
</div>

### Example 3.

<div>
$
\bs{A}=\begin{bmatrix}
    2 & 3 \\\
    1 & 4 \\\
    7 & 6
\end{bmatrix},
\bs{B}=\begin{bmatrix}
    5 \\\
    2
\end{bmatrix},
\bs{C}=\begin{bmatrix}
    4 \\\
    3
\end{bmatrix}
$
</div>


<div>
$
\begin{aligned}
\bs{A}(\bs{B}+\bs{C})&=\begin{bmatrix}
    2 & 3 \\\
    1 & 4 \\\
    7 & 6
\end{bmatrix}\times
\left(\begin{bmatrix}
    5 \\\
    2
\end{bmatrix}+
\begin{bmatrix}
    4 \\\
    3
\end{bmatrix}\right)=
\begin{bmatrix}
    2 & 3 \\\
    1 & 4 \\\
    7 & 6
\end{bmatrix}\times
\begin{bmatrix}
    9 \\\
    5
\end{bmatrix}\\\
&=
\begin{bmatrix}
    2 \times 9 + 3 \times 5 \\\
    1 \times 9 + 4 \times 5 \\\
    7 \times 9 + 6 \times 5
\end{bmatrix}=
\begin{bmatrix}
    33 \\\
    29 \\\
    93
\end{bmatrix}
\end{aligned}
$
</div>

is equivalent to

<div>
$
\begin{aligned}
\bs{A}\bs{B}+\bs{A}\bs{C} &= \begin{bmatrix}
    2 & 3 \\\
    1 & 4 \\\
    7 & 6
\end{bmatrix}\times
\begin{bmatrix}
    5 \\\
    2
\end{bmatrix}+
\begin{bmatrix}
    2 & 3 \\\
    1 & 4 \\\
    7 & 6
\end{bmatrix}\times
\begin{bmatrix}
    4 \\\
    3
\end{bmatrix}\\\
&=
\begin{bmatrix}
    2 \times 5 + 3 \times 2 \\\
    1 \times 5 + 4 \times 2 \\\
    7 \times 5 + 6 \times 2
\end{bmatrix}+
\begin{bmatrix}
    2 \times 4 + 3 \times 3 \\\
    1 \times 4 + 4 \times 3 \\\
    7 \times 4 + 6 \times 3
\end{bmatrix}\\\
&=
\begin{bmatrix}
    16 \\\
    13 \\\
    47
\end{bmatrix}+
\begin{bmatrix}
    17 \\\
    16 \\\
    46
\end{bmatrix}=
\begin{bmatrix}
    33 \\\
    29 \\\
    93
\end{bmatrix}
\end{aligned}
$
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


## Matrix mutliplication is associative

<div>
$\bs{A}(\bs{BC}) = (\bs{AB})\bs{C}$
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
$\bs{AB} \neq \bs{BA}$
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
$\bs{x^{ \text{T}}y} = \bs{y^{\text{T}}x} $
</div>

Let's try with the following example:


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

One way to see why the vector multiplication is commutative is to notice that the result of $\bs{x^{ \text{T}}y}$ is a scalar. We know that scalars are equal to their own transpose, so mathematically, we have:

<div>
$\bs{x^{\text{T}}y} = (\bs{x^{\text{T}}y})^{\text{T}} = \bs{ y^{\text{T}} (x^{\text{T}})^{\text{T}} } = \bs{y^{\text{T}}x}$
</div>


# System of linear equations

This is an important part of why linear algebra can be very useful to solve a large variety of problems. Here we will see that it can be used to represent systems of equations.

A system of equations is a set of multiple equations (at least 1). For instance we could have:

<div>
$
\begin{cases}
y = 2x + 1 \\\
y = \frac{7}{2}x +3
\end{cases}
$
</div>

It is defined by its number of equations and its number of unknowns. In this example, there are 2 equations (the first and the second line) and 2 unknowns ($x$ and $y$). In addition we call this a system of **linear** equations because each equation is linear. We can represent that in 2 dimensions: we have one straight line per equation and dimensions correspond to the unknowns. Here is the plot of the first equation:

<img src="../../assets/images/2.2/plot-linear-equation.png" width="300" alt="Representation of a line from an equation" title="Plot of a linear equation">
<em>Representation of a linear equation</em>

<span class='pquote'>
    In our system of equations, the unknowns are the dimensions and the number of equations is the number of lines (in 2D) or $n$-dimensional planes.
</span>

## Using matrices to describe the system

Matrices can be used to describe a system of linear equations of the form $\bs{Ax}=\bs{b}$. Here is such a system:

<div>
$
A_{1,1}x_1 + A_{1,2}x_2 + A_{1,n}x_n = b_1 \\\
A_{2,1}x_1 + A_{2,2}x_2 + A_{2,n}x_n = b_2 \\\
\cdots \\\
A_{m,1}x_1 + A_{m,2}x_2 + A_{m,n}x_n = b_n
$
</div>

The unknowns (what we want to find to solve the system) are the variables $x_1$ and $x_2$. It is exactly the same form as with the last example but with all the variables on the same side. $y = 2x + 1$ becomes $-2x + y = 1$ with $x$ corresponding to $x_1$ and $y$ corresponding to $x_2$. We will have $n$ unknowns and $m$ equations.

The variables are named $x_1, x_2, \cdots, x_n$ by convention because we will see that it can be summarised in the vector $\bs{x}$.

### Left-hand side

The left-hand side can be considered as the product of a matrix $\bs{A}$ containing weights for each variable ($n$ columns) and each equation ($m$ rows):

<div>
$
\bs{A}=
\begin{bmatrix}
    A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\\
    A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\\
    \cdots & \cdots & \cdots & \cdots \\\
    A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
$
</div>

with a vector $\bs{x}$ containing the $n$ unknowns

<div>
$
\bs{x}=
\begin{bmatrix}
    x_1 \\\
    x_2 \\\
    \cdots \\\
    x_n
\end{bmatrix}
$
</div>

The dot product of $\bs{A}$ and $\bs{x}$ gives a set of equations. Here is a simple example:

<img src="../../assets/images/2.2/system-linear-equations-matrix-form.png" width="400" alt="Matrix form of a system of linear equation" title="Matrix form of a system of linear equation">
<em>Matrix form of a system of linear equations</em>

We have a set of two equations with two unknowns. So the number of rows of $\bs{A}$ gives the number of equations and the number of columns gives the number of unknowns.

### Both sides

The equation system can be wrote like that:

<div>
$
\begin{bmatrix}
    A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\\
    A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\\
    \cdots & \cdots & \cdots & \cdots \\\
    A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
\begin{bmatrix}
    x_1 \\\
    x_2 \\\
    \cdots \\\
    x_n
\end{bmatrix}
=
\begin{bmatrix}
    b_1 \\\
    b_2 \\\
    \cdots \\\
    b_m
\end{bmatrix}
$
</div>

Or simply:

<div>
$\bs{Ax}=\bs{b}$
</div>

### Example 4.

We will try to convert the common form of a linear equation $y=ax+b$ to the matrix form. If we want to keep the previous notation we will have instead:

<div>
$x_2=ax_1+b$
</div>

Don't confuse the variable $x_1$ and $x_2$ with the vector $\bs{x}$. This vector contains all the variables of our equations:

<div>
$
\bs{x} =
\begin{bmatrix}
    x_1 \\\
    x_2
\end{bmatrix}
$
</div>

In this example we will use the following equation:

<div>
$
\begin{aligned}
&x_2=2x_1+1 \\\
\Leftrightarrow& 2x_1-x_2=-1
\end{aligned}
$
</div>

In order to end up with this system when we multiply $\bs{A}$ and $\bs{x}$ we need $\bs{A}$ to be a matrix containing the weights of each variable. The weight of $x_1$ is $2$ and the weights of $x_2$ is $-1$:

<div>
$
\bs{A}=
\begin{bmatrix}
    2 & -1
\end{bmatrix}
$
</div>

So we have

<div>
$
\begin{bmatrix}
    2 & -1
\end{bmatrix}
\begin{bmatrix}
    x_1 \\\
    x_2
\end{bmatrix}
=
\begin{bmatrix}
2x_1-1x_2
\end{bmatrix}
$
</div>

To complete the equation we have

<div>
$
\bs{b}=
\begin{bmatrix}
    -1
\end{bmatrix}
$
</div>

which gives

<div>
$
\begin{bmatrix}
    2 & -1
\end{bmatrix}
\begin{bmatrix}
    x_1 \\\
    x_2
\end{bmatrix}
=
\begin{bmatrix}
    -1
\end{bmatrix}
$
</div>

This system of equations is thus very simple and contains only 1 equation ($\bs{A}$ has 1 row) and 2 variables ($\bs{A}$ has 2 columns).

To summarise, $\bs{A}$ will be a matrix of dimensions $m\times n$ containing scalars multiplying these variables (here $x_1$ is multiplied by 2 and $x_2$ by -1). The vector $\bs{x}$ contains the variables $x_1$ and $x_2$. And the right-hand side is the constant $\bs{b}$:

<div>
$
\bs{A}=
\begin{bmatrix}
    2 & -1
\end{bmatrix}
$
</div>

<div>
$
\bs{x}=
\begin{bmatrix}
    x_1 \\\
    x_2
\end{bmatrix}
$
</div>

<div>
$
\bs{b}=
\begin{bmatrix}
    -1
\end{bmatrix}
$
</div>

We can write this system

<div>
$
\bs{Ax}=\bs{b}
$
</div>

We will see at the end of the [the next chapter](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) that this compact way of writing sets of linear equations can be very usefull. It provides a way to solve the equations.

{% include mailchimp.html %}


# References

- [Math is fun - Multiplying matrices](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

