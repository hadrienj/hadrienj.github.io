---
bg: "moss.jpg"
layout: post
mathjax: true
title: 2.3 Identity and Inverse Matrices
crawlertitle: "deep learning machine learning linear algebra python getting started numpy data sciences"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/tree/master/2.3%20Identity%20and%20Inverse%20Matrices
comments: true
---

# Introduction

This chapter is light but contains some important definitions. The identity matrix or the inverse of a matrix are some concepts that are simple but will be very useful in the next chapters. We will see at the end of this chapter that we can solve systems of linear equations by using the inverse matrix. So hang on!

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts.
</span>

# 2.3 Identity and Inverse Matrices


# Identity matrices

The identity matrix $\bs{I}_n$ is a special matrix of shape ($n \times n$) that has all 0 except the diagonal that is filled with 1.

<img src="../../assets/images/2.3/identity.png" width="150" alt="identity">

An identity matrix can be created with the Numpy function `eye()`:


```python
np.eye(3)
```

When multiplied with a vector the result is this same vector:

<div>
$$\bs{I}_n\bs{x} = \bs{x}$$
</div>

### Example 1.

<div>
$$
\begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & 1 & 0 \\\\
    0 & 0 & 1
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
    0 \times x_1 + 0 \times x_2 + 1\times x_3
\end{bmatrix}=
\begin{bmatrix}
    x_{1} \\\\
    x_{2} \\\\
    x_{3}
\end{bmatrix}
$$
</div>


```python
x = np.array([[2], [6], [3]])
x
```


```python
xid = np.eye(x.shape[0]).dot(x)
xid
```

## Intuition

You can think of a matrix as a way to transform objects in a $n$-dimensional space. It does a linear transformation of the space. We can say that we *apply* a matrix to an element: this means that we do the dot product between this matrix and the element (more details about the dot product in [2.2]()). We will see this important notion thoroughly in the next chapters but the identity matrix can be a first example of that. It is a particular example because the space doesn't change when we *apply* the identity matrix to it.

<span class='pquote'>
    The space doesn't change when we *apply* the identity matrix to it
</span>

We saw that $\bs{x}$ was not altered after being multiplied by $\bs{I}$.

# Inverse Matrices

The matrix inverse of $\bs{A}$ is denoted $\bs{A}^{-1}$ and corresponds to the matrix that results in the identity matrix when it is multiplied by $\bs{A}$:

<div>
$$\bs{A}^{-1}\bs{A}=\bs{I}_n$$
</div>

This means that if we apply a linear transformation to the space with $\bs{A}$, it is possible to go back with $\bs{A}^{-1}$. It provides a way to cancel the transformation.

### Example 2.

<div>
$$
\bs{A}=\begin{bmatrix}
    3 & 0 & 2 \\\\
    2 & 0 & -2 \\\\
    0 & 1 & 1
\end{bmatrix}
$$
</div>

For this example, we will use the Numpy function `linalg.inv()` to calculate the inverse of $\bs{A}$. Let's start by creating $\bs{A}$:


```python
A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
A
```

Now we calculate its inverse:


```python
A_inv = np.linalg.inv(A)
A_inv
```

We can check that $\bs{A_{inv}}$ is well the inverse of $\bs{A}$ with Python:


```python
A_bis = A_inv.dot(A)
A_bis
```

We will see that inverse of matrices can be very usefull. For example, to solve set of linear equations. We must note however that non square matrices (that is matrix with more columns than rows or more rows than columns) don't have inverse.

# Sovling a system of linear equations

Introduction on system of linear equations can be found in [2.2]().

The inverse matrix can be used to solve the equation $\bs{Ax}=\bs{b}$ by adding it to each term:

<div>
$$\bs{A}^{-1}\bs{Ax}=\bs{A}^{-1}\bs{b}$$
</div>

Since we know by definition that $\bs{A}^{-1}\bs{A}=\bs{I}$, we have:

<div>
$$\bs{I}_n\bs{x}=\bs{A}^{-1}\bs{b}$$
</div>

We saw that a vector is not changed when multiplied by the identity matrix. So we can write:

<div>
$$\bs{x}=\bs{A}^{-1}\bs{b}$$
</div>

This is great! We can solve a set of linear equation just by computing the inverse of $\bs{A}$!

Let's try that!

### Example 3.

We will take a simple solvable example:

<div>
$$
\begin{cases}
y = 2x \\\\
y = -x +3
\end{cases}
$$
</div>

We will use the notation we saw in [2.2]():

<div>
$$
\begin{cases}
A_{1,1}x_1 + A_{1,2}x_2 = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2= b_2
\end{cases}
$$
</div>

Here, $x_1$ corresponds to $x$ and $x_2$ corresponds to $y$. So we have:

<div>
$$
\begin{cases}
2x_1 - x_2 = 0 \\\\
x_1 + x_2= 3
\end{cases}
$$
</div>

Our matrix $\bs{A}$ of weights is:

<div>
$$
\bs{A}=
\begin{bmatrix}
    2 & -1 \\\\
    1 & 1
\end{bmatrix}
$$
</div>

And the vector $\bs{b}$ containing the solutions of individual equations is:

<div>
$$
\bs{b}=
\begin{bmatrix}
    0 \\\\
    3
\end{bmatrix}
$$
</div>

Under the matrix form, our systems becomes:

<div>
$$
\begin{bmatrix}
    2 & -1 \\\\
    1 & 1
\end{bmatrix}\cdot
\begin{bmatrix}
    x_1 \\\\
    x_2
\end{bmatrix}=
\begin{bmatrix}
    0 \\\\
    3
\end{bmatrix}
$$
</div>

Let's find the inverse of $\bs{A}$:


```python
A = np.array([[2, -1], [1, 1]])
A
```


```python
A_inv = np.linalg.inv(A)
A_inv
```

We have also:


```python
b = np.array([[0], [3]])
```

Since we saw that

<div>
$$\bs{x}=\bs{A}^{-1}\bs{b}$$
</div>

We have:


```python
x = A_inv.dot(b)
x
```

This is our solution! 

<div>
$$
\bs{x}=
\begin{bmatrix}
    1 \\\\
    2
\end{bmatrix}
$$
</div>

This means that the point (1, 2) is the solution and is at the intersection of the lines representing the equations. Let's plot them to check this solution:


```python
x = np.arange(-10, 10)
y = 2*x
y1 = -x + 3

plt.figure()
plt.plot(x, y)
plt.plot(x, y1)
plt.xlim(0, 3)
plt.ylim(0, 3)
# draw axes
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')
plt.show()
plt.close()
```


![png](../../assets/images/2.3/output_29_0.png)


We can see that the solution (corresponding to the line crossing) is when $x=1$ and $y=2$. It confirms what we found with the matrix inversion!

## BONUS: Coding tips - Draw an equation

To draw the equation with Matplotlib, we first need to create a vector with all the $x$ values. Actually, since this is a line, only two points would have been sufficient. But with more complex functions, the length of the vector $x$ corresponds to the sampling rate. So here we used the Numpy function `arrange()` (see the [doc](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)) to create a vector from $-10$ to $10$ (not included).


```python
np.arange(-10, 10)
```

The first argument is the starting point and the second the ending point. You can add a third argument to specify the step:


```python
np.arange(-10, 10, 2)
```

Then we create a second vector $y$ that is a transformation of the $x$ vector. Numpy will take each value of $x$ and apply the equation formula to it.


```python
x = np.arange(-10, 10)
y = 2*x + 1
y
```

Finally, you just need to plot these vectors.

# Singular matrices

Some matrices are not invertible. They are called **singular**.

# Conclusion

This introduces different cases according to the linear system because $\bs{A}^{-1}$ exists only if the equation $\bs{Ax}=\bs{b}$ has one and only one solution. [The next chapter]() is almost all about systems of linear equations and number of solutions.
