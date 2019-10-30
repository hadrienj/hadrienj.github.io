---
bg: "deadTrees.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series · 2.9 The Moore Penrose Pseudoinverse
crawlertitle: "Introduction to The Moore Penrose Pseudoinverse using Python/Numpy examples and drawings"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.9%20The%20Moore-Penrose%20Pseudoinverse/2.9%20The%20Moore-Penrose%20Pseudoinverse.ipynb
date: 2018-03-26 16:30:00
excerpt: In this post, we will learn about the Moore Penrose pseudoinverse as a way to find an approaching solution where no solution exists. In some cases, a system of equation has no solution, and thus the inverse doesn't exist. However it can be useful to find a value that is almost a solution (in term of minimizing the error). We will see for instance how we can find the best-fit line of a set of data points with the pseudoinverse.
excerpt-image: <img src="../../assets/images/2.9/overdetermined-system-equations.png" width="300" alt="Example of three linear equations in 2 dimensions. This is an overdetermined system" title="Overdetermined system of equations">
  <em>There is more equations (3) than unknowns (2) so this is an overdetermined system of equations</em>
deep-learning-book-toc: true
---

# Introduction

We saw that not all matrices have an inverse. It is unfortunate because the inverse is used to solve system of equations. In some cases, a system of equation has no solution, and thus the inverse doesn't exist. However it can be useful to find a value that is almost a solution (in term of minimizing the error). We will see for instance how we can find the best-fit line of a set of data points with the pseudoinverse.

{% include mailchimp.html %}

# 2.9 The Moore-Penrose Pseudoinverse

The Moore-Penrose pseudoinverse is a direct application of the SVD (see [2.8](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/)). But before all, we have to remind that systems of equations can be expressed under the matrix form.

As we have seen in [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/), the inverse of a matrix $\bs{A}$ can be used to solve the equation $\bs{Ax}=\bs{b}$:

<div>
$$
\bs{A}^{-1}\bs{Ax}=\bs{A}^{-1}\bs{b}
$$
</div>

<div>
$$
\bs{I}_n\bs{x}=\bs{A}^{-1}\bs{b}
$$
</div>

<div>
$$
\bs{x}=\bs{A}^{-1}\bs{b}
$$
</div>

But in the case where the set of equations have 0 or many solutions the inverse cannot be found and the equation cannot be solved. The pseudoinverse is $\bs{A}^+$ such as:

<div>
$$
\bs{A}\bs{A}^+\approx\bs{I_n}
$$
</div>

minimizing

<div>
$$
\norm{\bs{A}\bs{A}^+-\bs{I_n}}_2
$$
</div>

The following formula can be used to find the pseudoinverse:

<div>
$$
\bs{A}^+= \bs{VD}^+\bs{U}^T
$$
</div>

with $\bs{U}$, $\bs{D}$ and $\bs{V}$ respectively the left singular vectors, the singular values and the right singular vectors of $\bs{A}$ (see the SVD in [2.8](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/)). $\bs{A}^+$ is the pseudoinverse of $\bs{A}$ and $\bs{D}^+$ the pseudoinverse of $\bs{D}$. We saw that $\bs{D}$ is a diagonal matrix and thus $\bs{D}^+$ can be calculated by taking the reciprocal of the non zero values of $\bs{D}$.

This is a bit crude but we will see some examples to clarify all of this.

### Example 1.

Let's see how to implement that. We will create a non square matrix $\bs{A}$, calculate its singular value decomposition and its pseudoinverse.

<div>
$$
\bs{A}=\begin{bmatrix}
    7 & 2\\\\
    3 & 4\\\\
    5 & 3
\end{bmatrix}
$$
</div>


```python
A = np.array([[7, 2], [3, 4], [5, 3]])
U, D, V = np.linalg.svd(A)

D_plus = np.zeros((A.shape[0], A.shape[1])).T
D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))

A_plus = V.T.dot(D_plus).dot(U.T)
A_plus
```

<pre class='output'>
array([[ 0.16666667, -0.10606061,  0.03030303],
       [-0.16666667,  0.28787879,  0.06060606]])
</pre>


We can now check with the `pinv()` function from Numpy that the pseudoinverse is correct:


```python
np.linalg.pinv(A)
```

<pre class='output'>
array([[ 0.16666667, -0.10606061,  0.03030303],
       [-0.16666667,  0.28787879,  0.06060606]])
</pre>


It looks good! We can now check that it is really the near inverse of $\bs{A}$. Since we know that

<div>
$$\bs{A}^{-1}\bs{A}=\bs{I_n}$$
</div>

with

<div>
$$\bs{I_2}=\begin{bmatrix}
    1 & 0 \\\\
    0 & 1
\end{bmatrix}
$$
</div>


```python
A_plus.dot(A)
```

<pre class='output'>
array([[  1.00000000e+00,   2.63677968e-16],
       [  5.55111512e-17,   1.00000000e+00]])
</pre>


This is not bad! This is almost the identity matrix!

A difference with the real inverse is that $\bs{A}^+\bs{A}\approx\bs{I}$ but $\bs{A}\bs{A}^+\neq\bs{I}$.

Another way of computing the pseudoinverse is to use this formula:

<div>
$$
(\bs{A}^T\bs{A})^{-1}\bs{A}^T
$$
</div>

The result is less acurate than the SVD method and Numpy `pinv()` uses the SVD ([cf Numpy doc](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html)). Here is an example from the same matrix $\bs{A}$:


```python
A_plus_1 = np.linalg.inv(A.T.dot(A)).dot(A.T)
A_plus_1
```

<pre class='output'>
array([[ 0.16666667, -0.10606061,  0.03030303],
       [-0.16666667,  0.28787879,  0.06060606]])
</pre>


In this case the result is the same as with the SVD way.

## Using the pseudoinverse to solve a overdetermined system of linear equations

In general there is no solution to overdetermined systems (see [2.4](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.4-Linear-Dependence-and-Span/) ; [Overdetermined systems](https://en.wikipedia.org/wiki/Overdetermined_system)). In the following picture, there is no point at the intersection of the three lines corresponding to three equations:

<img src="../../assets/images/2.9/overdetermined-system-equations.png" width="300" alt="Example of three linear equations in 2 dimensions: this is an overdetermined system" title="Overdetermined system of equations">
<em>There is more equations (3) than unknowns (2) so this is an overdetermined system of equations</em>

The pseudoinverse solve the system in the least square error perspective: it finds the solution that minimize the error. We will see this more explicitly with an example.

<span class='pquote'>
    The pseudoinverse solve the system in the least square error perspective
</span>

### Example 2.

For this example we will consider this set of three equations with two unknowns:

<div>
$$
\begin{cases}
-2x_1 + 2 = x_2 \\\\
4x_1 + 8 = x_2 \\\\
-1x_1 + 2 = x_2
\end{cases}
\Leftrightarrow
\begin{cases}
-2x_1 - x_2 = -2 \\\\
4x_1 - x_2 = -8 \\\\
-1x_1 - x_2 = -2
\end{cases}
$$
</div>

Let's see their graphical representation:


```python
x1 = np.linspace(-5, 5, 1000)
x2_1 = -2*x1 + 2
x2_2 = 4*x1 + 8
x2_3 = -1*x1 + 2

plt.plot(x1, x2_1)
plt.plot(x1, x2_2)
plt.plot(x1, x2_3)
plt.xlim(-2., 1)
plt.ylim(1, 5)
plt.show()
```


<img src="../../assets/images/2.9/overdetermined-system-equations-python.png" width="300" alt="Plot of three equations in 2 dimensions done with Python, Numpy and Matplotlib" title="Overdetermined system of equations plotted with Python">
<em>Representation of our overdetermined system of equation</em>

We actually see that there is no solution.

Putting this into the matrix form we have:

<div>
$$
\bs{A}=
\begin{bmatrix}
    -2 & -1 \\\\
    4 & -1 \\\\
    -1 & -1
\end{bmatrix}
$$
</div>

<div>
$$
\bs{x}=
\begin{bmatrix}
    x_1 \\\\
    x_2
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{b}=
\begin{bmatrix}
    -2 \\\\
    -8 \\\\
    -2
\end{bmatrix}
$$
</div>

So we have:

<div>
$$
\bs{Ax} = \bs{b}
\Leftrightarrow
\begin{bmatrix}
    -2 & -1 \\\\
    4 & -1 \\\\
    -1 & -1
\end{bmatrix}
\begin{bmatrix}
    x_1 \\\\
    x_2
\end{bmatrix}
=
\begin{bmatrix}
    -2 \\\\
    -8 \\\\
    -2
\end{bmatrix}
$$
</div>

We will now calculate the pseudoinverse of $\bs{A}$:


```python
A = np.array([[-2, -1], [4, -1], [-1, -1]])
A_plus = np.linalg.pinv(A)
A_plus
```

<pre class='output'>
array([[-0.11290323,  0.17741935, -0.06451613],
       [-0.37096774, -0.27419355, -0.35483871]])
</pre>


Now that we have calculated the pseudoinverse of $\bs{A}$:

<div>
$$
\bs{A}^+=
\begin{bmatrix}
    -0.1129 &  0.1774 & -0.0645 \\\\
    -0.3710 & -0.2742 & -0.3548
\end{bmatrix}
$$
</div>

we can use it to find $\bs{x}$ knowing that:

<div>
$$
\bs{x}=\bs{A}^+\bs{b}
$$
</div>

with:

<div>
$$
\bs{x}
=
\begin{bmatrix}
    x1 \\\\
    x2
\end{bmatrix}
$$
</div>


```python
b = np.array([[-2], [-8], [-2]])
res = A_plus.dot(b)
res
```

<pre class='output'>
array([[-1.06451613],
       [ 3.64516129]])
</pre>


So we have

<div>
$$
\begin{align*}
\bs{A}^+\bs{b}&=
\begin{bmatrix}
    -0.1129 &  0.1774 & -0.0645 \\\\
    -0.3710 & -0.2742 & -0.3548
\end{bmatrix}
\begin{bmatrix}
    -2 \\\\
    -8 \\\\
    -2
\end{bmatrix}\\\\
&=
\begin{bmatrix}
    -1.06451613 \\\\
    3.64516129
\end{bmatrix}
\end{align*}
$$
</div>

In our two dimensions, the coordinates of $\bs{x}$ are

<div>
$$
\begin{bmatrix}
    -1.06451613 \\\\
    3.64516129
\end{bmatrix}
$$
</div>

Let's plot this point along with the equations lines:


```python
plt.plot(x1, x2_1)
plt.plot(x1, x2_2)
plt.plot(x1, x2_3)
plt.xlim(-2., 1)
plt.ylim(1, 5)

plt.scatter(res[0], res[1])

plt.show()
```


<img src="../../assets/images/2.9/mean-square-error-solution-overdetermined-system-equations-python.png" width="300" alt="Finding the solution that minimizes the mean square error with the pseudo-inverse" title="Finding a solution minimizing the mean square error">
<em>The pseudoinverse can be used to find the point that minimizes the mean square error</em>

Maybe you would have expected the point being at the barycenter of the triangle (cf. [Least square solution in the triangle center](https://math.stackexchange.com/questions/471812/is-the-least-squares-solution-to-an-overdetermined-system-a-triangle-center)). This is not the case becase the equations are not scaled the same way. Actually the point is at the intersection of the three [symmedians](https://en.wikipedia.org/wiki/Symmedian) of the triangle.

### Example 3.

This method can also be used to fit a line to a set of points. Let's take the following data points:

<img src="../../assets/images/2.9/dataset-representation.png" width="300" alt="Representation of a set of data points" title="Some data points">
<em>We want to fit a line to this set of data points</em>

We have this set of $\bs{x}$ and $\bs{y}$ and we are looking for the line $y=mx+b$ that minimizes the error. The error can be evaluated as the sum of the differences between the fit and the actual data points. We can represent the data points with a matrix equations:

<div>
$$
\bs{Ax} = \bs{b}
\Leftrightarrow
\begin{bmatrix}
    0 & 1 \\\\
    1 & 1 \\\\
    2 & 1 \\\\
    3 & 1 \\\\
    3 & 1 \\\\
    4 & 1
\end{bmatrix}
\begin{bmatrix}
    m \\\\
    b
\end{bmatrix}
=
\begin{bmatrix}
    2 \\\\
    4 \\\\
    0 \\\\
    2 \\\\
    5 \\\\
    3
\end{bmatrix}
$$
</div>

Note that here the matrix $\bs{A}$ represents the values of the coefficients. The column of 1 correspond to the intercepts (without it the fit would have the constraint to cross the origin). It gives the following set of equations:

<div>
$$
\begin{cases}
    0m + 1b = 2 \\\\
    1m + 1b = 4 \\\\
    2m + 1b = 0 \\\\
    3m + 1b = 2 \\\\
    3m + 1b = 5 \\\\
    4m + 1b = 3
\end{cases}
$$
</div>

We have the set of equations $mx+b=y$. The ones are used to give back the intercept parameter. For instance, in the first equation corresponding to the first point we have well $x=0$ and $y=2$. This can be confusing because here the vector $\bs{x}$ corresponds to the coefficients. This is because the problem is different from the other examples: we are looking for the coefficients of a line and not for $x$ and $y$ unknowns. We kept this notation to indicate the similarity with the last examples.

So we will construct these matrices and try to use the pseudoinverse to find the equation of the line minimizing the error (difference between the line and the actual data points).

Let's start with the creation of the matrix of $\bs{A}$ and $\bs{b}$:


```python
A = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [3, 1], [4, 1]])
A
```

<pre class='output'>
array([[0, 1],
       [1, 1],
       [2, 1],
       [3, 1],
       [3, 1],
       [4, 1]])
</pre>



```python
b = np.array([[2], [4], [0], [2], [5], [3]])
b
```

<pre class='output'>
array([[2],
       [4],
       [0],
       [2],
       [5],
       [3]])
</pre>


We can now calculate the pseudoinverse of $\bs{A}$:


```python
A_plus = np.linalg.pinv(A)
A_plus
```

<pre class='output'>
array([[ -2.00000000e-01,  -1.07692308e-01,  -1.53846154e-02,
          7.69230769e-02,   7.69230769e-02,   1.69230769e-01],
       [  6.00000000e-01,   4.00000000e-01,   2.00000000e-01,
          4.16333634e-17,   4.16333634e-17,  -2.00000000e-01]])
</pre>


and apply it to the result to find the coefficients with the formula:

<div>
$$
\bs{x}=\bs{A}^+\bs{b}
$$
</div>


```python
coefs = A_plus.dot(b)
coefs
```

<pre class='output'>
array([[ 0.21538462],
       [ 2.2       ]])
</pre>


These are the parameters of the fit. The slope is $m=0.21538462$ and the intercept is $b=2.2$. We will plot the data points and the regression line:


```python
x = np.linspace(-1, 5, 1000)
y = coefs[0]*x + coefs[1]

plt.plot(A[:, 0], b, '*')
plt.plot(x, y)
plt.xlim(-1., 6)
plt.ylim(-0.5, 5.5)

plt.show()
```


<img src="../../assets/images/2.9/line-fit-dataset.png" width="300" alt="Representation of the data points and the fitting line minimizing the mean square error" title="Representation of the fit">
<em>We found the line minimizing the error!</em>

If you are not sure about the result. Just check it with another method. For instance, I double-checked with R:

```r
a <- data.frame(x=c(0, 1, 2, 3, 3, 4),
                y=c(2, 4, 0, 2, 5, 3))

ggplot(data=a, aes(x=x, y=y)) +
  geom_point() +
  stat_smooth(method = "lm", col = "red") +
  xlim(-1, 5) +
  ylim(-1, 6)
```

outputs:

<img src="../../assets/images/2.9/linear-regression-r.png" width="300" alt="Fitting a line with another method (in R)" title="Fitting with R">
<em>Just checking with another method</em>

You can also do the fit with the Numpy `polyfit()` to check the parameters:


```python
np.polyfit(A[:, 0], b, 1)
```

<pre class='output'>
array([[ 0.21538462],
       [ 2.2       ]])
</pre>


That's good! We have seen how to use the pseudoinverse in order to solve a simple regression problem. Let's see with a more realistic case.

### Example 4.

To see the process with more data points we can generate data (see [this nice blog post](https://mec560sbu.github.io/2016/08/29/Least_SQ_Fitting/) for other methods of fitting).

We will generate a column vector (see `reshape()` bellow) containing 100 points with random $x$ values and pseudo-random $y$ values. The function `seed()` from the [Numpy.random package](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html) is used to freeze the randomisation and be able to reproduce the results:


```python
np.random.seed(123)
x = 5*np.random.rand(100)
y = 2*x + 1 + np.random.randn(100)

x = x.reshape(100, 1)
y = y.reshape(100, 1)
```

We will create the matrix $\bs{A}$ from $\bs{x}$ by adding a column of ones exactly like we did in the example 3.


```python
A = np.hstack((x, np.ones(np.shape(x))))
A[:10]
```

<pre class='output'>
array([[ 3.48234593,  1.        ],
       [ 1.43069667,  1.        ],
       [ 1.13425727,  1.        ],
       [ 2.75657385,  1.        ],
       [ 3.59734485,  1.        ],
       [ 2.1155323 ,  1.        ],
       [ 4.90382099,  1.        ],
       [ 3.42414869,  1.        ],
       [ 2.40465951,  1.        ],
       [ 1.96058759,  1.        ]])
</pre>


We can now find the pseudoinverse of $\bs{A}$ and calculate the coefficients of the regression line:


```python
A_plus = np.linalg.pinv(A)
coefs = A_plus.dot(y)
coefs
```

<pre class='output'>
array([[ 1.9461907 ],
       [ 1.16994745]])
</pre>


We can finally draw the point and the regression line:


```python
x_line = np.linspace(0, 5, 1000)
y_line = coefs[0]*x_line + coefs[1]

plt.plot(x, y, '*')
plt.plot(x_line, y_line)
plt.show()
```


<img src="../../assets/images/2.9/linear-regression-dataset.png" width="300" alt="Fitting a line to a set of data points" title="Regression line">
<em>Fitting a line to a set of data points</em>

Looks good!

# Conclusion

You can see that the pseudoinverse can be very useful for this kind of problems! The series is not completely finished since we still have 3 chapters to cover. However, we have done the hardest part! We will now see two very light chapters before going to a nice example using all the linear algebra we have learn: the PCA.

# References

## Intuition

- [Sean Owen - Pseudoinverse intuition](https://www.quora.com/What-is-the-intuition-behind-pseudo-inverse-of-a-matrix)

## Numpy

- [Numpy - linalg.pinv](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html)

- [Numpy random seed](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html)

## Systems of equations

- [Using the Moore-Penrose Pseudoinverse to Solve Linear Equations](https://www.youtube.com/watch?v=5bxsxM2UTb4)

- [Overdetermined systems](https://en.wikipedia.org/wiki/Overdetermined_system)

- [Least square solution in the triangle center](https://math.stackexchange.com/questions/471812/is-the-least-squares-solution-to-an-overdetermined-system-a-triangle-center)

- [Symmedian](https://en.wikipedia.org/wiki/Symmedian)

## Least square fit

- [Least square fitting](https://mec560sbu.github.io/2016/08/29/Least_SQ_Fitting/)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

