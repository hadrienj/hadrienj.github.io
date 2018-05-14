---
bg: "path.jpg"
layout: post
mathjax: true
title: Datacamp tutorial - Deep Learning Book Series
crawlertitle: "Datacamp tutorial - Deep Learning Book Series"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
date: 2018-05-12
excerpt: Tutorial of linear algebra for deep learning and machine learning
excerpt-image: <img src="../../assets/images/2.5/squared-l2-norm.png" width="500" alt="Representation of the squared L2 norm" title="The squared L2 norm">
  <em>The squared L2 norm</em>
---

# Syllabus

- Norm
- Special matrices
- Eigendecomposition
- Singular value decomposition
- Pseudoinverse
- Trace operator
- The determinant
- PCA


# Introduction

This tutorial will get you through some intermediates linear algebra concepts usefull in deep learning, machine learning and data science in general. The prerequisites are not high: just be sure that you know what is a matrix and how to do the dot product. You can get or refresh these notions on the first posts (1 to 4) of [my series]() on the deep learning book by Ian Goodfellow. I think that having a practical tutorial on theoretical topic like linear algebra can be useful because writing and reading code is a good way to truly understand mathematical concepts.

...

# 2.5 Norms

# A practical example

This first chapter concerns an important concept for machine learning and deep learning. The norm is extensively used in the context of supervised learning, to evaluate the goodness of a model. Imagine that you built a model that predicts the duration of a song. You trained the model with a lot of different song with a lot of features (style, instruments, lyrics vs no lyrics etc.). Now you want to know if you model is good to predicts song duration. One way to do it is to look at the difference between true and predicted duration for each observation. Imagine that you have the following results in seconds:

```python
errorModel1 = [22, -4, 2, 7, 6, -36, 12]
```

These differences can be tought of the error of the model. A perfect model would have only 0's while a very bad model would have huge values.

Now imagine that you try another model and you end up with the following differences between predicted and real song durations:

```python
errorModel2 = [14, 9, -13, 19, 8, -21, 4]
```

What can you do if you want to find the best model? A natural way to do it is to take the sum of the absolute value of these errors. The absolute value is used because a negative error (true duration smaller than predicted duration) is also an error. The model with the smaller total error is the better:

```python
totalErrorModel1 = np.sum(errorModel1)
totalErrorModel2 = np.sum(errorModel2)
```

It looks like the model 1 is far better than the model 2.

Congratulation! What you have just done is calculate the norm of the vector of errors!

You can think of the norm as the **length** of the vector. To have an idea of the graphical representation of this, let's take again our preceding example. The error vectors are multidimensional: there is one dimension per observation. In our simple case, there is 7 observations so there is 7 dimensions. It is quite hard to represent 7 dimensions so let's again simplify the example and keep only 2 observations:

```python
errorModel1 = [22, -4]
errorModel2 = [14, 9]
```

Now we can represent these vector considering that the first element of the array is the x-coordinate and the second element is the y-coordinate.

... plots

So we have two vectors and the way to see which one is the smaller (and hence which model has the smaller error) is to take the sum of each coordinate. It is what we have done earlier. Actually we need to take the absolute value of each coordinate because we don't want to cancel out coordinates.

# How to plot vectors with Python and Matplotlib

We just used a custom function to plot vectors in Python. Let's see how it works:

```python
def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        if (isinstance(alpha, list)):
            alpha_i = alpha[i]
        else:
            alpha_i = alpha
        if (len(vecs[i])==2):
            x = np.concatenate([[0,0],vecs[i]])
        elif (len(vecs[i])==4):
            x = vecs[i]
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                  alpha=alpha_i)
```

It takes an array of vectors that we want to plot (`vecs`) and their colors as input (`cols`). If 2 dimensions are specified in the vector, it starts at (0, 0). Under the hood, we iterate on this array of vectors and use `plt.quiver()` to plot them.

# Norm functions: definitions

This is one way to calculate the length of the vector but the norm is actually any function that maps a vector to a positive value. Different functions can be used and we will see few examples.

Norms are any functions that are characterized by the **following properties**:

- Norms are non-negative values. If you think of the norms as a length, you easily see why it can't be negative.

- Norms are $0$ if and only if the vector is a zero vector

- Norms respect the triangle inequity. See bellow.

- $\norm{\bs{k}\cdot \bs{u}}=\norm{\bs{k}}\cdot\norm{\bs{u}}$. The norm of a vector multiplied by a scalar is equal to the absolute value of this scalar multiplied by the norm of the vector.

It is usually written with two horizontal bars: $\norm{\bs{x}}$

# The triangle inequity

The norm of the sum of some vectors is less than or equal the sum of the norms of these vectors.

<div>
$$
\norm{\bs{u}+\bs{v}} \leq \norm{\bs{u}}+\norm{\bs{v}}
$$
</div>

### Example 1.

To show what this means, we will take two vectors containing each two elements (usefull to be represented as x and y coordinates). Our vectors are:

<div>
$$
\bs{u}=
\begin{bmatrix}
    1 & 6
\end{bmatrix}
$$
</div>

```python
u = np.array([1, 6])
u
```

<pre class='output'>
array([1, 6])
</pre>

and

<div>
$$
\bs{v}=
\begin{bmatrix}
    4 & 2
\end{bmatrix}
$$
</div>

```python
v = np.array([4, 2])
v
```

<pre class='output'>
array([4, 2])
</pre>

Let's compare:

<div>
$$
\norm{\bs{u}+\bs{v}}
$$
</div>

and:

<div>
$$
\norm{\bs{u}}+\norm{\bs{v}}
$$
</div>

using the $L^2$ norm. With the `linalg` methods from `numpy`, the $L^2$ norm can be calculated with `np.linalg.norm()`.

<div>
$$
\norm{\bs{u}+\bs{v}} = \sqrt{(1+4)^2+(6+2)^2} = \sqrt{89} \approx 9.43
$$
</div>

```python
np.linalg.norm(u+v)
```

<pre class='output'>
9.4339811320566032
</pre>

and

<div>
$$
\norm{\bs{u}}+\norm{\bs{v}} = \sqrt{1^2+6^2}+\sqrt{4^2+2^2} = \sqrt{37}+\sqrt{20} \approx 10.55
$$
</div>

```python
np.linalg.norm(u)+np.linalg.norm(v)
```

<pre class='output'>
10.554898485297798
</pre>


We can see that the triangle inequity is respected since:

<div>
$$
\norm{\bs{u}+\bs{v}} \leq \norm{\bs{u}}+\norm{\bs{v}}
$$
</div>


**Graphical explanation**

You will see that the graphical representation of this theorem makes it quite trivial. We will plot the vectors $\bs{u}$, $\bs{v}$ and $\bs{u}+\bs{v}$. We will not use the custom function `plotVectors()`.

```python
u = np.array([0,0,1,6])
v = np.array([0,0,4,2])
w = u+v

u_bis = [u[2], u[3], v[2],v[3]]


plotVectors([u, u_bis, w],
            [sns.color_palette()[0],
            sns.color_palette()[1],
            sns.color_palette()[2]])

plt.xlim(-2, 6)
plt.ylim(-2, 9)

plt.text(-1, 3.5, r'$||\vec{u}||$', color=sns.color_palette()[0], size=20)
plt.text(2.5, 7.5, r'$||\vec{v}||$', color=sns.color_palette()[1], size=20)
plt.text(2, 2, r'$||\vec{u}+\vec{v}||$', color=sns.color_palette()[2], size=20)

plt.show()
plt.close()
```


<img src="../../assets/images/2.5/triangle-inequity.png" width="300" alt="Vector illustration of the triangle inequity" title="Triangle inequity">
<em>Vector illustration of the triangle inequity</em>

You can see with this plot that if the norm function corresponds to the length of the vector, the length of $\bs{u}$ plus the length of $+\bs{v}$ is larger than the length of the vector $\bs{u}+\bs{b}$. Geometrically, this simply means that the shortest path between two points is a line.

# P-norms: general rules

Now that we have seen the conditions required by the function to be called norm. This means that there are multiple functions that can be used as norms. We will see later the pros and cons of these different norms. We call p-norm the following function that depends on $p$:

<div>
$$
\norm{\bs{x}}_p=(\sum_i|\bs{x}_i|^p)^{1/p}
$$
</div>

In pratical terms, here is the recipe to get the $p$-norm of a vector:

1. Calculate the absolute value of each element
2. Take the power $p$ of these absolute values
3. Sum all these powered absolute values
4. Take the power $\frac{1}{p}$ of this result

This will be clear with examples using these widely used $p$-norms.

# The $L^0$ norm

If $p=0$, the formula becomes:

<div>
$$
\norm{\bs{x}}_0=(\sum_i|\bs{x}_i|^0)^{1/0}
$$
</div>

Let's see what it means. All positive values will get you a $1$ if you calculate its power $0$ except $0$ that will get you another $0$. Therefore this norm corresponds to the number of non-zero elements in the vector. It is not really a norm because if you multiply the vector by $\alpha$, this number is the same (rule 4 above).

# The $L^1$ norm

$p=1$ so this norm is simply the sum of the absolute values:

<div>
$$
\norm{\bs{x}}_1=\sum_{i} |\bs{x}_i|
$$
</div>

# The Euclidean norm ($L^2$ norm)

The Euclidean norm is the $p$-norm with $p=2$. This may be the more used norm with the squared $L^2$ norm.

<div>
$$
\norm{\bs{x}}_2=(\sum_i \bs{x}_i^2)^{1/2}\Leftrightarrow \sqrt{\sum_i \bs{x}_i^2}
$$
</div>

Let's see an example of this norm:

### Example 2.

Graphically, the Euclidean norm corresponds to the length of the vector from the origin to the point obtained by linear combination (like applying Pythagorean theorem).

<div>
$$
\bs{u}=
\begin{bmatrix}
    3 \\\\
    4
\end{bmatrix}
$$
</div>

<div>
$$
\begin{align*}
\norm{\bs{u}}_2 &=\sqrt{|3|^2+|4|^2}\\\\
&=\sqrt{25}\\\\
&=5
\end{align*}
$$
</div>


So the $L^2$ norm is $5$.

The $L^2$ norm can be calculated with the `linalg.norm` function from numpy. We can check the result:


```python
np.linalg.norm([3, 4])
```

<pre class='output'>
5.0
</pre>


Here is the graphical representation of the vectors:


```python
u = [0,0,3,4]

plt.quiver([u[0]],
           [u[1]],
           [u[2]],
           [u[3]],
           angles='xy', scale_units='xy', scale=1)

plt.xlim(-2, 4)
plt.ylim(-2, 5)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.annotate('', xy = (3.2, 0), xytext = (3.2, 4),
             arrowprops=dict(edgecolor='black', arrowstyle = '<->'))
plt.annotate('', xy = (0, -0.2), xytext = (3, -0.2),
             arrowprops=dict(edgecolor='black', arrowstyle = '<->'))

plt.text(1, 2.5, r'$\vec{u}$', size=18)
plt.text(3.3, 2, r'$\vec{u}_y$', size=18)
plt.text(1.5, -1, r'$\vec{u}_x$', size=18)

plt.show()
plt.close()
```


<img src="../../assets/images/2.5/l2-norm-vectors.png" width="300" alt="Vector illustration of the l2 norm" title="Vectors of the example 2.">
<em>Vectors of the example 2.</em>

In this case, the vector is in a 2-dimensional space but this stands also for more dimensions.

<div>
$$
u=
\begin{bmatrix}
    u_1\\\\
    u_2\\\\
    \cdots \\\\
    u_n
\end{bmatrix}
$$
</div>

<div>
$$
||u||_2 = \sqrt{u_1^2+u_2^2+\cdots+u_n^2}
$$
</div>


# The squared Euclidean norm (squared $L^2$ norm)

<div>
$$
\sum_i|\bs{x}_i|^2
$$
</div>


The squared $L^2$ norm is convenient because it removes the square root and we end up with the simple sum of every squared values of the vector. 

The squared Euclidean norm is widely used in machine learning partly because it can be calculated with the vector operation $\bs{x}^\text{T}\bs{x}$. There can be performance gain due to the optimization See [here](https://softwareengineering.stackexchange.com/questions/312445/why-does-expressing-calculations-as-matrix-multiplications-make-them-faster) and [here](https://www.quora.com/What-makes-vector-operations-faster-than-for-loops) for more details.

### Example 3.

<div>
$$
\bs{x}=
\begin{bmatrix}
    2 \\\\
    5 \\\\
    3 \\\\
    3
\end{bmatrix}
$$
</div>

<div>
$$
\bs{x}^\text{T}=
\begin{bmatrix}
    2 & 5 & 3 & 3
\end{bmatrix}
$$
</div>

<div>
$$
\begin{align*}
\bs{x}^\text{T}\bs{x}&=
\begin{bmatrix}
    2 & 5 & 3 & 3
\end{bmatrix} \times
\begin{bmatrix}
    2 \\\\
    5 \\\\
    3 \\\\
    3
\end{bmatrix}\\\\
&= 2\times 2 + 5\times 5 + 3\times 3 + 3\times 3= 47
\end{align*}
$$
</div>


```python
x = np.array([[2], [5], [3], [3]])
x
```

<pre class='output'>
array([[2],
       [5],
       [3],
       [3]])
</pre>



```python
euclideanNorm = x.T.dot(x)
euclideanNorm
```

<pre class='output'>
array([[47]])
</pre>



```python
np.linalg.norm(x)**2
```

<pre class='output'>
47.0
</pre>


It works!

## Derivative of the squared $L^2$ norm

Another advantage of the squared $L^2$ norm is that its partial derivative is easily computed:

<div>
$$
u=
\begin{bmatrix}
    u_1\\\\
    u_2\\\\
    \cdots \\\\
    u_n
\end{bmatrix}
$$
</div>

<div>
$$
\norm{u}_2 = u_1^2+u_2^2+\cdots+u_n^2
$$
</div>

<div>
$$
\begin{cases}
\dfrac{d\norm{u}_2}{du_1} = 2u_1\\\\
\dfrac{d\norm{u}_2}{du_2} = 2u_2\\\\
\cdots\\\\
\dfrac{d\norm{u}_2}{du_n} = 2u_n
\end{cases}
$$
</div>

## Derivative of the $L^2$ norm

In the case of the $L^2$ norm, the derivative is more complicated and takes every elements of the vector into account:

<div>
$$
\norm{u}_2 = \sqrt{(u_1^2+u_2^2+\cdots+u_n^2)} = (u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}}
$$
</div>

<div>
$$
\begin{align*}
\dfrac{d\norm{u}_2}{du_1} &=
\dfrac{1}{2}(u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}-1}\cdot
\dfrac{d}{du_1}(u_1^2+u_2^2+\cdots+u_n^2)\\\\
&=\dfrac{1}{2}(u_1^2+u_2^2+\cdots+u_n^2)^{-\frac{1}{2}}\cdot
\dfrac{d}{du_1}(u_1^2+u_2^2+\cdots+u_n^2)\\\\
&=\dfrac{1}{2}\cdot\dfrac{1}{(u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}}}\cdot
\dfrac{d}{du_1}(u_1^2+u_2^2+\cdots+u_n^2)\\\\
&=\dfrac{1}{2}\cdot\dfrac{1}{(u_1^2+u_2^2+\cdots+u_n^2)^{\frac{1}{2}}}\cdot
2\cdot u_1\\\\
&=\dfrac{u_1}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\end{align*}
$$
</div>

<div>
$$
\begin{cases}
\dfrac{d\norm{u}_2}{du_1} = \dfrac{u_1}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\dfrac{d\norm{u}_2}{du_2} = \dfrac{u_2}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\cdots\\\\
\dfrac{d\norm{u}_2}{du_n} = \dfrac{u_n}{\sqrt{(u_1^2+u_2^2+\cdots+u_n^2)}}\\\\
\end{cases}
$$
</div>

One problem of the squared $L^2$ norm is that it hardly discriminates between 0 and small values because the increase of the function is slow.

We can see this by graphically comparing the squared $L^2$ norm with the $L^2$ norm. The $z$-axis corresponds to the norm and the $x$- and $y$-axis correspond to two parameters. The same thing is true with more than 2 dimensions but it would be hard to visualize it.

$L^2$ norm:

<img src="../../assets/images/2.5/l2-norm.png" width="500" alt="Representation of the L2 norm" title="The L2 norm">
<em>The L2 norm</em>

Squared $L^2$ norm:

<img src="../../assets/images/2.5/squared-l2-norm.png" width="500" alt="Representation of the squared L2 norm" title="The squared L2 norm">
<em>The squared L2 norm</em>

$L^1$ norm:

<img src="../../assets/images/2.5/l1-norm.png" alt="Representation of the L1 norm" title="The L1 norm" width="500">
<em>The L1 norm</em>

These plots are done with the help of this [website](https://academo.org/demos/3d-surface-plotter/). Go and plot these norms if you need to move them in order to catch their shape.

# The max norm

It is the $L^\infty$ norm and corresponds to the absolute value of the greatest element of the vector.

<div>
$$
\norm{\bs{x}}_\infty = \max\limits_i|x_i|
$$
</div>

# Matrix norms: the Frobenius norm

<div>
$$
\norm{\bs{A}}_F=\sqrt{\sum_{i,j}A^2_{i,j}}
$$
</div>

This is equivalent to take the $L^2$ norm of the matrix after flattening.

The same Numpy function can be use:


```python
A = np.array([[1, 2], [6, 4], [3, 2]])
A
```

<pre class='output'>
array([[1, 2],
       [6, 4],
       [3, 2]])
</pre>



```python
np.linalg.norm(A)
```

<pre class='output'>
8.3666002653407556
</pre>


# Expression of the dot product with norms

<div>
$$
\bs{x}^\text{T}\bs{y} = \norm{\bs{x}}_2\cdot\norm{\bs{y}}_2\cos\theta
$$
</div>


### Example 4.

<div>
$$
\bs{x}=
\begin{bmatrix}
    0 \\\\
    2
\end{bmatrix}
$$
</div>

and

<div>
$$
\bs{y}=
\begin{bmatrix}
    2 \\\\
    2
\end{bmatrix}
$$
</div>


```python
x = [0,0,0,2]
y = [0,0,2,2]

plt.xlim(-2, 4)
plt.ylim(-2, 5)
plt.axvline(x=0, color='grey', zorder=0)
plt.axhline(y=0, color='grey', zorder=0)

plt.quiver([x[0], y[0]],
           [x[1], y[1]],
           [x[2], y[2]],
           [x[3], y[3]],
           angles='xy', scale_units='xy', scale=1)

plt.text(-0.5, 1, r'$\vec{x}$', size=18)
plt.text(1.5, 0.5, r'$\vec{y}$', size=18)

plt.show()
plt.close()
```


<img src="../../assets/images/2.5/dot-product-norm.png" width="300" alt="Expression of the dot product with norms" title="Expression of the dot product with norms">
<em>Expression of the dot product with norms</em>

We took this example for its simplicity. As we can see, the angle $\theta$ is equal to 45°.

<div>
$$
\bs{x^\text{T}y}=
\begin{bmatrix}
    0 & 2
\end{bmatrix} \cdot
\begin{bmatrix}
    2 \\\\
    2
\end{bmatrix} =
0\times2+2\times2 = 4
$$
</div>

and

<div>
$$
\norm{\bs{x}}_2=\sqrt{0^2+2^2}=\sqrt{4}=2
$$
</div>

<div>
$$
\norm{\bs{y}}_2=\sqrt{2^2+2^2}=\sqrt{8}
$$
</div>

<div>
$$
2\times\sqrt{8}\times cos(45)=4
$$
</div>

Here are the operations using numpy:


```python
# Note: np.cos take the angle in radian
np.cos(np.deg2rad(45))*2*np.sqrt(8)
```

<pre class='output'>
4.0000000000000009
</pre>

# References

- [Norm - Wikipedia](https://en.wikipedia.org/wiki/Norm_(mathematics))

- [3D plots](https://academo.org/demos/3d-surface-plotter/)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>