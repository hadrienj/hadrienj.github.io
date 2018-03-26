---
bg: "moss.jpg"
layout: post
mathjax: true
title: Deep Learning Book Series 2.4 Linear Dependence and Span
crawlertitle: "deep learning machine learning linear algebra python getting started numpy data sciences"
categories: posts
tags: ['linear-algebra', 'python', 'numpy', 'deep-learning-book']
author: hadrienj
jupyter: https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.4%20Linear%20Dependence%20and%20Span/2.4%20Linear%20Dependence%20and%20Span.ipynb
date: 2018-03-24 13:00:00
skip_span: true
---

<span class='notes'>
    This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). It aims to provide intuitions/drawings/python code on mathematical theories and is constructed as my understanding of these concepts. You can check the syllabus in the [introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/).
</span>

# Introduction

This chapter is quite heavy by its size and its content but I did what I could to make it more intuitive and visual. We will see how to represent systems of equations graphically, how to interpret the number of solutions of a system, what is linear combination and more. As usual, we will use Numpy/Matplotlib as a tool to experiment these concepts and hopefully gain a more concrete understanding.

# 2.4 Linear Dependence and Span

Since it is all about systems of linear equations, let's start again with the set of equations:

<div>
$$\bs{Ax}=\bs{b}$$
</div>

We saw in [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/) that this system corresponds to:

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 + \cdots + A_{1,n}x_n = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2 + \cdots + A_{2,n}x_n = b_2 \\\\
\cdots \\\\
A_{m,1}x_1 + A_{m,2}x_2 + \cdots + A_{m,n}x_n = b_n
$$
</div>

So we have multiple equations with multiple unknowns. We know $A_{1,1}...A_{m,n}$ and $b_1...b_n$. To solve the system we need to find the values of the variables $x_1...x_n$ that satisfies all equations.

# Number of solutions

The first thing to ask when we face such a system of equations is: what is the number of solutions ?

Three cases can represent the number of solutions of the system of equations $\bs{Ax}=\bs{b}$.
 
 1. No solution
 2. 1 solution
 3. An infinite number of solutions
 
## Why there can't be more than 1 solution and less than an infinite number of solutions ?

### Intuition

Simply because we deal with **linear** systems! Two lines can't cross more than once.

To be able to visualize it, let's take two dimensions and two equations. The solutions of the system correspond to the intersection of the lines. One option is that the two lines never cross (parallel). Another option is that they cross once. And finally, the last option is that they cross everywhere (superimposed):

<img src="../../assets/images/2.4/numberSolutions.png" width="800" alt="numberSolutions">

<span class='pquote'>
    Two lines can't cross more than once but can be either parallel or superimposed
</span>

### Proof

Let's imagine that $\bs{x}$ and $\bs{y}$ are two solutions of our system. This means that

<div>
$$
\begin{cases}
\bs{Ax}=\bs{b}\\\\
\bs{Ay}=\bs{b}
\end{cases}
$$
</div>

In that case, we will see that $\bs{z}=\alpha \bs{x} + (1-\alpha \bs{y})$ is also a solution for any value of $\alpha$. If $\bs{z}$ is a solution, we can say that $\bs{Az}=\bs{b}$. Indeed, if we plug $\bs{z}$ into the left hand side of the equation we obtain:

<div>
$$
\begin{align*}
\bs{Az}&=\bs{A}(\alpha x + (1-\alpha y))\\\\
    &=\bs{Ax}\alpha + \bs{A}(1-\alpha y)\\\\
    &=\bs{Ax}\alpha + \bs{Ay}(1-\alpha)
\end{align*}
$$
</div>

And since $\bs{Ax}=\bs{Ay}=\bs{b}$. This leads to:

<div>
$$
\begin{align*}
\bs{Az}&=\bs{b}\alpha + \bs{b}(1-\alpha)\\\\
    &=\bs{b}\alpha + \bs{b}-\bs{b}\alpha\\\\
    &=\bs{b}
\end{align*}
$$
</div>

So $\bs{z}$ is also a solution.

# Matrix representation of the system

As we saw it, the equation $\bs{Ax}=\bs{b}$ can be represented by a matrix $\bs{A}$ containing the weigths of each variable and a vector $\bs{x}$ containing each variable (see [2.2](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/)). The product of $\bs{A}$ and $\bs{x}$ gives $\bs{b}$ that is another vector of size $m$:

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

Which corresponds to the set of linear equations

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 + \cdots + A_{1,n}x_n = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2 + \cdots + A_{2,n}x_n = b_2 \\\\
\cdots \\\\
A_{m,1}x_1 + A_{m,2}x_2 + \cdots + A_{m,n}x_n = b_n
$$
</div>

Here are some intuitions about what is represented by these matrices. The number of columns of $\bs{A}$ is the number of dimensions of our vector space. It is the number $n$ of directions we can travel by. The number of solutions of our linear system corresponds to the number of ways we can reach $\bs{b}$ by travelling through our $n$ dimensions.

But to understand this, we need to underline that two possibilities exist to represent the system of equations: ***the row figure*** and ***the column figure***.

# Graphical views: Row and column figures


I recommend to look at [this video lesson of Gilbert Strang](http://ia802205.us.archive.org/18/items/MIT18.06S05_MP4/01.mp4). It provides a very nice intuition about these two ways of looking at a system of linear equations.


When you are looking to the matrix $\bs{A}$:

<div>
$$
\bs{A}=\begin{bmatrix}
    A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\\\
    A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\\\
    \cdots & \cdots & \cdots & \cdots \\\\
    A_{m,1} & A_{m,2} & \cdots & A_{m,n}
\end{bmatrix}
$$
</div>

You can consider its rows or its columns separately. Recall that the values are the weights corresponding to each variable. Each row synthetizes one equation. Each column is the set of weights given to 1 variable.

It is possible to draw a different graphical represention of the set of equations looking at the rows or at the columns.

## Graphical view 1: the row figure

The row figure is maybe more usual because it is the representation used when we have only one equation. It can now be extended to an infinite number of equations and unknowns (even if it would be hard to represent a 9-dimensional hyperplane in a 10-dimensional space...).

We said that the solutions of the linear system of equations are the sets of values of $x_1...x_n$ that satisfies all equations, that is to say, the values taken by the unknowns. For instance, in the case of $\bs{A}$ being a ($2 \times 2$) matrix ($n=m=2$) the equations correspond to lines in a 2-dimensional space and the solution of the system is the intersection of these lines.

Note that associating one direction in space to one parameter is only one way to represent the equations. There are number of ways to represent more than 3 parameters systems. For instance, you can add colors to have the representation of a fourth dimension. It is all about **representation**.

<img src="../../assets/images/2.4/3dAxes.png" width="900" alt="3dAxes">

### Overdetermined and underdetermined systems

A linear system of equations can be viewed as a set of $(n-1)$-dimensional hyperplanes in a *n*-dimensional space. So the linear system can be characterized with its number of equations ($m$) and the number of unknown variables ($n$).

- If there are more equations than unknows the system is called **overdetermined**. In the following example we can see a system of 3 equations (represented by 3 lines) and 2 unknowns (corresponding to 2 dimensions). In this example there is no solution since there is no point belonging to the three lines:

<img src="../../assets/images/2.4/overdeterminedSystem.png" width="300" alt="overdeterminedSystem">

- If there is more unknowns than equations the system is called **underdetermined**. In the following picture, there is only 1 equation (1 line) and 2 dimensions. Each point that is on the line is a solution of the system. In this case there is an infinite number of solutions:

<img src="../../assets/images/2.4/underdeterminedSystem.png" width="300" alt="underdeterminedSystem">


Let's see few examples of these different cases to clarify that.

### Example 1. 

$m=1$, $n=2$: **1 equation and 2 variables**

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 = b_1
$$
</div>

The graphical interpretation of $n=2$ is that we have a 2-D space. So we can represent it with 2 axes. Since our hyperplane is of $n-1$-dimensional, we have a 1-D hyperplane. This is simply a line. As $m=1$, we have only one equation. This means that we have only one line characterizing our linear system.

Note that the last equation can also be written in a way that may be more usual:

<div>
$$
y = ax + b
$$
</div>

with $y$ corresponding to $x_2$, $x$ corresponding to $x_1$, $a$ corresponding to $A_{1,1}$ and $A_{1,2}=1$.

For this first example we will take the following equation:

<div>
$$
y = 2x + 1
$$
</div>

Let's draw the line of this equation with Numpy and Matplotlib (see BONUS in [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) for light tips to plot equations).


```python
x = np.arange(-10, 10)
y = 2*x + 1

plt.figure()
plt.plot(x, y)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_13_0.png)


#### Solutions

The solutions of this linear system correspond to the value of $x$ and $y$ such as $y=2x+1$. Graphically, it corresponds to each point on the line so there is an infinite number of solutions. For instance, one solution is $x=0$ and $y=1$, or $x=1$ and $y=3$ and so on.

### Example 2.

*m*=2, *n*=2: **2 equations and 2 unknowns**

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 = b_1\\\\
A_{2,1}x_1 + A_{2,2}x_2 = b_2
$$
</div>

The graphical interpretation of this system is that we still have lines in a 2-D space. However this time there are 2 lines since there are 2 equations.

Let's take these equations as example:

<div>
$$
\begin{cases}
y = 2x + 1\\\\
y = 6x - 2
\end{cases}
$$
</div>



```python
x = np.arange(-10, 10)
y = 2*x + 1
y1 = 6*x - 2

plt.figure()
plt.plot(x, y)
plt.plot(x, y1)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_17_0.png)


As we have seen, with 2 lines in a 2-D space, there are multiple possible cases. On the above figure, the two lines are crossing so there is one unique solution. If they are superimposed (same equation or equivalent, *cf*. linear dependance bellow) there are a infinite number of solutions since each points of the lines corresponds to an intersection. If they are parallel, there is no solution.

The same thing can be observed with other values of $m$ (number of equations) and $n$ (number of dimensions). For instance, two 2-D planes in a 3-D space can be superposed (infinitely many solutions), or crossed (infinitely many solutions since their crossing is a line), or parallel (no solution).

### Example 3.

*m*=3, *n*=2: **3 equations and 2 unknowns**

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 = b_1\\\\
A_{2,1}x_1 + A_{2,2}x_2 = b_2\\\\
A_{3,1}x_1 + A_{3,2}x_2 = b_3
$$
</div>

The same idea stands with more than 2 equations in a 2-D space. In that example we have the following 3 equations:

<div>
$$
\begin{cases}
y = 2x + 1\\\\
y = 6x - 2\\\\
y = \frac{1}{10}x+6
\end{cases}
$$
</div>


```python
x = np.arange(-10, 10)
y = 2*x + 1

y1 = 6*x - 2
y2 = 0.1*x+6

plt.figure()
plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_20_0.png)


In the above case, there is 3 equations and no solution because there is no point in space that is on each of these lines.

## Linear combination

Before going to the column figure, we need to talk about linear combination. The linear combination of 2 vectors corresponds to their weighted sum.

### Example 4.

Let's take two vectors

<div>
$$
\vec{u}=
\begin{bmatrix}
    1 \\\\
    3
\end{bmatrix}
$$
</div>

and

<div>
$$
\vec{v}=
\begin{bmatrix}
    2 \\\\
    1
\end{bmatrix}
$$
</div>

These two vectors have 2 dimensions and thus contain coordinates in 2-D.


The linear combination of $\vec{u}$ and $\vec{v}$ is

<div>
$$
a\vec{u}+b\vec{v}= a
\begin{bmatrix}
    1 \\\\
    3
\end{bmatrix} + b\begin{bmatrix}
    2 \\\\
    1
\end{bmatrix}
$$
</div>

with $a$ and $b$ the weights of the vectors.

Graphically, the vectors are added to reach a specific point in space. For example if $a=2$ and $b=1$:

<div>
$$
2\vec{u}+\vec{v}= 2
\begin{bmatrix}
    1 \\\\
    3
\end{bmatrix} +
\begin{bmatrix}
    2 \\\\
    1
\end{bmatrix} =
\begin{bmatrix}
    2 \cdot 1 + 2 \\\\
    2 \cdot 3 + 1
\end{bmatrix} =
\begin{bmatrix}
    4 \\\\
    7
\end{bmatrix}
$$
</div>

The sum of $\vec{u}$ and $\vec{v}$ is a vector that will reach the point of corrdinates $(4, 7)$. To show that on a plot, I will use the custom function `plotVectors()` that I defined at the beginning of [the notebook](https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.4%20Linear%20Dependence%20and%20Span/2.4%20Linear%20Dependence%20and%20Span.ipynb). It takes a set of coordinates and an array of colors as input and plot the corresponding vectors. So let's plot $\vec{u}$ and $\vec{v}$:


```python
orange = '#FF9A13'
blue = '#1190FF'
plotVectors([[1, 3], [2, 1]], [orange, blue])
plt.xlim(0, 5)
plt.ylim(0, 5)
```

<pre class='output'>
(0, 5)
</pre>



![png](../../assets/images/2.4/output_23_1.png)


We will now add these vectors and their weights. This gives:


```python
# Weigths of the vectors
a = 2
b = 1
# Start and end coordinates of the vectors
u = [0,0,1,3]
v = [2,6,2,1]

plt.quiver([u[0], a*u[0], b*v[0]],
           [u[1], a*u[1], b*v[1]],
           [u[2], a*u[2], b*v[2]],
           [u[3], a*u[3], b*v[3]],
           angles='xy', scale_units='xy', scale=1, color=[orange, orange, blue])
plt.xlim(-1, 8)
plt.ylim(-1, 8)
# Draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.scatter(4,7,marker='x',s=50)
# Draw the name of the vectors
plt.text(-0.5, 2, r'$\vec{u}$', color=orange, size=18)
plt.text(0.5, 4.5, r'$\vec{u}$', color=orange, size=18)
plt.text(2.5, 7, r'$\vec{v}$', color=blue, size=18)
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_25_0.png)


We can see that we end up with the coordinates ($4$, $7$).

## Span

Take the vectors $\vec{u}$ and $\vec{v}$ from the previous example and think about all the points you can reach by their combination changing $a$ and $b$. This set of points is the span of the set of vectors $\{\vec{u}, \vec{v}\}$.

## Note on spaces and subspaces

(For more details see Strang (2006), p.70)

The space of a vector determines all the values that can be taken by this vector. The vector spaces are denoted $\mathbb{R}$ because the values are real numbers. If there are multiple dimensions the space is denoted $\mathbb{R}^n$ with $n$ corresponding to the number of dimensions. For instance $\mathbb{R}^2$ is the space of the usual $x$-$y$ plane where $x$ and $y$ values are real numbers.

If you take a 2-dimensional plane in $\mathbb{R}^3$ (3-dimensional space), this plane is a **subspace** of your original $\mathbb{R}^3$ space. On the same manner, if you start with a $\mathbb{R}^2$ space and take a line in this space, this line is a subspace of the original space.

The linear combination of vectors gives vectors in the original space. Every linear combination of vectors inside a space will stay in this space. For instance, if you take 2 lines in a $\mathbb{R}^2$ space, any linear combinations will give you a vector in the same $\mathbb{R}^2$ space.

<span class='pquote'>
    The linear combination of vectors gives vectors in the original space
</span>

## Graphical view 2: the column figure

It is also possible to represent the set of equations by considering that the solution vector $\bs{b}$ corresponds to a linear combination of each columns multiplied by their weights.

From the set of equations:

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 + A_{1,n}x_n = b_1 \\\\
A_{2,1}x_1 + A_{2,2}x_2 + A_{2,n}x_n = b_2 \\\\
\cdots \\\\
A_{m,1}x_1 + A_{m,2}x_2 + A_{m,n}x_n = b_m
$$
</div>

The column form is then:

<div>
$$
x_1
\begin{bmatrix}
    A_{1,1}\\\\
    A_{2,1}\\\\
    A_{m,1}
\end{bmatrix}
+
x_2
\begin{bmatrix}
    A_{1,2}\\\\
    A_{2,2}\\\\
    A_{m,2}
\end{bmatrix}
+
x_n
\begin{bmatrix}
    A_{1,n}\\\\
    A_{2,n}\\\\
    A_{m,n}
\end{bmatrix}
=
\begin{bmatrix}
    b_1\\\\
    b_2\\\\
    b_m
\end{bmatrix}
$$
</div>

On a graphical point of view, we have to travel from the origin (zero on every dimensions) to the point of coordinate $\bs{b}$. The columns of $\bs{A}$ give us the directions we can travel by and their weights are the length of the way in that direction.

<span class='pquote'>
     The columns of $\bs{A}$ give us the directions we can travel by and their weights are the length of the way in each direction.
</span>

### Example 5. 

$m=2$, $n=2$: 2 equations and 2 variables

<div>
$$
A_{1,1}x_1 + A_{1,2}x_2 = b_1\\\\
A_{2,1}x_1 + A_{2,2}x_2 = b_2
$$
</div>

<div>
$$
\begin{cases}
y = \frac{1}{2}x+1\\\\
y = -x + 4
\end{cases}
\Leftrightarrow
\begin{cases}
\frac{1}{2}x-y = -1\\\\
x+y=4
\end{cases}
$$
</div>

So here is the matrix $\bs{A}$:

<div>
$$
\bs{A}=
\begin{bmatrix}
    \frac{1}{2} & -1 \\\\
    1 & 1
\end{bmatrix}
$$
</div>

The column figure gives us:

<div>
$$
x
\begin{bmatrix}
    \frac{1}{2} \\\\
    1
\end{bmatrix}
+
y
\begin{bmatrix}
    -1 \\\\
    1
\end{bmatrix}
=
\begin{bmatrix}
    -1 \\\\
    4
\end{bmatrix}
$$
</div>

The goal is to find the value of the weights ($x$ and $y$) for which the linear combination of the vector

<div>
$$
\begin{bmatrix}
    \frac{1}{2} \\\\
    1
\end{bmatrix}
$$
</div>

and

<div>
$$
\begin{bmatrix}
    -1 \\\\
    1
\end{bmatrix}
$$
</div>

gives the vector 

<div>
$$
\begin{bmatrix}
    -1 \\\\
    4
\end{bmatrix}
$$
</div>

We will solve the system graphically by plotting the equations and looking for their intersection:


```python
x = np.arange(-10, 10)
y = 0.5*x + 1

y1 = -x + 4

plt.figure()
plt.plot(x, y)
plt.plot(x, y1)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_31_0.png)


We can see that the solution (the intersection of the lines representing our two equations) is $x=2$ and $y=2$. This means that the linear combination is the following:

<div>
$$
2
\begin{bmatrix}
    \frac{1}{2} \\\\
    1
\end{bmatrix}
+
2
\begin{bmatrix}
    -1 \\\\
    1
\end{bmatrix}
=
\begin{bmatrix}
    -1 \\\\
    4
\end{bmatrix}
$$
</div>

Let's say that 

<div>
$$
\vec{u}=
\begin{bmatrix}
    \frac{1}{2} \\\\
    1
\end{bmatrix}
$$
</div>

and

<div>
$$
\vec{v}=
\begin{bmatrix}
    -1 \\\\
    1
\end{bmatrix}
$$
</div>

To talk in term of the column figure we can reach the point of coordinates $(-1, 4)$ if we add two times the vector $\vec{u}$ and two times the vector $\vec{v}$. Let's check that:


```python
u = [0,0,0.5,1]
u_bis = [u[2],u[3],u[2],u[3]]
v = [2*u[2],2*u[3],-1,1]
v_bis = [2*u[2]-1,2*u[3]+1,v[2],v[3]]

plt.quiver([u[0], u_bis[0], v[0], v_bis[0]],
           [u[1], u_bis[1], v[1], v_bis[1]],
           [u[2], u_bis[2], v[2], v_bis[2]],
           [u[3], u_bis[3], v[3], v_bis[3]],
           angles='xy', scale_units='xy', scale=1, color=[blue, blue, orange, orange])
# plt.rc('text', usetex=True)
plt.xlim(-1.5, 2)
plt.ylim(-0.5, 4.5)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.scatter(-1,4,marker='x',s=50)
plt.text(0, 0.5, r'$\vec{u}$', color=blue, size=18)
plt.text(0.5, 1.5, r'$\vec{u}$', color=blue, size=18)
plt.text(0.5, 2.7, r'$\vec{v}$', color=orange, size=18)
plt.text(-0.8, 3, r'$\vec{v}$', color=orange, size=18)
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_33_0.png)


We can see that it is working! We arrive to the point ($-1$, $4$).

## Determine if the system has one and only one solution for every value of $\bs{b}$

We will now see how to determine if a system of equations has one and only one solution. Note that this is only the general cases. This can be split into two requirements:

1. The system must have at least one solution
2. Then, the system must have **only** one solution

### Requirement 1. Overdetermined system: the system must have at least one solution for each value of $\bs{b}$: $m\geq n$


<span class='pquote'>
     An overdetermined system of equations is a system with more equations than unknowns
</span>

The column figure is helpful to understand why the linear system has usually no solution if $n$ (the number of unknowns) is smaller than $m$ (the number of equations). Let's add 1 equation to the above system in order to end up with a ($3\times2$) matrix (3 equations and 2 unknowns):

<div>
$$
\begin{cases}
y = \frac{1}{2}x+1\\\\
y = -x + 4\\\\
y = 7x + 2
\end{cases}
\Leftrightarrow
\begin{cases}
\frac{1}{2}x-y = -1\\\\
x+y=4\\\\
7x-y=2
\end{cases}
$$
</div>

This corresponds to:

<div>
$$
x
\begin{bmatrix}
    \frac{1}{2} \\\\
    1 \\\\
    7
\end{bmatrix}
+
y
\begin{bmatrix}
    -1 \\\\
    1 \\\\
    -1
\end{bmatrix}
=
\begin{bmatrix}
    -1 \\\\
    4 \\\\
    2
\end{bmatrix}
$$
</div>

So we are still traveling in our 2-dimensional space (see the plot of the column space above) but the point that we are looking for is defined by 3 dimensions. There are cases where the third coordinate does not rely on our 2-dimensional $x$-$y$ plane. In that case no solution exists.

<span class='pquote'>
     We are traveling in a 2D space but the solution is defined by 3 dimensions. If the third coordinate does not rely on our 2D $x$-$y$ plane then there is no solution.
</span>

### Linear dependence

The number of columns can thus provide information on the number of solutions. But the number that we have to take into account is the number of **linearly independent** columns. Columns are linearly dependent if one of them is a linear combination of the others. Thinking in the column picture, the direction of two linearly dependent vectors is the same. This doesn't add a dimension that we can use to travel and reach $\bs{b}$.

Here is an example of linear system containing linear dependency:

<div>
$$
\begin{cases}
y = 2x+6\\\\
y = 2x
\end{cases}
\Leftrightarrow
\begin{cases}
2x-y = -6\\\\
2x-y=0
\end{cases}
$$
</div>

The row figure shows that the system has no solution:


```python
x = np.arange(-10, 10)
y = 2*x + 6

y1 = 2*x

plt.figure()
plt.plot(x, y)
plt.plot(x, y1)
plt.xlim(-2, 10)
plt.ylim(-2, 10)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_38_0.png)


Since the lines are parallel, there is no point at their intersection.

The column figure illustrates the point as well:

<div>
$$
x
\begin{bmatrix}
    2 \\\\
    2
\end{bmatrix}
+
y
\begin{bmatrix}
    -1 \\\\
    -1
\end{bmatrix}
=
\begin{bmatrix}
    -6 \\\\
    0
\end{bmatrix}
$$
</div>



```python
u = [0,0,2,2]
v = [0,0,-1,-1]

plt.quiver([u[0], v[0]],
           [u[1], v[1]],
           [u[2], v[2]],
           [u[3], v[3]],
           angles='xy', scale_units='xy', scale=1, color=[blue, orange])
plt.xlim(-7, 3)
plt.ylim(-2, 3)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.scatter(-6,0,marker='x',s=150)
plt.text(-6, 0.5, r'$b$', color='b', size=18)
plt.show()
plt.close()
```


![png](../../assets/images/2.4/output_41_0.png)


We would like to go to $b$ but the only path we can take is the blue/orange line. The second equation doesn't provide us with a new direction to take since it is just a linear combination of the first one.

### Requirement 2. Underdetermined system: the system must have **only** one solution for each value of $\bs{b}$: $n=m$

<span class='pquote'>
     An underdetermined system of equations is a system with less equations than unknowns
</span>

We saw that a requirement is that $n$ (the number of unknowns) must not be inferior to $m$ (the number of equations). But if we want our system to have one and only one solution a second requirement is that $n$ must not be bigger than $m$.

Let's take the example of a ($2\times 3$) matrix that corresponds to a set of 2 equations with 3 unknowns variables:


<div>
$$
\begin{cases}
8x+y+z=1\\\\
x+y+z=1
\end{cases}
$$
</div>

<div>
$$
x
\begin{bmatrix}
    8 \\\\
    1
\end{bmatrix}
+
y
\begin{bmatrix}
    1 \\\\
    1
\end{bmatrix}
+
z
\begin{bmatrix}
    1 \\\\
    1
\end{bmatrix}
=
\begin{bmatrix}
    1 \\\\
    1
\end{bmatrix}
$$
</div>

Here is the representation of the planes plotted with the help of this [website](https://technology.cpm.org/general/3dgraph/):

<img src="../../assets/images/2.4/2planes.png" alt="2planes" width="500">

We can see that in the best case the two planes are not parallel and there are solutions to the set of equations. It means that it exists some points that rely on both planes. But we can also see that there is inevitably an infinite number of points on the intersection (a line that we can see on the figure). We need a third plane to have a unique solution.

The resulting of all of this is that the system needs a **square matrix** $\bs{A}$ ($m=n$) with linearly independant columns to have a unique solution for every values of $\bs{b}$.

<span class='pquote'>
     The system needs a **square matrix** $\bs{A}$ ($m=n$) with linearly independant columns to have a unique solution for every values of $\bs{b}$
</span>

The inverse of a matrix exists only if the set of equations has one and only one solution for each value of $\bs{b}$ because:

- The matrix $\bs{A}$ cannot have more than 1 inverse. Imagine that $\bs{A}$ has 2 inverses $\bs{B}$ and $\bs{C}$ such as $\bs{AB}=\bs{I}$ and $\bs{AC}=\bs{I}$. This would mean that $\bs{B}=\bs{C}$.

- The solution of the system $\bs{Ax}=\bs{b}$ is $\bs{x}=\bs{A} ^{-1} \bs{b}$. So if there are multiple solutions, there are multiple inverses and the first point is not met.

For more details about the row and the column figure, have a look at the books of Gilbert Strang (there are some ressources [here](http://math.mit.edu/~gs/dela/dela_4-1.pdf)). There are tons of really great examples and graphical explanations! And the *1.2 Geometry of linear equations* in 'Linear algebra and its applications' also from Gilbert Strang.

# References

## Books and videos of Gilbert Strang

- Strang, G. (2006). Linear Algebra and Its Applications, 4th Edition (4th edition). Belmont, CA: Cengage Learning.

- Strang, G. (2014). Differential Equations and Linear Algebra (UK ed. edition). Wellesley, Mass: Wellesley-Cambridge.

- [The column space of a matrix. Video from Gilbert Strang](https://ocw.mit.edu/resources/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/differential-equations-and-linear-algebra/vector-spaces-and-subspaces/the-column-space-of-a-matrix/)

## System of equations

- [Wikipedia - System of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations)

## Numpy

- [Numpy arange()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)

<span class='notes'>
    Feel free to drop me an email or a comment. The syllabus of this series can be found [in the introduction post](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-Introduction/). All the notebooks can be found on [Github](https://github.com/hadrienj/deepLearningBook-Notes).
</span>