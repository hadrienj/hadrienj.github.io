---
bg: "ch09_system_of_linear_equations/road.jpg"
layout: post
mathjax: true
categories: posts
tags: ['essential-math', 'python', 'numpy']
author: hadrienj
date: 2021-01-13
excerpt: ""
excerpt-image: <img src="../../assets/images/ch09_system_of_linear_equations/ch09_system_of_linear_equations_3_0.png" width=300><em>Plot of the equation $y=2x+1$.</em>
twitterImg: "ch09_system_of_linear_equations/ch09_solutions_systems_equations"
title: "Essential Math for Data Science: Introduction to Systems of Linear Equations"
crawlertitle: ""
essential-math-sample: true
---


Systems of Linear Equations
===========================

In this article, you'll be able to use what you learned about vectors and matrices, and linear combinations (respectively Chapter 05, 06 and 07 of <a href="https://bit.ly/3oGHXyR">Essential Math for Data Science</a>). This will
allow you to convert data into systems of linear equations. At the end
of this chapter (in <a href="https://bit.ly/3oGHXyR">Essential Math for Data Science</a>), you'll see how you can use systems of equations and
linear algebra to solve a linear regression problem.

Linear equations are formalizations of the relationship between
variables. Take the example of a linear relationship between two
variables $x$ and $y$ defined by the following equation:

<div>
$$
y = 2x + 1
$$
</div>

You can represent this relationship in a Cartesian plane:

```python
# create x and y vectors
x = np.linspace(-2, 2, 100)
y = 2 * x + 1
plt.plot(x, y)
# [...] Add axes and styles
```

![Figure 1: Plot of the equation
$y=2x+1$.](../../assets/images/ch09_system_of_linear_equations/ch09_system_of_linear_equations_3_0.png){:width="350px"}
<em>Figure 1: Plot of the equation
$y=2x+1$.</em>



Remember that each point on the line corresponds to a solution of this
equation: if you replace $x$ and $y$ with the coordinates of a point on
the line in this equation, the equality is satisfied. This means that
there is an infinite number of solutions (every point in the line).

It is also possible to consider more than one linear equation using the
same variables: this is a *system of equations*.

System of linear equations
--------------------------

A system of equations is a set of equations describing the relationship
between variables. For instance, let's consider the following example:

<div>
$$
\begin{cases}
y &= 2x + 1\\\\
y &= -0.5x + 3
\end{cases}
$$
</div>

You have two linear equations and they both characterize the
relationship between the variables $x$ and $y$. This is a system with
two equations and two variables (also called *unknowns* in this
context).

You can consider systems of linear equations (each row of the system) as
multiple equations, each corresponding to a line. This is called the
*row picture*.

You can also consider the system as different columns
corresponding to coefficients scaling the variables. This is called the
*column picture*. Let's see more details about these two pictures.

### Row Picture

With the row picture, each row of the system corresponds to an equation.
In the previous example, there are two equations describing the
relationship between two variables $x$ and $y$.

#### Graphical Representation of the Row Picture

Let's represent the two equations graphically:

```python
# create x and y vectors
x = np.linspace(-2, 2, 100)
y = 2 * x + 1
y1 = -0.5 * x + 3
plt.plot(x, y)
plt.plot(x, y1)
# [...]
```

![Figure 2: Representation of the two equations from our
system.](../../assets/images/ch09_system_of_linear_equations/ch09_system_of_linear_equations_6_0.png){:width="350px"}
<em>Figure 2: Representation of the two equations from our
system.</em>



Having more than one equation means that the values of $x$ and $y$ must
satisfy more equations. Remember that the $x$ and $y$ from the first
equation are the same as the $x$ and $y$ from the second equation.

All points on the blue line satisfy the first equation and all points on
the green line satisfy the second equation. This means that only the
point on both lines satisfies the two equations. The system of equations
is solved when $x$ and $y$ take the values corresponding to the
coordinates of the line intersection.

In this example, this point has an $x$-coordinate of 0.8 and a
$y$-coordinate of 2.6. If you replace these values in the system of
equations, you have:

<div>
$$
\begin{cases}
2.6 &= 2 \cdot 0.8 + 1\\\\
2.6 &= (-0.5) \cdot 0.8 + 3
\end{cases}
$$
</div>

This is a geometrical way of solving the system of equations. The linear
system is solved for $x=0.8$ and $y=2.6$.

### Column Picture

Viewing the system as columns is called the column picture: you consider
your system as unknown values ($x$ and $y$) that scale vectors.

To better see this, let's rearrange the equations to have the variables
on one side and the constants on the other side. For the first, you
have:

<div>
$$
\begin{aligned}
y = 2x + 1 \\\\
y - 2x = 1
\end{aligned}
$$
</div>

and for the second:

<div>
$$
\begin{aligned}
y = -0.5x + 3 \\\\
y + 0.5x = 3
\end{aligned}
$$
</div>

You can now write the system as:

<div>
$$
\begin{cases}
y - 2x &= 1\\\\
y + 0.5x &= 3
\end{cases}
$$
</div>

You can now look at Figure 3 to see
how to convert the two equations into a single *vector equation*.

![Figure 3: Considering the system of equations as column vectors scaled
by the variables $x$ and
$y$.](../../assets/images/ch09_system_of_linear_equations/ch09_system_column_picture.png){:width="600px"}
<em>Figure 3: Considering the system of equations as column vectors scaled
by the variables $x$ and
$y$.</em>

On the right of Figure 3, you have
the vector equation. There are two column vectors on the left-hand side
and one column vector on the right-hand side. As you saw in <a href="https://bit.ly/3oGHXyR">Essential Math for Data Science</a>, this
corresponds to a linear combination of the following vectors:

<div>
$$
\begin{bmatrix}
    1 \\\\
    1
\end{bmatrix}
$$
</div>

and

<div>
$$
\begin{bmatrix}
    -2 \\\\
    0.5
\end{bmatrix}
$$
</div>

With the column picture, you replace multiple equations with a single
vector equation. In this perspective, you want to find the linear
combination of the left-hand side vectors that gives you the right-hand
side vector.

The solution in the column picture is the same. Row and column pictures
are just two different ways to consider the system of equations:

<div>
$$
\begin{aligned}
2.6 \begin{bmatrix}
    1 \\\\
    1
\end{bmatrix} +
0.8 \begin{bmatrix}
    -2 \\\\
    0.5
\end{bmatrix}
&=\begin{bmatrix}
    2.6 \cdot 1 \\\\
    2.6 \cdot 1
\end{bmatrix} +
\begin{bmatrix}
    0.8 \cdot (-2) \\\\
    0.8 \cdot 0.5
\end{bmatrix} \\\\
&=\begin{bmatrix}
    2.6 \\\\
    2.6
\end{bmatrix} +
\begin{bmatrix}
    -1.6 \\\\
    0.4
\end{bmatrix} \\\\
&=\begin{bmatrix}
    2.6 - 1.6 \\\\
    2.6 + 0.4
\end{bmatrix}=\begin{bmatrix}
    1 \\\\
    3
\end{bmatrix}
\end{aligned}
$$
</div>

It works: you get the right-hand side vector if you use the solution you found geometrically.

#### Graphical Representation of the Column Picture

Let's represent the system of equations considering it as a linear
combination of vectors. Let's take again the previous example:

<div>
$$
y \begin{bmatrix}
    1 \\\\
    1
\end{bmatrix} +
x \begin{bmatrix}
    -2 \\\\
    0.5
\end{bmatrix} =
\begin{bmatrix}
    1 \\\\
    3
\end{bmatrix}
$$
</div>

Figure 4 shows the
graphical representation of the two vectors from the left-hand side (the
vectors you want to combine, in blue and red in the picture) and the
vector from the right-hand side of the equation (the vector you want to
obtain from the linear combination, in green in the picture).

![Figure 4: Linear combination of the vectors scaled by $x$ and $y$
gives the right-hand
vector.](../../assets/images/ch09_system_of_linear_equations/ch09_column_picture_linear_combination.png){:width="400px"}
<em>Figure 4: Linear combination of the vectors scaled by $x$ and $y$
gives the right-hand
vector.</em>

You can see in Figure 4
that you can reach the right-hand side vector by combining the left-hand
side vectors. If you scale the vectors with the values 2.6 and 0.8, the
linear combination gets you to the vector on the right-hand side of the
equation.

### Number of Solutions

In some linear systems, there is not a unique solution. Actually, linear
systems of equations can have either:

-   No solution.
-   One solution.
-   An infinite number of solutions.

Let's consider these three possibilities (with the row picture and the
column picture) to see how it is impossible for a linear system to have
more than one solution and less than an infinite number of solutions.

#### Example 1. No Solution

Let's take the following linear system of equations, still with two
equations and two variables:

<div>
$$
\begin{cases}
y &= 2x + 1\\\\
y &= 2x + 3
\end{cases}
$$
</div>

We'll start by representing these equations:

```python
# create x and y vectors
x = np.linspace(-2, 2, 100)
y = 2 * x + 1
y1 = 2 * x + 3

plt.plot(x, y)
plt.plot(x, y1)
# [...] Add axes, styles...
```

![Figure 5: Parallel equation
lines.](../../assets/images/ch09_system_of_linear_equations/ch09_system_of_linear_equations_13_0.png){:width="300px"}
<em>Figure 5: Parallel equation
lines.</em>



As you can see in Figure 5, there
is no point that is on both the blue and green lines. This means that
this system of equations has no solution.

You can also understand graphically why there is no solution through the
column picture. Let's write the system of equations as follows:

<div>
$$
\begin{cases}
y - 2x &= 1\\\\
y - 2x &= 3
\end{cases}
$$
</div>

Writing it as a linear combination of column vectors, you have:

<div>
$$
y \begin{bmatrix}
    1 \\\\
    1
\end{bmatrix} +
x \begin{bmatrix}
    2 \\\\
    2
\end{bmatrix} =
\begin{bmatrix}
    1 \\\\
    3
\end{bmatrix}
$$
</div>

![Figure 6: Column picture of a linear system with no
solution.](../../assets/images/ch09_system_of_linear_equations/ch09_column_picture_no_solution.png){:width="300px"}
<em>Figure 6: Column picture of a linear system with no
solution.</em>

Figure 6 shows the column
vectors of the system. You can see that it is impossible to reach the
endpoint of the green vector by combining the blue and the red vectors.
The reason is that these vectors are linearly dependent (more details in
<a href="https://bit.ly/3oGHXyR">Essential Math for Data Science</a>). The vector to reach is outside of the span of the vectors you
combine.

#### Example 2. Infinite Number of Solutions

You can encounter another situation where the system has an infinite
number of solutions. Let's consider the following system:

<div>
$$
\begin{cases}
y &= 2x + 1\\\\
2y &= 4x + 2
\end{cases}
$$
</div>

```python
# create x and y vectors
x = np.linspace(-2, 2, 100)
y = 2 * x + 1
y1 = (4 * x + 2) / 2

plt.plot(x, y)
plt.plot(x, y1, alpha=0.3)
# [...] Add axes, styles...
```

![Figure 7: The equation lines are
overlapping.](../../assets/images/ch09_system_of_linear_equations/ch09_system_of_linear_equations_16_0.png){:width="350px"}
<em>Figure 7: The equation lines are
overlapping.</em>



Since the equations are the same, an infinite number of points are on
both lines and thus, there is an infinite number of solutions for this
system of linear equations. This is for instance similar to the case
with a single equation and two variables.

From the column picture perspective, you have:

<div>
$$
\begin{cases}
y - 2x &= 1\\\\
2y - 4x &= 2
\end{cases}
$$
</div>

and with the vector notation:

<div>
$$
y \begin{bmatrix}
    1 \\\\
    2
\end{bmatrix} +
x \begin{bmatrix}
    2 \\\\
    4
\end{bmatrix} =
\begin{bmatrix}
    1 \\\\
    2
\end{bmatrix}
$$
</div>

![Figure 8: Column picture of a linear system with an infinite number of
solutions.](../../assets/images/ch09_system_of_linear_equations/ch09_column_picture_infinite_solutions.png){:width="300px"}
<em>Figure 8: Column picture of a linear system with an infinite number of
solutions.</em>

Figure 8 shows the
corresponding vectors graphically represented. You can see that there is
an infinite number of ways to reach the endpoint of the green vector
with combinations of the blue and red vectors.

Since both vectors go in the same direction, there is an infinite number
of linear combinations allowing you to reach the right-hand side vector.

#### Summary

To summarize, you can have three possible situations, shown with two
equations and two variables in Figure
9.

![Figure 9: Summary of the three situations for two equations and two
variables.](../../assets/images/ch09_system_of_linear_equations/ch09_solutions_systems_equations.png){:width="650px"}
<em>Figure 9: Summary of the three situations for two equations and two
variables.</em>

It is impossible to have two lines crossing more than once and less than
an infinite number of times.

The principle holds for more dimensions. For instance, with three planes
in $\setR^3$, at least two can be parallel (no solution), the three can
intersect (one solution), or the three can be superposed (infinite
number of solutions).

### Representation of Linear Equations With Matrices {#sec:ch07_section_representation_of_linear_equations_with_matrices}

Now that you can write vector equations using the column picture, you
can go further and use a matrix to store the column vectors.

Let's take again the following linear system:

<div>
$$
y \begin{bmatrix}
    1 \\\\
    1
\end{bmatrix} + x \begin{bmatrix}
    -2 \\\\
    0.5
\end{bmatrix} = \begin{bmatrix}
    1 \\\\
    3
\end{bmatrix}
$$
</div>

Remember from <a href="https://bit.ly/3oGHXyR">Essential Math for Data Science</a> that you can write linear combinations as a
matrix-vector product. The matrix corresponds to the two column vectors
from the left-hand side concatenated:

<div>
$$
\begin{bmatrix}
    1 & -2 \\\\
    1 & 0.5
\end{bmatrix}
$$
</div>

And the vector corresponds to the coefficients weighting the column
vectors of the matrix (here, $x$ and $y$):

<div>
$$
\begin{bmatrix}
    y \\\\
    x
\end{bmatrix}
$$
</div>

Your linear system becomes the following matrix equation:

<div>
$$
\begin{bmatrix}
    1 & -2 \\\\
    1 & 0.5
\end{bmatrix}
\begin{bmatrix}
    y \\\\
    x
\end{bmatrix}
= \begin{bmatrix}
    1 \\\\
    3
\end{bmatrix}
$$
</div>

#### Notation

This leads to the following notation widely used to write linear
systems:

<div>
$$
\mA \vx = \vb
$$
</div>

with $\mA$ the matrix containing the column vectors, $\vx$ the vector of
coefficients and $\vb$ the resulting vector, that we'll call the *target
vector*. It allows you to go from calculus, where equations are
considered separately, to linear algebra, where every piece of the
linear system are represented as vectors and matrices. This abstraction
is very powerful and brings vector space theory to solve systems of
linear equations.

With the column picture, you want to find the coefficients of the linear
combination of the column vectors on the left-hand side of the equation.
The solution exists only if the target vector is within their span.


<div style="text-align: center; font-size: 3.5rem; font-weight: bold; color: #c9c9c9">...</div>
