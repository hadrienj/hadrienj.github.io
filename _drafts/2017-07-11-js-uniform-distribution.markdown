---
bg: "tools.jpg"
layout: post
mathjax: true
title:  "Uniform distribution"
crawlertitle: "Stats"
summary: "Statistics"
date:   2017-07-11 20:09:47 +0700
categories: posts
tags: ['stats']
author: hadrien
---

This post is about checking that the `Math.random()` function in JS generate an uniform distribution.

I generated random numbers in JS:

```js
var arr = [];
for (var i=0; i<10000; i++) {
  arr.push(Math.random()*(1.5-0.1)+0.1);
}
```

I then exported this array in `R` and plotted the distribution.

```r
test <- read.csv('test.csv')
colnames(test) <- 'x'
ggplot(data=test, aes(x=x)) +
    geom_histogram()
```

I finally checked it with the `fitdistrplus` package.

```r
library(fitdistrplus)
descdist(test$x, discrete = FALSE)
```

Here is the resulting plot (broken legend):

![test](../../assets/images/distributionJS.png)

A very similar result is obtained with a uniform distribution created with `R`:

```r
test$x1 <- runif(10000, 0.1, 1.5)
descdist(test$x1, discrete = FALSE)
```

![](../../assets/images/distributionR.png)

