---
bg: "tools.jpg"
layout: post
mathjax: true
title: Use reduce function in js
date: 2017-12-05
categories: posts
tags: ['']
author: hadrienj
---

As an introduction to the reduce function, let's start with a simple exemple:

```js
const a = [5, 5, 5, 7.5, 11.25, 7.5, 5, 5, 7.5, 11.25, 7.5, 6, 6, 7.5];
a.reduce((total, curr)=>console.log(total, curr));
```

gives

```js
5 5
undefined 5
undefined 7.5
undefined 11.25
undefined 7.5
undefined 5
undefined 7.5
undefined 11.25
undefined 7.5
undefined 6
undefined 7.5
undefined
```

The first value of `total` is 5 (the first value of the array) and then `undefined`. It is because we return nothing of our function used in reduce. Let's try to return something:

```js
a.reduce((total, curr)=>{
    console.log(total, curr);
    return curr;
});
```

outputs

```js
5 5
5 7.5
7.5 11.25
11.25 7.5
7.5 5
5 5
5 7.5
7.5 11.25
11.25 7.5
7.5 6
6 6
6 7.5
7.5
```

We can see that `total` is now the preceding value since we return the current value (that becomes the preceding in the next iteration).

Let's change its argument name to make the things clear:

```js
a.reduce((prev, curr)=>{
    console.log(prev, curr);
    return curr;
});
```


## Example: calculate a rolling difference

We will use `reduce()`, `map()` and `filter()` to calculate the rolling difference of an array of values. The rolling difference is difference between the first and the second element, between the second and the third and so on. So we should end up with a new array containing `n-1` values if the originel array is of length `n`. Here is the object we will use:

```json
const obj = [
   {
       "id": "id0.6999864898224961",
       "time": 0.5854679153668751,
       "dur": 0.1,
       "freq": 490.4290771484375,
       "gain": 0.005
   },
   {
       "id": "id0.5836270474184808",
       "time": 0.9984754039796808,
       "dur": 0.1,
       "freq": 603.118896484375,
       "gain": 0.005
   },
   {
       "id": "id0.6999864898224961",
       "time": 1.0320680105058262,
       "dur": 0.1,
       "freq": 879.6334228515625,
       "gain": 0.005
   },
   {
       "id": "id0.6999864898224961",
       "time": 1.0624564315067886,
       "dur": 0.1,
       "freq": 1181.87158203125,
       "gain": 0.005
   },
   {
       "id": "id0.4568650934623717",
       "time": 1.0884754359174753,
       "dur": 0.1,
       "freq": 1060.1126708984375,
       "gain": 0.005
   },
   {
       "id": "id0.6999864898224961",
       "time": 1.5664517594370437,
       "dur": 0.1,
       "freq": 405.18475341796875,
       "gain": 0.005
   }
]
```

First, we want to keep only objects with a specific `id`:

```js
obj.filter(x=>{
    return x.id === "id0.6999864898224961";
})
```

returns

```json
[
    {
        "id":"id0.6999864898224961",
        "time":0.5854679153668751,
        "dur":0.1,
        "freq":490.4290771484375,
        "gain":0.005
    },
    {
        "id":"id0.6999864898224961",
        "time":1.0320680105058262,
        "dur":0.1,
        "freq":879.6334228515625,
        "gain":0.005
    },
    {
        "id":"id0.6999864898224961",
        "time":1.0624564315067886,
        "dur":0.1,
        "freq":1181.87158203125,
        "gain":0.005
    },
    {
        "id":"id0.6999864898224961",
        "time":1.5664517594370437,
        "dur":0.1,
        "freq":405.18475341796875,
        "gain":0.005
    }
]
```

Then we keep only the `time` property of these objects

```js
obj.filter(x=>{
    return x.id === "id0.6999864898224961";
}).map(x=>{
    return x.time;
})
```

returns

```json
[
    0.5854679153668751,
    1.0320680105058262,
    1.0624564315067886,
    1.5664517594370437
]
```

And now we actually compute the rolling difference

```js
const rollDiff = [];
obj.filter(x=>{
    return x.id === "id0.6999864898224961";
}).map(x=>{
    return x.time;
}).reduce((accu, curr, index, arr)=>{
  rollDiff.push(curr-arr[index-1]);
});
```

returns

```json
[
    0.44660009513895105,
    0.030388421000962396,
    0.5039953279302551
]
```

