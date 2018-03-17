---
bg: "eeg.jpg"
layout: post
mathjax: true
title:  "First pilot EEG"
crawlertitle: "eeg"
summary: "First pilot eeg."
date:   2017-07-18 20:09:47 +0700
categories: posts
tags: ['eeg']
author: hadrien
---

For the first pilot with eeg recording (14 July 2017) SOA was not stored because it was undefined after some change: I had to replace

```js
data.targetSOA = stim.targetSOA;
```

with

```js
data.targetSOA = SOA;
```

It is possible to recover the target's SOA from the tone cloud because the average SOA in the tone cloud is the same as the target SOA.

```js
function(doc) {
  if (doc._id.indexOf('infMask_141') !== -1) {
    if (doc.toneCloudParam.length<600) {
      emit(doc.time, [doc.toneCloudParam.length, '4Hz']);
    } else if (doc.toneCloudParam.length>800 && doc.toneCloudParam.length<1100) {
      emit(doc.time, [doc.toneCloudParam.length, '7Hz']);
    } else if (doc.toneCloudParam.length>1200) {
      emit(doc.time, [doc.toneCloudParam.length, '13Hz']);
    }
  }
}
```

Also 1h still have to be added to the timestamp.