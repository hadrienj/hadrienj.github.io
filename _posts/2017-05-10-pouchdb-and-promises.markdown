---
bg: "tools.jpg"
layout: post
mathjax: true
title:  "PouchDB and promises"
crawlertitle: "PouchDB and promises"
summary: "PouchDB and the use of promises."
date:   2017-05-10 20:09:47 +0700
categories: posts
tags: ['programming']
author: hadrien
---

PouchDB and the use of promises.

```js
function retryUntilWritten(doc) {
  return db.get(doc._id).then(function (origDoc) {
    doc._rev = origDoc._rev;
    return db.put(doc);
  }).catch(function (err) {
    if (err.status === 409) {
      return retryUntilWritten(doc);
    } else { // new doc
      return db.put(doc);
    }
  });
}
```

In this function the `catch` statement concerns the previous `db.get()` or `db.put()`. Errors threw by `db.put()` can be catched because there is a `return` statement.

It should be an habbit to return the next block from a `then()` statement (see [here](https://pouchdb.com/2015/05/18/we-have-a-problem-with-promises.html)).


# Revisions

A conflict can occur if the same `_rev` is used even with a different `_id`:

```js
var localDB = new PouchDB('test')
localDB.put({_id:'a'})

localDB.get('a').then((doc)=> {
    console.log('doc', doc);
  doc._id = 'b';
  return localDB.put(doc);
}).catch((err)=> {
  console.log('err', err);
});
```


