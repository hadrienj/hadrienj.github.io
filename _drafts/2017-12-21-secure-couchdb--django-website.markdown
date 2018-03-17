---
bg: "tools.jpg"
layout: post
mathjax: true
title:  "Secure couchdb + Django website"
crawlertitle: "Secure couchdb + Django website"
summary: "Use one 1-user 1-database scheme to secure CouchDB"
date:   2017-12-21 20:09:47 +0700
categories: posts
tags: ['couchdb', 'django', 'programming']
author: hadrien
---

CouchDB is great because everything is quite straigthforward and it works easily out of the box. However, some tricks are needed if you want to have a more precise control over your database. Here I'll introduce how to getting started with couchDB while securing your data. The aim is to end up with a database per user with each user having access to his own database only (read+write). This can be use to store data from a website after a user logged in. Since I'll use Django, I'll also rely on the Python package `couchdb-python`.

1. Getting started with couchDB
2. Adding the first layer of security: killing the admin party
3. Using `couchdb-python` to do basic operations
4. Create users
5. Create `_security` document for each database
6. Use SSL to secure the communication between the app and the database
7. Store credentials in the cookies
8. Use Pouchdb on the client-side and the library `Pouchdb-authentication` to use the database credentials and login.