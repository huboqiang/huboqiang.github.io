---
layout: page
title: Boqiang Hu's Blog
tagline: Big data programming about bioinformatics
---
{% include JB/setup %}


## Self-introduction
My [Curriculum Vitae(CV)](http://qcloud-1252801552.file.myqcloud.com/huboqiangCV_01.svg).

## This blog mainly includes my study notes about

### 1. programming language:

- Python.
- R.
- Scala & Java.

### 2. big-data related:
- My [kaggle](https://www.kaggle.com) solution.
- Spark.
- Docker.
- MongoDB.


### 3. network related:
- Django.
- Javascript.


## The bioinformatic related notes were also included:

### 1. The latest paper:
- Software.
- News.
- Journal Club.

### 2. Introduction for my bioinformatic analysis framework:
- RNA Seq.
- - [Single cell human retina Expression level](http://198.13.42.241:9231/)
- ChIP Seq.
- DNA Methylation Seq.
- Whole Genome Sequencing.

## Latest Posts

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; {{ post.draft_flag }} <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
