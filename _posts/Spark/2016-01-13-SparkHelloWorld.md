---
title: "[BigData-Spark]My hello world script for py-spark."
tagline: ""
last_updated: 2016-01-11
category: big-data related
layout: post
tags : [Spark, py-spark, groupByKey]
---

#My hello world script for py-spark

## 1. The installtion for Java-Hadoop-Scala-Spark in a cluster:

This step were processed via the instuction in 

- [For a single computer](http://www.powerxing.com/install-hadoop/).
- [For a cluster](http://www.powerxing.com/install-hadoop-cluster/).
- [For spark](http://wuchong.me/blog/2015/04/04/spark-on-yarn-cluster-deploy/)

## 2. The helloworld script for pyspark.

Here, this book were highly recommended [Machine Learning With Spark](http://www.amazon.com/Machine-Learning-Spark-Powerful-Algorithms-ebook/dp/B00TXBLFB0), and I used the code in this book to analysis my data.

Now I have a 10G ```tab-split``` text file like this:

```bash
cat chrInfo.bed
```

We get:

```
chr1    10441   10500   -0.60   19_DNase
chr1    10461   10520   -0.60   19_DNase
chr1    10481   10540   -0.93   19_DNase
chr1    10501   10560   -0.98   19_DNase
chr1    10521   10580   -1.89   19_DNase
chr1    10541   10600   -2.33   19_DNase
chr1    10561   10620   -2.08   19_DNase
chr1    10561   10620   -2.08   25_Quies
chr1    10581   10640   -0.76   19_DNase
chr1    10581   10640   -0.76   25_Quies
```

There are different types of region in the 5th column and the average value for each type in the 4th column were wanted. 

At the beginning, this file should be put into the HDFS:

```bash
# code to put file in HDFS
/usr/local/hadoop/bin/hdfs dfs -put chrInfo.bed ./

# Let's see the HDFS
/usr/local/hadoop/bin/hdfs dfs -ls hdfs://tanglab1:9000/user/hadoop
# Out:
#-rw-r--r--   3 hadoop group  xxxxxxxxx 2016-01-13 15:57 chrInfo.bed
```

Then, a helloworld script for my hadoop:

### First, read the input file:

```python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from pyspark import SparkContext

sc = SparkContext()

sns.set_style("ticks")
sns.despine()

func_notInf = lambda x: 1 if x != float("inf") else 0

# read file
state_data = sc.textFile('hdfs://tanglab1:9000/user/hadoop/chrInfo.bed')
state_data_used = state_data.map(lambda line: line.split("\t"))

```

### Second, replace inf with the max value
```python
# get the maximum value
maxVal = state_data_used.map(lambda fields: float(fields[3])).filter(func_notInf).reduce(lambda x, y: max(x, y))

def func_giveVal(x):
    return_val = float(x)
    if x == "inf":
        return_val = maxVal
    elif x == "-inf":
        return_val = -1*maxVal
    return return_val
```
####1. replace the inf value and map region to the value.
####2. preparing the input for the reduce function.
   e.g. 
```[('Reg1', 1), ('Reg1', 2), ('Reg2', 3), ('Reg2', 4)] => ["Reg1" : [1, 2], "Reg2" : [3, 4]]```

```python
state_groupped_pval = state_data_used.map(lambda fields: (fields[4], func_giveVal(fields[3]) )).groupByKey().mapValues(list)
```

####3. using numpy to get the average value and s.e.m value. 
Admittedly, for get the average value, ```reduce(lambda x, y: x+y)/LengthOfRegion``` could be a better choice, but as **no good way for s.e.m or some thing like median**, here numpy were used.

```python
state_groupped_cnt = state_groupped_pval.map( lambda (k, v): (k, np.array(v, dtype="float").mean(), np.array(v, dtype="float").std()/len(v) ) )
```

####4. Put the RDD into system memory.

```python
l_groupped_cnt = state_groupped_cnt.collect()
pd_groupped_cnt = pd.DataFrame(l_groupped_cnt, columns = ['Type', 'Value', 'SE', 'Count'])
```

####5. Sort the frame according to the value by decreasing order, then put it into a text file for plotting.
```python
pd_groupped_cnt_out = pd_groupped_cnt.sort(['Value'], ascending=False)
pd_groupped_cnt_out.to_csv('/data/hadoop/study_spark/test.out.xls', index=False, sep="\t")
```

##3. The output file:

```bash
head /data/hadoop/study_spark/test.out.xls
```

We get:

```
Type	Value	SE
1_TssA	6.829177492209111	0.00022375394832033929
3_PromD1	2.769258220502902	8.951337243585849e-05
2_PromU	2.4881563055584124	6.514531579507373e-05
23_PromBiv	1.5236738222765807	7.65498892835764e-05
13_EnhA1	1.1349146014735432	5.5104538564808086e-05
16_EnhW1	0.9953765022620666	8.462728409314212e-05
9_TxReg	0.9611427572951248	5.266687733311639e-05
```