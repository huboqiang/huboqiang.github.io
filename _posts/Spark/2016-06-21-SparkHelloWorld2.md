---
title: "[BigData-Spark] My first spark script for real data-analysis."
tagline: ""
last_updated: 2016-06-21
category: big-data related
layout: post
tags : [Spark]
---

#My first spark script for real data-analysis

## 1. Overview:

This is a revision version for my previous blog [My hello world script for py-spark](http://huboqiang.cn/2016/01/13/SparkHelloWorld) with the following revisions:

- From **pyspark** to pure **scala** spark
- Using **a new class with self-write new features** derived from a official class instead of **simply reduce**.

## 2. Input data

The input file were like this:

chr:begin-end|region|sub-region | value
---|----|----|---
chr1:10467-11448 | Satellite | telo | 0.00029762777551
chr1:11502-11676 | LINE | L1 | 0.0
chr1:11676-11781 | DNA | hAT-Charlie | 0.0
chr1:15263-15356 | SINE | MIR | 0.0
chr1:16711-16750 | Simple_repeat | Simple_repeat | 0.0
chr1:18905-19049 | LINE | L2 | 0.0
chr1:19946-20406 | LINE | CR1 | 0.0
chr1:20529-20680 | LINE | CR1 | 0.0
chr1:21947-22076 | LTR | ERVL-MaLR | 0.0
chr1:23118-23372 | SINE | MIR | 0.0


The output for all data is like:

Region | SubRegion | MeanValue | StdValue | CountValue | CountNA
-------|-----------|-----------|----------|------------|---
Low_complexity | Low_complexity | 0.008135001 | 0.055048064 | 360551 | 3693
Simple_repeat | Simple_repeat | 0.002655132 | 0.029989119 | 335446 | 3759
Other | Other | 0.001602282 | 0.002732717 | 3122 | 122
Satellite | Satellite | 6.49E-04 | 0.01428574 | 3942 | 399
rRNA | rRNA | 3.97E-04 | 0.004792962 | 1517 | 33
Satellite | centr | 3.10E-04 | 0.001632475 | 878 | 25
DNA | hAT? | 1.88E-04 | 0.005007405 | 2755 | 29
srpRNA | srpRNA | 1.77E-04 | 0.003470018 | 1324 | 25
tRNA | tRNA | 1.54E-04 | 0.002455723 | 1657 | 226
LTR | ERV | 1.40E-04 | 0.002945222 | 513 | 0
LTR | ERV1 | 1.11E-04 | 0.003349953 | 114367 | 2089
SINE | MIR | 8.56E-05 | 0.002599533 | 556387 | 4234
DNA | hAT-Tip100 | 5.16E-05 | 0.002051124 | 25863 | 357



## 3. The scala version script

The script ```bwIntervalAvg.scala``` is like this. Revising from the second chapter of [___Advanced Analytics with Spark___](https://github.com/sryza/aas/blob/master/ch02-intro/src/main/scala/com/cloudera/datascience/intro/RunIntro.scala)

```scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter


case class MatchData(chrpos1: String, chrpos2: String,
  chrpos3: String, value: Array[Double])


object bwIntervalAvg{

  def toDouble(s: String) = {
    if ("nan".equals(s)) Double.NaN else s.toDouble
  }

  def parse(line: String) = {
      val pieces = line.split("\t")
      val chrpos1 = pieces(0)
      val chrpos2 = pieces(1)
      val chrpos3 = pieces(2)
      val value   = pieces.slice(3, 4).map(toDouble)
      MatchData(chrpos1, chrpos2, chrpos3, value)
  }

  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("IntervalMeans")
    val sc = new SparkContext(conf)
    val inFile = args(0)
    val outFile = f"$inFile.out"

    val rawData = sc.textFile(inFile)
    val parsed = rawData.map(line => parse(line))
    val nasRDD = parsed.map(md => {
        ( (md.chrpos2, md.chrpos3), (md.value.map(d => NAStatCounter(d))) )
    })
    val rr = nasRDD.reduceByKey((n1, n2) => { n1.zip(n2).map { case (a, b) => a.merge(b) } })
    val results = rr.map{case(pos, statInfo) => (pos._1, pos._2, statInfo(0).stats.mean, statInfo(0).stats.stdev, statInfo(0).stats.count, statInfo(0).missing )}.map{case(chrpos2, chrpos3, meanval, stdevval, countval, missingval ) => f"$chrpos2\t$chrpos3\t$meanval\t$stdevval\t$countval\t$missingval"}

    results.saveAsTextFile(outFile)

  }
}


class NAStatCounter extends Serializable {
  val stats: StatCounter = new StatCounter()
  var missing: Long = 0

  def add(x: Double): NAStatCounter = {
    if (x.isNaN) {
      missing += 1
    } else {
      stats.merge(x)
    }
    this
  }

  def merge(other: NAStatCounter): NAStatCounter = {
    stats.merge(other.stats)
    missing += other.missing
    this
  }

  override def toString: String = {
    "stats: " + stats.toString + " NaN: " + missing
  }
}

object NAStatCounter extends Serializable {
  def apply(x: Double) = new NAStatCounter().add(x)
}


```

## 4. How to complile

Here, we used ```sbt``` for compiling, which is much easier than ```maven```

The organization for the file structure is like this:

```
├── build.sbt
├── src
    └── main
        └── scala
            └── bwIntervalAvg.scala
            
```

for building-up jar file:

```bash
sbt package
```

The result gets:

```
├── build.sbt
├── project
│   └── target
│       └── config-classes
│           ├── $768002c67292d8fd8de7$$anonfun$$sbtdef$1.class
│           ├── ...
├── src
│   └── main
│       └── scala
│           └── bwIntervalAvg.scala
└── target
    ├── resolution-cache
    │   ├── bw_interval_stat
    │   │   └── bw_interval_stat_2.10
    │   │       └── 1.0
    │   │           ├── resolved.xml.properties
    │   │           └── resolved.xml.xml
    │   └── reports
    │       ├── bw_interval_stat-bw_interval_stat_2.10-compile-internal.xml
    │       ├── ...
    ├── scala-2.10
    │   ├── bw_interval_stat_2.10-1.0.jar
    │   └── classes
    │       ├── bwIntervalAvg$$anonfun$1.class
    │       ├── ...
    └── streams
        ├── compile
        │   ├── ...
```

## 5. How to submit the job

The task could be initiated using  command ```spark-submit```:

```bash
spark-submit \
	bw_avg/target/scala-2.10/bw_interval_stat_2.10-1.0.jar \
	hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat
```

After a while, we can get the result:

```bash
hdfs dfs -ls "hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat.out"
```

```
Found 3 items
-rw-r--r--   3 hadoop supergroup          0 2016-06-21 16:36 hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat.out/_SUCCESS
-rw-r--r--   3 hadoop supergroup       4233 2016-06-21 16:36 hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat.out/part-00000
-rw-r--r--   3 hadoop supergroup       4385 2016-06-21 16:36 hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat.out/part-00001
```

To get the output data:

```bash
hdfs dfs -cat "hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat.out/*"
```

```
LINE?	Penelope?	0.0	0.0	51	0
Alpha,MIRb	SINE	0.0	0.0	1	0
Alpha,L1PA5,AluSc	Satellite	0.0	0.0	1	0
snRNA	snRNA	2.268034468340003E-5	4.7303688975106756E-4	3503	99
Beta,AluYa5	Satellite	0.0	0.0	1	1
Alpha,AluYc	Satellite	0.0	0.0	2	0
Alpha,LTR5_Hs	LTR	0.0	0.0	1	0
Alpha,AluSq	Simple_repeat	0.0	0.0	1	0
Alpha,HERVK11-int	LTR	0.0	0.0	1	0
Beta,MIR3	SINE	0.0	0.0	1	0
```

That's the result for what we want!


## 6. Plot in a local machine

As the output file were put in the HDFS of a remote machine, which made it hard for analysis, we need to fetch these data to a local PC and plot it using Rstudio. 

To make it easy, I installed hadoop on my laptop and using hdfs command to fetch data in a way recommand by [r-bloggers](http://www.r-bloggers.com/read-from-hdfs-with-r-brief-overview-of-sparkr/) 

```r
library(data.table)
library(ggplot2)
sdf_local <- fread('/Software/hadoop-2.6.0/bin/hdfs dfs -text "hdfs://tanglab1:9000/user/hadoop/genomics/MSC_KO_TM.mat.out/*"')

colnames(sdf_local) <- c("Region",	"SubRegion",	"MeanValue",	"StdValue",	"CountValue",	"CountNA")
df_order <- sdf_local[CountValue > 5000][order(MeanValue, decreasing = T)]

df_order$SubRegion <- factor(df_order$SubRegion, levels = unique(df_order$SubRegion))

theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black"),axis.ticks=element_line(colour="black"),plot.margin=unit(c(5,1,1,1),"line"))

p<-ggplot(df_order,aes(x=SubRegion,y=MeanValue, fill=factor(SubRegion)))
p<-p+geom_bar(position=position_dodge(), stat="identity")+
  ylab("Density")+labs(title="")+
  theme(axis.text.x=element_text(angle=45,hjust=1),legend.key=element_rect(fill=NA),legend.text = element_text(size=8))+theme

p +  coord_cartesian(ylim = c(0, 0.0002))
```

![png](/images/2016-06-21-SparkHelloWorld2/Fig1.png)