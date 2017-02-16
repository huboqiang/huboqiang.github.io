---
title: "[20170216 Python Spark]使用 IBM datascience 平台统计 hg38每条染色体基因,转录本的分布."
tagline: ""
last_updated: 2017-02-16
category: BigData
layout: post
tags : [BigData]
---

# 1. 前言

这是一篇以生物信息学入门习题为例的大数据教程。具体而言，就是在 IBM 云计算平台，使用 pySpark 完成一个很简单的任务。任务描述如下：

```
每条染色体基因个数的分布？

所有基因平均有多少个转录本？

所有转录本平均有多个exon和intron？

注释文件一般以gtf/gff格式记录着！

基础作业，就是对这个文件 ftp://ftp.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh38.87.chr.gtf.gz 进行统计，普通人只需要关心第1，3列就好了。
```

源地址来自生信技能树 [http://www.biotrainee.com/thread-626-1-1.html](http://www.biotrainee.com/thread-626-1-1.html)

这些代码可以使用 IBM data science 平台( [http://datascience.ibm.com/](http://datascience.ibm.com/) )的 Notebook 功能直接执行。

IBM data science 平台对注册用户首月免费，默认提供一个 2核 CPU，预装 Rstudio, Jupyter。同时 IBM 公司作为 Spark 项目的重要商业支持者，对自家的Spark 大数据计算完美支持。

更重要的是，这个平台提供了很好的数据科学家相互交流的机会。编写的代码可以轻松在技术人员之间直接传阅，写完代码，最后的结果可以直接发给老板。

如果需要使用，首先需要在网站完成注册：

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/step1.png)

注册完成后，选择 DataHub

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/step2.png)

然后建立 Notebook，建立后的 Notebook 会在下面列出。

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/step3.png)

如果希望写个 HelloWorld，推荐你去简单画个图，源码位于 [matplotlib 官方 gallery](http://matplotlib.org/examples/pylab_examples/barchart_demo.html)，我们开个选择 Python，生成Jupyter Notebook 画一下：

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/step4.png)

然后把 matplotlib 上的这个例子的代码复制粘贴下来：



```python
import numpy as np
import matplotlib.pyplot as plt

n_groups = 5

means_men = (20, 35, 30, 35, 27)
std_men = (2, 3, 4, 1, 2)

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=std_men,
                 error_kw=error_config,
                 label='Men')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=std_women,
                 error_kw=error_config,
                 label='Women')

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
plt.legend()

plt.tight_layout()
plt.show()

```

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/step5.png)

如图操作，就可以得到 matplotlib 官网上的图。

神马？没有出图像？来，这里有个特殊的地方，需要在 import 完所有库之后，加一行 ```%matplotlib inline``` 魔法，允许直接在代码块下面显示，就像我图中写的那样。

Jupyter Notebook 和 Rstudio 一样有很多方便好用的快捷键。具体可以点击 Help => ShortCut 看一下。Help 位于箭头挡住的 run 又上方。


如果不使用 IBM data science 平台，也可以自己下载 [anaconda](https://www.continuum.io/downloads) 安装科学计算版 Python。但这样 [spark](http://spark.apache.org/downloads.html) 需要自己预装，特别是Spark 通常需要[预装 Hadoop](http://wuchong.me/blog/2015/04/04/spark-on-yarn-cluster-deploy/)，对生信菜鸟比较费事，大型机上管理员也不一定让你装。不过 anaconda 本身不使用 spark 加成，开 Jupyter Notebook 就已经十分强大了，建议大家试一试。

我在我们的大型机的一个计算节点装好 anaconda 后，根据 Jupyter Notebook 官方文档，设定集群访问[http://jupyter-notebook.readthedocs.io/en/latest/public_server.html](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html)，需要分析项目，会首先 cd 到项目所在的分析文件夹(鄙视放进 /home 目录里的人)， 接着 cmd 输入 ```jupyter notebook```，这样jupyter 会在后端挂起，然后访问 ```https://IP:PORT```，IP 是该集群的内网 IP，端口在上一步指定，默认 8888，注意是这里是 https 不是 http，然后允许打开网页，输入集群访问密码，就会进入管理页面，如图。

新建Python Notebook 后，会直接进入该文件，管理页面里面会出现一个绿色的 ipynb 文件，linux shell 里面也能看见。

这个文件就是Jupyter Notebook所在的文件，用法与 IBM datascience 的完全相同，大家也可以照着上图 HelloWorld 一下。


![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/step6.png)

我这里建议，如果想体验一把 PySpark，使用 IBM data science ，即使是菜鸟，也可以来体验一把高大上的大数据+云计算。

默认环境配置如下：

```
Instance
DataSciX

Language
Python 2.7

Spark as a Service
Apache Spark 2.0

Preinstalled Libraries
biopython-1.66
bitarray-0.8.1
brunel-1.1
iso8601-0.1.11
jsonschema-2.5.1
lxml-3.5.0
matplotlib-1.5.0
networkx-1.10
nose-1.3.7
numexpr-2.4.6
numpy-1.10.4
pandas-0.17.1
Pillow-3.0.0
pip-8.1.0
pyparsing-2.0.6
pytz-2015.7
requests-2.9.1
scikit-learn-0.17
scipy-0.17.0
simplejson-3.8.1
```



当然，在运行程序前，需要预装 python 的类似 ggplot2 画图包，方便作图。

最前头的感叹号的意思是这一行执行 shell 脚本。也就是说这个命令本应在 linux shell 里面执行，但由于 jupyter 把 shell 也给完美的集成了进来，所以在 notebook 中写就 OK。



代码块【1】：

```python
!pip install --user seaborn
```

结果是：
```Requirement already satisfied (use --upgrade to upgrade): seaborn in /gpfs/global_fs01/sym_shared/YPProdSpark/user/sa9e-127e054d347dc8-cc04bd2554cc/.local/lib/python2.7/site-packages```
这里我是因为已经装过，否则会显示下载进度以及提示安装完成。


接下来的程序不以 ```！``` 开头，写的就是 python 代码了。这几行是载入需要用到的包。

代码块【2】：

```python
# -*- coding: utf-8 -*-
import pandas  as pd               
import matplotlib.pyplot as plt    
import seaborn as sns              
import requests, StringIO, json    
import re                         
import numpy as np                

%matplotlib inline                
sns.set_style("white")            
```

简单解释下：

```
import pandas  as pd               # 数据分析包
import matplotlib.pyplot as plt    # 作图包
import seaborn as sns              # 类似 ggplot2 的高级作图包
import requests, StringIO, json    # 在IBM datascience 的存储系统里读取数据必须
import re                          # 正则表达式
import numpy as np                 # 矩阵运算，这里用在最后算中位数

%matplotlib inline                 # 允许作图函数执行后，在代码块下面直接显示所画的图
sns.set_style("white")             # 类似 ggplot2 的 theme_bw()
```

# 2. Jupyter 可以借助 Spark 轻松实现 Python 的多核心编程

看起来 Jupyter 既可以像 Rstudio 一样轻松的写 python 代码，又可以在代码块的上下写各种注释，一边写程序，一边出图，然后直接写图注，文章的逻辑写在旁边，是非常完美的轻量级科研利器。

Jupyter + pyspark 虽然轻量，但其实力气一点都不小。写出来的性能，在__某种意义上甚至高于 C++ Java 这样的低级语言__。我说某种意义，指的是单核运算方面的瓶颈。因为自从本世纪初我们耳熟能详的奔腾4处理器的主频，已经高达1.4GHz了，大家可以去看看自己电脑的主频，会发现这么些年来虽然 CPU 各种进步，主频的提升实际上是非常小的。CPU 的摩尔定律，主要还是在 __核心数以及线程数__ 的提升。家用笔记本现在很多都是2核4线程，而服务器的单 CPU 线程数一般也都在 10 个以上。如果还是通过 for 循环读取各行，就会导致“一核干活，十核围观”，对计算资源造成闲置浪费。

![png](http://v.wxrw123.com/?url=http://qbwx.qpic.cn/mmbiz_jpg/XelFhOre4tZFBEtjRnc8BNUHN5btObokloic92ibaguo8picPp0wAOhdIV8kFAWibuHYATSyFsibJSLNQdHNQgC31YA/0?wx_fmt=jpeg)


所以，为了进一步跟上时代潮流，重要的软件程序，我们都使用多核心编程技术。我们生物信息领域很多耳熟能详的软件，如比对用的 bwa bowtie 的参数，都有使用几个核心的选项。

那么我们能不能也轻松写一个多核心程序出来呢？C++ 的套路是，用多核心就要调 pthread 库，通常是先在 IO 读取数据，缓存1,000,000行，然后通过 pthread 把缓存里面的各行扔给需要调用的函数。具体把哪一行扔给函数，也需要自己指定，比如当前的行数取余数，余几就扔给几号CPU。然后还需要预留一块内存接各个CPU 执行函数的输出结果，不能直接输出。。。可能菜鸟已经听晕了，不知道在说什么，而听懂的人想必是清楚其中的麻烦是我这几行远远没有说明白的。

这一问题在 Python 和 R 中也或多或少的存在。这两种语言系统支持的多线程技术调起来也只是稍微简单一些，而性能却没法和C++比。于是乎，在这个大数据的时代背景下，他们抱上了 Hadoop Spark 这些最新的大数据工具的大腿。特别是 Spark。

Spark 源码是通过一种叫做 Scala 的语言编写的。Scala 是脱胎于 java 的一种更高效的编程语言，一般人还真不会用，于是 Spark 项目就打通了 Python R 的使用接口。然而为了保证版本升级的进度，Spark 的新功能一般是首先 Java Scala 能用，然后轮到 Python，最后才到 R。比如 Spark 的机器学习库，目前 Python 已经能很好支持了，而 R语言得等到 2.2.0（16年11月 IBM 的 Spark机器学习库编写人员亲口所说）。

虽然 PySpark 用的是一种不完整的 Spark，但用它对列式数据（R 中的 dataframe 类型）搞分组求和、文件清洗，已经足够了。更重要的是，这里由于是和数据科学界接轨，强烈推荐把数据简单处理后（抓取信息，规定每一列的名称，扔掉某些行），放进 SparkSQL中，用 SQL 语句，用 __人话__ 而不是代码，去人机交互，分析数据。

最后，多说一句，就是实际上 Spark 能做的已经不是单机多核心了，哪怕是上百台电脑，处理和本文一模一样的一个 TB 级的基因注释GTF文件（就当是外星人的），代码怎么写？一模一样，只要 Spark 指挥的 Hadoop 集群被合理的配置好，PySpark 代码方面一模一样，上百台电脑，上千个 CPU 核心，共同处理同一文件。当然这个文件需要被放入 [HDFS 分布式存储系统](https://www-01.ibm.com/software/data/infosphere/hadoop/hdfs/)中，命令也很简单：

```/hadoop/bin/hdfs dfs -put 外星人.GTF hdfs://[HDFS系统IP]:[HDFS系统端口]:[文件路径/外星人.GTF]```

继续说正事，分析数据之前，必须做的一件事，就是上传数据。而上传数据的第一步，是得把数据先给下载下来。

我们的数据，就是从 ftp://ftp.ensembl.org/pub/releas ... RCh38.87.chr.gtf.gz 下载的压缩状态的gtf 文件，不解压缩，直接[上传到 IBM data 平台](http://datascience.ibm.com/blog/upload-files-to-ibm-data-science-experience-using-the-command-line-2/)。

方法如下：

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/UploadFile.png)

选择 Insert SparkSession Step 后，系统会自动生成一系列代码，如下，除 ```spark.read.text(path_1).show()```：

代码块【3】：

```python
from pyspark.sql import SparkSession

# @hidden_cell
# This function is used to setup the access of Spark to your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_4fadd3c3d7a24e3b9b78d895f084bb84(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '个人保密')
    hconf.set(prefix + '.username', '个人保密')
    hconf.set(prefix + '.password', '个人保密')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', False)

# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_4fadd3c3d7a24e3b9b78d895f084bb84(name)

spark = SparkSession.builder.getOrCreate()

# Please read the documentation of PySpark to learn more about the possibilities to load data files.
# PySpark documentation: https://spark.apache.org/docs/2.0.1/api/python/pyspark.sql.html#pyspark.sql.SparkSession
# The SparkSession object is already initalized for you.
# The following variable contains the path to your file on your Object Storage.
path_1 = "swift://mydemo." + name + "/Homo_sapiens.GRCh38.87.chr.gtf.gz"

spark.read.text(path_1).show()
```



结果：

    +--------------------+
    |               value|
    +--------------------+
    |#!genome-build GR...|
    |#!genome-version ...|
    |#!genome-date 201...|
    |#!genome-build-ac...|
    |#!genebuild-last-...|
    |1	havana	gene	118...|
    |1	havana	transcri...|
    |1	havana	exon	118...|
    |1	havana	exon	126...|
    |1	havana	exon	132...|
    |1	havana	transcri...|
    |1	havana	exon	120...|
    |1	havana	exon	121...|
    |1	havana	exon	126...|
    |1	havana	exon	129...|
    |1	havana	exon	132...|
    |1	havana	exon	134...|
    |1	havana	gene	144...|
    |1	havana	transcri...|
    |1	havana	exon	295...|
    +--------------------+
    only showing top 20 rows
    


直接读，发现其实行是识别出来了，是个 __DataFrame__，但是列不对。首先是前几行注释需要扔掉，其次是我们需要的基因名称、外显子名称这些内容需要单独被分出一列。

于是我们通过 Python 的正则表达式 re 包，配合 PySpark 的 __RDD__ 相关操作，做数据清洗以及特征提取。

## Q: Spark 的 RDD DataFrame 都是什么？

简单说，RDD 可以理解成我们以前开文件后逐行读取每行信息，不直接处理，好多行给缓存成了个列表，这个列表就类似是 RDD。而 DataFrame 则类似是R 中的 DataFrame，RDD + 表头。

但是 这里的 RDD 虽然类似列表，DataFrame 虽然也跟 R 很像，却都不支持行列操作。只可以显示最上面的几行， 如 

```rdd.take(5)``` 或者 ```DataFrame.show(5)```

显示最上面的5行，却不支持显示例如第250行这样的命令。原因是， __RDD DataFrame 这里并不位于内存__，而是位于前面提到的 HDFS，在硬盘里！所以一个行下标是不能直接调出数据的。内存只是存了指针指向了硬盘，多个CPU来要数据时，内存的指针快速给他们在分布式的存储系统给他们分配任务。这也是为什么 Spark 可以Hold住海量数据的真实原因，数据不需要全扔进内存。

更多他们之间的区别联系，推荐阅读：[https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)

代码块【4】：

```python
pat_gene = '''gene_id\s+\"(\S+)\";'''
pat_tran = '''transcript_id\s+\"(\S+)\";'''
pat_exon = '''exon_number\s+\"*(\w+)\"*'''

pattern_gene = re.compile( pat_gene )
pattern_tran = re.compile( pat_tran )
pattern_exon = re.compile( pat_exon )

def parseEachLine(f_line):
    match_gene = pattern_gene.search( f_line[-1] )
    match_tran = pattern_tran.search( f_line[-1] )
    match_exon = pattern_exon.search( f_line[-1] )
         
    gene = "NULL"
    tran = "NULL"
    exon = "NULL"
    if match_gene:
        gene = match_gene.group(1)
    if match_tran:
        tran = match_tran.group(1)
    if match_exon:
        exon = match_exon.group(1)
    
    return [gene, tran, exon, f_line[0]]

rdd = spark.read.text(path_1).rdd\
            .filter(lambda x: x.value[0]!= "#")\
            .map(lambda x: x.value.split("\t"))\
            .map(lambda x: parseEachLine(x))

rdd.take(5)

```


结果：


```[[u'ENSG00000223972', 'NULL', 'NULL', u'1'],```
``` [u'ENSG00000223972', u'ENST00000456328', 'NULL', u'1'],```
``` [u'ENSG00000223972', u'ENST00000456328', u'1', u'1'],```
``` [u'ENSG00000223972', u'ENST00000456328', u'2', u'1'],```
``` [u'ENSG00000223972', u'ENST00000456328', u'3', u'1']]```



简单解释下：

## 第一部分，设定正则表达式：

```python
pat_gene = '''gene_id\s+\"(\S+)\";'''
pat_tran = '''transcript_id\s+\"(\S+)\";'''
pat_exon = '''exon_number\s+\"*(\w+)\"*'''

pattern_gene = re.compile( pat_gene )
pattern_tran = re.compile( pat_tran )
pattern_exon = re.compile( pat_exon )

def parseEachLine(f_line):
    match_gene = pattern_gene.search( f_line[-1] )
    match_tran = pattern_tran.search( f_line[-1] )
    match_exon = pattern_exon.search( f_line[-1] )
         
    gene = "NULL"
    tran = "NULL"
    exon = "NULL"
    if match_gene:
        gene = match_gene.group(1)
    if match_tran:
        tran = match_tran.group(1)
    if match_exon:
        exon = match_exon.group(1)
    
    return [gene, tran, exon, f_line[0]]
```

这部分是正则表达式。前几行规定我们从 gene_id transcript_id exon_id 这几个字段后面抓数据，并且抓引号里面的内容。

## 第二部分，正则表达式函数用于rdd 的各行，提取我们需要的信息

这个被进一步写成了一个函数。这个函数是可以直接用于逐行读取结构的，如下：

```python
with gzip.open("input.gtf.gz", "rb") as f_gtf:
    for line in f_gtf:
        if line[0] == "#":
            continue
        f_line = f_gtf.split("\t")
        [gene, tran, exon, chrom] = parseEachLine(f_line)
```

也可以被直接用在进过类似 split 处理的 RDD 字段上。

```python
rdd = spark.read.text(path_1).rdd\
            .filter(lambda x: x.value[0]!= "#")\
            .map(lambda x: x.value.split("\t"))\
            .map(lambda x: parseEachLine(x))

```

处理后的 rdd 长这样：

```python
[[u'ENSG00000223972', 'NULL', 'NULL', u'1'],
 [u'ENSG00000223972', u'ENST00000456328', 'NULL', u'1'],
 [u'ENSG00000223972', u'ENST00000456328', u'1', u'1'],
 [u'ENSG00000223972', u'ENST00000456328', u'2', u'1'],
 [u'ENSG00000223972', u'ENST00000456328', u'3', u'1']]
```

得到这个以后，后续处理想必大家都跃跃欲试了。传统的 Hadoop 使用的 MapReduce 结构，有这个就够了。但写出的代码终归不太好看。而我们需要的，是 __说人话__，怎么说人话呢，就是给RDD加一个表头，变成 DataFrame，然后通过 SparkSQL ，用 SQL 语句进行人机交互。代码如下：

代码块【5】：

```python
from pyspark.sql.types import *

schema=StructType(
    [StructField("Gene",  StringType())] + 
    [StructField("Tran",  StringType())] + 
    [StructField("Exon",  StringType())] +
    [StructField("Chrom", StringType())]
)

df = sqlCtx.createDataFrame(rdd, schema)
df.show()
```

结果：

    +---------------+---------------+----+-----+
    |           Gene|           Tran|Exon|Chrom|
    +---------------+---------------+----+-----+
    |ENSG00000223972|           NULL|NULL|    1|
    |ENSG00000223972|ENST00000456328|NULL|    1|
    |ENSG00000223972|ENST00000456328|   1|    1|
    |ENSG00000223972|ENST00000456328|   2|    1|
    |ENSG00000223972|ENST00000456328|   3|    1|
    |ENSG00000223972|ENST00000450305|NULL|    1|
    |ENSG00000223972|ENST00000450305|   1|    1|
    |ENSG00000223972|ENST00000450305|   2|    1|
    |ENSG00000223972|ENST00000450305|   3|    1|
    |ENSG00000223972|ENST00000450305|   4|    1|
    |ENSG00000223972|ENST00000450305|   5|    1|
    |ENSG00000223972|ENST00000450305|   6|    1|
    |ENSG00000227232|           NULL|NULL|    1|
    |ENSG00000227232|ENST00000488147|NULL|    1|
    |ENSG00000227232|ENST00000488147|   1|    1|
    |ENSG00000227232|ENST00000488147|   2|    1|
    |ENSG00000227232|ENST00000488147|   3|    1|
    |ENSG00000227232|ENST00000488147|   4|    1|
    |ENSG00000227232|ENST00000488147|   5|    1|
    |ENSG00000227232|ENST00000488147|   6|    1|
    +---------------+---------------+----+-----+
    only showing top 20 rows
    


这里表头已经加上去了。懂得 R 语言 melt ddply dcast 套路的人一定知道该怎么做了。但其实更多的数据从业者不懂 R，而是 SQL 专家，所以 SparkSQL 作为通用处理框架，提供的是 SQL 语句作为解决思路。

我们思考一下提问者的几个问题：

- 每条染色体基因个数的分布？

- 所有基因平均有多少个转录本？

- 所有转录本平均有多个exon和intron？

我们思考一下怎么翻译这几句话成 SQL 语句：

### 每条染色体基因个数的分布？

思考一下，问的其实是：

每个 Chrom 值，对应 几种、不重复的Gene？

```SQL
    SELECT Chrom, COUNT(DISTINCT(Gene)) FROM GTF
    GROUP BY Chrom
```

"每个" Chrom 意味着 GROUP BY Chrom, 与此同时，前面SELECT 就得加上 Chrom，这样最后的结果才会显示是哪个染色体。



"几种" 翻译成 COUNT，不重复翻译成 "DISTINCT"，于是合并后就是 COUNT(DISTINC(Gene))

然后我们要保证只看 Gene Tran Exon 都非空的行，即 WHERE (Gene != "NULL")  AND (Tran != "NULL") AND (Exon != "NULL")。

于是写出SQL:


```SQL
    SELECT Chrom, COUNT(DISTINCT(Gene)) FROM GTF
    WHERE (Gene != "NULL")  AND (Tran != "NULL") AND (Exon != "NULL")
    GROUP BY Chrom

```

了解更多 SQL 可以在这个网站 [https://www.w3schools.com/sql/](https://www.w3schools.com/sql/) 在线学习基本的SQL语句。

SQL 语句在调取 UCSC 数据集中同样作用巨大，具体这里不表。

这句话怎么在 DataFrame 执行？需要先 ```registerTempTable("GTF")``` 把 df 这个 dataFrame 给 SparkSQL，取一个表名，叫做 “GTF”。这样 df 就可以直接用 SQL语句分析了。

更多内容参考文档[http://spark.apache.org/docs/2.0.2/sql-programming-guide.html](http://spark.apache.org/docs/2.0.2/sql-programming-guide.html)


代码块【6】：

```python
df.registerTempTable("GTF")
sqlDF_genesInEachChr = spark.sql("""
    SELECT Chrom, COUNT(DISTINCT(Gene)) AS Cnt FROM GTF
    WHERE (Gene != "NULL")  AND (Tran != "NULL") AND (Exon != "NULL")
    GROUP BY Chrom
""")
sqlDF_genesInEachChr.show()
```

结果：

    +-----+----+
    |Chrom| Cnt|
    +-----+----+
    |    7|2867|
    |   15|2152|
    |   11|3235|
    |    3|3010|
    |    8|2353|
    |   22|1318|
    |   16|2511|
    |    5|2868|
    |   18|1170|
    |    Y| 523|
    |   17|2995|
    |   MT|  37|
    |    6|2863|
    |   19|2926|
    |    X|2359|
    |    9|2242|
    |    1|5194|
    |   20|1386|
    |   10|2204|
    |    4|2505|
    +-----+----+
    only showing top 20 rows
    
运行过程时间有点长，请耐心等待。因为 IBM 的免费机器是 2 核心单机模式，体现不出 Spark 大数据分析的威力。如果你在Spark集群模式下，几台 48 线程的机器上对一个大文件执行SparkSQL（前提是没人使用 + 满CPU使用），在等待的过程中去后台 top 一下，会看见计算节点上全部都是恐怖的 4800% 的 CPU 使用率，共同执行同一个任务。

好啦，SparkSQL 的结果已经只有20+行了，此时可以收进内存里面了。

不过 SparkSQL 的结果是个 DataFrame, R 语言倒是能直接收进去，Python 默认的数据类型，没有这个，怎么办？来，我们先抑制住重复造轮子、准备自己写一个的冲动，由于我们最开始 Import 了 pandas，这个包引入后， Python 也就支持 DataFrame 了。这里直接用SparkSQL 的 toPandas 方法，就可以得到Pandas 的 DataFrame 了：

代码块【7】：

```python
pd_genesInEachChr = sqlDF_genesInEachChr.toPandas()
pd_genesInEachChr.head()
```

结果：


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Chrom</th>
      <th>Cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>2867</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>2152</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>3235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2353</td>
    </tr>
  </tbody>
</table>
</div>



得到表了，有人要说，你最开始就讲 Jupyter 能画图，有个包叫做 seaborn 的还跟 ggplot2 一样简单，记忆力强的还念叨着 set_style('white') 相当于 theme_bw()，现场画一个呗？

没问题。首先，Pandas 的DataFrame 没有R语言的 factor 这种让人又爱又恨的东西（掉过这个坑的在下面举手）。所以如果要调整顺序，得自己想办法。我就用了高阶函数做这个事情。具体大家参考 廖雪峰大神的Python 教程之[匿名函数篇](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431843456408652233b88b424613aa8ec2fe032fd85a000) 加 [高阶函数篇](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014317852443934a86aa5bb5ea47fbbd5f35282b331335000)。简单说， 下面的 lambda 属于匿名函数，对我这种懒人而言不用写 def 定义函数了。map 是对一个列表每个值执行一个函数， reduce 把返回结果一个接在另一个尾巴上。有Python基础的注意，由于 map 返回的是 pandas 的 DataFrame 而不是 Python 默认的list，实际上 reduce 的 append 是 [Pandas的append](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html) 而不是系统 append。

还不清楚的，这里写一个 shell 的同义语句：

```bash
rm input.chrSort.txt
for chr in {1..22} X Y MT
do
    grep -w ${chr} input.txt >>input.chrSort.txt
done
```

代码块【8】：

```python
l_plotChrOrder = map(lambda x: str(x), range(1, 23)) + ['X', 'Y', 'MT']
pd_genesInEachChrSort = reduce(lambda x,y: x.append(y), 
                               map(lambda x: pd_genesInEachChr[pd_genesInEachChr['Chrom'] == x], l_plotChrOrder)
                        )
pd_genesInEachChrSort.head()
```


结果：

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Chrom</th>
      <th>Cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>5194</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>3971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3010</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>2505</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>2868</td>
    </tr>
  </tbody>
</table>
</div>


代码块【9】：

```python
sns.barplot(data=pd_genesInEachChrSort, x="Cnt", y="Chrom")
```

结果：


    <matplotlib.axes._subplots.AxesSubplot at 0x7fb1a31e9110>




![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/output_18_1.png)


大家看是不是实现了 ggplot2 的效果？更多例子请查看 [seaborn文档](http://seaborn.pydata.org/examples/index.html)。 

OK，快速解决剩下来的问题：

## 所有基因平均有多少个转录本？

代码块【10】：

```python
sqlDF_transInEachGene = spark.sql("""
    SELECT Gene, COUNT(DISTINCT(Tran)) AS Cnt FROM GTF
    WHERE (Gene != "NULL")  AND (Tran != "NULL") AND (Exon != "NULL")
    GROUP BY Gene
""")
pd_transInEachGene = sqlDF_transInEachGene.toPandas()
sns.distplot(pd_transInEachGene['Cnt'])
```

结果：


    <matplotlib.axes._subplots.AxesSubplot at 0x7fb1a09a6bd0>




![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/output_20_1.png)


画好了，拿给老板看，这个肯定得挨骂，不好看啊，长尾效应太明显。Python 作图微调效果如何？好用的话你画一个 0~25 的柱状分布呗？

既然要微调，我就用原始的python 作图 matplotlib 库了，他和 seaborn 的关系如同 R 的 plot 与 ggolot2。
matplotlib 库有非常精美的 [gallery](http://matplotlib.org/gallery.html)，代码拿来就能在jupyter上画，再次强调，如果不显示图像请像我这里最开始import 一样加 %matplotlib inline 魔法。

画之前先简单看下数据分布，类似 R 的 summary

代码块【11】：

```python
pd_transInEachGene['Cnt'].describe()
```


结果：

    count    57992.000000
    mean         3.413143
    std          5.103533
    min          1.000000
    25%          1.000000
    50%          1.000000
    75%          4.000000
    max        170.000000
    Name: Cnt, dtype: float64


代码块【12】：

```python
plt.hist(pd_transInEachGene['Cnt'], max(pd_transInEachGene['Cnt']), histtype='bar')
plt.xlim(0, 25)
```


结果：

    (0, 25)



![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/output_23_1.png)


OK，这个应该拿给老板不会挨骂了，下一个问题：

## 所有转录本平均有多个exon？

代码块【13】：

```python
sqlDF_exonsInEachTran = spark.sql("""
        SELECT Tran, COUNT(DISTINCT(Exon))  AS Cnt FROM GTF
        WHERE (Gene != "NULL")  AND (Tran != "NULL") AND (Exon != "NULL")
        GROUP BY Tran
""")
pd_exonsInEachTran = sqlDF_exonsInEachTran.toPandas()
pd_exonsInEachTran.head()
```


结果：

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tran</th>
      <th>Cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ENST00000487835</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENST00000309519</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENST00000463042</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENST00000490603</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENST00000492025</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


代码块【14】：

```python
print("Median Value %d " % (pd_exonsInEachTran.median(0)))
plt.hist(pd_exonsInEachTran['Cnt'], max(pd_exonsInEachTran['Cnt']), histtype='bar')
plt.xlim(0, 15)
```

结果：
    Median Value 4 

    (0, 15)
    

![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/output_26_2.png)



# 老板觉得似乎不对，想起了什么……

“这里看的是所有基因，我要你看的是编码基因。那么多非编码的看他干嘛！除了把中位数往下带跑偏影响结果，有什么意义？！”

此时，听话的人就直接 ```grep protein_coding``` 去了。而对此，我认为，如果长期以往，只能一直做菜鸟。我们要多长一个心眼，里面不还有 lincRNA 嘛，也挺重要，万一老板哪天让我比一下lincRNA 和编码的，我是不是还得再算一次？万一又要看其他的呢？

防止这种情况，很简单，把基因类型那一列加进去，分不同基因类别，全算出来放那里就好了。

如果是用 perl 的 hash表做这件事，就会出来个似乎是（原谅我几年不写perl全忘光了）这样的数据架构：

```
push(@{$TypeTranExons{$gtype}{$tran}}, $exon);
```

相信有过这种噩梦般经历的读者此时会是懵逼的。哪地方该有括号，用 $ @ 还是%，小骆驼根本就没有，写错一个就报错，想深入学习，要么去看大神的代码，要么就得去看一本叫做 《Perl高级编程》的书，[京东购买链接](https://item.jd.com/10698190.html) 在这里，点开发现无货的别急，这本书我几年前学这个的时候，就早已断货了。

Python 就没有这么多规矩，我最早就为的这个转的 python。

```python
TypeTranExons[gtype][tran].append(exon)
```

当然我们现在有了 pyspark，更不用去折腾 Hash 结构去了，直接在 SQL 里，说人话。

代码块【15】：

```python
pat_gene = '''gene_id\s+\"(\S+)\";'''
pat_tran = '''transcript_id\s+\"(\S+)\";'''
pat_exon = '''exon_number\s+\"*(\w+)\"*'''
pat_type = '''gene_biotype\s+\"*(\w+)\"*'''

pattern_gene = re.compile( pat_gene )
pattern_tran = re.compile( pat_tran )
pattern_exon = re.compile( pat_exon )
pattern_type = re.compile( pat_type )

def parseEachLineV2(f_line):
    match_gene = pattern_gene.search( f_line[-1] )
    match_tran = pattern_tran.search( f_line[-1] )
    match_exon = pattern_exon.search( f_line[-1] )
    match_type = pattern_type.search( f_line[-1] )
         
    gene = "NULL"
    tran = "NULL"
    exon = "NULL"
    gtype = "NULL"
    if match_gene:
        gene = match_gene.group(1)
    if match_tran:
        tran = match_tran.group(1)
    if match_exon:
        exon = match_exon.group(1)
    if match_type:
        gtype = match_type.group(1)

        
        
    return [gene, tran, exon, gtype,f_line[0]]

rdd = spark.read.text(path_1).rdd\
            .filter(lambda x: x.value[0]!= "#")\
            .map(lambda x: x.value.split("\t"))\
            .map(lambda x: parseEachLineV2(x))

rdd.take(5)
```

结果：

    [[u'ENSG00000223972',
      'NULL',
      'NULL',
      u'transcribed_unprocessed_pseudogene',
      u'1'],
     [u'ENSG00000223972',
      u'ENST00000456328',
      'NULL',
      u'transcribed_unprocessed_pseudogene',
      u'1'],
     [u'ENSG00000223972',
      u'ENST00000456328',
      u'1',
      u'transcribed_unprocessed_pseudogene',
      u'1'],
     [u'ENSG00000223972',
      u'ENST00000456328',
      u'2',
      u'transcribed_unprocessed_pseudogene',
      u'1'],
     [u'ENSG00000223972',
      u'ENST00000456328',
      u'3',
      u'transcribed_unprocessed_pseudogene',
      u'1']]


代码块【16】：

```python

from pyspark.sql.types import *

schema2=StructType(
    [StructField("Gene",  StringType())] + 
    [StructField("Tran",  StringType())] + 
    [StructField("Exon",  StringType())] +
    [StructField("Type",  StringType())] +
    [StructField("Chrom", StringType())]
)

df2 = sqlCtx.createDataFrame(rdd, schema2)
df2.show()
```

结果：

    +---------------+---------------+----+--------------------+-----+
    |           Gene|           Tran|Exon|                Type|Chrom|
    +---------------+---------------+----+--------------------+-----+
    |ENSG00000223972|           NULL|NULL|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000456328|NULL|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000456328|   1|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000456328|   2|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000456328|   3|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|NULL|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|   1|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|   2|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|   3|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|   4|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|   5|transcribed_unpro...|    1|
    |ENSG00000223972|ENST00000450305|   6|transcribed_unpro...|    1|
    |ENSG00000227232|           NULL|NULL|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|NULL|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|   1|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|   2|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|   3|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|   4|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|   5|unprocessed_pseud...|    1|
    |ENSG00000227232|ENST00000488147|   6|unprocessed_pseud...|    1|
    +---------------+---------------+----+--------------------+-----+
    only showing top 20 rows
    


OK, 新的一列成果加入表格，然后写SQL 分析数据。既然要看各种基因类型、每个转录本有几种外显子，那么 GROUP BY 就加一个 Type 列，SELECT 也加一个 Type 列显示出来。

代码块【17】：

```python
df2.registerTempTable("GTF2")
sqlDF_exonsInEachTran = spark.sql("""
        SELECT Tran, Type, COUNT(DISTINCT(Exon))  AS Cnt FROM GTF2
        WHERE (Gene != "NULL")  AND (Tran != "NULL") AND (Exon != "NULL")
        GROUP BY Tran, Type
""")
pd_exonsInEachTran = sqlDF_exonsInEachTran.toPandas()
pd_exonsInEachTran.head()
```

结果：

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tran</th>
      <th>Type</th>
      <th>Cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ENST00000434641</td>
      <td>protein_coding</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENST00000454975</td>
      <td>protein_coding</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENST00000463950</td>
      <td>protein_coding</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENST00000485040</td>
      <td>protein_coding</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENST00000367635</td>
      <td>protein_coding</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



Pandas 也可以进行分组信息统计，如同 R 的 ddply。

代码块【18】：

```python
pd_sort = pd_exonsInEachTran[['Type', 'Cnt']].groupby(['Type'])\
                    .agg([len,np.median, np.mean, np.std])['Cnt']\
                    .sort_values(['median'], ascending=False)
pd_sort.head()
```

结果：

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>len</th>
      <th>median</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>protein_coding</th>
      <td>145348</td>
      <td>5</td>
      <td>7.232511</td>
      <td>7.383970</td>
    </tr>
    <tr>
      <th>transcribed_unitary_pseudogene</th>
      <td>283</td>
      <td>5</td>
      <td>6.392226</td>
      <td>5.536296</td>
    </tr>
    <tr>
      <th>TR_C_gene</th>
      <td>14</td>
      <td>5</td>
      <td>5.214286</td>
      <td>1.625687</td>
    </tr>
    <tr>
      <th>processed_transcript</th>
      <td>2814</td>
      <td>4</td>
      <td>4.515281</td>
      <td>4.707923</td>
    </tr>
    <tr>
      <th>transcribed_unprocessed_pseudogene</th>
      <td>2395</td>
      <td>4</td>
      <td>5.578706</td>
      <td>4.741607</td>
    </tr>
  </tbody>
</table>
</div>



再排序画图看看

代码块【19】：

```python
pd_exonsInEachTran_sort = reduce(lambda x,y: x.append(y), 
                                 map(lambda x: pd_exonsInEachTran[pd_exonsInEachTran['Type']==x], pd_sort.index[0:10])
                        )
pd_exonsInEachTran_sort.head()
```

结果：

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tran</th>
      <th>Type</th>
      <th>Cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ENST00000434641</td>
      <td>protein_coding</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENST00000454975</td>
      <td>protein_coding</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENST00000463950</td>
      <td>protein_coding</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENST00000485040</td>
      <td>protein_coding</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENST00000367635</td>
      <td>protein_coding</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



最后画一个复杂点的啊

代码块【20】：

```python
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
sns.boxplot(data=pd_exonsInEachTran_sort, y="Type", x="Cnt", ax=ax1)
sns.violinplot(data=pd_exonsInEachTran_sort, y="Type", x="Cnt", ax=ax2, bw=0.5)
ax1.set_xlim(0, 20)
ax2.set_xlim(-5, 60)
ax2.set_yticklabels([])
ax2.set_xticks([0, 20, 40, 60])
ax2.set_xticklabels([0, 20, 40, 60])
ax1.set_title("Boxplot")
ax2.set_title("Violinplot")
```

结果：

    <matplotlib.text.Text at 0x7fb19e203050>




![png](/images/2017-02-16-AnalyseTheAnnotationOfHumanGenome/output_37_1.png)

IBM data science 上完整版本：

[https://apsportal.ibm.com/analytics/notebooks/d3400624-fd7f-483b-96f3-b9d07876f455/view?access_token=499996f6a4e6f93e448907bf219bae6310975c0d02521c7c67ef02b79b1ccf77](https://apsportal.ibm.com/analytics/notebooks/d3400624-fd7f-483b-96f3-b9d07876f455/view?access_token=499996f6a4e6f93e448907bf219bae6310975c0d02521c7c67ef02b79b1ccf77)