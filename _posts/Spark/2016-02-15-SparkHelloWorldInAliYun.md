---
title: "[BigData-Spark]The hello world script for AliYun EMR service with pyspark."
tagline: ""
last_updated: 2016-02-15
category: big-data related
layout: post
tags : [Spark, py-spark, aliyun]
---

# Using Hadoop/Spark in Aliyun EMR service

## 开通阿里云 OSS 对象存储服务
在这里开通 [https://www.aliyun.com/product/oss](https://www.aliyun.com/product/oss)

OSS 前缀地址列表

位置|外网|内网
---|----|-------
青岛|oss-cn-qingdao.aliyuncs.com	|	oss-cn-qingdao-internal.aliyuncs.com
北京|oss-cn-beijing.aliyuncs.com	|	oss-cn-beijing-internal.aliyuncs.com
杭州|oss-cn-hangzhou.aliyuncs.com	|	oss-cn-hangzhou-internal.aliyuncs.com
上海|oss-cn-shanghai.aliyuncs.com	|	oss-cn-shanghai-internal.aliyuncs.com
香港|oss-cn-hongkong.aliyuncs.com	|	oss-cn-hongkong-internal.aliyuncs.com
深圳|oss-cn-shenzhen.aliyuncs.com	|	oss-cn-shenzhen-internal.aliyuncs.com
美西|oss-us-west-1.aliyuncs.com	|	oss-us-west-1-internal.aliyuncs.com
新加坡|oss-ap-southeast-1.aliyuncs.com	|	oss-ap-southeast-1-internal.aliyuncs.com

上传、下载文件方法：

[https://help.aliyun.com/document_detail/oss/sdk/python-sdk/install.html?spm=5176.docoss/sdk/python-sdk/download.6.260.xhXWR2](https://help.aliyun.com/document_detail/oss/sdk/python-sdk/install.html?spm=5176.docoss/sdk/python-sdk/download.6.260.xhXWR2)

这里使用 python SDK oss2 包。上传部分和下载部分都写在了这里：

```python
# -*- coding: utf-8 -*-
import os
import oss2

auth = oss2.Auth('OSS-keyID', 'OSS-securateKeyID')
service = oss2.Service(auth, 'oss-cn-beijing.aliyuncs.com')
print([b.name for b in oss2.BucketIterator(service)])

endpoint = 'oss-cn-beijing.aliyuncs.com'
bucket = oss2.Bucket(auth, endpoint, 'hubqaliossbj')
print(bucket.get_bucket_acl().acl)

## Upload from local to OSS

key = 'SRR015287.1.fq.gz'
filename = '/Users/hubq/Downloads/FileZilla/Study/00.fastq/SRR015287.1.fq.gz'


bucket.put_object_from_file(key, filename)
# or
oss2.resumable_upload(bucket, key, filename,
    store=oss2.ResumableStore(root='/tmp'),
    multipart_threshold=100*1024,
    part_size=100*1024)
    
## Download from OSS to remote EMR machine
import shutil

endpoint = 'oss-cn-beijing-internal.aliyuncs.com'
bucket = oss2.Bucket(auth, endpoint, 'hubqaliossbj')
key = 'merge.GCA.GCC.GCT.GlobalPvalue.anno.bed'

remote_stream = bucket.get_object('merge.GCA.GCC.GCT.GlobalPvalue.anno.bed')
with open('merge.GCA.GCC.GCT.GlobalPvalue.anno.bed', "wb") as local_fileobj:
    shutil.copyfileobj(remote_stream, local_fileobj)
    
    
remote_stream = bucket.get_object('analysis.py')
with open('analysis.py', "wb") as local_fileobj:
    shutil.copyfileobj(remote_stream, local_fileobj)
```

## 使用阿里云 EMR 弹性计算服务

### 开通
遵循 EMR 官方手册的 快速开始部分 [准备工作](https://help.aliyun.com/document_detail/emr/quick-start/prepare.html?spm=5176.docemr/trouble-shooting/trouble-shooting.6.90.2HE9i4) [创建集群](https://help.aliyun.com/document_detail/emr/quick-start/create-cluster.html?spm=5176.docemr/quick-start/create-job.6.91.XcAZJX) 两个环节，注意最后允许使用 ssh 登陆 master 节点。

购买时，遇到两次差错，一次说卖完了不卖给我，一次说余额不足 300 块，然后成功建立集群。

### 预设
使用阿里云的 pyspark 之前，需要先注意阿里云的 python 版本是 2.6，并且没有 pip， 没有 ```numpy``` ```scipy``` ```pandas``` ```matplotlib``` 等包， 需要手动装上 pip，然后安装这些常用包。

安装各种包，需要下载资源，而下载资源会产生相应的流量费用，并且速度相对内网慢得多。因此这里先改镜像站地址，从内网镜像[http://mirrors.aliyuncs.com](http://mirrors.aliyuncs.com) 获得相应的资源。

首先改好 yum 源

```bash
cd /etc/yum.repos.d/
mkdir bak
mv /etc/yum.repos.d/CentOS-Base.repo bak

wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
sed -i 's/aliyun.com/aliyuncs.com/'  /etc/yum.repos.d/CentOS-Base.repo
yum clean all && yum makecache
yum install zsh
```

用 yum 装 easy_install，然后装 pip

```bash
yum install python-devel
yum install python-setuptools
easy_install pip
```


然后改 pip, 由于是阿里云内网，用 ```http://mirrors.aliyuncs.com``` 镜像

```bash
mkdir ~/.pip
cat > ~/.pip/pip.conf <<EOF\
[global]\
index-url = http://mirrors.aliyuncs.com/pypi/simple/\
EOF
```

与通常的 pip 源不同，这种情况下，用 pip 装 numpy pandas 等会提示报错，说需要信任 http://mirrors.aliyuncs.com 这个源。于是信任并按照：

```bash
pip install numpy  --trusted-host mirrors.aliyuncs.com
pip install pandas --trusted-host mirrors.aliyuncs.com
```



现阶段，我在直接读取 OSS 中上传的文件遇到了些麻烦， 后来采用了直接 ssh 登陆然后将文件用上一部分的脚本放入 hdfs 中，再分析的方法，成功的运行了程序。

运行方法如下：

```bash
### Before doing this
### Download merge.GCA.GCC.GCT.GlobalPvalue.anno.bed 
### from OSS using oss2 python Aliyun SDK

# put the file into HDFS 
hdfs dfs -mkdir ./
hdfs dfs -put merge.GCA.GCC.GCT.GlobalPvalue.anno.bed ./

# submit the script to spark
cd /opt/apps/spark-1.4.1-bin-hadoop2.6/bin

./spark-submit analysis.py \
	merge.GCA.GCC.GCT.GlobalPvalue.anno.bed \
	merge.GCA.GCC.GCT.GlobalPvalue.anno.outInfo.xls \
	--driver-memory 7G --executor-memory 5G \
	--master yarn-cluster --num-executors 50
```

多说几句，[上一篇日志](http://huboqiang.github.io/2016/01/13/SparkHelloWorld) 里面，如果不设置 ```--driver-memory 7G``` ```--executor-memory 5G``` 的话，会造成程序最多使用 2G 内存，然后这个输入文件为 3G 的话，程序无法进行，会自动退出。这里使用这两个参数，可以将程序调用的内存增大，进而保证程序成功运行。

这个 submit 命令中， ```analysis.py ``` 如下， 与[上一篇日志](http://huboqiang.github.io/2016/01/13/SparkHelloWorld) 基本相同。

```python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf


sc = SparkContext("local", "Simple App")

func_notInf = lambda x: 1 if x != float("inf") else 0

state_data = sc.textFile(sys.argv[1])
state_data_used = state_data.map(lambda line: line.split("\t"))

# non-inf max value
maxVal = state_data_used.map(lambda fields: float(fields[3])).filter(func_notInf).reduce(lambda x, y: max(x, y))


def func_giveVal(x):
    return_val = float(x)
    if x == "inf":
        return_val = maxVal
    elif x == "-inf":
        return_val = -1*maxVal
    return return_val


state_groupped_pval = state_data_used.map(lambda fields: (fields[4], func_giveVal(fields[3]) )).groupByKey().mapValues(list)
state_groupped_cnt = state_groupped_pval.map( lambda (k, v): (k, np.array(v, dtype="float").mean(), np.array(v, dtype="float").std()/len(v), len(v) ) )
l_groupped_cnt = state_groupped_cnt.collect()
pd_groupped_cnt = pd.DataFrame(l_groupped_cnt, columns = ['Type', 'Value', 'SE', 'Count'])
pd_groupped_cnt_out = pd_groupped_cnt.sort(['Value'], ascending=False)
pd_groupped_cnt_out.to_csv(sys.argv[2], index=False, sep="\t")

```

