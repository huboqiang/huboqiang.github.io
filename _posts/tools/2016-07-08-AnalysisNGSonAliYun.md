---
title: "[Docker]使用阿里云分析高通量测序数据—— RNA-Seq 与 ChIP-Seq."
tagline: ""
last_updated: 2016-07-08
category: tools
layout: post
tags : [tools, environments, docker]
---

# 使用阿里云分析高通量测序数据—— RNA-Seq 与 ChIP-Seq

## 1. 在阿里云购买服务器

### 1.1 进入控制台，购买 ECS 云服务器
首先选择区域。我的 OSS 网络存储放在了华北2区，因此购买华北2区的服务器。

![Figure 1](/images/2016-07-08-AnalysisNGSonAliYum/Fig1.Console.png)

购买选项如下填写。

![Figure 2](/images/2016-07-08-AnalysisNGSonAliYum/Fig2.Buy.png)

注意CPU 内存 选择较大的 4CPU x 16Gb 内存。这是进行生物信息学分析的最低配置，更高配置的机器可以在亚马逊 aws 上租到。同时，操作系统使用  CentOS 7 ，可以直接安装 docker。

购买成功显示：

![Figure 3](/images/2016-07-08-AnalysisNGSonAliYum/Fig3.success.png)

### 1.2 登陆服务器

在终端中登陆服务器：

```bash
ssh  root@101.200.158.36
```

![Figure 4](/images/2016-07-08-AnalysisNGSonAliYum/Fig4.loginServer.png)

安装并启动 docker

```bash
yum install -y docker
service docker start
```

下载内网镜像:

```bash
docker pull registry.aliyuncs.com/hubq/tanginstall
```

好了，稍等一段时间即可安装完成。

运行镜像：

```bash
docker run -i -t registry.aliyuncs.com/hubq/tanginstall /bin/zsh
```

## 2. 分析数据

建立项目文件夹

```bash
pip install --user oss2
mkdir -p /home/analyzer/project/ChIP_test/00.0.raw_data
cd /home/analyzer/project/ChIP_test
```



### 2.1 将数据上传至服务器

用以下脚本 ```load_data.py``` ，将数据从 OSS 网络存储上传至服务器。当然也可以根据官方文档使用其他方法。

```python
# -*- coding: utf-8 -*-
import oss2
import os
from oss2 import SizedFileAdapter, determine_part_size
from oss2.models import PartInfo
import shutil

auth = oss2.Auth('ACCESS_KEY', 'KEY_PASSWORD')
service = oss2.Service(auth, 'oss-cn-beijing.aliyuncs.com')
endpoint = 'oss-cn-beijing.aliyuncs.com'
bucket = oss2.Bucket(auth, endpoint, 'hubqaliossbj')


# as if the Object dir is in the root of the bucket:
l_oss = ["12w-brain-k9me2-2.1.fq.gz", "12w-brain-k9me2-2.2.fq.gz", "input.1.fq.gz", "input.2.fq.gz"]
for oss in l_oss:
  remote_file = "project_"
  remote_stream = bucket.get_object(oss)
  server_file = "/home/analyzer/project/ChIP_test/00.0.raw_data/%s" % (oss)
  with open(server_file, "wb") as local_fileobj:
    shutil.copyfileobj(remote_stream, local_fileobj)
```

然后把文件夹放成如下格式：

```
ls ~/project/ChIP_test/00.0.raw_data/*
/home/analyzer/project/ChIP_test/00.0.raw_data/12w-brain-k9me2-2:
12w-brain-k9me2-2.1.fq.gz  12w-brain-k9me2-2.2.fq.gz

/home/analyzer/project/ChIP_test/00.0.raw_data/input:
input.1.fq.gz  input.2.fq.gz
```

### 2.2 启动分析

写一个 ```sample.tab.xls```, 格式如下，注意使用 tab 分割：

```
sample  stage   type    tissue  brief_name      merge_name      end_type        control
12w-brain-k9me2-2 12Week  H3K9me2 brain Week12_brain_H3K9me2_rep1 Week12_brain_H3K9me2  PE  Week12_brain_input
input 12Week  H3K9me2 brain Week12_brain_input_rep1 Week12_brain_input  PE  Week12_brain_input
```

由于是人类样本，执行如下命令：

```bash
sed 's/is_debug=1/is_debug=0/g' ~/module/ChIP/run_chipseq.py  >run_chipseq.py
python run_chipseq.py  --ref hg19 --TSS_genebody_up 5000 --TSS_genebody_down 5000 --TSS_promoter_up 5000 --TSS_promoter_down 5000 --Body_extbin_len 50 --Body_bincnt 100 --TSS_bin_len 1 --top_peak_idr 100000 sample.tab.xls
```

这个命令第一步会下载网上的 fasta 文件，并且建立 bwa 软件的 index。建立的 index 放在 ```/home/analyzer/database_ChIP```， 这个文件建立后，可以传入 oss 中，下次分析时可以直接从 oss 取出放入 ```/home/analyzer/database_ChIP```， 避免重复运算。

如果没有问题，则这个程序会一直继续往下跑，直到得出最终结果。

## 3. 汇总结果

重要结果包括:

```bash
ls /home/analyzer/project/ChIP_test/03.2.Peak_mrg/*/*bw /home/analyzer/project/ChIP_test/03.3.Peak_idr /home/analyzer/project/ChIP_test/StatInfo
```

以上结果请放入 OSS 中。至此，基本分析流程结束，下一阶段可以进行高级分析。
