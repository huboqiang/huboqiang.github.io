---
title: "[Docker]使用阿里云 + Docker 分析高通量测序数据—— RNA-Seq 与 ChIP-Seq."
tagline: ""
last_updated: 2017-03-20
category: tools
layout: post
tags : [tools, environments, docker]
---

# 使用阿里云+Docker分析高通量测序数据—— RNA-Seq 与 ChIP-Seq


写这篇文章的原因，是上一期大数据与生物信息结合的介绍 [使用 IBM datascience 平台统计 hg38每条染色体基因,转录本的分布](http://huboqiang.cn/2017/02/16/AnalyseTheAnnotationOfHumanGenome) 在生信技能树公众号发布以后，很多人问阿里云和 IBM 相比优势在哪里。我当时在文章下面回复说 IBM data science 平台主要是面向后续的数据分析部分，如果跑流程还是推荐阿里云，这个推荐综合了经济因素以及网络速度。

在阿里云租服务器，这里介绍租阿里云的 ECS 服务。ECS 主要分为包月和按小时计费两种方式。包月可以租到更好的机器，但价格动辄一个月上千RMB。按小时租，则最好的机器也就 4核16G 内存，勉强满足生物信息比对的最低要求，但一小时3、4块钱的收费，对于总原始 fastq 数据量在 10GB 以内的小型项目而言，还是十分经济划算的。

项目|信息
---|---
地域 : | 华北2
CPU : | 4 核 16 GB ( 通用n1 )
GPU : | 无
网络 : | 10兆网络
存储 : | 100GB SSD
镜像 :| Ubunto 16.04
购买量 :|1台
配置费用：|¥3.625/时

如果数据量更多，也推荐使用阿里的 [hpc 服务器](https://hpc-buy.aliyun.com)，这个服务器购买的话价格会在10w以上，性能足以在几天时间完成几十个、上百个样本的分析，当然租的话比ECS 的顶配高10倍。

项目|信息
---|---
地域 : | 华东1
CPU : | 64 核 128 GB ( G4 )
GPU : | Tesla M40 x2
网络 : |千兆网络
存储 : |1.92TB SSD x2
镜像 :| centos7
购买量 :|1台
配置费用：|¥37.50/时

本人并非土豪，这篇文章的测试用的是顶配的按小时收费 ECS 服务器。比对一个 200MB 的 SRA 原始文件，总计运行时间在两个半小时左右，其中下载数据约占用40%时间，比对占用50%时间，cufflinks HTSeq 等可以很快完成，总计花费3.625x3=10.875 元。如果用 Hisat 代替 TopHat 比对，速度可以更快，但这个 Docker 基于的脚本是我们实验室15年的分析框架，这个Docker 主要任务还是完整重复之前工作的分析流程，所以没有改动。

这就不得不提到写这篇文章的另一个原因————Docker 的使用目的，这期的 NBT 发表了一篇文章:

![FigureConsole](/images/2016-07-08-AnalysisNGSonAliYum/Fig.DockerNBT.png)

简而言之，Docker 的作用，快速部署服务只是一个方面，这个可以用简单的 shell 脚本完成。而保证整个流程可以重复，就必须保证系统环境的一致性，上到系统库，下到 R 包、Python库，版本必须保持完全一致，才可以保证 **最终结果完全一致、完全可重复**。，

综上所述，Docker + 阿里云实时收费，可以以较低的成本————包括经济成本和时间成本，重复出原先的工作。如果读者有同样的数据，可以在使用同样的 Docker 在阿里云算出自己的数据，然后拿着中间结果去找人求助，比如去[生信技能树](http://www.biotrainee.com/forum-93-1.html) ，打通整个分析流程。


## 1. 在阿里云购买服务器

### 1.1 进入控制台，购买 ECS 云服务器

阿里云的网页位于 [https://www.aliyun.com/](https://www.aliyun.com/)，进入注册后，点击登录控制台：

![FigureConsole](/images/2016-07-08-AnalysisNGSonAliYum/Fig.AliMain.png)

首先选择区域。我的 OSS 网络存储放在了华北2区，因此购买华北2区的服务器。

![Figure 1](/images/2016-07-08-AnalysisNGSonAliYum/Fig1.Console1.png)

![Figure 1](/images/2016-07-08-AnalysisNGSonAliYum/Fig1.Console2.png)


购买选项如下填写。

![Figure 2](/images/2016-07-08-AnalysisNGSonAliYum/Fig2.Buy.png)

注意CPU 内存 选择较大的 4CPU x 16Gb 内存。这是进行生物信息学分析的最低配置，更高配置的机器可以在亚马逊 aws 上租到。同时，操作系统使用  ubuntu 16.04 64位 ，可以直接安装 docker：




购买成功显示：

![Figure 3](/images/2016-07-08-AnalysisNGSonAliYum/Fig3.success.png)

等了几分钟，收到了手机短信：

![Figure 3](/images/2016-07-08-AnalysisNGSonAliYum/Fig.Messege.png)

### 1.2 登陆服务器

在终端中登陆服务器：

```bash
ssh  root@123.57.10.97
```

![Figure 4](/images/2016-07-08-AnalysisNGSonAliYum/Fig4.loginServer.png)

安装并启动 docker

```bash
apt-get install docker docker.io
```

下载内网镜像:

```bash
docker pull registry.aliyuncs.com/hubq/tanginstall
```

特别说明，这个命令只能在阿里云上使用。不使用阿里云的读者（如AWS），如果装了 Docker 也想试试，这里可以用 DockerHub 的官方镜像:

```bash
docker pull hubq/tanginstall
```

阿里云上当然也可以输入这个命令，不过网速就呵呵了，所以这里还是使用阿里云的 Docker镜像，而不是官方的。

好了，稍等一段时间即可安装完成。

## 2. 分析数据


现在来进行项目实战。拉下来的 Docker 文件如下：[https://github.com/huboqiang/tangEpiNGSInstall](https://github.com/huboqiang/tangEpiNGSInstall)


这个 Docker 可以对人鼠执行 RNA-Seq ChIP-Seq 的生物信息学基本流程分析。

对于 ChIP-Seq 部分，对应的脚本使用Encode 的 IDR 流程。 IDR 流程详细介绍见[ENCODE 主页](https://www.encodeproject.org/software/idr/)，简而言之，就是一个考虑了多个重复样本的 ChIP-Seq 分析流程。


今天我们着重介绍 RNA-Seq 部分。对于 RNA-Seq 部分，对应的脚本曾用于 [The Transcriptome and DNA Methylome Landscapes of Human Primordial Germ Cells](http://www.cell.com/cell/abstract/S0092-8674(15)00563-2) 这篇文章，相应描述如下：


>Read pairs with more than 10% low-quality bases, adapter contaminants or artificial sequences introduced during the experimental processes were trimmed, and the cleaned reads were aligned to the human hg19 reference using Tophat (v2.0.12) with default settings (Trapnell et al., 2009). Additionally, 92 ERCC spike-ins were added to the reference annotation as the extra artificial transcripts. Cufflinks (v2.2.1) with default parameters was further used to assemble the transcripts and quantify transcription levels (FPKM, fragments per kilobase of transcript per million mapped reads) of annotated genes (Trapnell et al., 2010). Linear regression was applied to fit the data points between the averaged transcription levels of the 92 exogenous ERCC spike- in RNAs (log2 transformed) in each single-cell RNA-seq dataset and the provided number of molecules per lysis reaction for each single cell, and the absolute mRNA abundance in each single cell was calculated by normalizing against the spike-in RNAs (Treutlein et al., 2014). The expression level of repetitive elements was quantified using the read counts of repetitive elements per million RefSeq mappable reads only if the unique mapped reads were located in the annotated repetitive elements. Other published data, including those from human implantation embryos, human naïve ESCs, in vitro human PGCLCs, and mouse PGCs, were downloaded from the GEO datasets (Irie et al., 2015; Seisenberger et al., 2012; Takashima et al., 2014; Yamaguchi et al., 2013; Yan et al., 2013), and only the raw fastq reads were downloaded and incorporated into our analysis pipelines.


OK，我们使用这篇文章的一个样本，来跑一下这个 RNA-Seq 的流程。

### 2.1 获取数据


建立项目文件夹

```bash
cd ~
git clone https://github.com/huboqiang/tangEpiNGSInstall

```

这一步是下载数据，如果读者有 RNASeq 数据，这里替换成自己的 fastq 文件即可。这里下载的数据因为是 SRA 格式，需要转换成 fastq:

```bash
mkdir fastq
cd fastq

wget ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByExp/sra/SRX/SRX102/SRX1021247/SRR2013442/SRR2013442.sra

cd ~
wget http://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.8.2-1/sratoolkit.2.8.2-1-ubuntu64.tar.gz

tar -zxvf sratoolkit.2.8.2-1-ubuntu64.tar.gz

mkdir -p ./fastq/SRR2013442 && ./sratoolkit.2.8.2-1-ubuntu64/bin/fastq-dump --split-3 --outdir ./fastq/SRR2013442  -gzip ./fastq/SRR2013442.sra
```

然后需要给样本起一个名字。这里直接借鉴 [https://github.com/huboqiang/tangEpiNGSInstall/blob/master/test_fq_RNA/sample.tab.xls](https://github.com/huboqiang/tangEpiNGSInstall/blob/master/test_fq_RNA/sample.tab.xls) 这个文件

```bash
cp ./tangEpiNGSInstall/test_fq_RNA/sample.tab.xls fastq/ &&\
sed -i 's/SampleA1/SRR2013442/g' fastq/sample.tab.xls &&\
sed -i 's/A1/M_PGC_4W_embryo3_sc1/g' fastq/sample.tab.xls

cat fastq/sample.tab.xls
```

sample|	brief_name	|stage	|sample_group	|ERCC_time	|RFP_polyA	|GFP_polyA	|CRE_polyA	|end_type|	rename
---|---|---|---|---|---|---|---|---|---|---
SRR2013442|	M\_PGC\_4W\_embryo3\_sc1|	Group1	|RNA	|0.0|	0.0	|0.0	|0.0|	PE	|M\_PGC\_4W\_embryo3\_sc1

如果有更多的输入文件，在这里依次写下来即可。

Docker 流程的第一步会下载网上的 fasta 文件，并且建立 bwa 软件的 index。建立的 index 放在 docker 内部的 ```/home/analyzer/database_RNA```， 这个文件建立后，可以传入 oss 中，下次分析时可以直接从 oss 取出放入 ```/home/analyzer/database_RNA```， 避免重复运算。

我这里由于已经有了 ref 文件，所以我直接从 OSS 下载，然后直接把这个文件夹挂在 Docker 的 ```/home/analyzer/database_ChIP``` 目录下即可。读者如果想尝试，这里的几步应调过，直接生成 ref 文件。当然时间可能会很久，得几个小时，如果想继续使用，这里需要注意保存输出文件，以便以后使用。

```bash
pip install alioss
osscmd.py config --host=hubqgenomeref.oss-cn-beijing-internal.aliyuncs.com --id=我的AccessKeyID --key=我的AccessKeySecret
mkdir ./RefRNA
osscmd.py downloadtodir oss://hubqgenomeref/Database_RNA_v2/hg19 ./RefRNA

# 这里 RNA 部分不需要
osscmd.py downloadtodir oss://hubqgenomeref/Database_ChIP_v2/mm10 ./RefChIP

```

### 2.2 执行流程

此时文件结构如下：

```
.
├── fastq
│   ├── sample.tab.xls
│   ├── SRR2013442
│   │   ├── SRR2013442_1.fastq.gz
│   │   └── SRR2013442_2.fastq.gz
│   └── SRR2013442.sra
├── RefChIP
│   ├── chrom.sort.bed
│   ├──....... 
│   └── region.Intragenic.bed
├── RefRNA
│   ├── all.exon.sort.ERCC.gene.bed
│   ├──.......
│   └── splicing_sites.sh
├── sratoolkit.2.8.2-1-ubuntu64
│   ├── bin
│   │   ├── abi-dump -> abi-dump.2
│   ├──.......
├── sratoolkit.2.8.2-1-ubuntu64.tar.gz
└── tangEpiNGSInstall
    ├── Dockerfile
    ├── README.md
    ├── settings
    │   ├── run_chipseq.py
    │   ├── run_mRNA.py
    │   ├── scripts_chipseq.py
    │   └── scripts_mRNA.py
    ├── src
    │   └── run_sample.sh
    ├── test_fq
    │   ├── H3K4me3
    │   │   ├── test.1.fq.gz
    │   │   └── test.2.fq.gz
    │   ├── Input
    │   │   ├── test.1.fq.gz
    │   │   └── test.2.fq.gz
    │   └── sample.tab.xls
    └── test_fq_RNA
        ├── SampleA1
        │   ├── test.1.fastq.gz
        │   └── test.2.fastq.gz
        └── sample.tab.xls
```

后台执行命令：

```bash
nohup docker run  \
	-v /root/fastq:/fastq \
	-v /root/outRNA:/home/analyzer/project \
	-v /root/RefRNA/:/home/analyzer/database_RNA/hg19\
	-v /root/tangEpiNGSInstall/settings/:/settings/ \
	--env ref=hg19 --env type=RNA     \
	registry.aliyuncs.com/hubq/tanginstall &
```

输入 

```bash
top
```

发现程序真的运行起来了。

![Figure 4](/images/2016-07-08-AnalysisNGSonAliYum/Fig5.running.png)


根据 ```~/tangEpiNGSInstall/settings/run_mRNA.py```文件73 行到 80 行的设定，在 QC以及比对后的表达定量步骤，会 4 个程序并行，而在比对这一步，同时只有 1 个程序并行。

```python
    part1 = m01.Map_From_raw(ref, sam_RNAinfo, is_debug=0)
    part1.s01_QC(core_num=4)
    part1.s02_Tophat(core_num=1)

    part2 = m02.RNA_Quantification(ref, sam_RNAinfo, core_num=4, is_debug=0)
    part2.run_pipeline(extra_GTF, given_GTF, is_MergeSam=1)

    part3 = m03.SampStat(ref, sam_RNAinfo, given_GTF, is_debug=0)
```

这是因为根据 ```~/tangEpiNGSInstall/settings/scripts_mRNA.py``` 这个文件 183-198 行的设定，

```python
l_sh_info.append("""
if [ $data_dype == "1" ]
    then $tophat_py                                                         \\
           -p 8 -G $gtf_file --library-type fr-unstranded                   \\
           --transcriptome-index $genome.refGene                            \\
           -o $tophat_dir/$brief_name  $genome                              \\
           $cln_dir/$samp_name/1.cln.fq.gz
fi
if [ $data_dype == "2" ]
    then $tophat_py                                                         \\
           -p 8 -G $gtf_file --library-type fr-unstranded                   \\
           --transcriptome-index $genome.refGene                            \\
           -o $tophat_dir/$brief_name  $genome                              \\
           $cln_dir/$samp_name/1.cln.fq.gz $cln_dir/$samp_name/2.cln.fq.gz
fi
        """) 
```

Tophat 使用了 8 个线程并行。然而服务器只有4个线程，所以我们依次就只并行一个程序了，防止运行速度过慢。并且 Tophat 运行最多可能会占用 7G 左右的内存，这样也是防止内存使用过多。

如果以后阿里云提供更强大的计算资源，并行度的这几个参数就可以增加，加速程序运算。

如果没有问题，则这个程序会一直继续往下跑，直到得出最终结果。

顺便测试下 ChIP-Seq 的流程表现如何：

```bash
docker run  -v /root/tangEpiNGSInstall/test_fq:/fastq -v /root/outChIP:/home/analyzer/project -v /root/RefChIP/:/home/analyzer/database_ChIP/mm10  -v /root/tangEpiNGSInstall/settings/:/settings/ --env ref=mm10 --env type=ChIP   registry.aliyuncs.com/hubq/tanginstall
```

## 3. 汇总结果

测试的 ChIP-Seq 流程很快能结束，重要结果如下：

```
outChIP/ChIP_test/result/
├── bigwig
│   └── mESC_H3K4me3_treat_minus_control.sort.norm.bw
├── peaks
│   ├── mESC_H3K4me3_VS_Input_peaks.conservative.regionPeak.gz
│   └── mESC_H3K4me3_VS_Input_peaks.conservative.regionPeak.gz.tbi
└── tables
    ├── 01.Basic_info.sample.tab.xls
    └── IDR_result.sample.tab.xls
```

RNA 实战案例的重要结果包括:

```bash
outRNA/RNA_test/result/
├── [4.0K]  count  
│   ├── [224K]  merge.dexseq_clean.gene.xls
│   ├── [  26]  merge.dexseq_clean_lncRNA.gene.xls
│   ├── [224K]  merge.dexseq_clean_refseq.gene.xls #后续用 DESeq 分析差异表达
│   ├── [ 127]  merge.dexseq_clean.stat.xls
│   ├── [1.3K]  merge.dexseq_ERCC_RGCPloyA.gene.xls
│   ├── [1.7K]  merge.dexseq_ERCC_RGCPloyA.RPKM.xls
│   ├── [  26]  merge.dexseq_ERCC_RGCPloyA.stat.xls
│   └── [   0]  merge.dexseq_NeoPass.gene.xls
├── [4.0K]  fpkm
│   └── [426K]  merge.FPKM.gene.xls #FPKM 表达量作图
├── [4.0K]  repeat
│   ├── [226M]  merge.Repeat.Count.xls
│   ├── [237M]  merge.Repeat.RPKM.xls 
│   ├── [1.1K]  merge.Repeat.SumCount.element.xls #重复序列 表达量作图
│   ├── [ 428]  merge.Repeat.SumCount.group.xls
│   └── [ 23K]  merge.Repeat.SumCount.subgroup.xls
└── [4.0K]  table
    ├── [ 323]  01.BasicInfo_QC_map_SpikeIn.xls
    └── [ 182]  02.ERCC_Mols.xls
```



以上结果请放入 OSS 中。至此，基本分析流程结束，下一阶段可以进行高级分析。
