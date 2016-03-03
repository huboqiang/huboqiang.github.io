---
title: "R plot PCA using ggplot2"
tagline: ""
last_updated: 2016-03-03
category: programming language
layout: post
tags : [R ggplot2]
---

# PCA and ggplot2

# 1. Load data
Using iris data for analysis.

```r
head(iris)
df <- iris
df <- as.data.frame(iris)
```

Dimension conversion for futher analysis:

```r
row.names(df) <- paste(df$Species, row.names(df), sep="_") 
df$Species <- NULL

head(df)
```


# 2. PCA analysis using prcomp

```r
df_pca <- prcomp(df)
```

## 2.1 plot directly

```r
plot(df_pca$x[,1], df_pca$x[,2])
```

![png](/images/2016-03-03-PCAggplot2/Fig1.png)

# 3. Using ggplot2 to revise this plot:

First, a new dataframe should be created, with the information of sample-group.

```r
df_out <- as.data.frame(df_pca$x)
df_out$group <- sapply( strsplit(as.character(row.names(df)), "_"), "[[", 1 )
head(df_out)
```

Second, some prepartions for ggplot2:

```r
library(ggplot2)
library(grid)
library(gridExtra)
```


Try:

```{r}
p<-ggplot(df_out,aes(x=PC1,y=PC2,color=group ))
p<-p+geom_point()
p
```

![png](/images/2016-03-03-PCAggplot2/Fig2.png)


## 3.2 plot with a theme
For format of picture. 

```r
theme<-theme(panel.background = element_blank(),panel.border=element_rect(fill=NA),panel.grid.major = element_blank(),panel.grid.minor = element_blank(),strip.background=element_blank(),axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black"),axis.ticks=element_line(colour="black"),plot.margin=unit(c(1,1,1,1),"line"))
```

For plot:

```r
p<-ggplot(df_out,aes(x=PC1,y=PC2,color=group ))
p<-p+geom_point()+theme
p
```

![png](/images/2016-03-03-PCAggplot2/Fig3.png)


## 3.3 Put the words on the figure:

```{r}
p<-ggplot(df_out,aes(x=PC1,y=PC2,color=group, label=row.names(df) ))
p<-p+geom_point()+ geom_text(size=3)+theme
p

```

![png](/images/2016-03-03-PCAggplot2/Fig4.png)


## 3.4 The percentage:

```r
percentage <- round(df_pca$sdev / sum(df_pca$sdev) * 100, 2)
percentage <- paste( colnames(df_out), "(", paste( as.character(percentage), "%", ")", sep="") )

p<-ggplot(df_out,aes(x=PC1,y=PC2,color=group ))
p<-p+geom_point()+theme + xlab(percentage[1]) + ylab(percentage[2])
p
```

![png](/images/2016-03-03-PCAggplot2/Fig5.png)


## 3.5 Change the order for Sample group:

```r

df_out$group <- factor(df_out$group, levels = c("virginica", "setosa", "versicolor"))

p<-ggplot(df_out,aes(x=PC1,y=PC2,color=group ))
p<-p+geom_point()+theme + xlab(percentage[1]) + ylab(percentage[2]) + scale_color_manual(values=c("#FFFF00", "#00FFFF", "#FF00FF"))
p
```

![png](/images/2016-03-03-PCAggplot2/Fig6.png)

## 3.5 Save in PDF file


```r
pdf(  "file_out.pdf",width = 10,height = 10)
library(gridExtra)
yy <- grid.arrange(p,nrow=1)
op <- par(no.readonly=TRUE)
par(op)
dev.off()
```

## 3.6 Plot features that contribute to the classification

```r
df_out_r <- as.data.frame(df_pca$rotation)
df_out_r$feature <- row.names(df_out_r)

df_out_r

p<-ggplot(df_out_r,aes(x=PC1,y=PC2,label=feature,color=feature ))
p<-p+geom_point()+theme + geom_text(size=3)
p

```

![png](/images/2016-03-03-PCAggplot2/Fig7.png)
