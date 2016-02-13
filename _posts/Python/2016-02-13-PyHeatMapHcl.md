---
title: "Python matplotlib 作图方法"
tagline: ""
last_updated: 2016-02-13
category: programming language
layout: post
tags : [Python, matplotlib, hclust]
---

# Hierarchical Clustering, Heatmaps, and Gridspec

[http://nbviewer.jupyter.org/github/ucsd-scientific-python/user-group/blob/master/presentations/20131016/hierarchical_clustering_heatmaps_gridspec.ipynb](http://nbviewer.jupyter.org/github/ucsd-scientific-python/user-group/blob/master/presentations/20131016/hierarchical_clustering_heatmaps_gridspec.ipynb)

Chris DeBoever

cdeboeve@ucsd.edu

UCSD Scientific Python User Group

10/16/2013

This is the post-presentation notebook with bug fixes. There may be more, use at your own risk.

This notebook shows how to make a simple annotated heatmap based on hierarchical clustering using matplotlib\'s `GridSpec` class.


```python
import brewer2mpl
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch

%matplotlib inline  
np.random.seed(5)
```


```python

# font size for figures
matplotlib.rcParams.update({'font.size': 16})
# Arial font
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
```


```python

# helper for cleaning up axes by removing ticks, tick labels, frame, etc.
def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
```

## Gridspec

See matplotlib\'s user\'s guide [gridspec page](http://matplotlib.org/users/gridspec.html) for an introduction to matplotlib\'s `Gridspec` class.

## Test data

We\'ll make some test data from two obviously different clusters.


```python
# test data
testL = []
# 5 samples from one group
for i in range(5):
    # 20 measurements from normal with mean 10, stdev 2
    testL.append(np.random.normal(10,2,20))
# 8 samples from another group
for i in range(8):
    # 20 measurements from normal with mean 4, stdev 4
    testL.append(np.random.normal(4,4,20))
```


```python
# permute test data and make dataframe
testA = np.array(testL)[np.random.permutation(range(len(testL)))]
testDF = pd.DataFrame(testA)
```


```python
testDF.shape
```




    (13, 20)




```python
# look at raw data
axi = plt.imshow(testDF,interpolation='nearest',cmap=matplotlib.cm.RdBu)
ax = axi.get_axes()
clean_axis(ax)
```

    /data/Analysis/huboqiang/software/anaconda/lib/python2.7/site-packages/matplotlib/artist.py:221: MatplotlibDeprecationWarning: This has been deprecated in mpl 1.5, please use the
    axes property.  A removal date has not been set.
      warnings.warn(_get_axes_msg, mplDeprecation, stacklevel=1)



![png](/images/2016-02-13-PyHeatMapHcl/output_13_1.png)


## Row clustering and dendrogram


```python
# calculate pairwise distances for rows
pairwise_dists = distance.squareform(distance.pdist(testDF))
print 'Number of rows: {0}'.format(testDF.shape[0])
print 'Size of distance matrix: {0}'.format(pairwise_dists.shape)
```

    Number of rows: 13
    Size of distance matrix: (13, 13)


`sch.linkage` performs hierarchical clustering. As you would expect, you can specify the method of clustering with the `method` parameter and the metric with the `metric`.


```python
# cluster
clusters = sch.linkage(pairwise_dists,method='complete')
```

`sch.linkage` returns a matrix with 4 columns and n-1 rows. Each row records which two clusters were combined as the heirarchical clustering was performed. For instance, if the first row is

    [  5.        ,   8.        ,  13.91203052,   2.        ],

then clusters 5 and 8 were combined because they had a distance of 13.91203052. The 2 means that there are two original observations in the newly formed cluster. If the second row is 

    [  3.           9.          17.77496623   2.        ],

then clusters 3 and 9 were combined because they had a distance of 17.8 and there are two original observations in the new cluster etc.

We can use `sch.dendrogram` to draw a dendrogram from these results and also provide a nice representation of the clustering.


```python
# dendrogram
den = sch.dendrogram(clusters)
```

    /data/Analysis/huboqiang/software/anaconda/lib/python2.7/site-packages/matplotlib/font_manager.py:1288: UserWarning: findfont: Font family [u'sans-serif'] not found. Falling back to Bitstream Vera Sans
      (prop.get_family(), self.defaultFamily[fontext]))



![png](/images/2016-02-13-PyHeatMapHcl/output_19_1.png)


Running dendrogram will by default plot a dendrogram with some funky coloring. If we want to have just a single color, we can use the option `color_threshold=np.inf`. We can also set an `sch` parameter to start coloring with the color black:


```python
# make dendrograms black rather than letting scipy color them
sch.set_link_color_palette(['black'])
# or 
den = sch.dendrogram(clusters,color_threshold=np.inf)
```


![png](/images/2016-02-13-PyHeatMapHcl/output_21_0.png)


If we don't want the dendrogram, we can set `no_plot=True`.


```python
# dendrogram without plot
den = sch.dendrogram(clusters,color_threshold=np.inf,no_plot=True)
```

`sch.dendrogram` returns a dict with some useful information. In particular, the key `leaves` holds the indices into our original data ordered by the clustering:


```python
den['leaves']
```




    [11, 1, 4, 7, 12, 10, 2, 8, 6, 9, 0, 3, 5]



## Heatmap

Let\'s reorder our original data with these indices and plot:


```python
axi = plt.imshow(testDF.ix[den['leaves']],interpolation='nearest',cmap=matplotlib.cm.RdBu)
ax = axi.get_axes()
clean_axis(ax)
```


![png](/images/2016-02-13-PyHeatMapHcl/output_28_0.png)


We can see that we recovered our two clusters. If we\'d like, we can plot our dendrogram along with the heatmap:


```python
fig = plt.figure()
heatmapGS = gridspec.GridSpec(1,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1])

### row dendrogram ###
denAX = fig.add_subplot(heatmapGS[0,0])
denD = sch.dendrogram(clusters,color_threshold=np.inf,orientation='right')
clean_axis(denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[0,1])
axi = heatmapAX.imshow(testDF.ix[den['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)
```


![png](/images/2016-02-13-PyHeatMapHcl/output_30_0.png)


I used a gridspec to make multiple subplots of different sizes and then put the dendrogram in one subplot and the heatmap in another. This is an easy way to control their relative sizes. Note that I had to use `aspect='auto'` and `origin=lower` for the heatmap.

We can follow the same process to cluster and make a dendrogram for the columns:


```python
# rename row clusters
row_clusters = clusters
```


```python
# calculate pairwise distances for columns
col_pairwise_dists = distance.squareform(distance.pdist(testDF.T))
# cluster
col_clusters = sch.linkage(col_pairwise_dists,method='complete')
```


```python
# plot the results
fig = plt.figure()
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ####
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[den['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

fig.tight_layout()
```


![png](/images/2016-02-13-PyHeatMapHcl/output_34_0.png)


Not too bad. There aren't any relationships among the columns, so that clustering doesn\'t really show anything meaningful.

## Heatmap annotations

### Row and column labels

The simplest things we might want to add to our heatmap are row and column names. Let\'s add some row and column names to the test data:


```python
testDF.index = [ 'Sample ' + str(x) for x in testDF.index ]
testDF.columns = [ 'c' + str(x) for x in testDF.columns ]
```

There are a few things we have to consider here. First, we need reorder the labels to fit the dendrogram. We can do this easily with 

```python
testDF.index[row_denD['leaves']]
```

We also need to switch the y-axis on the heatmap from the left side of the axis to the right side. This is done with

```python
heatmapAX.yaxis.set_ticks_position('right')
```

The last thing we have to worry about is the tick locations. My function `clean_axis` wiped out the tick locations, so we\'ll reset them with

```python
heatmapAX.set_yticks(arange(testDF.shape[0]))
```


```python
# heatmap with row names
fig = plt.figure()
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ###
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[den['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

fig.tight_layout()
```


![png](/images/2016-02-13-PyHeatMapHcl/output_41_0.png)


Column names are along the same lines:


```python
# heatmap with row names
fig = plt.figure()
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ###
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### heatmap ####
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])

## col labels ##
heatmapAX.set_xticks(np.arange(testDF.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']])
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

fig.tight_layout()
```


![png](/images/2016-02-13-PyHeatMapHcl/output_43_0.png)


And now we have row and column labels.

### Scale colorbar

Another reasonable addition is a scale for the heatmap.


```python
# heatmap with row names
fig = plt.figure(figsize=(12,8))
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ###
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### heatmap ####
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])

## col labels ##
heatmapAX.set_xticks(np.arange(testDF.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']])
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

### scale colorbar ###
scale_cbGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[0,0],wspace=0.0,hspace=0.0)
scale_cbAX = fig.add_subplot(scale_cbGSSS[0,0]) # colorbar for scale in upper left corner
cb = fig.colorbar(axi,scale_cbAX) # note that we tell colorbar to use the scale_cbAX axis
cb.set_label('Measurements')
cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
cb.outline.set_linewidth(0)

fig.tight_layout()
```


![png](/images/2016-02-13-PyHeatMapHcl/output_47_0.png)


The above looks pretty good, but there are a few stylistic changes we might want to make. For instance, you might want the color bar to be \"centered\" on zero (not a good choice for this data, but instructive for the tutorial). You might also want to change the ticks on the color bar so they aren\'t overlapping and move the scale closer to the heatmap. Let\'s try it out.

To force `imshow` to use a symmetric scale, we need to define an instance of `matplotlib.colors.Normalize` or provide imshow with the `vmin` and `vmax` parameters. However, defining `vmin` and `vmax` will rescale our data so that the minimum is `vmin` and the maximum is `vmax`, so we \'ll define a norm. This might be a little more useful because you can use the norm elsewhere. 

Changing the size of the colorbar ticks isn\'t too hard, but it\'s worth pointing out that when you draw a colorbar, you are actually creating an `axis` object and drawing the colorbar onto the new axis. Thus, while a call to `colorbar` creates a `matplotlib.colorbar.Colorbar` instance, you also create an axis. Here, however, I\'ve explicitly told `colorbar` to use an axis that I\'ve already created with `gridspec`:

    cb = fig.colorbar(axi,scale_cbAX)

The second option, `scale_cbAX`, sets the `cax` option as the axis I\'ve made for the colorbar. `colorbar` also has an `ax` option which specifies an axis to steal space from and make a new colorbar axis. You can access the colobar axis with `cb.ax`. Since I told `colorbar` to use `scale_cbAX`, `scale_cbAX` and `cb.ax` are the same.


```python
# make norm
vmin = np.floor(testDF.min().min())
vmax = np.ceil(testDF.max().max())
vmax = max([vmax,abs(vmin)]) # choose larger of vmin and vmax
vmin = vmax * -1
my_norm = matplotlib.colors.Normalize(vmin, vmax)
```


```python
# heatmap with row names
fig = plt.figure(figsize=(12,8))
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ###
col_denAX = fig.add_subplot(heatmapGS[0,1])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### heatmap ####
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',norm=my_norm,cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])

## col labels ##
heatmapAX.set_xticks(np.arange(testDF.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']])
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

### scale colorbar ###
scale_cbGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[0,0],wspace=0.0,hspace=0.0)
scale_cbAX = fig.add_subplot(scale_cbGSSS[0,1]) # colorbar for scale in upper left corner
cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
cb.set_label('Measurements')
cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
cb.outline.set_linewidth(0)
# make colorbar labels smaller
tickL = cb.ax.yaxis.get_ticklabels()
for t in tickL:
    t.set_fontsize(t.get_fontsize() - 3)

fig.tight_layout()
```


![png](/images/2016-02-13-PyHeatMapHcl/output_50_0.png)


In some instances, you may notice that a scale colorbar looks weird (discrete rectangles with space between them rather than continuous color distribution) when you save to PDF (maybe other formats too?). I\'ve run into this several times and have been able to solve it with

    cb.solids.set_edgecolor("face")
    
See [this Stack Overflow post](http://stackoverflow.com/questions/15003353/why-does-my-colorbar-have-lines-in-it) for more information.

### Sample colorbars

We might also want to add some colorbars that label some discrete covariates for our samples or columns (e.g. categorical data). First, we\'ll assign colors randomly to samples and columns, but you could also imagine that these labels correspond to labels for mutant and wild-type samples, labels for genes from different pathways, etc.


```python
# run dendrogram without color_threshold=np.inf to define some clusters
row_cbSE = pd.Series([brewer2mpl.get_map('Set1','Qualitative',3).mpl_colors[0]] * (testDF.shape[0] / 2) + \
    [brewer2mpl.get_map('Set1','Qualitative',3).mpl_colors[1]] * (testDF.shape[0] / 2 + testDF.shape[0] % 2))
col_cbSE = pd.Series([brewer2mpl.get_map('Set2','Qualitative',3).mpl_colors[0]] * (testDF.shape[1] / 2) + \
    [brewer2mpl.get_map('Set2','Qualitative',3).mpl_colors[1]] * (testDF.shape[1] / 2 + testDF.shape[1] % 2))
```


```python
# heatmap with row names
fig = plt.figure(figsize=(12,8))
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ###
colGSSS = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=heatmapGS[0,1],wspace=0.0,hspace=0.0,height_ratios=[1,0.25])
col_denAX = fig.add_subplot(colGSSS[0,0])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### col colorbar ###
col_cbAX = fig.add_subplot(colGSSS[1,0])
col_axi = col_cbAX.imshow([list(col_cbSE.ix[col_denD['leaves']])],interpolation='nearest',aspect='auto',origin='lower')
clean_axis(col_cbAX)

### row dendrogram ###
rowGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[1,0],wspace=0.0,hspace=0.0,width_ratios=[1,0.25])
row_denAX = fig.add_subplot(rowGSSS[0,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### row colorbar ###
row_cbAX = fig.add_subplot(rowGSSS[0,1])
row_axi = row_cbAX.imshow([ [x] for x in row_cbSE.ix[row_denD['leaves']].values ],interpolation='nearest',aspect='auto',origin='lower')
clean_axis(row_cbAX)

### heatmap ####
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',norm=my_norm,cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])

## col labels ##
heatmapAX.set_xticks(np.arange(testDF.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']])
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

### scale colorbar ###
scale_cbGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[0,0],wspace=0.0,hspace=0.0)
scale_cbAX = fig.add_subplot(scale_cbGSSS[0,1]) # colorbar for scale in upper left corner
cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
cb.set_label('Measurements')
cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
cb.outline.set_linewidth(0)
# make colorbar labels smaller
tickL = cb.ax.yaxis.get_ticklabels()
for t in tickL:
    t.set_fontsize(t.get_fontsize() - 3)

fig.tight_layout()
```


![png](/images/2016-02-13-PyHeatMapHcl/output_55_0.png)


Our row and column colorbars don\'t have any meaning, but they sure are pretty. However, the heatmap is looking a cramped with all of the different parts of the image adjacent to each other. Let\'s add some space and make the row and column colorbars a little smaller. I\'m also going to reduce the width of the scale colorbar.


```python
# heatmap with row names
fig = plt.figure(figsize=(12,8))
heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

### col dendrogram ###
colGSSS = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=heatmapGS[0,1],wspace=0.0,hspace=0.1,height_ratios=[1,0.15])
col_denAX = fig.add_subplot(colGSSS[0,0])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### col colorbar ###
col_cbAX = fig.add_subplot(colGSSS[1,0])
col_axi = col_cbAX.imshow([list(col_cbSE.ix[col_denD['leaves']])],interpolation='nearest',aspect='auto',origin='lower')
clean_axis(col_cbAX)

### row dendrogram ###
rowGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[1,0],wspace=0.1,hspace=0.0,width_ratios=[1,0.15])
row_denAX = fig.add_subplot(rowGSSS[0,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### row colorbar ###
row_cbAX = fig.add_subplot(rowGSSS[0,1])
row_axi = row_cbAX.imshow([ [x] for x in row_cbSE.ix[row_denD['leaves']].values ],interpolation='nearest',aspect='auto',origin='lower')
clean_axis(row_cbAX)

### heatmap ####
heatmapAX = fig.add_subplot(heatmapGS[1,1])
axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',norm=my_norm,cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right') 
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])

## col labels ##
heatmapAX.set_xticks(np.arange(testDF.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']])
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

### scale colorbar ###
scale_cbGSSS = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=heatmapGS[0,0],wspace=0.0,hspace=0.0)
scale_cbAX = fig.add_subplot(scale_cbGSSS[0,1]) # colorbar for scale in upper left corner
cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
cb.set_label('Measurements')
cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
cb.outline.set_linewidth(0)
# make colorbar labels smaller
tickL = cb.ax.yaxis.get_ticklabels()
for t in tickL:
    t.set_fontsize(t.get_fontsize() - 3)

#fig.tight_layout()
heatmapGS.tight_layout(fig,h_pad=0.1,w_pad=0.5)
```


![png](/images/2016-02-13-PyHeatMapHcl/output_57_0.png)


As you can see in the code above, there are a couple of options for adding some spacing between the various gridspec axes. When making gridspec objects, you can specify spacing. However, using `tight_layout` will interfere with the spacing, so I find it easier to add some padding with `tight_layout`. For the `GridSpecFromSubplotSpec` objects, however, I added the padding when I made the gridspecs because I didn\'t use `tight_layout` there.

One annoying thing is that the spacing is specified proportionally, so if your image isn\'t square, you have to specify spacing differently vertically and horizontally to make the spacing even (the same goes for the gridspecs for dendrograms and row/column colorbars). You could use the image size to decide the scaling of the various gridspec instances to make everything the same size, but I won\'t do that here.

I used `GridSpecFromSubplotSpec` to make the dendrogram and row/column colorbar axes here, but it\'s probably better to specify these axes as part of the main gridspec (i.e. `heatmapGS` in my examples). I just used `GridSpecFromSubplotSpec` here for demonstration. I will do that below. Notice my use below of 

    scale_cbAX = fig.add_subplot(heatmapGS[0:2,0])
    
for specifying the axis for the scale colorbar. This combines two gridspecs together into one axis.


```python
# heatmap with row names
fig = plt.figure(figsize=(12,8))
heatmapGS = gridspec.GridSpec(3,3,wspace=0.0,hspace=0.0,width_ratios=[0.25,0.05,1],height_ratios=[0.25,0.05,1])

### col dendrogram ###
colGSSS = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=heatmapGS[0,1],wspace=0.0,hspace=0.1,height_ratios=[1,0.15])
col_denAX = fig.add_subplot(heatmapGS[0,2])
col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
clean_axis(col_denAX)

### col colorbar ###
col_cbAX = fig.add_subplot(heatmapGS[1,2])
col_axi = col_cbAX.imshow([list(col_cbSE.ix[col_denD['leaves']])],interpolation='nearest',aspect='auto',origin='lower')
clean_axis(col_cbAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[2,0])
row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
clean_axis(row_denAX)

### row colorbar ###
row_cbAX = fig.add_subplot(heatmapGS[2,1])
row_axi = row_cbAX.imshow([ [x] for x in row_cbSE.ix[row_denD['leaves']].values ],interpolation='nearest',aspect='auto',origin='lower')
clean_axis(row_cbAX)

### heatmap ####
heatmapAX = fig.add_subplot(heatmapGS[2,2])
axi = heatmapAX.imshow(testDF.ix[row_denD['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',norm=my_norm,cmap=matplotlib.cm.RdBu)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(testDF.shape[0]))
heatmapAX.yaxis.set_ticks_position('right') 
heatmapAX.set_yticklabels(testDF.index[row_denD['leaves']])

## col labels ##
heatmapAX.set_xticks(np.arange(testDF.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(testDF.columns[col_denD['leaves']])
# rotate labels 90 degrees
for label in xlabelsL:
    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
    l.set_markersize(0)

### scale colorbar ###
scale_cbAX = fig.add_subplot(heatmapGS[0:2,0]) # colorbar for scale in upper left corner
cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
cb.set_label('Measurements')
cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
cb.outline.set_linewidth(0)
# make colorbar labels smaller
tickL = cb.ax.yaxis.get_ticklabels()
for t in tickL:
    t.set_fontsize(t.get_fontsize() - 3)

#fig.tight_layout()
heatmapGS.tight_layout(fig,h_pad=0.1,w_pad=0.5)
```


![png](/images/2016-02-13-PyHeatMapHcl/output_59_0.png)


Stylistically, the colorbar is probably too wide. One easy way to alter the width would be to use `GridSpecFromSubplotSpec` to make a smaller axis for the colorbar. You can try it out.

I\'ve collected a minimal set of code from this notebook below for use as a template. This code along with the plotting commands in the above cell should be a good start for making your own heatmaps.


```python
# Stop those who "Run All"
3 +
```


```python
# helper for cleaning up axes by removing ticks, tick labels, frame, etc.
def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# make dendrograms black rather than letting scipy color them
sch.set_link_color_palette(['black'])

# calculate pairwise distances for rows
row_pairwise_dists = distance.squareform(distance.pdist(testDF))
# cluster
row_clusters = sch.linkage(row_pairwise_dists,method='complete')
# calculate pairwise distances for columns
col_pairwise_dists = distance.squareform(distance.pdist(testDF.T))
# cluster
col_clusters = sch.linkage(col_pairwise_dists,method='complete')

# run dendrogram without color_threshold=np.inf to define some clusters
row_cbSE = pd.Series([brewer2mpl.get_map('Set1','Qualitative',3).mpl_colors[0]] * (testDF.shape[0] / 2) + \
    [brewer2mpl.get_map('Set1','Qualitative',3).mpl_colors[1]] * (testDF.shape[0] / 2 + testDF.shape[0] % 2))
col_cbSE = pd.Series([brewer2mpl.get_map('Set2','Qualitative',3).mpl_colors[0]] * (testDF.shape[1] / 2) + \
    [brewer2mpl.get_map('Set2','Qualitative',3).mpl_colors[1]] * (testDF.shape[1] / 2 + testDF.shape[1] % 2))

# make norm
vmin = np.floor(testDF.min().min())
vmax = np.ceil(testDF.max().max())
vmax = max([vmax,abs(vmin)]) # choose larger of vmin and vmax
vmin = vmax * -1
my_norm = matplotlib.colors.Normalize(vmin, vmax)
```

