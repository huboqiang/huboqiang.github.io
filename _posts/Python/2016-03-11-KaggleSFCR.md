---
title: "Kaggle San Francisco Criminal Data Visualization"
tagline: ""
last_updated: 2016-03-11
category: programming language
layout: post
tags : [Python, matplotlib, kaggle, kde]
---

# Kaggle San Francisco Criminal Data Visualization

Copy from my kaggle site:[https://www.kaggle.com/johnhu/sf-crime/geo-comparision-for-different-crimes/notebook](https://www.kaggle.com/johnhu/sf-crime/geo-comparision-for-different-crimes/notebook)

In this script, the geocode distribution would be plot for different catagories of crimes.
Futhermore, the comparision between two categories of criminals could also be viewed using this script.
The statsmodels nonparametric module were used to calculate the distribution for the given geocode. And the results were shown into 100 bins contour-plot. For 2-samples comparision, the shown result were one 100x100 bins density distribution minus the other.
Thanks to [https://www.kaggle.com/swbevan/sf-crime/a-history-of-crime-python/code](https://www.kaggle.com/swbevan/sf-crime/a-history-of-crime-python/code) because this is my first kaggle-script and I really benifit a lot from him. Also, some of this code were revised from a small part of seaborn

# 1. Loading modules, parameters and functions.

```python

# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import contextlib
import statsmodels.nonparametric.api as smnp
import seaborn as sns


from ipywidgets import widgets  
from IPython.display import display

%matplotlib inline  
matplotlib.rcParams['figure.figsize'] = (15, 25)

sns.despine(fig=None, left=False, right=False, top=False, bottom=False, trim=True)


lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

def parse_timeInfo(pd_input, colname="Dates"):
    l_y = []
    l_m = []
    l_d = []
    l_h = []
    for i in range(pd_input.shape[0]):
        dt = pd_input[colname][i]
        l_y.append(pd.Timestamp(dt).year)
        l_m.append(pd.Timestamp(dt).month)
        l_d.append(pd.Timestamp(dt).day)
        l_h.append(pd.Timestamp(dt).hour)
    
    pd_input['Year'] = l_y
    pd_input['Month'] = l_m
    pd_input['Day'] = l_d
    pd_input['Hour'] = l_h


def num_l_word(l_word, M_wordDict=None):
    if M_wordDict is None:
        l_word_unique = sorted(list(set(l_word)))
        M_wordDict = dict(zip(l_word_unique, range(len(l_word_unique))))
    l_word_idx = [ M_wordDict[w] for w in l_word ]
    return l_word_idx, M_wordDict

def Df_wordParseToNum(pd_input, colname, M_wordDict=None):
    l_word = list(pd_input[colname])
    l_word_idx, M_wordDict = num_l_word(l_word, M_wordDict)
    pd_input[colname] = l_word_idx
    return M_wordDict

def kde_support(data, bw, gridsize, cut, clip):
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    print(bw, support_min, support_max)
    return np.linspace(support_min, support_max, gridsize)

def smnp_kde(pd_input, cut, gridsize, clipsize, bw="scott"):
    bw_func = getattr(smnp.bandwidths, "bw_" + bw)
    x_bw = bw_func(pd_input["X"].values)
    y_bw = bw_func(pd_input["Y"].values)
    bw = [x_bw, y_bw]
    kde = smnp.KDEMultivariate( pd_input.T.values, "cc", bw)
    x_support = kde_support(pd_input['X'].values, x_bw, gridsize, cut, clipsize[0])
    y_support = kde_support(pd_input['Y'].values, y_bw, gridsize, cut, clipsize[1])
    
    xx, yy = np.meshgrid(x_support, y_support)
    Z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, Z
    
    
@contextlib.contextmanager
def quitting(thing):
    yield thing
    thing.quit()


def get_heatmap(pd_train, clipsize, category):
    l_colNameUsed = [ 'X', 'Y']
    pd_train_used = pd_train[pd_train['Category']==category][ l_colNameUsed ]
    cut = 10
    gridsize = 100
    xx, yy, Z = smnp_kde(pd_train_used, cut=cut, gridsize=gridsize, clipsize=clipsize)
    return xx, yy, Z


def remove_axis(ax):
    ax.get_xaxis().set_ticks( [] )
    ax.get_xaxis().set_ticklabels( [] )
    ax.get_yaxis().set_ticks( [] )
    ax.get_yaxis().set_ticklabels( [] )

    
def plot_one_heatmap(xx, yy, Z, clipsize, pdf_name):
    mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
    up_max = np.percentile(Z, 99)
    Z[Z > up_max] = up_max
    cut = 10
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
    ax1.contourf(xx, yy, Z, cut, cmap="jet", shade=True, alpha=0.5).collections[0].set_alpha(0)
    remove_axis(ax1)
    fig.savefig(pdf_name)
    fig.show()
    

def plot_cmp_heatmap(xx, yy, Z1, Z2, clipsize, pdf_name):
    mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
    up_max1 = np.percentile(Z1, 99)
    Z1[Z1 > up_max1] = up_max1
    up_max2 = np.percentile(Z2, 99)
    Z2[Z2 > up_max2] = up_max2
    
    cut = 10
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax1.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
    ax2.imshow(mapdata,extent=lon_lat_box, cmap=plt.get_cmap('gray'))
    delta_Z_h = Z1-Z2
    delta_Z_l = Z2-Z1
    delta_Z_h[delta_Z_h < 0] = 0
    delta_Z_l[delta_Z_l < 0] = 0
    ax1.contourf(xx, yy, delta_Z_h, cut, cmap="jet", shade=True, alpha=0.5).collections[0].set_alpha(0)
    ax2.contourf(xx, yy, delta_Z_l, cut, cmap="jet", shade=True, alpha=0.5).collections[0].set_alpha(0)
    remove_axis(ax1)
    remove_axis(ax2)
    fig.savefig(pdf_name)
    fig.show()
    
```

# 2. Loading the input data

```python
infile_train = "../input/train.csv"
pd_train = pd.read_csv(infile_train)
parse_timeInfo(pd_train)
l_colNameUsed = [ 'X', 'Y']
y, M_categoryDict = num_l_word(list(pd_train['Category']) ) 
CateGoryDict = {"Category":sorted(M_categoryDict.keys())}
````

# 3. Plot for one category of crime

```python
xx, yy, Z = get_heatmap(pd_train, clipsize, "ARSON")
plot_one_heatmap(xx, yy, Z, clipsize, "ARSON.pdf")
```

![png](/images/2016-03-11-KaggleSFCR/Fig1.single.png)


# 4. Comparing different kind of crimes

```python
xx, yy, Z1 = get_heatmap(pd_train, clipsize, "ARSON")
xx, yy, Z2 = get_heatmap(pd_train, clipsize, "ASSAULT")
plot_cmp_heatmap(xx, yy, Z1, Z2, clipsize, "ARSON__ASSAULT.pdf")
```


![png](/images/2016-03-11-KaggleSFCR/Fig2.compare.png)