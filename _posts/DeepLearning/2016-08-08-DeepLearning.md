---
title: "[20160808 Journal Club]Using Deep Learning to study the gene-regulatory elements."
tagline: ""
last_updated: 2016-08-08
category: Journal Club
layout: post
tags : [Deep Learning]
---

# Using Deep Learning to detect DNR-regulatory elements


<script src="/assets/blog20160808/js/foreign/d3.v3.min.js" charset="utf-8"></script>
<script src="/assets/blog20160808/js/foreign/jquery-1.7.0.min.js" charset="utf-8"></script>
<script src="/assets/blog20160808/js/foreign/jquery-ui.min.js" charset="utf-8"></script>
<script src="/assets/blog20160808/js/three.min.js"></script>
<script src="/assets/blog20160808/js/foreign/TrackballControls.js"></script>
<script src="/assets/blog20160808/js/BasicVis.js" type="text/javascript"></script>
<script src="/assets/blog20160808/js/MnistVis.js" type="text/javascript"></script>
<script src="/assets/blog20160808/js/data/MNIST.js" type="text/javascript"></script>
<script src="/assets/blog20160808/js/data/mnist_pca.js" type="text/javascript"></script>
<script src="/assets/blog20160808/js/data/MNIST-SNE-good.js"></script>
<script type="text/javascript" src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<link rel="stylesheet" href="http://code.jquery.com/ui/1.11.0/themes/cupertino/jquery-ui.css">

<script type="text/x-mathjax-config">
      MathJax.Hub.Register.StartupHook("TeX Jax Ready",function () {
        var TEX = MathJax.InputJax.TeX,
            MML = MathJax.ElementJax.mml;
        var CheckDimen = function (dimen) {
          if (dimen === "" ||
              dimen.match(/^\s*([-+]?(\.\d+|\d+(\.\d*)?))\s*(pt|em|ex|mu|px|mm|cm|in|pc)\s*$/))
                  return dimen.replace(/ /g,"");
          TEX.Error("Bad dimension for image: "+dimen);
        };
        TEX.Definitions.macros.img = "myImage";
        TEX.Parse.Augment({
          myImage: function (name) {
            var src = this.GetArgument(name),
                valign = CheckDimen(this.GetArgument(name)),
                width  = CheckDimen(this.GetArgument(name)),
                height = CheckDimen(this.GetArgument(name));
            var def = {src:src};
            if (valign) {def.valign = valign}
            if (width)  {def.width  = width}
            if (valign) {def.height = height}
            this.Push(this.mmlToken(MML.mglyph().With(def)));
          }
        });
      });
</script>

<script type="text/javascript">
      function mult_img_display (div, data) {
        var N = 7;
        div.style('width', '100%');
        var W = parseInt(div.style('width'));
        div.style('height', W/N);
        div.style('position', 'relative');
        for (var n = 0; n < 4; n++) {
          var div2 = div.append('div')
            .style('position', 'absolute')
            .style('left', (n+(N-4)/2)*W/N);
          //  .style('position', 'absolute')
          //  .left(n*W/5);
          var img_display = new BasicVis.ImgDisplay(div2)
            .shape([28,28])
            .imgs(data)
            .show(n);
          img_display.canvas
            .style('border', '2px solid #000000')
            .style('width', W/N*0.85);
        }
      }

      var mnist_tooltip = new BasicVis.ImgTooltip();
      mnist_tooltip.img_display.shape([28,28]);
      mnist_tooltip.img_display.imgs(mnist_xs);
      setTimeout(function() {mnist_tooltip.hide();}, 3000);
</script>

<script type="text/javascript">
      (function () {
        var div = d3.select("#mnist_image_examples");
        mult_img_display(div, mnist_xs)
      })()
</script>





<script type="text/javascript">
        var raw_mnist = null;
        mnist_pca.W1 = mnist_pca.W.subarray(0, 784);
        mnist_pca.W2 = mnist_pca.W.subarray(784, 2*784);
        var mnist_pca_plot;
        setTimeout(function(){
          mnist_pca_plot = new DirExploreMNIST("#pca_mnist");
          mnist_pca_plot.plot.b0(mnist_pca.W1);
          mnist_pca_plot.plot.b1(mnist_pca.W2);
          mnist_pca_plot.plot.scatter.yrange([-4,6]);
          mnist_pca_plot.plot.scatter.xrange([-2,10]);
          setTimeout(function() {
            for (var i = 0; i < 28; i++)
            for (var j = 0; j < 28; j++) {
              mnist_pca_plot.x.pixel_display.pixel_values[i][j] = 12*mnist_pca.W1[i+28*(28-j)];
              mnist_pca_plot.y.pixel_display.pixel_values[i][j] = 12*mnist_pca.W2[i+28*(28-j)];
            }
            mnist_pca_plot.x.pixel_display.render();
            mnist_pca_plot.y.pixel_display.render();
          }, 50);
        }, 2000);
</script>

<script type="text/javascript">
        setTimeout(function(){
          var test = new GraphLayout("#tsne_mnist");
          test.scatter.size(3.1);
          var test_wrap = new AnimationWrapper(test);
          test_wrap.button.on("mousemove", function() { mnist_tooltip.hide(); d3.event.stopPropagation();});

          setTimeout(function() {
            test.scatter.xrange([-35,35]);
            test.scatter.yrange([-35,35]);
            mnist_tooltip.bind(test.scatter.points);
            mnist_tooltip.bind_move(test.scatter.s);
            test_wrap.layout();
          }, 50);

          var W = new Worker("/assets/blog20160808/js/CostLayout-worker.js");

          test_wrap.bindToWorker(W);

          W.postMessage({cmd: "init", xs: mnist_xs, N: test.sne.length/2, D: 784, cost: "tSNE", perplexity:40});
          test_wrap.run   = function(){ W.postMessage({cmd: "run", steps: 1600, skip: 2, Kstep: 18.0, Kmu: 0.85})};

        }, 500);
</script>



## 0. Authors

### 0.1 Corresponding Authors:

![png](/images/2016-08-08-DeepLearning/In1.png)
![png](/images/2016-08-08-DeepLearning/In2.png)

- Predicting the sequence specificities of DnA- and RnA-binding proteins by deep learning

![png](/images/2016-08-08-DeepLearning/In3.png)
![png](/images/2016-08-08-DeepLearning/In4.png)

- Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks

### 0.2 Important contributor to Deep Learning:

**BIG THREE** during the development of Deep Learning theory:

 ![png](/images/2016-08-08-DeepLearning/In5.png)

**Geoffrey Hinton**: 

![png](/images/2016-08-08-DeepLearning/In6.png)

- Big contribution **Two Times**

![png](/images/2016-08-08-DeepLearning/In7.png)

by Ran Bi, NYU <br>
[http://www.kdnuggets.com/2014/10/deep-learning-make-machine-learning-algorithms-obsolete.html](http://www.kdnuggets.com/2014/10/deep-learning-make-machine-learning-algorithms-obsolete.html)




## 1. A brief introduction to DL

### 1.1. Data vitualization: From PCA to tSNE

Example from Colah's Blog [Visualizing MNIST: An Exploration of Dimensionality Reduction](http://colah.github.io/posts/2014-10-Visualizing-MNIST/):

- PCA performed well overall,  but not well in some detail region, like to divide 4, 7and  9

<div id="pca_mnist" class="figure" style="margin-bottom:0px;">
</div>
<div class="caption" style="margin-bottom:10px;">
	<strong>Visualizing MNIST with PCA</strong>
</div>

- tSNE performed much better.
<div id="tsne_mnist" class="figure" style="width: 60%; margin: 0 auto; margin-bottom: 8px;">
</div>

<div class="caption">
      <strong>Visualizing MNIST with t-SNE</strong>
</div>


### 1.2. tSNE were invented during the development of Deep learning theory

![png](/images/2016-08-08-DeepLearning/In8.png)

by Laurens van der Maaten <br>[https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)

### 1.3. Deep Learning includes:

- **Multi-layer network** (The inception of Deep learning)
	- More Layers.
	- Using GPU(显卡) for calculation with many many trials in a parallel way.
- **Convolutional Neuron Network** (CNN， 卷积神经网络) 
- **Recurrent Neuron Network** (RNN， 循环神经网络)
	- Time serial information(Video, Audio)
	- Using LSTM algorithm to make the calculation easier.
-  **Reinforce Learning Neuron Network**（增强学习）
    - Value network：To FEEL the environment.
    - Policy network：To Decide what is the best solution.
    - AlphaGO, Unpiloted cars


   
### 1.4 Deep Learning ABC:
- Three basic kinds of calculation: multiplication, addiction and transformation.

![png](/images/2016-08-08-DeepLearning/In9.png)

by Deep Learning Udacity Course <br>[https://cn.udacity.com/course/deep-learning--ud730/](https://cn.udacity.com/course/deep-learning--ud730/)




- For example: Colah's blog [Neural Networks, Manifolds, and Topology
](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

	- The transforming process:
![gif](/images/2016-08-08-DeepLearning/spiral.1-2.2-2-2-2-2-2.gif)

	- The key point is the training for parameters to be multiplied, added and so on. Bad parameters could get bad results:
![gif](/images/2016-08-08-DeepLearning/spiral.2.2-2-2-2-2-2-2.gif)

	- More dimension were required for complex datasets. For example, it is hard to divide these points in a 2D graph: 
![gif](/images/2016-08-08-DeepLearning/topology_2D-2D_train.gif)
However, after adding one demension, these two groups could be divided by a layer instead of a line.
![png](/images/2016-08-08-DeepLearning/topology_3d.png)

### 1.5 Convolution Kernal

![png](/images/2016-08-08-DeepLearning/In11.png)
![png](/images/2016-08-08-DeepLearning/In12.png)

by iOS Developer Guide<br>[https://developer.apple.com/library/ios/documentation/Performance/Conceptual/vImage/ConvolutionOperations/ConvolutionOperations.html](https://developer.apple.com/library/ios/documentation/Performance/Conceptual/vImage/ConvolutionOperations/ConvolutionOperations.html)


### 1.6 Deep Learning frame work

Framework | Core Programming Language | Interfaces from Other Languages | Programming Paradigm |Wrappers
---|---|---|---|---
Caffe | C++/CUDA | Python, Matlab | Imperative | -
TensorFlow |  C++/CUDA | Python | Declarative | Pretty Tensor, Keras
Theano |  Python (compiled to C++/CUDA) | – | Declarative | Keras, Lasagne, or Blocks
Torch7 | LuaJIT (with C/CUDA backend) | C | Imperative | -

TensorFlow: Biology’s Gateway to Deep Learning?<br>[http://www.cell.com/cell-systems/pdf/S2405-4712(16)00010-7.pdf](http://www.cell.com/cell-systems/pdf/S2405-4712(16)00010-7.pdf)

## 2. Predicting the sequence specificities of DNA- and RNA-binding proteins

### 2.1 Using published Data to train the model

- 12 TB of sequence data(Protein Binding Micro-array,  RNA-compete,  ChIP-seq and HT-SELEX)
    - PBM(Protein Binding Micro-array), [DREAM5 competition ](http://hugheslab.ccbr.utoronto.ca/supplementary-data/DREAM5/).
    - RBP(RNA binding protein), [RNAcompete](http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/).
    - Encode ChIPSeq [Table S4](http://www.nature.com/nbt/journal/v33/n8/extref/nbt.3300-S6.xlsx) in [http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/](http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/)
- Input: X, DNA Sequence grabed by TF;  
- Label: Y, is there any peak on this region. 



### 2.2. The structure of the neural networks:

![png](/images/2016-08-08-DeepLearning/NBT1.png)

Let's see it in detail:

![png](/images/2016-08-08-DeepLearning/NBT2.png)

#### 2.2.1 conv

e.g. Using motif  with length of 3 to convolve the input sequence: ATGG

![png](/images/2016-08-08-DeepLearning/NBT3.png)

#### 2.2.2 recifity

![png](/images/2016-08-08-DeepLearning/NBT4.png)

Vanessa's blog <br>[https://imiloainf.wordpress.com/2013/11/06/rectifier-nonlinearities/](https://imiloainf.wordpress.com/2013/11/06/rectifier-nonlinearities/)

#### 2.2.3 pooling
![png](/images/2016-08-08-DeepLearning/NBT5.png)

Deep Learning For Java <br>[http://deeplearning4j.org/convolutionalnets.html](http://deeplearning4j.org/convolutionalnets.html)

#### 2.2.4 neural network

- Fully connected Layer
	- Multiplication + Addiction + Transformation. Scale to sum of one at last.

![jpg](/images/2016-08-08-DeepLearning/NBT6.jpg)

Visual Studio Magzine<br>[https://visualstudiomagazine.com/articles/2014/11/01/~/media/ECG/visualstudiomagazine/Images/2014/11/1114vsm_mccaffreyfig2.ashx](https://visualstudiomagazine.com/articles/2014/11/01/~/media/ECG/visualstudiomagazine/Images/2014/11/1114vsm_mccaffreyfig2.ashx)

- The process for calculation:

$$  tanh \begin{pmatrix}\begin{bmatrix} 
1 \\ 5 \\ 9 
\end{bmatrix}^T \times  \begin{bmatrix} 
0.01 & 0.02 & 0.03  & 0.04 \\
0.05 & 0.06 & 0.07  & 0.08 \\
0.09 & 0.10 & 0.11  & 0.12 \\
\end{bmatrix} + \begin{bmatrix}
0.13 \\ 0.14 \\ 0.15 \\ 0.16 \end{bmatrix}^T  \end{pmatrix} = \begin{bmatrix}
0.8337 \\ 0.8764 \\ 0.9087 \\ 0.9329
\end{bmatrix}^T $$

$$\begin{bmatrix}
0.8337 \\ 0.8764  \\ 0.9087  \\ 0.9329 
\end{bmatrix}^T  \times \begin{bmatrix} 
0.17 & 0.18\\
0.19 & 0.20\\
0.21 & 0.22\\
0.23 & 0.24\\
\end{bmatrix} + \begin{bmatrix} 0.25 \\ 0.26 \end{bmatrix}^T = \begin{bmatrix}   0.963 \\ 1.009 \end{bmatrix} ^T $$

$$\begin {bmatrix} 0.963 / (0.963+1.009) \\ 1.009  / (0.963+1.009)
\end{bmatrix}^T = \begin{bmatrix}  0.4886 \\ 0.5114\end{bmatrix}^T $$

### 2.3. Optimizing parameters to get the best performance:

![png](/images/2016-08-08-DeepLearning/NBT7.png)

- Calibrate: Using 3xCV to estimate 30 groups of parameters and select the best one
- Train: Repeat Calibrate process several times.
- Test: Using the best parameters in a non-used data for testing.
- Store these group of parameters for predicting new data without training.

### 2.4. Quantitative performance on various types of held-out experimental test data.

#### 2.4.1 DNA binding

___in vitro___ Micro-array ,  ___in vivo___  ChIP , better performance

![png](/images/2016-08-08-DeepLearning/NBT8.png)

#### 2.4.2 RNAcomplete micro-array

better than formal methods

![png](/images/2016-08-08-DeepLearning/NBT9.png)

Check in TF level

![png](/images/2016-08-08-DeepLearning/NBT10.png)

#### 2.4.3 Using all peaks rather than top500 peaks will get better result

![png](/images/2016-08-08-DeepLearning/NBT11.png)


### 2.5 Potentially disease-causing genomic variants

![png](/images/2016-08-08-DeepLearning/NBT12.png)

- A disrupted SP1 binding site in the LDL-R promoter that leads to familial hypercholesterolemia
- A gained GATA1 binding site that disrupts the original globin cluster promoters.

### 2.6 RNA binding proteins preference for up-stream and down-stream information
- Exons known to be downregulated by Nova had higher Nova scores in their upstream introns, and exons known to be upregulated by Nova had higher Nova scores in their downstream intron.

![png](/images/2016-08-08-DeepLearning/NBT13.png)

- TIA has been shown to upregulate exons when bound to the downstream intron

![png](/images/2016-08-08-DeepLearning/NBT14.png)

### 2.7 What are motifs like in the convolution kernal after training.
- Comparing with known databases(DNA, jaspar. RNA, CISBP-RNA):

  ![png](/images/2016-08-08-DeepLearning/NBT15.png)


## 3. Learning the regulatory code of the accessible genome with Deep CNN.
- A very familiar Deep Learning structure comparing with the NBT 3300 article. 
- Source code available. 

### 3.1 Data source
- Encode Project Consortium + Roadmap Project, 164 samples' BED file for peaks.
- Hg19 genome sequences.

X | Y
---|---
200 million x (600bp*4) | 200 million x 164

e.g, Y is like:

![png](/images/2016-08-08-DeepLearning/GR1.png)


### 3.2 The structure of the neural network

- Familiar structure

![png](/images/2016-08-08-DeepLearning/GR2.png)

- More layers 

![png](/images/2016-08-08-DeepLearning/GR3.png)

This architecture is recommended by [Spearmint](https://github.com/HIPS/Spearmint)

#### 3.2.1: SGD:
divide all training-samples into many subsets. Using one set to update parameters in order to speed up.

![gif](/images/2016-08-08-DeepLearning/SGD.gif)

Sebastian Ruder's blog: <br>[http://sebastianruder.com/optimizing-gradient-descent/](http://sebastianruder.com/optimizing-gradient-descent/)

#### 3.2.2: Batch Normalization(BN):


Definition:
**Input**: Values of x over a mini-batch: 
$$ B = \{x_{1..m}\} $$

Parameters to be learned: 
$$\beta, \gamma$$ 
**Output**: 

$${y_i = BN_{\gamma, \beta}(x_i)}$$

$$ \mu_B = \frac{1}{m}\sum^m_{i=1} x_i $$   

$$ \sigma_B^2 = \frac{1}{m}\sum^m_{i=1} (x_i-\mu_B)^2 $$

$$ \hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\varepsilon}} $$

$$  y  = \gamma \hat{x_i} + \beta $$ 

Yes, BN is just like z-score, which can scale values for training to the center of the optimizer, which can help speed up the optimization as well as get a higher accuracy.

#### 3.2.3: Drop-out

Randomly choosing a subset of nodes to train in order to guarantee the robustness of the network.

![png](/images/2016-08-08-DeepLearning/GR4.png)

Journal of Machine Learning Research 15(2014) 1929-1958: <br>[http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

### 3.3  Basset accurately predicts cell-specific DNA accessibility

![png](/images/2016-08-08-DeepLearning/GR5.png)

- Better than formal method.
- Differences for AUC between cell types.

### 3.4 The convolution kernal

![png](/images/2016-08-08-DeepLearning/GR6.png)

For A:

- x axis Information content is:

$$ -\sum_{i, j} p_{i,j} log(p_{i, j}/ p_{background})$$

- y axis Influences reflects the  accessibility prediction changes over all cells.

- high influeces but unannotated includes CpG and ATAT boxes.
- 45% kernals could be annotated.

For  C:

- cell-specific patterns.

### 3.5 Accessibility and Binding-Sites
![png](/images/2016-08-08-DeepLearning/GR7.png)

- AP-1 complex members includes JUN and JUND
- The open region inclueds JUN/JUND peaks.
- Basset result showed a mutation in FOS motif will induce to the loss of the accessibility.

![png](/images/2016-08-08-DeepLearning/GR8.png)

- Conservation also showed a correlation with signal.


### 3.6 Using GWAS data to validate.
![png](/images/2016-08-08-DeepLearning/GR9.png)

- Basset score for general GWAS SNP vs causal GWAS SNP.

![png](/images/2016-08-08-DeepLearning/GR10.png)

- Basset report T>C a 85% for causality for vitiligo(白癜风) for rs4409785.
- DNA were opened and CTCF could bind.

![png](/images/2016-08-08-DeepLearning/GR11.png)

- Encode CTCF data for raw reads in rs4409785.
- 21 / 88 Samples were sequences here with 11 have peaks, and almost all sequenced samples have T>C mutation.

## 4. Deep Learning needs GPU

type | Tesla K20m GPU | Mac Intel i7-CPU 
---|---|---
Seeded single-task | 18m | 6h37m
Full multi-task | 85h | -