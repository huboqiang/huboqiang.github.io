---
title: "20170220 Journal Club]Using Deep Learning to study classify different types of skin cancer."
tagline: ""
last_updated: 2017-02-20
category: Journal Club
layout: post
tags : [Deep Learning]
---

# [20170220 Journal Club]Using Deep Learning to study classify different types of skin cancer.

![png](/images/2017-02-20-DeepLearningSkinCancer/Nature21056.jpeg)


## 0. Author

### Corresponding Authors:

![png](/images/2017-02-20-DeepLearningSkinCancer/author_ST.jpeg)

### About Sebastian Thrun

- Founder and CEO of **Udacity**

- Founder of **Google X**, where he founded Google Glass and Google's self driving car among many other projects. 

- Professor in Department of Computer Science, **Stanford University**, research on robotics, artificial intelligence, education, human computer interaction, and medical devices.  


## 1. More about Deep Learning

In our previous introduction, [a structure of convolution neural network(CNN)](http://huboqiang.cn/2016/08/08/DeepLearning#the-structure-of-the-neural-network) has already been introduced. 

However, this structure used a 1D-structure for converlutionary layers for DNA sequence, which could not be generalized for an 2D inputs, like photographs. 

So at first, the general structure and related theories for image-classification will be introduced. 

### 1.1 ImageNet competition

Since 2010, the annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is a competition where research teams submit programs that classify and detect objects and scenes. 

- millions of photographs with 1000 labels for its description.

- Around 2011, a good ILSVRC classification error rate was 25%. 

- In 2012, a deep convolutional neural net (AlexNet) achieved 16%; in the next couple of years, error rates fell to a few percent.

![png](/images/2017-02-20-DeepLearningSkinCancer/ImageNet2012.jpeg)

After 2012, all winners for ImageNet used the [basic elements described in AlexNet](http://huboqiang.cn/2016/08/08/DeepLearning#the-structure-of-the-neural-networks) to build up their new networks with different structures.

![png](/images/2017-02-20-DeepLearningSkinCancer/MxnetStructure.svg)


### 1.2 Convolutionary kernel after training using ImageNet data

First, let's see the structure of AlexNet in detail:

![png](/images/2017-02-20-DeepLearningSkinCancer/ConvolutionaryKernels.png)

The structure of AlexNet, as well as other CNN networks, could be roughly divided into two parts. One is convolutionary part and the other is fully connected part. 

- The convolutionary layers contained a group of features(kernal) that were trained from the training set and in other words, **feature engineering were not necessary in deep learning because features will generate during the training process automatically**. These features were then scored step by step along the whole input matrix in each layer. 

- The output of the convolutionary part could be the input features for a smaller neural network or other machine learning models to train for the results.

### 1.3 Transfer learning, from ImageNet to EVERYTHING 

Transfer learning is a machine learning technique, where knowledge gain during training in one type of problem is used to train in other similar type of problem. 

Transfer learning usually performed by:

1. Training convolutionary neural network on a large dataset. It's OK for a non-related database like ImageNet. This step could be skipped if you already have the resulting model's parameters for 
the large dataset.

2. Using your training pictures as input and get the result of the convolutionary part of the model. In other words, we just used the convolutionary part to perform feature engineering to generated features as a complete new input file.

3. Using the results of the convolutionary part's output as the input file and training a new model based on these features.

## 2. Dermatologist-level classification of skin cancer using Deep Neural Network

### 2.1 Collecting images of skin cancer from multiple resources.

Information about input data:

- **127,463 (classified by eye)** and **1,942 biopsy-labelled** images from 18 different clinician-curated, open-access online repositories, as well as from clinical data from Stanford University Medical Center.

- The dataset is composed of dermatologist-labelled images organized in a tree-structured taxonomy of 2,032 diseases, in which the individual diseases form the leaf nodes. 

- Merge the children nodes of 2,032 classes of diseases into **757 classes** recursively with the ```maxClassSize < 1000``` 

- Top layers of tree-structured taxonomy were shown as FigureS1 and Figure2a

**Extended Data Figure1** for a example of calculating the probability for 3 classes from 757 classes by summing the probabilities for all children nodes recursivlys.

![png](/images/2017-02-20-DeepLearningSkinCancer/FigureS1.png)

**Figure2a** for top layers of the taxonomy tree with classifications annotated as benign(green), malignant(red), conditions that can be either(orange) and melanoma(black).
![png](/images/2017-02-20-DeepLearningSkinCancer/Figure2A.png)


### 2.2 Building-up the neural network for training

Steps of transfer learning includes:

- Using Google’s Inception v3 CNN architecture pretrained to 93.33% top-five accuracy on the 1,000 object classes (1.28 million images) of the 2014 ImageNet Challenge
- Removing the final classification layer from the network and retrain it with our dataset, fine-tuning the parameters across all layers. 
- Resize each image to 299 × 299 pixels while training in order to make it compatible with the original dimensions of the Inception v3 network architecture and leverage the natural-image features learned by the ImageNet.

For model training, the important parameters includes:

- Learning rate: Same global learning rate of 0.001 and a decay factor of 16 every 30 epochs. 
- Optimizer: RMSProp with a decay of 0.9, momentum of 0.9 and epsilon of 0.1. 
- Framework: Google’s TensorFlow. 
- Augment: Rotated randomly between 0° and 359° with the largest upright inscribed rectangle cropped from the image; flipped vertically with a probability of 0.5.
 
![png](/images/2017-02-20-DeepLearningSkinCancer/Figure1.png)


### 2.3 Results

#### Task1: classify pictures into 3 major class disease partition
- Labeling 757 classes into 3 major classes: **benigh**, **non-plastic** and **malignant**(First level nodes in the taxonomy tree) 
- 72.1% overall accuracy.
- Dermatologists: 65.56% and 66%.

#### Task2: classify pictures into 9 sub-major class disease partition
- Labeling 757 classes into 3 major classes: **benigh**, **non-plastic** and **malignant**(Second level nodes in the taxonomy tree) 
- 55.4% overall accuracy.
- Dermatologists: 53.3% and 55.0%.

#### Task3: using biopsy-proven images to distinguish maligant from benign lesion
 - The majority training sets (127,463 images) were simply classified by eye without biopsy-proven
 - Only testing the **biopsy-labelled** images, including:
 - Multi-class classification => binary classification
 	
Images Used | Epidermal | Melanocytic| Melanocytic (dermoscopy) 
---|---|---|---
Benign        | 70 benign seborrheic keratoses | 97 benign nevi | 40 benign
Malignant 	  | 65 keratinocyte carcinomas | 33 malignant melanomas    | 71 malignant
Total        | 135 Carcinoma  | 130 melanoma | 111 melanoma (demoscopy) 
   
![png](/images/2017-02-20-DeepLearningSkinCancer/Figure2B.png)

![png](/images/2017-02-20-DeepLearningSkinCancer/Figure3.png)

### 2.4 The evidence used by deep learning network for classification

#### Importance of each pixel for diagnosis
 - Generate saliency maps for example images of the nine classes
 - $$ y_{1..9} $$ as the true values and $$ \hat{y}_{1..9} $$ as the predicted values
 - The loss gradient can also be backpropagated to the input data layer. 
 - By taking the L1 norm of this input layer loss gradient across the RGB channels (sum the absolute values for loss gradient across Red, Green and Blue channels), the resulting heat map intuitively represents the importance of each pixel for diagnosis.

![png](/images/2017-02-20-DeepLearningSkinCancer/FigureS3.png)

#### tSNE visualization of the results of convolutionary layers

 - After processing the pretrained InceptionV3 convolutionary layers, all images results into a vector with length of 2048.
 - tSNE: 2048 dims => 2 dims
 - Plotting
![png](/images/2017-02-20-DeepLearningSkinCancer/Figure4.png)