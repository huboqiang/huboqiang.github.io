---
title: "[BigData-Spark]Classification using Spark."
tagline: ""
last_updated: 2016-03-03
category: big-data related
layout: post
tags : [Spark, Zeppelin]
---

# Classification using Spark

Learning note for [Machine learning with spark](http://www.amazon.com/Machine-Learning-Spark-Nick-Pentreath/dp/1783288515).

Besides, thanks to [Zeppelin](https://github.com/apache/incubator-zeppelin). Although it is not so user-friendly like RStudio or Jupyter, it __really__ makes the learning of Spark much easier.

# 1. Data Loading from HDFS

First, download the data from [https://www.kaggle.com/c/stumbleupon](https://www.kaggle.com/c/stumbleupon).

Then upload data to HDFS:

```
tail -n +2 train.tsv >train_noheader.tsv
hdfs dfs -mkdir hdfs://tanglab1:9000/user/hadoop/stumbleupon
hdfs dfs -put train_noheader.tsv hdfs://tanglab1:9000/user/hadoop/stumbleupon
```

```scala
val rawData = sc.textFile("/user/hadoop/stumbleupon/train_noheader.tsv")
val records = rawData.map(line => line.split("\t"))
records.first()
```

# 2. Data Process
Select the column for label(last column) and Feature(5 ~ last but one column)
Data cleanning and convert NA to 0.0
Save the label and feature in vector into MLlib.

__As naive bayesian model do not accept negative input value, convert negtive input into 0__

```scala
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

val data = records.map{ r => 
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val features = trimmed.slice(4, r.size - 1).map(d => 
    	if (d=="?") 0.0 else d.toDouble)
    LabeledPoint(label, Vectors.dense(features))
}

val nbData = records.map{ r => 
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val features = trimmed.slice(4, r.size - 1).map(d => 
	    if(d=="?") 0.0 else d.toDouble).map( d=> if(d<0.0) 0.0 else d)
    LabeledPoint(label, Vectors.dense(features))
}

data.cache
data.count
```

# 3. Model training
Import modules required. 
Then define the parameters required by the models.

```scala
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy

val numIterations = 10
val maxTreeDepth = 5
```


## 3.1 Training logistic regression

```scala
val lrModel = LogisticRegressionWithSGD.train(data, numIterations)

val dataPoint = data.first
val prediction = lrModel.predict(dataPoint.features)
val trueLabel = dataPoint.label
```


## 3.2 Training SVM

```scala
val svmModel = SVMWithSGD.train(data, numIterations)
```


## 3.3 Training the naive bayesian model
```scala
val nbModel = NaiveBayes.train(nbData)
```

## 3.4 Training the decision model
```scala
val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)
```

# 4. Evaluating the preformance of the classification models

## 4.1 Accuracy

```scala
val lrTotalCorrect = data.map{ point =>
    if( lrModel.predict(point.features) == point.label) 1 else 0
}.sum

val svmTotalCorrect = data.map{ point =>
    if( svmModel.predict(point.features) == point.label ) 1 else 0
}.sum

val nbTotalCorrect = nbData.map{ point =>
    if( nbModel.predict(point.features)  == point.label ) 1 else 0
}.sum

val dtTotalCorrect = data.map{ point =>
    val score = dtModel.predict(point.features)
    val predicted = if(score > 0.5) 1 else  0
    if (predicted == point.label) 1 else 0
}.sum

val lrAccuracy = lrTotalCorrect / data.count
val svmAccuracy    = svmTotalCorrect / data.count
val nbTotalAccuracy= nbTotalCorrect  / data.count
val dtTotalAccuracy= dtTotalCorrect  / data.count
```

```
lrAccuracy: Double = 0.5146720757268425
svmAccuracy: Double = 0.5146720757268425
nbTotalAccuracy: Double = 0.5803921568627451
dtTotalAccuracy: Double = 0.6482758620689655
```

## 4.2 Calculating the region under the Precision and recall(PR) and FP-TP(ROC) curve

```scala
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val metrics = Seq(lrModel, svmModel).map{ model =>
    val scoreAndLabels = data.map{ point =>
        (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
```


Naive bayesian need another dataset which have no negative feature. 
And the prediction of naive bayesian is a ratio range from 0 to 1, which needs to be cut to 0 or 1.

```scala
val nbmetrics = Seq(nbModel).map{ model =>
    val scoreAndLabels = nbData.map{ point =>
        val score = model.predict(point.features)
        (if(score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
```

The prediction of decision tree also have a cutoff.

```scala
val dtmetrics = Seq(dtModel).map{ model =>
    val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (if(score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
```

For all model, the Precision/Recall and FP-TP ROC were summarized as below:

```scala
val allMetrics = metrics ++ nbmetrics ++ dtmetrics
allMetrics.foreach{ case (m, pr, roc) =>
    println(f"$m, Area under PR: $pr, Area under ROC: $roc")
}
```

which gived:

```
LogisticRegressionModel, Area under PR: 0.7567586293858841, Area under ROC: 0.5014181143280931
SVMModel, Area under PR: 0.7567586293858841, Area under ROC: 0.5014181143280931
NaiveBayesModel, Area under PR: 0.6808510815151734, Area under ROC: 0.5835585110136261
DecisionTreeModel, Area under PR: 0.7430805993331199, Area under ROC: 0.6488371887050935
```

__As the preformance is not well enough, some adjustment were required to promote the performance.__

# 5. The improvement the performance of model and the optimization of the parameters.

Drawbacks for current model:

- Only the values were included, not all features.
- No analysis for the features of the data.
- Non-optimized parameter.

## 5.1 The standardization for features

Try to calculate the mean value and variation for each column of the data

```scala
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val vectors = data.map(lp => lp.features)
val matrix = new RowMatrix(vectors)
val matrixSummary = matrix.computeColumnSummaryStatistics()

case class MatrixInfo(index:Int, mean: Double, variation: Double)
val value_RowMean = matrixSummary.mean.toArray
val value_RowVar  = matrixSummary.variance.toArray

val Info = (0 to value_RowMean.length-1 toList).map{i =>
    MatrixInfo(i, value_RowMean(i), value_RowVar(i))
}.toDF()

Info.registerTempTable("Info")
```

These results can be shown directly with Zeppelin:

```sql
SELECT index, mean FROM Info
ORDER BY index
```
![png](/images/2016-03-03-SparkMLlibClassification/FigVar.png)

```sql
SELECT index, variation FROM Info 
ORDER BY index
```
![png](/images/2016-03-03-SparkMLlibClassification/FigVar.png)

Let's see the mean and variation. In the raw format, the distribution of data did not follow the Gaussian distribution. So let's make a z-score normalization:

```scala
import org.apache.spark.mllib.feature.StandardScaler

val scaler = new StandardScaler(
	withMean = true, 
	withStd = true
).fit(vectors)

val scaledData = data.map(lp => 
	LabeledPoint(lp.label, scaler.transform(lp.features))
)

```

As only logistic regression would be influenced by normalization, here logistic regression will be re-preformed to see the influence of normalization to the result:

```scala
val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
val lrTotalCorrectScaled = scaledData.map{ point => 
    if(lrModelScaled.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaled = lrTotalCorrectScaled / data.count
val lrPredictionsVsTrue = scaledData.map{ point => 
    (lrModelScaled.predict(point.features), point.label)
}
val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
val lrPr = lrMetricsScaled.areaUnderPR
val lrROC = lrMetricsScaled.areaUnderROC
println(f"${lrModelScaled.getClass.getSimpleName}, Area under PR: $lrPr, Area under ROC: $lrROC")
```

```
LogisticRegressionModel, Area under PR: 0.7272540762713375, Area under ROC: 0.6196629669112512
```

## 5.2 Using other features.
The text in the field of __category__ and __boilerplate__ were ignored.
So it is necessary to convert those texts into numbers.


Let's highlights 

```scala 
val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
```

which can automatically convert k-item text into numbers.

All code:

```scala
val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
val numCategories = categories.size

val dataCategories = records.map{ r =>
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val categoryIdx = categories(r(3))
    val categoryFeatures = Array.ofDim[Double](numCategories)
    categoryFeatures(categoryIdx) = 1.0
    val otherFeatures = trimmed.slice(4, r.size-1).map( d => if(d=="?") 0 else d.toDouble)
    val features = categoryFeatures ++ otherFeatures
    LabeledPoint(label, Vectors.dense(features))
}
println(dataCategories.first)
```

Then normalize all features.

```scala
val scalerCats = new StandardScaler(withMean = true, withStd = true).
    fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp => 
    LabeledPoint(lp.label, scalerCats.transform(lp.features)))

println(scaledDataCats.first.features)
```

```scala
val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
val lrTotalCorrectScaledCats = scaledDataCats.map{ point => 
    if(lrModelScaledCats.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaledCats = lrTotalCorrectScaledCats / data.count
val lrPredictionsVsTrueCats = scaledDataCats.map{ point => 
    (lrModelScaledCats.predict(point.features), point.label)
}
val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
val lrPrCats = lrMetricsScaledCats.areaUnderPR
val lrROCCats = lrMetricsScaledCats.areaUnderROC
println(f"${lrModelScaledCats.getClass.getSimpleName}, Area under PR: $lrPrCats, Area under ROC: $lrROCCats")
```

## 5.3 Using the correct format for data

```scala
val dataNB = records.map{ r => 
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val categoryIdx = categories(r(3))
    val categoryFeatures = Array.ofDim[Double](numCategories)
    categoryFeatures(categoryIdx) = 1.0
    LabeledPoint(label, Vectors.dense(categoryFeatures))
}
```

Then trainning the naive bayesian model:

```scala
val nbModelCats = NaiveBayes.train(dataNB)
val nbTotalCorrectCats = dataNB.map{ point =>
    if(nbModelCats.predict(point.features) == point.label) 1 else 0
}.sum
val nbAccuracyCats = nbTotalCorrectCats / data.count
val nbPredictionsVsTrueCats = dataNB.map{ point => 
    (nbModelCats.predict(point.features), point.label)
}
val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
val nbPrCats = nbMetricsCats.areaUnderPR
val nbROCCats = nbMetricsCats.areaUnderROC

println(f"${nbMetricsCats.getClass.getSimpleName}, Area under PR: $nbPrCats, Area under ROC: $nbROCCats")
```


## 5.4.Optimize the parameter

### 5.4.1 The linear model
Linear model includes logistic regression and SVM. Both logrestic regression and SVM shared the same parameters because they all used __SGD__ as the basic optimization method. The different between them is that they used the different Loss-function.

In LogisticRegressionWithSGD:

```scala
class LogisticRegressionWithSGD private(
    private var stepSize : Double,
    private var numIterations : Int,
    private var regParam : Double,
    private var miniBatchFraction : Double)
    extends GeneralizedLinearAlgorithm[LogisticRegressionModel] ...
)
```

stepSize, numIteration regParam and miniBatchFraction can be passed inth the init function. 

To investigate the influence of these parameters, some function should be build to help the training process.

```scala
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.classification.ClassificationModel

def trainWithParams(input : RDD[LabeledPoint], regParam:Double, numIterations: Int, updater : Updater, stepSize : Double) = {
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).
    setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
}

def createMetrics(label : String, data : RDD[LabeledPoint], model : ClassificationModel) = {
    val scoreAndLabels = data.map{ point => 
        (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
}
```

#### (1) Iteration

```scala
val iterResults = Seq(1, 5, 10, 50).map{ param => 
    val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
    createMetrics(s"$param iterations", scaledDataCats, model)
}
iterResults.foreach{ case(param, auc) => println(f"$param, AUC = $auc") }
```

```
1 iterations, AUC = 0.6495198950299683
5 iterations, AUC = 0.6661609623443581
10 iterations, AUC = 0.6654826844243996
50 iterations, AUC = 0.6681425454500738
```

#### (2) StepSize

```scala
val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
    val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
    createMetrics(s"$param step size", scaledDataCats, model)
}
stepResults.foreach{ case(param, auc) => println(f"$param, AUC = $auc") }
```

```
0.001 step size, AUC = 0.6496588225098238
0.01 step size, AUC = 0.6496444027450547
0.1 step size, AUC = 0.6552106515362099
1.0 step size, AUC = 0.6654826844243996
10.0 step size, AUC = 0.6192278852778154
```

#### (3) Regularization

```scala
val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
    val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
    createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
}
regResults.foreach{ case(param, auc) => println(f"$param, AUC = $auc") }
```

```
0.001 L2 regularization parameter, AUC = 0.6654826844243996
0.01 L2 regularization parameter, AUC = 0.665475474542015
0.1 L2 regularization parameter, AUC = 0.6663378789506862
1.0 L2 regularization parameter, AUC = 0.6603745376525676
10.0 L2 regularization parameter, AUC = 0.3532533843993077
```

### 5.4.2 The decision tree model

For decision tree model, the most important issue is the depth of the tree. The higher depth, the higher AUC.
However, when the depth is __too high__, the model will __overfit the data__.

```scala
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini

def trainDTWithParams(input : RDD[LabeledPoint], maxDepth : Int, impurity : Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth )
}

val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param => 
    val model = trainDTWithParams(data, param, Entropy)
    val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param, tree depth", metrics.areaUnderROC)
}
dtResultsEntropy.foreach{ case(param, auc) => println(f"$param, AUC = $auc") }
```

```
1, tree depth, AUC = 0.5932683560677638
2, tree depth, AUC = 0.6168392183052838
3, tree depth, AUC = 0.6260699538655363
4, tree depth, AUC = 0.6363331299438932
5, tree depth, AUC = 0.6488371887050935
10, tree depth, AUC = 0.7625521856410764
20, tree depth, AUC = 0.9845371811804648

```

### 5.4.3 The naive bayesian model

For naive bayesian model, lambda can control the __additive smoothing__ to solve the problem that the co-deficiency for the combination of one classification and one feature.

```scala
def trainNBWithParams(input : RDD[LabeledPoint], lambda : Double) = {
    val nb = new NaiveBayes
    nb.setLambda(lambda)
    nb.run(input)
}
val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map{ param =>
    val model = trainNBWithParams(dataNB, param)
    val scoreAndLabels = dataNB.map{ point =>
        (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param lambda", metrics.areaUnderROC)
}
nbResults.foreach{ case(param, auc) =>
    println(f"$param, AUC = $auc") 
}
```

```
0.001 lambda, AUC = 0.6051384941549446
0.01 lambda, AUC = 0.6051384941549446
0.1 lambda, AUC = 0.6051384941549446
1.0 lambda, AUC = 0.6051384941549446
10.0 lambda, AUC = 0.6051384941549446
```

### 5.4.4 Cross Validation
Here the data were divided into 60% training set and 40% testing set.

For the training set:

```scala
val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
val train = trainTestSplit(0)
val test  = trainTestSplit(1)

val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map{ param =>
    val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
    createMetrics(s"$param L2 regularization parameter", test, model)
}
regResultsTest.foreach{ case (param, auc) => 
    println(f"$param, AUC = $auc")
}
```

```
0.0 L2 regularization parameter, AUC = 0.6717311017792245
0.001 L2 regularization parameter, AUC = 0.6717311017792245
0.0025 L2 regularization parameter, AUC = 0.6717311017792245
0.005 L2 regularization parameter, AUC = 0.6714060042499658
0.01 L2 regularization parameter, AUC = 0.671759861291721
```