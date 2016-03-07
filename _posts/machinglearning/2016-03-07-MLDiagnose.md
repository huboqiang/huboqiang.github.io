# Machine Learning Diagnose：

[http://www.holehouse.org/mlclass/10_Advice_for_applying_machine_learning.html](http://www.holehouse.org/mlclass/10_Advice_for_applying_machine_learning.html)

## Diagnosing bias vs Variance
_high bias_ means __UNDER FIT__
_high variance_ means __OVER FIT__

B => U 
V => O

![png](/images/2016-03-07-MLDiagnose/1457341545509.png)

![png](/images/2016-03-07-MLDiagnose/1457341777493.png)

##Regularization and Bias/Variance
Low lambda => overfit
High lambda => underfit

![png](/images/2016-03-07-MLDiagnose/1457341842806.png)

![png](/images/2016-03-07-MLDiagnose/1457341978208.png)


## Learning curves
Condition 1： High  bias
![png](/images/2016-03-07-MLDiagnose/1457341307212.png)
more training data is __not__ likely to help.

Condition 2：High variance
![png](/images/2016-03-07-MLDiagnose/1457341349249.png)
more training data is likely to help.

## Deciding what to do next revisited
Solution for bias/variance.

![png](/images/2016-03-07-MLDiagnose/1457342126653.png)
Action | Result  | Reason
-------|--------|-------
more training sets | fix high variance | More training sets so no overfit
Less features | fix high variance | Less parameters so no overfit
More features | fix high bias | More parameters so no underfit
More polynomial features | fix high bias | More parameters so no under-fit
Decreasing $\lambda$ | fix high bias | low $\lambda => $ more parameters
Increasing $\lambda$ | fix high variance | high $\lambda => $ less parameters


# Improved model selection
Given a training set instead split into three pieces
1. Training set (60%) - m values
2. Cross validation (CV) set (20%)mcv
3. Test set (20%) mtest 

As before, we can calculate:
- Training error
- Cross validation error
- Test error

Using CV to train the degree of polynomial d or lambda:
- The degree of a model will increase as you move towards overfitting
- Lets define training and cross validation error as before
- Now plot 
	- x = degree of polynomial d
	- y = error for both training and cross validation (two lines)
		- CV error and test set error will be very similar 

![png](/images/2016-03-07-MLDiagnose/1457343469186.png)
