---
layout: post
title: K-Nearest Neighbors Algorithm
category: ML/DL
use_math: True
---
Reference by 

https://medium.com/analytics-vidhya/k-nearest-neighbors-algorithm-7952234c69a4

> KNN is a **non-parametric** and lazy learning algorithm. Non-parametric means there is no assumption for underlying data distribution.
> In other words, the model structure determined from the dataset.
> This will be very helpful in practice where most of the real-world datasets do not follow mathematical theoretical assumptions.


> KNN is one of the most simple and traditional non-parametric techniques to classify samples.
> Given an input vector, KNN calculates the approximate distances between the vectors and then assign the points which are not yet labeled to the class of its K-nearest neighbors.


> The lazy algorithm means it does not need any training data points from model generation.
> All training data used in the testing phase.
> This makes training faster and the testing phase slower and costlier.
> The costly testing phase means time and memory.
> In the worst case, KNN needs more time to scan all data points, and scanning all data points will require more memory for stroing training data.

In summary, KNN is a non-parametric and lazy learning algorithm.

## K-NN for classification

In K-NN, K is the number of nearest neighbors. This is the hyperparameter to deciding the number of neighbors.
K is typically an odd number if the number of classes is 2. When K=1, then the algorithm is known as the nearest neighbor algorithm.
K=1 case is the simplest case. 

### K-NN has the following basic steps:

1. Calcuate distance
2. Find closest neighbors
3. Vote for labels
4. Take the majority Vote

### Distance Measures in K-NN

There are maninly four distance measures in Machine Learning Listed below.

1. Euclidean Distance
2. Manhattan Distance
3. Minkowski Distance
4. Hamming Distance


#### Euclidean Distance
The most commonly used Euclidean distances are:

$d = \sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$

#### Manhattan Distance

$\displaystyle{d(x,y)=\sum^m_{i=1}\|x_i-y_i\|}$

#### Minkowski Distance

$\displaystyle{d(x,y)=(\sum^n_{i=1}\|x_i-y_i\|^p)^{1/p}}$

#### Euclidean distance from Minkowski distance when p = 2.
#### Manhattan distance from Minkowski distance when p = 1.
