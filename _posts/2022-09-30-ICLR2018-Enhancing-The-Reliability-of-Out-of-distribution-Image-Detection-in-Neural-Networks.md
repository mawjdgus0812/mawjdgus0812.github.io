---
layout: post
title: ICLR2018-Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks
use_math: True
---


## Abstract

본 논문에서는 ODIN이라는 쉽고 효율적인 방법을 제안합니다. 이 방법은 pre-trained된 neural network의 어떤 변화도 필요로 하지 않습니다.(perturbation에 할텐데 ?) 이 방법은 temperature scaling과 입력 이미지에 small perturbations을 더해줌으로써 in- and out-of-distribution사이의 softmax score를 더 효과적으로 분리할 수 있습니다

## Introduction

딥러닝이 발전함에 있어서 여러 문제들이 있지만, 이 논문에서 다루고자 하는 문제는 처음보는 입력을 어떻게 처리할 것인지에 대한 것입니다. 딥러닝이 신뢰할 수 있을만한 수준이 되기 위해서는 처음보는 입력(Out-of-distribution)에 대해서 불확실성을 가질 수 있는 것이 중요합니다.

Hendrycks & Gimpel은 이러한 문제(Detecting out-of-distribution)를 해결하기 위한 baseline을 제시했는데, 이 때, 모델의 재훈련 없이 이를 가능케 했습니다. 바로, 잘 훈련된 neural-networks가 in-distribution 입력들을 out-of-distribution보다 더 높은 softmax scores를 갖도록 만들어 분리하는 것입니다.

그리고 이 논문(ODIN)에서는 여기에 덧붙여, softmax function에서 temperature scaling을 적용하고 조작가능한 작은 perturbations를 입력에 더해주는 것으로 in- and out-of-distribution softmax score의 차이를 더욱 크게 만들었습니다.

## Problem Statement

$P_x$ : in-distribution 

$Q_x$ : out-of-distribution

$\mathbb{P}_{x\times{z}}$

$\mathbb{P}_{x\|z=0}=P_x$

$\mathbb{P}_{x\|z=1}=Q_x$

>Given an image $X$ drawn from the mixture distribution $\mathbb{P}_{x\times{z}}$, can we distinguish whether the image is from in-distribution $P_x$ or not?

## Temperature scaling

> In this section, we present our method, ODIN, for detecting out-of-distribution samples. The detector is built on two components: temperature scaling and input preprocessing. We describe the details of both components below.

> Temperatur Scaling. Assume that the neural network $\mathbf{f}=(f_1,...,f_N)$ is trained to classify $N$ classes. For each input $x$, the neural network assigns a label $\hat{y}(x)=argmax_iS_i(x;T)$ by computing the softmax output for each class. Specifically,

$\displaystyle{S_i(x;T)={\exp{(f_i(x)/T)}\over{\Sigma^N_{j=1}\exp{(f_j(x)/T)}}}}$
$ Px

이러한 temperature scaling을 통해 저자들은 in- and out-of-distribution 이미지들 사이의 softmax score를 잘 분리함으로써 OoD detection을 진행합니다.

```python
images, _ = data
        
inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
outputs = net1(inputs)
# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
nnOutputs = outputs.data.cpu()
nnOutputs = nnOutputs.numpy()
nnOutputs = nnOutputs[0]
nnOutputs = nnOutputs - np.max(nnOutputs)
nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
```

여기서 중간 부분에 정규화가 들어가는데, `nnOutputs - np.max(nnOutputs)` softmax를 구하는 과정에서 값이 무한대가 될 수 있는 걸 방지해 주기 위한 테크닉이다.