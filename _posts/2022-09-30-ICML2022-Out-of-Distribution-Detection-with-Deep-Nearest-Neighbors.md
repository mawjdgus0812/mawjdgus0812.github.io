---
layout: post
title: (ICLR2022) Out-of-Distribution Detection with Deep Nearest Neighbors
category: paper
use_math: True
---

[pdf](https://arxiv.org/pdf/2204.06507.pdf)

This paper describes **the effectiveness of non-parametric nearest-neighbor distance for OoD detection**.
The reason why *this method can be effective for generality and flexibility* is when, unlike other methods, **it does not impose
any distributional assumption**. The authors applied nearest-neighbor OoD detection to several benchmarks and establish superior performance.


## Abstract

> Distance-based methods have demonstared promise, where testing samples are detected as OoD if they are relatively far from in-distribution data. However prior methods impose a strong distributional assumption of the underlying feature space, which may not always hold.

So this paper describes the effectiveness of non-parametric nearest-neighbor distance for OoD detection.

Then, what is the non-parametric nearest-neighbor distance ?

See [K-NN](https://mawjdgus0812.github.io/2022/09/30/K-Nearest-Neighbors-Algorithm/)

## Introduction

> Modern machine learning models deployed in the open world often struggle with Out-of-distribution(OoD) inputs - samples from a different distribution that the network has not been exposed to during training, and therefore should not be predicted at test time.

> A reliable classifier should not only accurately classify known in-distribution samples, but also identify as "unknown" any OoD input.

That is why we need the methods for OoD detection.

> Distance-based methods leverage feature embeddings extracted from a model, and operate under the assumption that the test OoD samples are relatively far away from the ID data.

> However, all these approaches make a strong distributional assumption of the underlying feature space being class conditional Gaussian.

What is the class-conditional Gaussian ??

> In this paper, we challenge the status quo by presenting the first study exploring and demonstrating the efficacy of the non-parametric nearest-neighbor distance for OOD detection. To detect OOD samples, we compute the k-th nearest neighbor (KNN) distance between the embedding of test input and the embeddings of the training set and use a threshold-based criterion to determine if the input is OOD or not.

### Contributions

1. We present the **first study** exploring and demonstrating the efficacy of non-parametric density estimation with nearest neighbors for OOD detection - a simple, flexible yet overlooked approach in literature. We hope our work draws attention to the strong promise of the non-parametric approach, which obviates data assumption on the feature space.

2. We demonstrate the superior performance of the KNN-based method on several OOD detection benchmarks, different model architectures (including CNNs and ViTs), and different training losses. Under the same model trained on ImageNet-1k, our method substantially reduces the false positive rate(FPR@TPR95) by 24.77% compared to a strong baseline SSD+, which uses a parametric approach (i.e., Mahalanobis distance) for detection.

3. We offer new insights on the key components to make KNN effective in practice, including feature normalization and a compact representation space. Our findings are supproted by extensive ablations and experiments. We believe these insights are valuable to the community in carrying out feature research.

4. We provide theoretical analysis, showing that KNN-based OOD detection ca reject inputs equivalent to the Bayes optimal estimator. By modeling the nearest neighbor distance in the feature space, out theory (1) directly connects to our method which also operates in the feature space, and (2) complements our experiments by considering the universality of OOD data.

## Preliminaries







