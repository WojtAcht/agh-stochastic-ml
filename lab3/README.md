# Adversarial Examples

**Source Paper**  
[Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572). _ICLR, 2015_.

## 1. Motivation

Machine learning models, especially deep neural networks, have achieved remarkable accuracy in many domains. However, **adversarial examples** demonstrate a surprising vulnerability of these models: tiny, carefully chosen perturbations to an input can cause confident misclassifications. Importantly, these adversarial perturbations are often *unnoticeable* or *nearly unnoticeable* to the human eye.

This phenomenon is intriguing because it reveals that high-performing models do **not** necessarily learn robust, human-like concepts of their input data.

In this lab, we will focus on a single, seminal paper:

> **Explaining and Harnessing Adversarial Examples**  
> _Ian J. Goodfellow, Jonathon Shlens & Christian Szegedy. ICLR, 2015._

**Why This Paper?**  
1. It introduced a clear explanation of why adversarial examples exist, attributing the phenomenon largely to the near-linear behavior of modern networks in high-dimensional input space.  
2. It proposed an efficient and straightforward method — the **Fast Gradient Sign Method (FGSM)** — to generate adversarial examples. This method leverages the gradient of the loss function with respect to the input to create minimal perturbations that cause misclassification. FGSM demonstrates how optimization techniques can be applied beyond model training, serving as versatile tools in a machine learning researcher's arsenal.

Because we are focusing on a _single paper lab_, our goal is to understand **only** these core ideas and replicate some simplified experiments without delving into other adversarial attack or defense strategies proposed later in the literature.

## 2. Core Concept: Adversarial Examples

### 2.1 Definition

An **adversarial example** is an input that has been **intentionally perturbed** so that a target model misclassifies it, typically with very **high confidence**. The perturbation is usually **small** and is often visually (or otherwise) difficult for humans to detect.

In mathematical terms, let:
- $\mathbf{x}$ be a “clean” input (e.g., an original image).
- $y$ be the correct label of $\mathbf{x}$.
- $\mathbf{\theta}$ be the parameters of the model $f_\theta(\cdot)$.

An adversarial example $\mathbf{x}_{adv}$ is created by applying a small perturbation $\eta$ to $\mathbf{x}$:
```math
\mathbf{x}_{adv} = \mathbf{x} + \eta
```
such that the model predicts the wrong label:
$$\arg\max_i f_\theta(\mathbf{x}_\text{adv})_i \neq y$$
while keeping the perturbation $\boldsymbol{\eta}$ small enough.

### 2.2 Linear Explanation in High-Dimensional Spaces

A key insight of Goodfellow *et al.* is that **nonlinear** neural networks often behave *locally* in ways that are close to **linear**. Even more crucially, high-dimensional spaces allow many small, correlated changes to add up — leading to a significant change in the final classification output. In other words, while each individual pixel may be altered by only a tiny amount (small enough to be imperceptible), those changes, when combined in a specific direction aligned with the gradients of the model, effectively push the decision score toward a wrong class.

## 3. Fast Gradient Sign Method (FGSM)

One of the paper’s main practical contributions is a fast, simple method to generate adversarial examples. The method is based on the gradient of the loss function with respect to the input:

```math
\mathbf{x}_\text{adv} = \mathbf{x} + \epsilon \cdot \text{sign}\bigl(\nabla_{\mathbf{x}} J(\mathbf{\theta}, \mathbf{x}, y)\bigr)
```

where:
- $J(\mathbf{\theta}, \mathbf{x}, y)$ is the training loss (e.g., cross-entropy),
- $\nabla_{\mathbf{x}} J(\mathbf{\theta}, \mathbf{x}, y)$ is the gradient of that loss with respect to the input,
- $\text{sign}(\cdot)$ is the element-wise sign function,
- $\epsilon$ is a small scalar controlling the amount of perturbation.

This simple, single-step approach often suffices to produce high-confidence misclassifications on neural networks that have been trained in a “standard” (non-adversarial) manner.

### Recommended Reading: Neural Network Fundamentals

If you need to refresh your understanding of neural networks before diving into adversarial examples, here are some excellent resources:

1. **[PyTorch Quickstart Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)** - A practical introduction to implementing neural networks with PyTorch
   
2. **[Micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy - A minimal neural network library built from scratch. Accompanied by an [explanatory video](https://www.youtube.com/watch?v=VMj-3S1tku0) that walks through the implementation

3. **3Blue1Brown's Neural Network Series** - An intuitive visual explanation of neural networks -[But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (Part 1 of the series)