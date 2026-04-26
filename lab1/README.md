## Introduction to Optimization in Machine Learning

In machine learning, many problems boil down to **optimization**. This means finding the best settings (or parameters) for a model to make its predictions as accurate as possible. We measure accuracy using a **loss function**, which tells us how far off our predictions are from the actual data. The process of "training" or "fitting" a model is really about tweaking its parameters to make this loss as small as possible.

For this lab, we’ll use **linear regression**—the simplest model for predicting continuous values (like temperatures or prices). Even though it’s basic, it’s a great way to see how optimization works in action.

## The Linear Regression Model

Imagine you have some data: each sample has a set of **features** (like house size or number of bedrooms) represented as a vector $\mathbf{x}^{(i)}$ with $d$ values, and a **target** value $y^{(i)}$ (like the house price). Linear regression predicts the target with a straight-line equation:

$$
\hat{y}^{(i)} = \sum_{j=1}^{d} w_j x_j^{(i)} + b = \mathbf{w}^\top \mathbf{x}^{(i)} + b
$$

- $\mathbf{w}$ is a vector of **weights**, one for each feature, controlling how much each feature affects the prediction.
- $b$ is the **bias**, a constant that shifts the line up or down.
- $\hat{y}^{(i)}$ is the predicted value for the $i$-th sample.

### Simplifying with Bias Included

To make coding easier, we can roll the bias $b$ into the weight vector. We do this by adding a "1" to every feature vector:

$$
\widetilde{\mathbf{x}}^{(i)} = \begin{pmatrix} 1, & x_1^{(i)}, & x_2^{(i)}, & \ldots, & x_d^{(i)} \end{pmatrix}
$$

$$
\widetilde{\mathbf{w}} = \begin{pmatrix} b, & w_1, & w_2, & \ldots, & w_d \end{pmatrix}
$$

Now, the prediction is just:

$$
\hat{y}^{(i)} = b \cdot 1 + w_1 \cdot x_1^{(i)} + w_2 \cdot x_2^{(i)} + \ldots + w_d \cdot x_d^{(i)} = \widetilde{\mathbf{w}}^\top \widetilde{\mathbf{x}}^{(i)}
$$

This trick lets us handle all parameters (including bias) in one vector, which simplifies our math and code.

## Loss Function: Mean Squared Error (MSE)

To figure out how good our predictions are, we need a loss function. For linear regression, we use the **Mean Squared Error (MSE)**, which measures the average squared difference between actual targets $y^{(i)}$ and predictions $\hat{y}^{(i)}$:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2
$$

- $N$ is the number of samples.
- The smaller the MSE, the better our model fits the data.

Our goal is to adjust $\widetilde{\mathbf{w}}$ to make this loss as small as possible.

## Gradient Descent: The Optimization Tool

The MSE is a function of our parameters, and we want to find the lowest point of this function — think of it as finding the bottom of a bowl. For simple problems like this, we could solve it directly with math, but in machine learning, the functions are often too complex for that. Instead, we use **gradient descent**, an iterative method that "walks" toward the minimum.

### The fundamental idea of gradient descent

Gradient descent is a simple but powerful method for finding the minimum of a function. Here's how it works:

1. Start at some point on the function.
2. Find which direction leads downhill most steeply (the negative gradient).
3. Take a small step in that direction.
4. Repeat until you reach the bottom.

Mathematically, we update our position using:

$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

Where:
- $w_t$ is our current position
- $\nabla L(w_t)$ is the gradient (slope) at that position
- $\alpha$ is the learning rate (how big a step to take)
- $w_{t+1}$ is our new position

The learning rate $\alpha$ is important - too large and we might overshoot, too small and progress will be slow.

Finally:

1. **Calculate the Gradient**: The gradient tells us the direction of the steepest increase in the loss. We want to go the opposite way to decrease it. For MSE with the bias included, the gradient is:

Let's derive this gradient formula step by step:

First, recall that our loss function is:
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2$$

And our predictions are:
$$\hat{y}^{(i)} = \widetilde{\mathbf{w}}^\top \widetilde{\mathbf{x}}^{(i)}$$

Substituting the prediction into the loss:
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\widetilde{\mathbf{w}}^\top \widetilde{\mathbf{x}}^{(i)} - y^{(i)})^2$$

In matrix form, this is:
$$\mathcal{L} = \frac{1}{N} (\widetilde{\mathbf{X}}\widetilde{\mathbf{w}} - \mathbf{y})^\top(\widetilde{\mathbf{X}}\widetilde{\mathbf{w}} - \mathbf{y})$$

To find the gradient, we expand this:
$$\mathcal{L} = \frac{1}{N} (\widetilde{\mathbf{w}}^\top\widetilde{\mathbf{X}}^\top\widetilde{\mathbf{X}}\widetilde{\mathbf{w}} - 2\mathbf{y}^\top\widetilde{\mathbf{X}}\widetilde{\mathbf{w}} + \mathbf{y}^\top\mathbf{y})$$

Taking the gradient with respect to $\widetilde{\mathbf{w}}$:
$$\nabla_{\widetilde{\mathbf{w}}} \mathcal{L} = \frac{1}{N} (2\widetilde{\mathbf{X}}^\top\widetilde{\mathbf{X}}\widetilde{\mathbf{w}} - 2\widetilde{\mathbf{X}}^\top\mathbf{y})$$

Since $\widetilde{\mathbf{X}}\widetilde{\mathbf{w}} = \hat{\mathbf{y}}$, we get:
$$\nabla_{\widetilde{\mathbf{w}}} \mathcal{L} = \frac{2}{N} \widetilde{\mathbf{X}}^\top (\mathbf{\hat{y}} - \mathbf{y})$$

- $\widetilde{\mathbf{X}}$ is the matrix of all $\widetilde{\mathbf{x}}^{(i)}$ samples stacked together.
- $\mathbf{y}$ is the vector of all true targets.
- $\hat{\mathbf{y}}$ is the vector of all predictions.

2. **Update the Parameters**: Move the weights a small step in the opposite direction of the gradient:

$$
\widetilde{\mathbf{w}} \leftarrow \widetilde{\mathbf{w}} - \alpha \nabla_{\widetilde{\mathbf{w}}} \mathcal{L}
$$

3. **Repeat**: Keep updating until the loss stops getting smaller (or we've done enough steps).

Gradient descent is like hiking down a hill by always taking a step downhill — eventually, you reach the bottom.

## Recommended Reading

For a comprehensive overview of gradient descent and its variants, we recommend the following resources:

- DeepLearning.AI [Parameter optimization in neural networks](https://www.deeplearning.ai/ai-notes/optimization/index.html)
- Ng, Andrew. [CS229 Lecture 2: Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8) + [Notes](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf) - Stanford's machine learning course with clear explanations of the fundamental concepts.
- Ruder, Sebastian. [An overview of gradient descent optimization algorithms.](https://arxiv.org/pdf/1609.04747)
- Doshi, Ketan. [Neural Network Optimizers Made Simple - Core algorithms and why they are needed](https://ketanhdoshi.github.io/Optimizer-Techniques/)
- Smith, Samuel L., et al. [Don't decay the learning rate, increase the batch size.](https://arxiv.org/pdf/1711.00489)
- Farina, Gabriele [Lecture 7 - Gradient descent](https://www.mit.edu/~gfarina/2024/67220s24_L07_gradient_descent/L07.pdf)


