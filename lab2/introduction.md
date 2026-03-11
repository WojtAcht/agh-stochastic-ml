# Gradient Descent and Its Extensions

Below is an introductory overview of Gradient Descent and three key extensions: Momentum, Adagrad, and Adam. The goal is to minimize a scalar function $f(w)$ with respect to the parameter vector $w$.


## 1. Gradient Descent

**Goal:** Find $w$ that minimizes $f(w)$.

**Update Rule:**
$$w_{t+1} = w_t - \alpha \nabla f(w_t)$$
where
- $\nabla f(w_t)$ is the gradient of $f$ evaluated at $w_t$.
- $\alpha > 0$ is the learning rate.

**Issue:** Vanilla Gradient Descent can oscillate along steep directions or converge slowly when different parameters have different scales.


## 2. Momentum

**Motivation:** Reduce oscillations and speed up convergence by "remembering" previous gradients.  
We introduce a velocity term $v_t$ that accumulates gradients via an exponential decay.

**Update Equations:**  
1. **Velocity Update:**
   $$v_{t} = \beta v_{t-1} + \alpha \nabla f(w_t)$$
2. **Parameter Update:**
   $$w_{t+1} = w_t - v_{t}$$
where $\beta \in [0,1)$ is the momentum coefficient (sometimes denoted $\gamma$ in some references). A larger $\beta$ applies more smoothing from previous steps.


## 3. Adagrad

**Motivation:** Adapt the step size for each parameter dimension. Parameters that have received large gradients in the past get smaller future updates, while parameters with smaller historical gradients get comparatively larger updates.

**Key Idea:** Accumulate the sum of squared gradients in a vector $G_t$.  

1. **Accumulator Update:**
   $$G_t = G_{t-1} + (\nabla f(w_t))^2$$
   (the square here is applied element-wise).
2. **Parameter Update:**
   $$w_{t+1} = w_t - \frac{\alpha}{\sqrt{G_t} + \varepsilon} \nabla f(w_t)$$
   where $\varepsilon$ is a small constant (e.g. $10^{-8}$) to avoid division by zero.

This per-parameter scaling is especially effective when some features are sparse or have different magnitudes.


## 4. Adam

**Motivation:** Combines the benefits of Momentum and Adagrad:
- Momentum: Keep an exponential moving average of gradients (the **first moment**).
- Adagrad-like adaptation: Keep an exponential moving average of squared gradients (the **second moment**).

**Update Equations:**  
1. **First Moment (mean of gradients):**  
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla f(w_t)$$
2. **Second Moment:**  
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla f(w_t))^2$$
3. **Bias Correction:**  
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
   $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. **Parameter Update:**  
   $$w_{t+1} = w_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

where $\beta_1, \beta_2 \in [0,1)$ are decay rates (typically $\beta_1 = 0.9$ and $\beta_2 = 0.999$) controlling how quickly old information is discarded, and $\varepsilon$ (typically $10^{-8}$) ensures numerical stability.

**Bias Correction Explanation:** The moving averages $m_t$ and $v_t$ are initialized with zeros, causing them to be biased toward zero, especially during the early steps of training. The bias correction terms adjust for this initialization bias by scaling the estimates. As $t$ increases, the correction becomes less significant since $(1 - \beta^t) \to 1$ as $t \to \infty$. This correction ensures more accurate adaptive learning rates, particularly in the early stages of optimization.


## Summary

1. **Gradient Descent**: Straightforward method that can suffer from slow or oscillatory convergence.  
2. **Momentum**: Uses a velocity term to damp oscillations and potentially accelerate convergence.  
3. **Adagrad**: Scales learning rates per parameter via historical gradient magnitudes, beneficial for sparse/differently scaled features.  
4. **Adam**: Combines Momentum and Adagrad ideas to provide robust and adaptive updates in most deep learning tasks.

Each method builds on the basic gradient step to address different challenges in optimization, ultimately helping gradients guide the search more efficiently.

## Recommended Reading

1. Kingma, Diederik P., and Jimmy Ba. [Adam: A method for stochastic optimization.](https://arxiv.org/pdf/1412.6980)
2. Ruder, Sebastian. [An overview of gradient descent optimization algorithms.](https://arxiv.org/pdf/1609.04747)
3. Sutskever, Ilya, et al. [On the importance of initialization and momentum in deep learning.](https://proceedings.mlr.press/v28/sutskever13.pdf)
4. [Adam — PyTorch 2.6 documentation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
5. AdamW: Loshchilov, Ilya, and Frank Hutter. [Decoupled weight decay regularization.](https://arxiv.org/abs/1711.05101)
6. [The State of Machine Learning Competitions](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
