# Model-Based Offline Optimization

Model-Based Offline Optimization (MBO) is a powerful approach to find the "best" design, represented as a parameter vector $w$ that maximizes (or minimizes) a costly scalar objective function $f(w)$, using solely a fixed, pre-collected dataset. The function $f(w)$ is considered a "black-box" because we only observe its inputs and outputs without access to its internal workings, analytical form, or derivatives. Unlike online methods that iteratively query $f$, offline MBO prohibits additional evaluations during optimization, relying entirely on existing data to propose improved designs. This makes it uniquely suited for real-world problems where further evaluations are expensive, risky, or impossible, such as drug discovery or materials science.
## 1. Definition
Given a pre-collected dataset of parameter-score pairs $\mathcal{D} = \{ (w_i, f(w_i)) \}_{i=1}^N$, identify a new design (set of parameters) $w^\ast$ such that $f(w^\ast)$ is as large as possible (for maximization) or as small as possible (for minimization), without ever evaluating $f$ on new inputs during the optimization process. This constraint distinguishes offline optimization from traditional approaches that iteratively query the objective function.

Key Components:
1. **Designs:** $\{ w_i \}_{i=1}^N$, where each $w_i$ is a parameter vector (e.g., a molecule configuration, neural network architecture, or material property set).

2. **Scores:** $\{ f(w_i) \}_{i=1}^N$, where $f(w_i)$ is the observed, expensive to compute output of the black-box function $f$.

3. **No New Queries:** Optimization must proceed using only $\mathcal{D}$, with no opportunity to evaluate $f$ at new points.

Core Idea: 
1. Train a surrogate model (e.g. neural net) $\hat{f}(w)$ to approximate $f(w)$ based on $\mathcal{D}$
2. Optimize $\hat{f}(w)$ to propose $w^\ast= \arg\min_w \hat{f}(w)$ using gradient-based methods. Since $\hat{f}(w)$ is a neural network, we can leverage automatic differentiation frameworks (e.g., PyTorch) to obtain gradients and apply optimization algorithms like Adam. The challenge lies in ensuring $w^\ast$ performs well under the true $f$, despite limited data and no further feedback.

## 2. The Offline MBO Process

The workflow of offline MBO is straightforward:

1. **Start with a Fixed Dataset:**
   - Use $\mathcal{D}=\{(w_i,f(w_i))\}_{i=1}^N$, collected prior to optimization (e.g., from past experiments or simulations).
   - This dataset is the only source of information about $f$.

2. **Train a Surrogate Model:**
   - Build $\hat{f}(w)$ (a neural network) to predict $f(w)$ for any $w$.
   - Ensure $\hat{f}$ is accurate and computationally cheap to evaluate.

3. **Optimize the Surrogate:**
   - Apply an optimization algorithm to find $w^\ast=\arg\min_w\hat{f}(w)$. Since $\hat{f}(w)$ is a neural network, we can leverage automatic differentiation frameworks (e.g., PyTorch) to obtain gradients and apply optimization algorithms like Adam.
   - Leverage $\hat{f}$'s low cost for extensive searches.

4. **Propose the Design:**
   - Output $w^\ast$ as the recommended design, with its true score $f(w^\ast)$ unknown unless post-hoc evaluation is feasible.

**Key Constraint:** No additional evaluations of $f$ are allowed during training or optimization, distinguishing offline MBO from iterative methods like Bayesian optimization.

## 3. Motivation and Real-World Relevance

### Why Offline MBO Matters:
- **No Additional Queries:** In domains like drug design, robotics hardware, or materials science, evaluating $f$ (e.g., synthesizing a compound, building a prototype) is costly or risky. Offline MBO uses existing data, sometimes years of prior experiments, to propose new designs without further expense.
- **Leveraging Existing Data:** Organizations often have vast databases of past results. Offline MBO turns this static data into actionable insights, recommending designs that improve on what’s already known.
- **Broad Applicability:** From optimizing neural network hyperparameters to designing novel proteins, offline MBO tackles problems where live experimentation is impractical.

## 4. Challenges:
- **Limited View of Design Space:** A small or unrepresentative $\mathcal{D}$ restricts $\hat{f}$'s ability to model $f$ accurately across all $w$.
- **Out-of-Distribution (OOD) Issues:** $\hat{f}$ may overestimate scores for designs unlike those in $\mathcal{D}$, leading the optimizer to propose suboptimal or invalid $w^\ast$.
- **Surrogate Accuracy:** Simple regression (e.g., minimizing mean squared error) can falter in OOD regions, risking poor generalization.

## 5. Recommended Reading:
1. Tan, Rong-Xi, et al. [Offline Model-Based Optimization by Learning to Rank.](https://openreview.net/forum?id=sb1HgVDLjN)
2. Trabucco, Brandon, et al. [Conservative objective models for effective offline model-based optimization.](https://arxiv.org/abs/2107.06882)
3. Momeni, Ali, et al. [Locality-aware Surrogates for Gradient-based Black-box Optimization.](https://arxiv.org/html/2501.19161v1)
4. Trabucco, Brandon, et al. [Design-bench: Benchmarks for data-driven offline model-based optimization.](https://arxiv.org/pdf/2202.08450)
5. Karpathy, Andrej [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)