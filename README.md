# Stochastic Methods in Machine Learning

This repository contains laboratory materials for the "Stochastic Methods in Machine Learning" course at AGH University of Science and Technology in Krakow.

## Course Description

This course explores various problems at the intersection of optimization and machine learning.

## Labs Overview

| Lab | Title | Description |
|-----|-------|-------------|
| 1 | Gradient Descent | **Basic (lab1):** Implements gradient descent algorithm to train a linear regression model from scratch. <br><br> **Advanced (lab1_new):** Reproduces key experiments from ["Understanding Deep Learning Requires Rethinking Generalization"](https://arxiv.org/abs/1611.03530) — investigates memorization of random labels, effect of model capacity, and weight decay regularization on MNIST. |
| 2 | Gradient Descent Extensions | Covers gradient descent extensions including Momentum, AdaGrad, and Adam. Students test these optimizers on standard benchmark test functions: Sphere, Rosenbrock, Rastrigin. |
| 3 | Adversarial Examples | Investigates the vulnerability of neural networks to adversarial attacks, implementing the Fast Gradient Sign Method (FGSM) to generate perturbations that cause misclassification. |
| 4 | Model-Based Offline Optimization | Explores optimization of black-box functions using pre-collected datasets without additional function evaluations. Involves training neural network surrogate models to approximate benchmark functions and implementing gradient-based optimization techniques on these surrogate models to find optimal solutions. |
| 5 | Hyperparameter Optimization | Utilizes Optuna framework to fine-tune CatBoost model hyperparameters on the Covertype dataset. Demonstrates practical application of HPO to maximize classification performance in a multiclass problem. |
| 6 | Bayesian Optimization | Covers various acquisition functions (e.g., Expected Improvement, UCB), focusing on how they manage the exploration-exploitation trade-off. |
| 7 | Loss Landscape Visualization | **New (lab7_new):** Reproduces key results from ["Visualizing the Loss Landscape of Neural Nets"](https://arxiv.org/abs/1712.09913).<br><br> **Legacy (lab7):** Explores the fundamental properties and significance of the Normal Distribution. |
| 8 | CMA-ES | Two-part lab on the pycma library. Part 1 explores CMA-ES on benchmark functions (Sphere, Rosenbrock, Rastrigin), studying sensitivity to the starting point, the initial step-size $\sigma_0$, and visualizing covariance matrix adaptation. Part 2 applies CMA-ES to reinforcement learning on Gymnasium's LunarLander-v3, evolving MLP policy weights and benchmarking against Random Search and PPO under a shared environment-step budget. |
| 9 | Reviewing a Paper With (and Without) an LLM | Group lab with no code. Each group reads a nature-inspired ML paper, runs an LLM reviewer under a neutral and an aggressive prompt, compares verdicts, and presents a conference-style accept/reject decision. Probes how good — and how prompt-sensitive — LLM peer reviews are. |
| 10 | Differential Evolution | Implements Differential Evolution from scratch. |
| 11 | LLM × EA | Evolution of Heuristics |
| 12 | Cuckoo Search | Explores the common issue of seemingly new optimization algorithms that may not offer genuinely novel ideas, focusing on Cuckoo Search and its critical analysis. |
| 13 | Multiobjective Optimization | Demonstrates multiobjective optimization techniques to find optimal asset allocations for investment portfolios. |

## Issues and Contributions

If you find any mistakes or have suggestions for improvements:

1. **Create an Issue**: Open a new issue in the repository describing the problem or suggestion in detail.
2. **Submit a Pull Request**: If you have a fix or improvement, feel free to fork the repository and submit a pull request with your changes.

Your contributions help improve the quality of these materials for all students.


