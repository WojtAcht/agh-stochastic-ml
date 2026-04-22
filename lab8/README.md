# Lab 8 — CMA-ES

Covariance Matrix Adaptation Evolution Strategy, applied to black-box optimization and to training a LunarLander-v3 policy.

## Setup

This lab uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv first:

- **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Then, from this `lab8/` directory:

```bash
uv sync
```

This creates a `.venv/` with the exact versions pinned in `uv.lock`. To run the notebook:

```bash
uv run jupyter lab instructions.ipynb
```

Or select the `.venv` kernel from your existing Jupyter / VS Code setup.

### Platform notes

- **Python 3.11 or 3.12** is required (3.13 is not yet supported by all deps).
- `gymnasium[box2d]` pulls in `box2d-py`, which is a C++ extension:
  - **Linux / macOS:** pre-built wheels exist — `uv sync` should just work.
  - **Windows:** if the wheel is unavailable for your Python version, install [SWIG](https://www.swig.org/download.html) and the "Desktop development with C++" workload from the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), then re-run `uv sync`.

---

# CMA-ES: Covariance Matrix Adaptation Evolution Strategy

*Goal:* To understand the basic idea behind CMA-ES, a powerful algorithm for finding the minimum (or maximum) of complex functions, especially when we don't know the gradient.

## What are Evolutionary Algorithms (EAs)?

EAs are inspired by biological evolution. They work with a population of candidate solutions. In each "generation" (iteration):

1. Good solutions are selected (survival of the fittest).
2. New solutions are created based on the good ones (reproduction, mutation).
3. This process repeats, hopefully evolving the population towards better and better solutions.

## What is CMA-ES?

The Core Idea: Adapting the Search Distribution

The "magic" of CMA-ES lies in how it adapts its search strategy. It uses a multivariate normal distribution (think of an ellipse or ellipsoid in multiple dimensions) to generate new candidate points.

1. Mean - $\mu$ - This is the center of the distribution.
2. Step-size - $\sigma$ - This controls the overall size of the search ellipse. If steps are consistently successful, it might increase $\sigma$ to explore further; if it seems to be overshooting or oscillating, it might decrease $\sigma$.
3. Covariance Matrix - $C$ - This controls the shape and orientation of the search ellipse. This is the most sophisticated part. CMA-ES learns correlations between variables. If the valley is a long, narrow ridge, the covariance matrix will adapt to make the search ellipse long and narrow and align it with the ridge, making the search much more efficient.

![CMA-ES Adaptation](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/CMA-ES-illustration.png)
*Images: Visualization of how CMA-ES adapts its search distribution to the landscape of the objective function.*

![CMA-ES Pseudocode](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/CMA-ES-algorithm.png)
*Image: Pseudocode of CMA-ES algorithm.*

## Strengths of CMA-ES

- Very effective on non-linear, non-convex, rugged, ill-conditioned problems.
- Handles correlations between variables automatically.
- Robust to noise in the function evaluation.
- Requires relatively few parameter settings (defaults often work well).
- Doesn't need the function's gradient/derivative.

## Weaknesses of CMA-ES

- Computationally more expensive per generation than simpler EAs due to matrix operations (especially in high dimensions).
- Can be slower than gradient-based methods if gradients are available and reliable.
- Performance can degrade significantly in very high dimensions (e.g., >>100)..

## Recommended Reading

1. Hansen, Nikolaus. [The CMA evolution strategy: A tutorial.](https://arxiv.org/abs/1604.00772)
2. Lil'Log [Evolution Strategies](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)
3. Such, Felipe Petroski, et al. [Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning.](https://arxiv.org/abs/1712.06567)
4. OpenAI [Evolution strategies as a scalable alternative to reinforcement learning](https://openai.com/index/evolution-strategies/)
