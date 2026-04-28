# Lab 8 (new) — Reviewing a Paper With (and Without) an LLM

*Goal:* Read a research paper critically, compare your judgment against an LLM reviewer, probe how stable that LLM review is across different prompts, and finally produce a real conference-style review with an accept/reject recommendation.

---

## Why This Lab?

The course is full of nature-inspired and stochastic ML methods (PSO, ACO, CMA-ES, NAS, ...). The literature in this area is large and uneven in quality. As researchers and engineers you will increasingly use LLMs to help triage and review papers. Two questions worth asking:

1. **How good are LLMs at peer review?** Do they catch the things a human reviewer would? Do they hallucinate strengths? Do they miss obvious weaknesses?
2. **How prompt-sensitive is an LLM reviewer?** If you re-ask the same model in a more adversarial tone, does the verdict change — and what does that tell you about trusting LLM judgments?

## The Papers

Pick **one** paper from `papers/`. All four sit at the intersection of evolutionary/swarm methods and deep learning, which connects directly to the rest of the course:

- `pso-cnn.pdf` — Particle Swarm Optimization for CNN architecture/training
- `pso-dnn.pdf` — PSO applied to deep neural network training
- `aco-cnn.pdf` — Ant Colony Optimization for CNNs
- `eco-nas.pdf` — Eco / efficiency-oriented Neural Architecture Search

Read the paper *first*, before touching any LLM. Your own first impression is the baseline you will compare against later.

## What You Will Do

### 1. Read the paper and write your own review

Take notes as you read. Then write a short (~1 page) review that covers:

- **Problem & contribution.** What is the paper claiming? In one sentence.
- **Strengths.** What is genuinely good or interesting? 2-3 bullet points. Be specific.
- **Weaknesses.** Be specific again. 2-3 bullet points.
- **Questions for the authors.** Two concrete questions you'd ask if you were the reviewer.
- **Your verdict (gut feeling).** Accept / weak accept / weak reject / reject. Don't overthink it yet.

### 2. Get an LLM review of the same paper

Use Claude Code, Codex, or any chat LLM you prefer. The point is to give the model the *full* paper, not just an abstract.

- Check whether the paper is on **arXiv** (search by title). If yes, **prefer the LaTeX source** over the PDF — `https://arxiv.org/e-print/<id>` lets you download the source tarball. LaTeX is cleaner for an LLM to parse than a 2-column PDF, and you can pass the unpacked project directly to a coding agent (`claude` or `codex` from inside the unpacked directory).
- If only PDF is available, attach the PDF directly.
- Use a **neutral reviewer prompt**, e.g.:

  > You are a reviewer for a top-tier ML conference. Read the attached paper and produce a structured review with: summary, strengths, weaknesses, questions for authors, soundness (1–4), presentation (1–4), contribution (1–4), and an overall recommendation (1–10) with justification.

- Save the full LLM review verbatim.

### 3. Compare your review to the LLM's review

Write a short side-by-side comparison:

- Which weaknesses did you both catch? Which did only one of you catch?
- Did the LLM hallucinate anything that isn't actually in the paper? **Verify suspicious claims against the PDF.**
- Did the LLM avoid making a hard judgment? Did *you*?
- Net assessment: **how good was the LLM as a reviewer for this specific paper?**

### 4. Re-run the LLM with an aggressive / "cruel" prompt

Now change the persona. Something like:

> You are a famously harsh Area Chair at NeurIPS. Your job is to find every reason this paper should be rejected. Be blunt, skeptical, and uncharitable. Assume the authors are overclaiming until proven otherwise. Demand specific evidence for every claim.

Re-submit the **same** paper. Then compare:

- Does the recommendation score change? By how much?
- Are the *facts* the LLM cites about the paper consistent across the two runs, or does the harsh persona invent new flaws?
- Which review do you trust more, and why?

This is the core empirical question of the lab: **how prompt-stable is LLM peer review?**

### 5. Produce a conference-style review

Pick one major ML venue and use its **official reviewer form** as your template. Recommended (all use OpenReview, all have public reviews you can browse):

- **ICLR** — reviewer form is the easiest to find and most explicit. See the reviewer guide: <https://iclr.cc/Conferences/2025/ReviewerGuide>
- **NeurIPS** — review form is similar; guide: <https://neurips.cc/Conferences/2024/ReviewerGuidelines>

Fill in the template **as yourself**, using your own analysis informed by (but not copied from) the LLM reviews. Include:

- Summary
- Strengths
- Weaknesses
- Questions
- Soundness / Presentation / Contribution scores
- Overall recommendation **with a clear accept/reject decision**
- Confidence
