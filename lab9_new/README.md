# Lab 8 (new) — Reviewing a Paper With (and Without) an LLM

*Goal:* Read a research paper critically, compare your judgment against an LLM reviewer, probe how stable that LLM review is across different prompts, and present a real conference-style accept/reject decision to the rest of the class.

This lab has **no code**. It is **group work** with a fixed schedule, ending in a short presentation per group.

---

## Why This Lab?

The course is full of nature-inspired and stochastic ML methods (PSO, ACO, CMA-ES, NAS, ...). The literature in this area is large and uneven in quality. As researchers and engineers you will increasingly use LLMs to help triage and review papers. Two questions worth asking:

1. **How good are LLMs at peer review?** Do they catch the things a human reviewer would? Do they hallucinate strengths? Do they miss obvious weaknesses?
2. **How prompt-sensitive is an LLM reviewer?** If you re-ask the same model in a more adversarial tone, does the verdict change — and what does that tell you about trusting LLM judgments?

## Group Setup

- Form groups of **3–4 people**.
- Each group picks **one** paper from `papers/`. Two groups *may* pick the same paper — comparing how different groups end up rating the same paper is itself interesting.
- Each member of the group reads the paper individually first. The discussion happens *after* everyone has formed an independent opinion.

## The Papers

- `pso-cnn.pdf` — Particle Swarm Optimization for CNN architecture/training
- `pso-dnn.pdf` — PSO applied to deep neural network training
- `aco-cnn.pdf` — Ant Colony Optimization for CNNs
- `eco-nas.pdf` — Eco / efficiency-oriented Neural Architecture Search

## Schedule

The lab is time-boxed. Stick to the clock — the point is decisive triage, not a perfect review.

| Phase | Time | What happens |
|---|---|---|
| 1. Individual reading | **20–25 min** | Each person reads the paper alone and takes private notes. No LLMs yet. |
| 2. LLM experiments | **10–15 min** | Each person individually runs an LLM reviewer — neutral prompt, then aggressive prompt. |
| 3. Group discussion | **20 min** | Reconcile individual reads, compare against LLM, agree on a verdict. |
| 4. Presentations | **5–10 min per group** | Each group presents to the class. |

---

### Phase 1 — Individual reading (20–25 min)

**Do not talk to your group, and do not open an LLM.** Just read.

While reading, jot down (in your own notes):

- **Contribution** — what is the paper claiming? One sentence.
- **Strengths** — 2–3 specific points. Cite a section, figure, or table.
- **Weaknesses** — 2–3 specific points.
- **Two questions** you would ask the authors.
- **Your gut verdict** — accept / weak accept / weak reject / reject.

These private notes are what you bring to the group discussion.

### Phase 2 — LLM experiments (10–15 min)

**Do this individually, on your own machine.** Each member of the group runs their own LLM experiments before the discussion — that way you bring independent transcripts to compare in Phase 3.

1. **Get the paper into an LLM.** Search for the title on arXiv. If it exists, grab the LaTeX source: `curl -L -o paper.tar.gz https://arxiv.org/e-print/<arxiv_id> && tar -xzf paper.tar.gz`. From the unpacked directory you can run `claude` or `codex` and ask the agent to review the project. If there is no arXiv version, attach the PDF directly to a chat LLM.
2. **Neutral prompt run.** For example:
   > You are a reviewer for a top-tier ML conference. Read the attached paper and produce a structured review with: summary, strengths, weaknesses, questions for authors, soundness (1–4), presentation (1–4), contribution (1–4), and an overall recommendation (1–10) with justification.
3. **Aggressive prompt run.** Same paper, harsher persona:
   > You are a famously harsh Area Chair at NeurIPS. Your job is to find every reason this paper should be rejected. Be blunt, skeptical, and uncharitable. Assume the authors are overclaiming until proven otherwise. Demand specific evidence for every claim.
4. **Save both transcripts.** You will need them for the presentation.
5. **Spot-check claims.** When the LLM cites a number or a citation, grep the PDF/LaTeX. Hallucinations are most common around numbers and references.

### Phase 3 — Group discussion (20 min)

Suggested agenda:

- ~5 min: Each member shares their gut verdict and the **single weakness** they care most about. Don't debate yet — just collect.
- ~5 min: Compare **neutral LLM reviews** across the group. Did everyone get a similar score, or did the same model give wildly different verdicts to different members? What did the LLM catch that nobody did? What did *you* catch that the LLM missed? Any hallucinations?
- ~5 min: Compare neutral vs. **aggressive** runs (across members too). Did the score move? Did the *facts* the LLM cited change, or just the tone? Does the neutral review now look sycophantic? Which run do you trust more?
- ~5 min: Agree on a **group verdict** using a real conference template (pick one):
  - **ICLR** reviewer guide: <https://iclr.cc/Conferences/2025/ReviewerGuide>
  - **NeurIPS** reviewer guidelines: <https://neurips.cc/Conferences/2024/ReviewerGuidelines>

  Fill in soundness / presentation / contribution scores, an overall recommendation, a clear accept/reject decision, and a confidence.

If you have any minutes left, browse a few real reviews on OpenReview to calibrate tone and specificity:

- ICLR 2025: <https://openreview.net/group?id=ICLR.cc/2025/Conference>
- ICLR 2024: <https://openreview.net/group?id=ICLR.cc/2024/Conference>
- NeurIPS 2024: <https://openreview.net/group?id=NeurIPS.cc/2024/Conference>

Open one accepted and one rejected paper. Look at how reviewers cite line / figure / equation numbers, how scores cluster, and how the meta-reviewer reconciles disagreement.

### Phase 4 — Presentations (5–10 min per group)

Each group presents to the class. Suggested structure:

1. **The paper** — title, claim, why it fits the course.
2. **Our review** — top strengths, top weaknesses, group verdict.
3. **Neutral vs. aggressive LLM** — score delta, one example where the two runs disagreed on a *fact* (not just tone).
4. **How good was the LLM?** — did it help, mislead, or both? One concrete example each.
5. **Final decision** — conference template scores + accept/reject + confidence.

Keep to time — aim for ~5 minutes of content + a couple of minutes of questions. The interesting part is rarely the paper itself; it is **how your group's verdict compared to the LLM's, and how stable the LLM was across prompts**.

---

## Practical Tips

- Don't try to read every section in 25 minutes. Read the **abstract, intro, method overview, main experimental table, and conclusion** first. Skim the rest only if time allows.
- Always verify LLM-quoted numbers and citations against the PDF/LaTeX before repeating them in your presentation.
- Keep both LLM transcripts. The diff between neutral and aggressive mode is the most interesting artifact you will produce.
- Don't outsource the verdict. The LLM's score is one data point; **your group's decision is what you present and defend**.
