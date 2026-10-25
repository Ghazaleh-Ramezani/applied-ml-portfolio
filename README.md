# Applied ML Portfolio — Reliable Evaluation, Optimization and LLM Systems

Projects in verifiable LLM agents, uncertainty under distribution shift,
preference optimization, retrieval-augmented generation and Bayesian
experimental design, applied to scientific, engineering and financial data.
Every flagship repository runs on a CPU from a clean checkout and commits the
outputs its README reports.

## Flagship projects

| Repository | What it establishes | Tests |
|---|---|---|
| [financial-claim-audit-agent](https://github.com/Ghazaleh-Ramezani/financial-claim-audit-agent) | Tool-using agent over versioned financial claims — restatements and as-of answers without look-ahead — behind a six-check verification gate. Red-team evaluation with 30 attacks from a planner that follows injected instructions: 1/30 succeed with free-text answers, 0/30 with answers rendered from the cited claims, and no honest answer blocked in that mode; FastAPI, PostgreSQL, Docker Compose, CI | 69 |
| [shift-aware-rul](https://github.com/Ghazaleh-Ramezani/shift-aware-rul) | Under operating-condition shift, 90% conformal intervals for remaining useful life cover 28–31% of target engines; health-anchored normalization restores 0.90–0.92 and cuts FD002 RMSE from 66.4 to 22.6 (NASA C-MAPSS, 3 seeds), unchanged with early-cycle-only clustering, statistics from other engines only, and anchors of 5–50 cycles | 9 |
| [prefopt-repro](https://github.com/Ghazaleh-Ramezani/prefopt-repro) | SFT → reward model → DPO/GRPO with position-swapped judging, bootstrap CIs and reward-hacking diagnostics, verified end to end on CPU; Goodhart gap at every checkpoint; ZeRO-3 sharded training matches single-process training to 1.8 × 10⁻⁶; eleven defects found and fixed | 43 |
| [action-isa](https://github.com/Ghazaleh-Ramezani/action-isa) | Typed action slots lift Qwen3-0.6B multi-step tool use from 0% to 87.5% without fine-tuning (exact McNemar p = 5.8 × 10⁻¹¹), with lower-entropy and prompt-matched random controls and a model ladder; NeurIPS 2026 SLM-Agents workshop | — |
| [materials-mobo](https://github.com/Ghazaleh-Ramezani/materials-mobo) | Retrospective campaign over 60 measured CNC/CNF/rGO films: qLogEHVI finds 99.8% of the pool's hypervolume and 14.4 of 15 Pareto-optimal films in 30 experiments, versus 90.5% and 7.2 for random selection (20/20 paired seeds), compared with qEHVI and ParEGO and with posterior-predictive calibration | 15 |
| [data-driven-materials-optimization](https://github.com/Ghazaleh-Ramezani/data-driven-materials-optimization) | Nested CV on the same 60 films: LASSO R² 0.957, multi-output network 0.988; the published LASSO result (R² 0.841) reproduces; bootstrap selection stability and grouped SHAP under strong collinearity | 20 |
| [materials-rag](https://github.com/Ghazaleh-Ramezani/materials-rag) | BM25 + FAISS retrieval fused with RRF, optional cross-encoder reranking, entity-aware semantic cache, FastAPI and Docker; Recall@k, Hit@k and MRR harness | 29 |
| [long-context-llm-experiments](https://github.com/Ghazaleh-Ramezani/long-context-llm-experiments) | Response-only LoRA training for long-context QA and a strict needle-in-a-haystack harness, verified on a local Llama-architecture model | 23 |

Project notes with problem, approach, evaluation and results: [`projects/`](./projects/).

## Learning project

### [Kronecker-factorized linear layers](./learning/kronecker-linear/)

A NumPy-from-scratch study of how Kronecker structure cuts a linear layer's
parameters and inference cost, and what it costs in expressivity: the
vec-trick forward pass, the Van Loan–Pitsianis approximation, and a benchmark
with 512× and 2048× fewer parameters at 1024² and 4096², a forward pass about
30–40× and 110–195× faster depending on the machine, and a rank sweep showing
that a random matrix is the worst case for the structure.

## Research interests

Efficient and long-context LLM inference: structured attention, adaptive
inference, planner–worker reasoning, retrieval-augmented inference and
memory-augmented transformers, with links to the code that touches each
thread → [`research-interests`](./research-interests/)

## Publications

13 peer-reviewed publications (h-index 6) across machine learning and
materials science → [`publications`](./publications/) · Google Scholar:
https://scholar.google.com/citations?user=H3ML498AAAAJ

## Contact

GitHub: [Ghazaleh-Ramezani](https://github.com/Ghazaleh-Ramezani)
