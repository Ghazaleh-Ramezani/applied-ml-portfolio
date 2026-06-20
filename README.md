# Applied ML Portfolio — Retrieval, Optimization, and Efficient LLM Systems

Selected projects and research in retrieval-augmented generation, multi-objective
Bayesian optimization, and efficient/long-context LLM systems, applied largely to
scientific domains. The repository is organized so it is clear what is **built and
defensible**, what is a **learning project**, and what is an **active research
interest** — everything here is something I can discuss in detail: problem setup,
modeling choices, evaluation, tradeoffs, and limitations.

## Flagship projects (built, with evaluation)

### Materials-RAG — hybrid retrieval for scientific literature
A production-style retrieval pipeline combining **BM25 lexical retrieval** and
**dense vector retrieval (FAISS)**, fused with **reciprocal rank fusion (RRF)**
and refined by **cross-encoder reranking**, evaluated with **Recall@k and MRR**.
→ [`projects/materials-rag`](./projects/) · repo: https://github.com/Ghazaleh-Ramezani/materials-rag

### Materials-MOBO — multi-objective Bayesian optimization
Multi-objective optimization for materials design using **BoTorch / qLogEHVI**,
producing a **Pareto front** over competing objectives on a real CNC/CNF/rGO
nanocomposite dataset, with the analysis tying the Pareto structure to the
thickness axis.
→ [`projects/materials-mobo`](./projects/)

These two carry hard, reproducible numbers and are the right place for a reviewer
to start.

## Learning project (from scratch, with measured results)

### [Kronecker-factorized linear layers](./learning/kronecker-linear/)
A NumPy-from-scratch study of how Kronecker structure cuts a linear layer's
parameters and inference cost — and what it costs in expressivity. Includes the
vec-trick forward pass, the Van Loan–Pitsianis approximation, and a real
benchmark (**512×–2048× fewer parameters and 41×–195× faster forward passes** at
1024×1024 and 4096×4096, with an honest analysis of when the structure actually
helps). Clearly labeled as a learning exercise, written to be explained
line by line.

## Research interests

Reading-driven notes and directions in **efficient and long-context LLM
inference** — structured attention, adaptive inference, planner–worker reasoning,
retrieval-augmented inference, and memory-augmented transformers. These are
framed as interests and ongoing study, not as shipped work.
→ [`research-interests`](./research-interests/)

## Publications

13 peer-reviewed publications (h-index 6) across ML and materials science,
including a recent ML-for-materials workshop paper.
→ [`publications`](./publications/) · Google Scholar:
https://scholar.google.com/citations?user=H3ML498AAAAJ

## How to read this repo

- **Flagship projects** — built and defensible in depth; start here.
- **Learning project** — a real implementation I wrote to learn the mechanism;
  small, measured, fully explainable.
- **Research interests** — current reading and directions, not finished claims.
- **Publications** — peer-reviewed research background.

## Contact

GitHub: [Ghazaleh-Ramezani](https://github.com/Ghazaleh-Ramezani)
