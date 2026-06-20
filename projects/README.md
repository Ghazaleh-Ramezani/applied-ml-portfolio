# Projects

Built work that I can defend in depth. Each project README should follow the same
pattern: problem, approach, stack, evaluation, what I personally did, limitations,
next steps. The two below are your real repositories — drop your actual metrics
into the marked spots (don't leave placeholders in the public version).

## Materials-RAG — hybrid retrieval for scientific literature
Repo: https://github.com/Ghazaleh-Ramezani/materials-rag

**Problem.** Retrieving and grounding answers over scientific literature where
both exact-term matching and semantic similarity matter.

**Approach.** Hybrid retrieval — BM25 lexical retrieval + dense vector retrieval
(FAISS), combined with reciprocal rank fusion (RRF), then cross-encoder reranking
of the fused candidates.

**Evaluation.** Recall@k and MRR on a held-out query set.

**Results.** *(fill in: Recall@k and MRR for BM25-only, dense-only, fused, and
fused+reranked — the comparison is the story, and it's yours to state.)*

**My role / limitations / next steps.** *(your words.)*

## Materials-MOBO — multi-objective Bayesian optimization
**Problem.** Choosing materials-design candidates under competing objectives
(e.g. conductivity vs. tensile vs. thickness) from limited experimental data.

**Approach.** Multi-objective Bayesian optimization with BoTorch / qLogEHVI;
Gaussian-process surrogates; Pareto-front analysis and candidate selection.

**Data.** Real CNC/CNF/rGO nanocomposite dataset (samples from your Micromachines
tables).

**Results.** *(fill in: hypervolume / Pareto-front summary; the insight that
conductivity and tensile are positively correlated so the Pareto structure is
driven by the thickness axis is a strong, specific point to make.)*

**My role / limitations / next steps.** *(your words.)*
