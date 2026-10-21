# Project notes

Each note gives the problem, the approach, how it is evaluated and what the
committed results show. Every number links back to a file in the repository
that produced it.

## shift-aware-rul — reliable remaining-useful-life prediction under shift

Repo: https://github.com/Ghazaleh-Ramezani/shift-aware-rul

**Problem.** Prognostic models are trained under one set of operating
conditions and deployed under others; their uncertainty estimates have to stay
honest when that happens.

**Approach.** Hybrid CNN–LSTM on NASA C-MAPSS FD001, deep-ensemble and
MC-dropout uncertainty with split-conformal calibration on held-out source
engines only, label-free test-time adaptation (AdaBN, per-condition
normalization) and health-anchored normalization from each engine's first 20
cycles.

**Evaluation.** Transfer to FD002–FD004, three seeds, paired engine bootstrap.

**Results.** Without adaptation, 90% conformal intervals cover 28% (FD002) and
31% (FD004) of target engines. Health-anchored normalization cuts FD002 RMSE
from 66.4 to 22.6 and restores coverage to 0.90 (FD002) and 0.92 (FD004);
anchored + AdaBN gives the best FD004 RMSE, 31.2.

**Next.** Anchor-length sweep and automatic selection of the adaptation.

## prefopt-repro — preference optimization with trustworthy evaluation

Repo: https://github.com/Ghazaleh-Ramezani/prefopt-repro

**Problem.** Comparing alignment methods needs evaluation that separates a
better policy from one that has learned what its reward model rewards.

**Approach.** SFT → Bradley–Terry reward model → DPO and GRPO on Transformers,
TRL and PEFT; position-swapped judging, bootstrap confidence intervals,
multi-seed generation, and reward-hacking diagnostics (length–reward
correlation, KL from the reference, distinct-4, stylistic tells, Goodhart gap).

**Evaluation.** `scripts/cpu_validation.sh` runs every stage end to end on an
offline fixture; `prefopt.goodhart` traces proxy-RM versus external-judge
win-rate at every saved checkpoint; `scripts/zero3_cpu_check.py` checks
DeepSpeed ZeRO-3 sharded training against single-process training.

**Results.** Every stage hands its checkpoint to the next and evaluation
rebuilds the model each stage trained; eleven code defects found and fixed,
each with a regression test (43 tests). ZeRO-3 on two ranks: each rank holds
half of the parameters, and losses and final parameters match single-process
training to 4.8 × 10⁻⁷ and 1.8 × 10⁻⁶.

**Next.** The 1.5B ablation grid on GPU: β sweep, GRPO group size and
reward-model quality.

## materials-mobo — multi-objective Bayesian optimization

Repo: https://github.com/Ghazaleh-Ramezani/materials-mobo

**Problem.** Choosing the next materials experiment under competing objectives
and a small experimental budget.

**Approach.** Independent GP surrogates per objective, qLogEHVI over a box
decomposition of the non-dominated region, compared with qEHVI, qLogNParEGO
and random selection; posterior-predictive calibration, leave-one-out and on
the films each campaign chooses.

**Data.** The 60 measured CNC/CNF/rGO films of Ramezani et al., *Micromachines*
16(4), 393 (2025): maximize conductivity and tensile strength, minimize
thickness.

**Results.** In a retrospective campaign over the 60 films (20 seeds, 30
experiments each), qLogEHVI finds 99.8% of the pool's hypervolume and 14.4 of
the 15 Pareto-optimal films, against 90.5% and 7.2 for random selection and
97.0% for ParEGO, and reaches 95% after a median of 18 films. Leave-one-out
90% coverage of the GP is 0.90–0.93; on the films the campaign chooses it
drops to 0.69–0.88, the selection effect to correct for next.

## data-driven-materials-optimization — small-data regression done carefully

Repo: https://github.com/Ghazaleh-Ramezani/data-driven-materials-optimization

**Problem.** Predicting conductivity, tensile strength and thickness from
composition and processing for 60 films with strongly collinear inputs.

**Approach.** Nested cross-validation stratified by reducing agent, with
scaling and hyperparameter search inside each training fold; composition-
grouped CV; LASSO, linear regression and a multi-output neural network;
bootstrap selection frequency; Kernel SHAP with grouped attribution.

**Results.** Nested-CV R²: LASSO 0.957, multi-output network 0.988. The
paper's LASSO setting reproduces its published R² (0.841; median 0.843 over 200
splits). Composition carries 83–91% of the SHAP attribution in both models.

## materials-rag — hybrid retrieval for scientific literature

Repo: https://github.com/Ghazaleh-Ramezani/materials-rag

**Problem.** Grounded answers over scientific papers, where exact terms
(materials, units) and semantic similarity both matter.

**Approach.** BM25 and FAISS retrieval fused with reciprocal rank fusion,
optional cross-encoder reranking, an entity-aware semantic cache scoped to
retrieval settings, an embedder check on the dense index, FastAPI `/qa` and
Docker; runs offline out of the box.

**Evaluation.** Recall@k, Hit@k and MRR on a 15-question benchmark (12
single-document, 3 cross-document), plus an optional LLM faithfulness judge.

**Results.** On the bundled corpus with the offline embedder: Recall@3 0.967,
Hit@3 1.000, MRR 0.967; 29 tests.

**Next.** A 30–50-paper corpus with sentence-transformers embeddings and the
reranker.

## long-context-llm-experiments — long-context fine-tuning and evaluation

Repo: https://github.com/Ghazaleh-Ramezani/long-context-llm-experiments

**Problem.** Adapting a small LLM to answer from long documents, and measuring
retrieval from context without lenient scoring.

**Approach.** LoRA with response-only loss (instruction and context tokens
masked, EOS kept under PAD = EOS), context-side truncation, and a
needle-in-a-haystack harness with strict single-passcode scoring and a
distinct haystack per trial.

**Results.** 23 tests on a locally built Llama-architecture model cover the
label masking, EOS handling, truncation and scoring.

**Next.** Llama-3.2-1B accuracy heatmap over context length and needle depth,
base versus LoRA, on GPU.
