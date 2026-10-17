# Technical Interview Guide — Applied ML Portfolio

A truth-first guide for explaining the projects in this portfolio during technical interviews. It distinguishes **implemented**, **executed**, and **empirically established** work so every statement remains defensible.

## Interview answer structure

For every project, prepare three levels:

1. **30 seconds:** problem, method, strongest defensible result.
2. **2 minutes:** data, architecture, evaluation, result, limitation.
3. **Deep dive:** equations, implementation choices, trade-offs, failure modes, and next experiments.

---

# 1. shift-aware-rul

Repository: <https://github.com/Ghazaleh-Ramezani/shift-aware-rul>

## 30-second answer

> I studied the reliability of remaining-useful-life prediction when deployment conditions differ from training. I trained a hybrid CNN–LSTM on NASA C-MAPSS FD001, calibrated uncertainty only on held-out source engines, and evaluated transfer to FD002–FD004. Under operating-condition shift, nominal 90% deep-ensemble conformal intervals collapsed to 28–31% coverage. I developed health-anchored normalization using each engine's first 20 cycles, reducing FD002 RMSE from 66.4 to 22.6 and FD004 RMSE from 63.5 to 32.8 while restoring coverage to 0.90–0.92 without target labels.

## 2-minute answer

> FD001 has one operating condition and one fault mode, so I used it as the source domain. FD002 introduces six operating conditions, FD003 introduces a second fault mode, and FD004 combines both shifts. The pipeline uses 14 informative sensors, 30-cycle windows, and a piecewise-linear RUL target capped at 125 cycles. The model has parallel convolutional and recurrent paths: the CNN captures local sensor patterns and the LSTM captures longer temporal degradation trends.
>
> I compared MC dropout, three-member deep ensembles, and an exact Gaussian process fitted on learned embeddings. For each method, I calibrated normalized split-conformal intervals using held-out source engines only. Standard target normalization and AdaBN sometimes helped under shift but damaged in-domain performance because truncated target trajectories have a different health-state mixture. Health-anchored normalization estimates per-condition statistics only from each target engine's first 20 cycles, where engines are assumed healthy. It removes operating-condition offsets without normalizing away later degradation. It was strongest for operating-condition shift, but AdaBN was better for pure fault-mode shift, showing that adaptation must match the shift mechanism.

## Claim audit

| Claim | Defensible scope |
|---|---|
| Coverage collapsed to 28–31% | Deep-ensemble conformal intervals, no adaptation, FD002 and FD004 |
| FD002 RMSE 66.4 → 22.6 | Three-seed deep-ensemble mean, health-anchored normalization |
| FD004 RMSE 63.5 → 32.8 | Three-seed deep-ensemble mean, health-anchored normalization |
| Coverage restored to 0.90–0.92 | Deep-ensemble conformal coverage on FD002 and FD004 |
| In-domain cost of 2.5 cycles | FD001 ensemble ΔRMSE +2.47; paired-bootstrap 95% CI [1.49, 3.39] |
| Three-seed validation | Seed variability reported separately; bootstrap resamples engines, not seeds |

## Data and preprocessing

### What is C-MAPSS?

C-MAPSS is a simulated turbofan run-to-failure benchmark. Each observation contains an engine ID, cycle index, three operating settings, and 21 sensor channels. The target is remaining useful life.

| Subset | Conditions | Fault modes | Role |
|---|---:|---:|---|
| FD001 | 1 | 1 | Source train/calibration and in-domain test |
| FD002 | 6 | 1 | Operating-condition shift |
| FD003 | 1 | 2 | Fault-mode shift |
| FD004 | 6 | 2 | Combined shift |

### How is RUL defined?

For training engine \(i\), ending at cycle \(T_i\):

\[
RUL_{i,t}=T_i-t,
\qquad
y_{i,t}=\min(T_i-t,125).
\]

The cap avoids asking the model to distinguish very large early-life RUL values whose degradation signals are nearly indistinguishable.

### Why 14 sensors?

Seven channels are constant or nearly constant within an operating condition. Keeping the 14 informative channels reduces irrelevant dimensions and dependence on non-degradation signals.

### Why 30-cycle windows?

A single row does not reveal temporal degradation. A 30-cycle window gives local temporal context while remaining computationally manageable. Evaluation uses the last available window of each test engine.

### How is leakage prevented?

Splitting is performed by engine ID, not by window. Windows from one engine cannot appear in both training and validation. Calibration also uses held-out source engines.

## CNN–LSTM architecture

- Input BatchNorm over sensor channels.
- CNN path: Conv1D(kernel 5) → BatchNorm → ReLU → dropout → Conv1D(kernel 3) → BatchNorm → ReLU → temporal mean pooling.
- LSTM path: one LSTM over the normalized sequence; the last hidden state represents temporal evolution.
- Fusion: concatenated CNN and LSTM representations → 64-dimensional dense layer → ReLU → dropout.
- Regression head: one scalar RUL prediction, clipped to [0, 125] at inference.

### Why parallel paths?

The CNN emphasizes local motifs and sensor interactions, while the LSTM summarizes sequential evolution. Parallel fusion preserves both inductive biases instead of forcing all information through only one path.

### Why input BatchNorm?

It provides an inspectable adaptation point. AdaBN freezes model weights while re-estimating target-domain BatchNorm statistics.

## Distribution shift

### Covariate versus concept shift

- Operating-condition shift changes sensor distributions and can often be reduced by condition-aware normalization.
- Fault-mode shift may alter \(P(y\mid x)\); normalization alone cannot guarantee correction.
- FD004 combines both effects.

### What is AdaBN?

Adaptive Batch Normalization recomputes BatchNorm running means and variances from unlabeled target windows while leaving learned parameters frozen.

### Why can AdaBN hurt?

Target trajectories are truncated and contain a different health-state mixture. Re-estimating moments over all target windows can classify degradation-related changes as nuisance shifts and remove useful signal. On FD001, ensemble RMSE rose from 14.45 to 26.49.

### What is condition normalization?

K-means clusters the three operating settings using the known number of regimes. Each sensor is then standardized using statistics from its assigned target-domain condition.

### What is health-anchored normalization?

Per-condition statistics are estimated only from cycles 1–20 of each target engine:

\[
\tilde{x}_{i,t,j}=
\frac{x_{i,t,j}-\mu^{early}_{c(i,t),j}}
{\sigma^{early}_{c(i,t),j}}.
\]

The method uses no RUL or fault labels, but it does use a domain prior: engines begin in a healthy state.

### Why 20 cycles?

Twenty is a design choice balancing statistical stability against contamination by degradation. It is not theoretically optimal and should be followed by an anchor-length sensitivity sweep. The shortest FD004 test engine has 19 cycles, so all its available cycles are used.

## Uncertainty methods

### MC dropout

Dropout remains active at inference while BatchNorm remains in evaluation mode. Thirty stochastic predictions produce an empirical mean and standard deviation.

### Deep ensemble

For independently trained models:

\[
\mu(x)=\frac{1}{M}\sum_{m=1}^{M}\hat y_m(x),
\qquad
\sigma(x)=\operatorname{Std}_m[\hat y_m(x)].
\]

The experiment uses three members. Ensembles capture variation across trained solutions but do not automatically produce calibrated intervals.

### GP head

An exact Gaussian process with constant × RBF + white-noise kernel is fitted on up to 1,000 standardized penultimate neural embeddings. It is post-hoc GP regression, not jointly trained deep-kernel learning.

## Conformal prediction

The normalized score is:

\[
s_i=\frac{|y_i-\mu_i|}{\max(\sigma_i,0.5)}.
\]

For \(n\) calibration samples and error level \(\alpha\):

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil.
\]

The interval is:

\[
C(x)=[\mu(x)-q\sigma(x),\mu(x)+q\sigma(x)].
\]

### Why does source conformal validity fail?

Split-conformal validity relies on exchangeability between calibration and test observations. Deployment shift breaks that condition. The source-calibrated nominal 90% label is therefore not guaranteed on FD002–FD004.

### Does restored coverage create a new guarantee?

No. Adaptation empirically makes target residuals resemble source residuals, but it does not establish distribution-free validity under arbitrary target shift.

### Why is coverage alone inadequate?

Intervals can obtain high coverage by becoming uselessly wide. Report coverage together with width, RMSE, NASA score, and calibration error. The GP sometimes reaches 1.00 coverage with intervals wider than 230 cycles.

## Main ensemble results

| Target | No-adapt RMSE | Anchored RMSE | No-adapt coverage | Anchored coverage |
|---|---:|---:|---:|---:|
| FD001 | 14.45 ± 0.27 | 16.89 ± 1.18 | 0.89 ± 0.04 | 0.87 ± 0.02 |
| FD002 | 66.40 ± 0.87 | 22.64 ± 0.14 | 0.28 ± 0.03 | 0.90 ± 0.04 |
| FD003 | 42.62 ± 5.68 | 40.29 ± 5.33 | 0.83 ± 0.11 | 0.88 ± 0.12 |
| FD004 | 63.51 ± 0.49 | 32.80 ± 2.29 | 0.31 ± 0.03 | 0.92 ± 0.04 |

### Strongest negative result

No adaptation is universally best. Health anchoring strongly corrects operating-condition shift, but AdaBN is substantially better for FD003's new fault mode. The intervention must match the mechanism of shift.

## Statistics

### Why paired bootstrap?

All methods are evaluated on the same engines. Engines are resampled with replacement, and metric differences are computed within each replicate. Pairing preserves engine-level correlation.

### Why resample engines rather than windows?

Windows from one engine are strongly dependent. Treating them as independent would produce artificially narrow confidence intervals.

### How are seeds handled?

Per-engine errors are averaged across seeds before bootstrapping. Seed-to-seed standard deviation is reported separately; the bootstrap CI represents engine variation, not seed uncertainty.

## Likely interview questions

### Why not train on target subsets?

The experiment represents deployment into an unlabeled fleet with unseen conditions or fault modes. Target-label training would answer supervised transfer rather than label-free adaptation.

### Did target labels select the best adaptation?

Target labels are used retrospectively for benchmark evaluation. The project does not yet solve label-free adaptation selection; production use would need shift detection, prior regime knowledge, or delayed-label validation.

### Why clip RUL predictions?

The benchmark target lies in [0, 125]. Clipping enforces physical target bounds, although un-clipped predictions should also be logged to expose extreme extrapolation.

### What would you do next?

Sweep anchor length, reverse transfer directions, add shift detection and automatic strategy selection, evaluate weighted or online conformal methods, increase seeds and ensemble size, and test real fleets with sensor faults and censoring.

## Code walkthrough

1. `src/rul/data.py`: labels, sensors, splitting, normalization, windows.
2. `src/rul/models.py`: hybrid architecture and early stopping.
3. `src/rul/tta.py`: AdaBN.
4. `src/rul/uq.py`: UQ and conformal intervals.
5. `src/rul/experiment.py`: orchestration and artifacts.
6. `src/rul/stats.py`: paired engine bootstrap.
7. `results/three_seeds/summary.md`: complete results.

---

# 2. prefopt-repro

Repository: <https://github.com/Ghazaleh-Ramezani/prefopt-repro>

## Evidence boundary

The repository implements the SFT → reward model → DPO/GRPO → evaluation → diagnostics framework, with PEFT, DeepSpeed, optional vLLM, configurations, and tests. It currently does not publish completed full-grid run artifacts or a populated top-level `RESULTS.md`. Say **“I implemented a reproducible pipeline and designed controlled ablations”**, not **“I demonstrated a specific DPO/GRPO gain”** until real results are published.

## 30-second answer

> I built a reproducible preference-optimization framework supporting SFT, scalar reward modeling, DPO, and GRPO with Transformers, TRL, PEFT, DeepSpeed ZeRO-3, and optional vLLM rollouts. The focus is trustworthy evaluation: position-swapped pairwise judging, bootstrap confidence intervals, multi-seed generation, and reward-hacking diagnostics including length–reward correlation, KL from the SFT reference, Goodhart gap, diversity, and stylistic artifacts. The framework is implemented and tested; I do not claim final ablation outcomes until full run artifacts are published.

## 2-minute answer

> The project asks how KL strength, GRPO group size, and reward-model quality affect policy quality and proxy over-optimization. SFT creates the baseline and reference adapter, the reward model learns pairwise preferences, DPO optimizes directly from chosen/rejected pairs, and GRPO samples groups of completions and uses standardized group-relative rewards without a value network.
>
> Each optimized response is compared against the SFT response twice, once in each order. A win counts only when the judge is consistent across both orders; positional flips become ties and are reflected in swap consistency. I bootstrap prompts, repeat generation across seeds, compare reward-model preference with held-out LLM-judge preference, track per-token KL, and measure length correlation, distinct 4-grams, and formatting tendencies. These diagnostics separate external quality gains from learning the reward model's shortcuts.

## Pipeline

| Stage | Input | Objective | Output |
|---|---|---|---|
| SFT | Prompt–response demonstrations | Token cross-entropy | Baseline/reference adapter |
| Reward model | Prompt + chosen/rejected responses | Pairwise ranking loss | Scalar sequence scorer |
| DPO | Preference pairs + reference | Direct preference objective | Optimized adapter |
| GRPO | Prompts + sampled groups + reward | Group-relative policy optimization | RL-optimized adapter |
| Evaluation | Policy and SFT responses | Swapped judging + bootstrap | Win rate, CI, consistency |
| Diagnostics | Samples and checkpoints | Proxy-failure tests | KL, RM score, diversity/style metrics |

## SFT

For response tokens \(y\) conditioned on prompt \(x\):

\[
\mathcal{L}_{SFT}=-\sum_t\log\pi_\theta(y_t\mid x,y_{<t}).
\]

SFT establishes instruction-following behavior and a stable reference before preference optimization.

## Reward model

For chosen response \(y_w\) and rejected response \(y_l\):

\[
\mathcal{L}_{RM}=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
\]

Held-out pairwise accuracy tests ranking performance, but it does not prove that the reward generalizes to policy-generated outputs.

### Why ablate RM data fraction?

Using 10%, 50%, and 100% of preference pairs creates controlled reward-quality levels and tests whether downstream behavior depends on reward quality rather than only the optimizer.

## DPO

Define the policy-reference preference margin:

\[
\Delta_\theta=
[\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x)]
-[\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x)].
\]

Then:

\[
\mathcal{L}_{DPO}=-\mathbb{E}[\log\sigma(\beta\Delta_\theta)].
\]

### What does beta control?

Beta controls preference strength relative to the reference. Its realized effect must be checked through measured KL. Equal numeric beta values in DPO and GRPO do not imply equal policy movement.

### Why is the reference memory-efficient with PEFT?

The policy is a LoRA adapter over a frozen base. TRL can disable the adapter to recover reference behavior, avoiding a second full copy of base weights.

## GRPO

For \(G\) sampled completions with rewards \(r_i\):

\[
A_i=\frac{r_i-\bar r}{s_r+\epsilon}.
\]

GRPO increases the probability of above-group-average completions while constraining drift from the reference. Group-relative normalization removes the need for a learned value network but still requires a reward signal.

### Why does group size matter?

Larger groups can stabilize within-prompt ranking and advantages, but rollout cost grows approximately linearly with group size. If all rewards are nearly equal, the learning signal remains weak.

### Why use vLLM?

GRPO is generation-heavy. vLLM improves rollout throughput using optimized batching and KV-cache management. Without it, generation can dominate step time.

## DPO versus GRPO

| Dimension | DPO | GRPO |
|---|---|---|
| Data | Offline chosen/rejected pairs | Online sampled completion groups |
| Reward | Implicit in pairs | Explicit RM/reward functions |
| Value model | No | No |
| Rollout cost | Low | High |
| Main risk | Dataset artifacts | Reward hacking and optimization instability |
| Key controls | Beta, loss type, smoothing | Beta, group size, RM quality, temperature |

## LoRA and systems

LoRA learns:

\[
W=W_0+\frac{\alpha}{r}BA.
\]

It reduces trainable parameters and optimizer state but limits policy capacity. ZeRO-3 partitions parameters, gradients, and optimizer states across workers; it primarily enables memory scaling and may add communication overhead. Gradient checkpointing trades additional recomputation for lower activation memory.

## Position-swapped evaluation

Every pair is judged in A/B and B/A order. A policy win is accepted only when both orders identify the same semantic winner. A positional flip is scored as a tie and lowers `swap_consistency`.

Position swapping targets order bias; it does not remove judge verbosity bias, self-preference, knowledge gaps, or stylistic preference.

### Why bootstrap?

Prompt resampling estimates finite-sample uncertainty without assuming normal per-prompt outcomes. Report win rate with prompt count, CI, baseline, judge, decoding parameters, and seed variability.

### Why multiple seeds?

Nonzero-temperature generation is stochastic. Seed spread captures generation variability, while the bootstrap CI captures prompt-sampling uncertainty.

## Reward-hacking diagnostics

### Length–reward correlation

Spearman correlation between completion length and RM score checks whether verbosity is rewarded. High correlation is diagnostic but not causal; inspect task-conditioned examples.

### KL from reference

The code estimates mean token-level forward KL on policy-generated continuations:

\[
D_{KL}(\pi_\theta\|\pi_{ref})=
\mathbb{E}_{y\sim\pi_\theta}
\left[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right].
\]

Rising KL while external win rate plateaus suggests inefficient drift or over-optimization.

### Goodhart gap

\[
\text{Goodhart gap}=
\text{RM-judged win rate}-
\text{external-judge win rate}.
\]

A widening positive gap across checkpoints suggests increasing optimization of the proxy without corresponding external improvement.

### Distinct-4 and style tests

Distinct-4 detects repetition through unique 4-gram ratio. Style tests track headings, bullets, enthusiasm phrases, and response length. They are screening tools and cannot detect all factual or semantic failures.

### Strongest reward-hacking evidence

Use a checkpoint trajectory: learned reward rises, external quality flattens or falls, KL grows, Goodhart gap widens, and qualitative samples reveal the exploited shortcut.

## Likely interview questions

### Why train an RM if DPO does not require it?

DPO consumes offline preference pairs directly; GRPO needs explicit rewards for sampled completions. The RM also enables reward-quality ablations and Goodhart analysis.

### Is evaluating GRPO with its training RM circular?

Yes. It measures proxy optimization, not external quality. It is useful as one axis of a Goodhart analysis, never as the headline result.

### Why use one LLM judge?

Cost and project scope. Position swapping tests consistency, not validity. Stronger evaluation needs humans, multiple judge families, and agreement analysis.

### Why not PPO?

PPO requires a value model and more stabilization components. GRPO removes the critic through group-relative baselines, while DPO is a lower-cost offline alternative. PPO remains a useful future baseline.

### What happens if all GRPO rewards are equal?

Centered advantages approach zero, producing little policy-gradient signal. Numerical stabilization avoids division by zero but cannot create preference information.

### What must be added before claiming empirical results?

Publish resolved configs, environment metadata, run artifacts, checkpoint metrics, all seed outputs, judge records, bootstrap intervals, RM held-out accuracy, compute budget, failure examples, and a populated `RESULTS.md`.

## Interview traps

- Do not claim a completed DPO/GRPO performance improvement without run artifacts.
- Do not equate reward score with human preference.
- Do not say position swapping eliminates all judge bias.
- Do not treat configured beta as measured KL.
- Do not say GRPO requires no reward signal.
- Do not say ZeRO-3 always makes training faster.
- Do not report win rate without CI, sample size, seed variation, baseline, judge, and decoding settings.

## Code walkthrough

1. `src/prefopt/config.py`: typed config and validation.
2. `src/prefopt/data.py`: stage-specific datasets.
3. `src/prefopt/modeling.py`: model, tokenizer, quantization, LoRA.
4. `train_sft.py` and `train_rm.py`: baseline and RM.
5. `train_dpo.py`: DPO and reference handling.
6. `train_grpo.py`: groups, rewards, vLLM, divisibility check.
7. `evaluate.py`: generation, swapped judging, bootstrap.
8. `diagnostics.py`: KL, length correlation, diversity, style.
9. `report.py`: aggregation and cards.

---

# 3. action-isa

Repository: <https://github.com/Ghazaleh-Ramezani/action-isa>

## Opening answer

> This repository contains code and per-task records for a study separating structural from parametric decisions in sub-billion-parameter tool use. Analysis scripts regenerate reported statistics from released records, block-aware analysis handles grouped dependence, and preregistered predictions are checked against a SHA-256 digest. Statistical reproduction is CPU-only; rerunning model arms requires a GPU.

### Likely questions

- **Structural versus parametric:** structure selects tool/schema/action form; parameters fill concrete argument values.
- **Why per-task records:** aggregate metrics hide failures and prevent paired or block-aware reanalysis.
- **Why a digest:** it commits to preregistered predictions before evaluation.
- **Why can regenerated files hash differently:** timestamps change bytes even when prediction values match.
- **Why block-aware inference:** related tasks are correlated; treating them as independent understates uncertainty.

---

# 4. materials-rag

Repository: <https://github.com/Ghazaleh-Ramezani/materials-rag>

## Opening answer

> I built an offline-runnable scientific RAG pipeline with document ingestion, metadata-aware chunking, BM25 and FAISS retrieval, reciprocal-rank fusion, optional cross-encoder reranking, semantic caching, cited generation, FastAPI serving, Docker packaging, and Recall@k/MRR evaluation. The tiny bundled corpus proves reproducibility, not production retrieval quality.

### Why hybrid retrieval?

BM25 captures exact terminology, formulas, and rare material names. Dense retrieval handles semantic paraphrases. Hybrid retrieval improves robustness across both query types.

### RRF

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+rank_r(d)}.
\]

RRF combines rank order without requiring BM25 and cosine scores to be calibrated on the same numeric scale.

### Why rerank?

A cross-encoder jointly reads the query and candidate and is more precise but too expensive for a full corpus. Retrieve cheaply first, then rerank a small candidate set.

### Evaluation limitation

The included benchmark has four synthetic documents and 12 questions, so near-perfect metrics are expected. Strong claims need a larger real corpus, hard negatives, leakage checks, citation evaluation, and answer-faithfulness analysis.

---

# 5. long-context-llm-experiments

Repository: <https://github.com/Ghazaleh-Ramezani/long-context-llm-experiments>

## Opening answer

> This project combines 4-bit LoRA adaptation with needle-in-a-haystack evaluation. LoRA targets attention query and value projections with rank 16 and alpha 32, while evaluation maps retrieval accuracy over context length and needle depth to expose positional failures such as lost-in-the-middle behavior.

### Key points

- LoRA update: \(W=W_0+(\alpha/r)BA\); rank controls capacity and \(\alpha/r\) scales the update.
- `q_proj` and `v_proj` affect attention routing and aggregation while keeping adapters small.
- NF4 stores frozen base weights in a 4-bit format designed for neural weight distributions.
- Needle-in-a-haystack tests positional retrieval, not full long-document reasoning or synthesis.

---

# 6. materials-mobo

Repository: <https://github.com/Ghazaleh-Ramezani/materials-mobo>

## Opening answer

> I implemented multi-objective Bayesian optimization for a seven-dimensional CNC/CNF/rGO synthesis space, using Gaussian-process surrogates and qLogEHVI to trade off conductivity and tensile strength. The output is a Pareto set rather than a single universal optimum.

### Key points

- Bayesian optimization is useful when experiments are expensive and objectives are black-box.
- A Pareto point cannot improve one objective without worsening another.
- Hypervolume measures dominated objective-space volume relative to a reference point.
- qLogEHVI targets batched Pareto-front improvement with better numerical behavior for small improvements.
- A precise “40% fewer experiments” claim needs saved baselines, seeds, uncertainty, and a fixed protocol.

---

# 7. data-driven-materials-optimization

Repository: <https://github.com/Ghazaleh-Ramezani/data-driven-materials-optimization>

## Opening answer

> This is a small-data methodological demonstration connecting materials formulation variables with regression, interaction features, LASSO selection, and coefficient interpretation. The simplified anonymized dataset demonstrates workflow rather than a deployable materials model.

LASSO solves:

\[
\min_\beta\frac{1}{n}\|y-X\beta\|_2^2+\lambda\|\beta\|_1.
\]

With roughly 30 samples, major risks are high variance, unstable selection, and optimistic evaluation. Use training-only preprocessing pipelines, repeated or nested cross-validation, simple baselines, and bootstrap stability analysis.

---

# 8. evidencerag

Status: private and under active development.

Safe wording:

> I am developing a production-oriented scientific RAG system around asynchronous ingestion, hybrid retrieval, reranking, citation-grounded generation, evaluation, CI/CD, and observability. It remains private because reproducibility and measured end-to-end quality are release criteria.

Distinguish it from `materials-rag`: Materials-RAG is a compact offline-runnable portfolio implementation; EvidenceRAG targets service lifecycle, asynchronous jobs, monitoring, and production reliability.

---

# Cross-project questions

## How do you select metrics?

- RUL: RMSE, NASA score, coverage, width, calibration.
- Preference optimization: external win rate, CI, seeds, KL, hacking diagnostics.
- Retrieval: Recall@k, MRR, reranking gain, latency.
- RAG generation: correctness, faithfulness, citation quality, abstention.
- MOBO: hypervolume, sample efficiency, Pareto-front quality.

No single metric establishes reliability.

## What is your reproducibility standard?

Record code SHA, environment, resolved config, random seed, data version, raw predictions, aggregate metrics, and scripts regenerating tables. Use synthetic tests for mathematical logic and preserve auditable benchmark outputs.

## How do you avoid overclaiming?

- **Implemented:** code exists.
- **Executed:** a run produced artifacts.
- **Observed:** a metric appeared under a specified protocol.
- **Established:** stable across seeds, baselines, confidence intervals, and alternative explanations.
- **Generalized:** holds across datasets, model families, and deployment conditions.

## ML debugging sequence

1. Validate schema, labels, units, splits, and leakage.
2. Overfit a tiny batch.
3. Compare with a trivial baseline.
4. Inspect per-example predictions.
5. Add component tests and ablations.
6. Check seeds, precision, libraries, and hardware.
7. Profile latency and memory after correctness.

# Final honesty checklist

- `shift-aware-rul`: completed results exist; always name the subset, adaptation, and UQ method.
- `prefopt-repro`: framework exists; do not invent final DPO/GRPO gains.
- `materials-rag`: pipeline is runnable; tiny sample metrics are not production evidence.
- `long-context-llm-experiments`: positional retrieval is not general long-context reasoning.
- `materials-mobo`: quantitative efficiency claims require exact artifacts.
- `data-driven-materials-optimization`: small-data methodology demo.
- `evidencerag`: private active development; keep external claims proportional to completed evidence.
