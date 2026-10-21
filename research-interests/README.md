# Research Interests — Efficient and Long-Context LLM Inference

Directions I am actively studying. Where I have written and run code against
one of these themes, it is linked in the last section below.

The unifying question across these threads: how do we make large language models
cheaper to run, stronger at reasoning, and more capable over long contexts —
without sacrificing quality?

## Threads

**Structured attention.** Kronecker- and other low-rank/structured
factorizations of attention and projection matrices to reduce compute and
KV-cache footprint. I have a concrete from-scratch study of Kronecker-factorized
linear layers with measured parameter/speed/quality tradeoffs:
→ [`../learning/kronecker-linear`](../learning/kronecker-linear/)

**Adaptive inference.** When and how model structure (e.g. factorization rank or
sparsity) might adapt at inference time to spend compute only where it helps.

**Planner–worker reasoning.** Decoupling planning from generation — for example a
planner that proposes intermediate reasoning structure and an autoregressive
worker that executes it — as a route to better reasoning efficiency.

**Retrieval-augmented inference.** Integrating external knowledge into the
inference path for factual consistency and long-context behavior. My applied RAG
work lives in [`../projects/materials-rag`](../projects/).

**Memory-augmented transformers.** Memory operations, the capacity/scalability
tradeoff, and implications for long-context post-training.

## Where these threads meet my code

- **Long-context behaviour:** response-only LoRA training and a strict
  needle-in-a-haystack harness in
  [long-context-llm-experiments](https://github.com/Ghazaleh-Ramezani/long-context-llm-experiments).
- **Structured decisions for small models:** typed action slots that separate
  structural from parametric choices lift Qwen3-0.6B multi-step tool use from
  0% to 87.5% without fine-tuning, in
  [action-isa](https://github.com/Ghazaleh-Ramezani/action-isa) (NeurIPS 2026
  SLM-Agents workshop).
- **Preference optimization and reward hacking:** SFT → RM → DPO/GRPO with
  position-swapped judging and per-checkpoint Goodhart analysis, in
  [prefopt-repro](https://github.com/Ghazaleh-Ramezani/prefopt-repro).
- **Structured linear layers:** the Kronecker study in
  [`../learning/kronecker-linear`](../learning/kronecker-linear/).
