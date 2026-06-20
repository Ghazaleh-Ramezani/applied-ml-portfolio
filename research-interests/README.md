# Research Interests — Efficient and Long-Context LLM Inference

Reading-driven notes and directions I am actively studying. **These are
interests and ongoing investigation, not shipped or independently validated
work.** Where I have written real code against one of these themes, it is linked
explicitly and lives under [`../learning/`](../learning/) as a labeled learning
project.

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

## Reading log

A short, dated list of papers I have read against each thread, with one-line
takeaways. *(Add entries as you go — this is what makes the section credible:
specific papers, specific takeaways, honestly your own notes.)*

| date | thread | paper | one-line takeaway |
|---|---|---|---|
| | structured attention | | |
| | planner–worker | | |
| | memory | | |
