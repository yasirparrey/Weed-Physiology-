# Stanford CME 295 — Transformers & Large Language Models

### Complete Reading Notes for the Autumn 2025 Lecture Series

*Instructors: Afshine Amidi & Shervine Amidi*

---

## How to read this document

These notes are written to **replace watching the nine lecture videos**. Every
section follows the order in which the material is actually taught, and each
technical idea is written out in full prose rather than as bullet-point slide
fragments — because a slide that says "RoPE. Rotate query and key vectors" is
useless without the twenty sentences the lecturer says on top of it.

Three kinds of text appear throughout, and it is worth knowing which is which:

- **Plain body text** is the lecture content itself: definitions, mechanisms,
  formulas, numbers, paper references.

- **Boxes labelled "Intuition"** are added explanation. They exist to answer
  the question "*but why does that work?*", which is usually the part that
  makes a technique stick in memory.

- **Boxes labelled "Watch out"** flag the places where notation is confusing,
  where two similar-sounding ideas are actually different, or where a naive
  reading of the slide would leave you with a wrong mental model.

Formulas are written in linear Unicode notation (for example
$\mathrm{softmax}\!\big(QK^{\top}\!/\sqrt{d_k}\big)V$) so that they read cleanly without needing rendered
LaTeX. Where the original slides showed a diagram, the diagram is described in
words, because the description is what you actually need in order to reason
about the mechanism.

## Prerequisites the course assumes

You need machine-learning basics (loss functions, gradient descent,
backpropagation, softmax, cross-entropy) and linear algebra (matrix
multiplication, dot products, the idea of a low-rank factorisation). Nothing
beyond that. No prior NLP knowledge is assumed — the course builds from
tokenisation upwards.

## The shape of the whole course

The nine lectures form one long argument, and it helps enormously to see the
skeleton before the detail:

1. **Lectures 1–2** build the Transformer from nothing. Why we need
   tokenisation, why static word vectors are insufficient, why recurrence is
   slow, what attention actually computes, and how the encoder–decoder stack is
   assembled. Then the engineering that made the architecture practical:
   position encodings, normalisation placement, cheaper attention variants, and
   BERT as the canonical encoder-only model.

2. **Lectures 3–4** turn a Transformer into a *Large Language Model*.
   Decoder-only architectures, mixture-of-experts, how text is actually sampled
   at inference, what makes inference fast, and then how these models are
   trained at scale — pretraining, parallelism, precision, supervised
   finetuning, and LoRA/QLoRA for people without a GPU cluster.

3. **Lectures 5–6** are about *alignment* and *reasoning*. Preference data,
   reward models, RLHF with PPO, the DPO shortcut, and then how the same RL
   machinery — with verifiable rewards and GRPO instead of PPO — produces
   reasoning models like DeepSeek-R1.

4. **Lectures 7–8** are about *systems*. Retrieval-augmented generation, tool
   calling, MCP, agents and the ReAct loop; then the hardest practical problem
   of all, evaluating any of it: rule-based metrics, LLM-as-a-judge, agentic
   failure modes, and benchmarks.

5. **Lecture 9** steps back: Transformers outside text (ViT, VLMs), diffusion
   language models as a genuine alternative to autoregressive decoding, and
   where the field appears to be heading.

A single running example is used across all nine lectures — a *cute teddy bear
that is reading* — and it is genuinely helpful. Every mechanism in the course
is demonstrated on some variant of that sentence, so when you see
`"A cute teddy bear is reading."` you should read it as "here comes a concrete
worked example".

## Notation used consistently throughout

| Symbol | Meaning |
|---|---|
| $V$ | vocabulary size (number of distinct tokens the model knows) |
| $n$ | sequence length, in tokens |
| $d_{\mathrm{model}}$ | width of the model's residual stream / embedding dimension |
| $d_k$, $d_v$ | per-head key and value dimensions |
| $d_{\mathrm{FF}}$ | hidden width of the feed-forward sub-layer |
| $h$ | number of attention heads |
| $N$ | number of stacked encoder or decoder layers |
| $Q$, $K$, $V$ | query, key and value matrices inside an attention layer |
| $\theta$ | model parameters |
| $\pi_\theta$ | the model viewed as a policy (used from Lecture 5 onwards) |
| $T$ | sampling temperature |
| $\beta$ | strength of a KL / regularisation term |

