# Lecture 9 — Current Trends

**Video length: 1h52.** Covered here: a full recap of the quarter, Transformers
applied beyond text (Vision Transformer, vision-language models, diffusion
transformers), diffusion LLMs as an alternative to autoregressive decoding,
cross-pollination between modalities, open foundational questions, hardware
directions, current and expected applications, ongoing challenges, and how to
stay current.

---

## 9.1 Recap of the quarter

The lecture walks back through all eight previous lectures. Consolidated:

**Lecture 1 — Transformers.** Tokenising `A cute teddy bear is reading.`,
embeddings, the query/key/value attention mechanism, and the full
encoder–decoder architecture from *Attention Is All You Need*.

**Lecture 2 — Transformer-based models & tricks.** RoPE (rotating queries and
keys so that attention sees only relative distance), GQA (*Training Generalized
Multi-Query Transformer Models from Multi-Head Checkpoints*, Ainslie et al.,
2023), and the encoder-only / decoder-only / encoder-decoder taxonomy.

**Lecture 3 — Large Language Models.** Mixture of experts replacing the FFNN with
`FFNN₁...FFNNₙ` plus a gate `G`, routed per token; the Mixtral analysis of what
experts specialise in; sampling, temperature and decoding strategies.

**Lecture 4 — LLM training.** Kaplan scaling laws; the Chinchilla
compute-optimal law; FlashAttention exploiting the HBM/SRAM hierarchy (*big and
slow* versus *small and fast*); and the lifecycle Pretraining → Finetuning →
Preference tuning.

**Lecture 5 — LLM tuning.** The RL mapping (LLM = agent, tokens = environment,
input so far = state, next token = action, next-token probability = policy, human
preference = reward); Bradley–Terry; the RLHF loop with a frozen reward model
and a trained policy; the PPO objective balancing *maximise rewards* against
*don't deviate too much from the old/base model*.

**Lecture 6 — LLM reasoning.** Question → LLM → reasoning chain → answer; chain
of thought; GRPO versus PPO; the DeepSeek-R1 recipe.

**Lecture 7 — Agentic LLMs.** RAG; tool calling (prompt + function API → LLM →
arguments to run with the API); agents and the ReAct loop.

**Lecture 8 — LLM evaluation.** LLM-as-a-Judge (prompt + model response +
criteria → rationale + score); the four benchmark categories with MMLU, AIME/PIQA,
SWE-bench and HarmBench.

---

## 9.2 Transformers beyond text

### Why they generalise at all

**Context.** Introduced in 2017 for machine translation; relies on
self-attention; built on the concepts of query, key and value.

**Benefits.** **Weaker inductive biases** and therefore **more
generalisability**.

> **Intuition — "weaker inductive biases" is the whole answer, and it is a
> double-edged one.** A convolutional network has strong built-in assumptions:
> locality (nearby pixels are related) and translation equivariance (a cat is a cat
> wherever it appears). Those assumptions are correct for images, which is why CNNs
> learn efficiently from modest datasets. An RNN assumes sequential locality. A
> Transformer assumes almost nothing — every element can interact with every other,
> and the relationships are learned from data. On small datasets this is a
> liability: the model must learn from scratch what a CNN was given for free. On
> **large** datasets it becomes a decisive advantage, because the model can learn
> the *actual* structure of the data rather than being constrained by a hand-designed
> approximation of it. This is why Transformers took over everywhere at roughly the
> same time — the datasets crossed the threshold where flexibility beats built-in
> assumptions.

### Attention for images

The attention mechanism does not care what its inputs mean. In text, tokens are
word pieces. **For images**, the tokens can be **patches**. Everything else — the
query/key/value machinery, the softmax, the value averaging — is unchanged.

### Vision Transformer (ViT)

**Dosovitskiy et al., 2020, *An Image is Worth 16×16 Words: Transformers for
Image Recognition at Scale*.**

**The end-to-end example from the lecture:**

1. Take the image and cut it into fixed-size **patches** of `P × P` pixels
   (16×16 in the paper).

2. **Flatten** each patch into a vector of length `P·P·C` (C = colour channels)
   and pass it through a **linear projection** to dimension `D`. Each patch is
   now a token.

3. **Prepend a `[CLS]` token**, exactly as in BERT.
4. Add the patch **embedding** and the **position embedding** → *position-aware
   embeddings*.

5. Feed the sequence into the **encoder** (the ViT architecture is an
   encoder-only Transformer).

6. Read the **encoded embedding at the `[CLS]` position**, pass it through a
   **feed-forward network**, and get the class: `teddy bear`.

> **Intuition — notice that this is BERT with a different tokeniser.** That is not
> a glib comparison; it is exactly what ViT is. `[CLS]` for pooling, learned
> position embeddings, a bidirectional encoder stack, a classification head. The
> only genuinely new component is the patch projection that turns a grid of pixels
> into a sequence of vectors. Patching is also what makes it affordable: a 224×224
> image has 50,176 pixels, and attention over that many tokens is impossible at
> `O(n²)`, but 16×16 patches reduce it to 196 tokens, which is an ordinary
> sequence length. The paper's central finding was that ViT *underperforms* CNNs on
> ImageNet-scale data and *outperforms* them once pretrained on much larger
> datasets — the inductive-bias trade-off, measured.

### Vision Language Models (VLM)

**Goal:** *"How cute is this teddy bear?"* + an image → *"Very cute!"*

Two methods:

**Method 1: recycle the decoder-only architecture** (*Visual Instruction
Tuning* — LLaVA, Liu et al., 2023). Encode the image into a sequence of vectors
(with a vision encoder such as ViT/CLIP), project them into the LLM's embedding
space, and **concatenate them with the text tokens**. The LLM then treats image
patches as if they were just more tokens in its context — a "typical" LLM,
unchanged, fed a mixed sequence.

**Method 2: leverage a cross-attention layer** (as in *The Llama 3 Herd of
Models*, 2024). Keep the text decoder as it is and insert **cross-attention**
layers that attend from the text stream into the image representation, exactly as
the original Transformer's decoder attended into its encoder.

> **Intuition — the trade-off between the two.** Method 1 is simpler and cheaper
> to build: no architectural change, so you can bolt vision onto an existing LLM
> with a small projection layer and some instruction tuning. Its cost is that image
> tokens **consume context window** — hundreds or thousands of tokens per image —
> and they participate in self-attention with everything else, which is expensive.
> Method 2 keeps image representations *outside* the main sequence, so the text
> context is untouched and cost scales better with many or high-resolution images;
> the price is new parameters and a modified architecture, meaning you cannot simply
> upgrade a text model in place. Method 1 dominated the open ecosystem for exactly
> that reason.

### Where else Transformers appear

- **Text generation** — CME 295's subject.
- **Vision understanding** — Vision Transformer (ViT).
- **Image generation** — Diffusion Transformer (DiT), MultiModal-DiT (MM-DiT).
- **And many others** — recommendation, speech.

**Transformers are actually used in a lot of different places.**

---

## 9.3 Diffusion LLMs

### The limitation being attacked

The current approach is **AutoRegressive Modeling (ARM)**: predict one token,
append it, predict the next.

**Problem: inference-time generation is not parallelisable** — *although training
is*.

> **Intuition — this is the same complaint that killed RNNs, resurfacing.**
> Lecture 1's argument against recurrence was "you cannot parallelise along the
> sequence". Transformers fixed that **for training**, where all positions are
> processed at once with a causal mask. But at *inference* an autoregressive model
> is irreducibly sequential: token 100 cannot be computed before token 99 exists.
> Every trick in Lecture 3 — KV caching, speculative decoding, multi-token
> prediction — is a way of *softening* this constraint without removing it.
> Diffusion LLMs attack the constraint itself.

### Alternative in the news

The lecture notes the commercial momentum: an Inception announcement (TechCrunch,
November 2025), a diffusion-based LLM showcased by **Google at I/O** (May 2025),
and a **ByteDance** announcement (July 2025). This is no longer purely academic.

### Background: how image diffusion works

**Intuition given in the lecture:**

- noise is easy to sample,
- the transformation is learned,
- it is mathematically well-defined.

**And the analogy — building a sculpture:**

> *"The sculpture is already complete within the marble block, before I start my
> work. It is already there, I just have to chisel away the superfluous
> material."* — Michelangelo

**High-level goal:** learn some transformation that goes from noise to the
desired data distribution.

**The mechanism** (*Denoising Diffusion Probabilistic Models*, Ho et al., 2020):

1. **Forward process** — add noise to an image, progressively, until it is pure
   noise. This requires no learning; it is a fixed schedule.

2. **Reverse process** — **learn to denoise** it. Train a network to take a noisy
   image and predict a slightly less noisy one. At generation time, start from
   pure noise and run the learned reverse process repeatedly.

### From images to text

The translation is one substitution: where images use **noise**, text uses
**MASK**.

**1. Forward process — mask tokens with some probability:**

```
A teddy bear is reading   →   A MASK bear is MASK
```

**2. Reverse process — learn to unmask tokens:**

```
A MASK bear is MASK   →   A teddy bear is reading
```

**MDM = Masked Diffusion Model.** At generation time you start from an
all-`MASK` sequence of the desired length and iteratively unmask:

```
MASK MASK MASK MASK MASK   ──Diffusion LLM──→   A teddy bear is reading
```

**Decoding is done in fewer forward passes**, because each pass can fill in
**many** positions at once, in any order.

*(Suggested readings: Lou et al., 2023, on discrete diffusion by estimating
data-distribution ratios; Sahoo et al., 2024, *Simple and Effective Masked
Diffusion Language Models*; Nie et al., 2025, *Large Language Diffusion
Models*.)*

### Discussion

**Advantages:**

- **better suited for some tasks**,
- **~10× output tokens per second compared to ARM**.

**Challenges and current work:**

- **adapting ARM-based techniques** — the entire toolbox from the previous eight
  lectures (KV caching, RLHF, GRPO, tool calling) assumes autoregressive
  generation and must be reinvented,

- **performance** — quality does not yet match the best autoregressive models.

> **Intuition — where the speedup comes from, and what it costs.** An ARM needs
> `n` forward passes to produce `n` tokens. A diffusion LM needs however many
> denoising steps you choose — typically far fewer than `n` — and each step fills
> in many positions. That is the 10× figure.
>
> There is a second, more conceptual advantage. Autoregressive generation commits
> irrevocably: once token 5 is emitted it can never be revised, even if token 40
> reveals it was a mistake. Diffusion generation is **iterative refinement over the
> whole sequence**, so it can revisit and change earlier positions. That should be
> a natural fit for tasks where global structure matters more than left-to-right
> flow — code editing, infilling, constrained generation, planning.
>
> The cost is that independence assumptions inside each denoising step make it hard
> to model token dependencies as sharply as a chain rule factorisation does, which
> is where the current quality gap comes from. Note also how deeply the ecosystem
> assumes autoregression: KV caching has no direct analogue when every position may
> change at every step, and even "streaming the response to the user" is a
> different proposition when the text materialises all at once rather than
> left to right.

---

## 9.4 Cross-pollination between modalities

**Trend: modalities trade ideas about concepts that work.** The lecture's examples
are deliberately symmetric:

**Architecture:**

- **"Diffusion" training for text output** — an image technique moving to text
  (*Large Language Diffusion Models*, Nie et al., 2025).

- **Transformers to deal with images** — a text architecture moving to images
  (*Scalable Diffusion Models with Transformers* — DiT, Peebles et al., 2022).

**Input representation: DeepSeek-OCR** (*Contexts Optical Compression*, Wei et
al., 2025) — render text **as an image** and feed it to a vision encoder, because
a picture of a paragraph can be represented in fewer tokens than the paragraph
itself. Text becoming an image in order to be cheaper as text.

**Tricks: RoPE variants** — e.g. Multimodal Scalable RoPE for images
(*Qwen-Image*, Wu et al., 2025), adapting the rotary position idea from 1-D
sequences to 2-D grids.

> **Intuition.** Notice the pattern: the *architecture* went from text to images
> (DiT), the *training objective* went from images to text (diffusion LMs), the
> *input representation* went from text to images and back (OCR compression), and
> the *positional trick* generalised from 1-D to 2-D. The boundary between
> modalities is dissolving — increasingly there is one toolbox, and modality is
> just a choice of how you tokenise. DeepSeek-OCR is the most delightfully
> counterintuitive item: it says that for long documents, an image of the text can
> be a *more efficient* representation than the text, which inverts everything you
> would assume about which modality is compact.

---

## 9.5 Back to the basics: foundational research is still open

**Foundational research is still very much ongoing.**

**Microscopic level — design choices still vary widely across papers:**

- optimisers: **AdamW vs. MuonClip** vs. others (*Kimi K2: Open Agentic
  Intelligence*, Kimi Team, 2025),

- normalisation — where, and which kind, or whether at all,
- MHA/MQA/GQA design choices,
- activation functions,
- MoE or not,
- number of layers.

**"Fuel" — future streams of high-quality data?** With the reference to *The
Curse of Recursion: Training on Generated Data Makes Models Forget* (Shumailov et
al., 2023).

**Taking a step back — is the Transformer the best architecture?**

> **Intuition — two things worth taking seriously here.** First, the fact that
> labs still disagree about the optimiser and the normalisation scheme in 2025
> tells you that the recipe is empirical, not derived. There is no theory saying
> AdamW is right; it is what worked. That should make you appropriately sceptical
> of confident claims about *why* any of this works.
>
> Second, the data question may be the real ceiling. Scaling laws say loss keeps
> falling with more data — but high-quality human text is finite and largely
> already consumed. The obvious workaround, training on model-generated text, has
> a documented failure mode: **model collapse**, where each generation trained on
> the previous one loses distributional tails and drifts towards the mean, forgetting
> the rare and the specific. It is a photocopy of a photocopy. Whether synthetic
> data can be made safe (by filtering, by verification, by mixing with real data)
> is one of the genuinely open questions in the field, and it interacts with
> Lecture 6's rejection sampling, which *is* a form of verified synthetic data
> generation and does work.

---

## 9.6 New frontiers and hardware

### From best performance to best trade-off

**Until now:** the focus was on **best performance**.
**More and more:** the focus is on the **best quality/cost trade-off** — i.e.
moving along the **Pareto frontier** from Lecture 8 rather than only pushing its
upper-right corner.

### Hardware optimisation: analog in-memory computing

*Leroux et al., 2025, "Analog in-memory computing attention mechanism for fast
and energy-efficient large language models".*

**Observation:** current GPUs optimise for matrix-vector and matrix-matrix
operations.

**Challenges — Transformer blocks have special needs:**

- **attention requires frequent KV reads/writes**,
- **memory movement dominates cost on GPUs**.

**Idea: "natively" support attention operations in hardware:**

- **store the KV cache in dedicated cells**,
- **use analog signals to model the computations**.

**Results: up to ~100× latency and ~70,000× energy savings** with respect to
NVIDIA's H100.

> **Intuition — this is FlashAttention's lesson taken to its logical conclusion.**
> Lecture 4 established that attention is bound by *memory movement*, not
> arithmetic, and FlashAttention's response was to restructure the algorithm to
> move less data. Analog in-memory computing's response is more radical: **don't
> move the data at all — compute where it is stored.** A crossbar of analog memory
> cells performs a matrix-vector multiply as a physical consequence of applying
> voltages, in one step, with no fetch. Store the KV cache in those cells and the
> dot-product-and-weighted-sum that *is* attention happens in place.
>
> The 70,000× energy figure should be read as an upper bound from a research
> prototype on a specific operation, not a product claim. Analog computing has real
> obstacles — limited precision, device drift, difficulty of writing values — which
> is why it has repeatedly failed to displace digital. But the argument for why
> attention specifically is a good fit is sound, and energy is becoming the binding
> constraint on the whole field, which is why this appears in a lecture on trends.

---

## 9.7 Applications: today and tomorrow

**Daily life has already dramatically changed. Key applications today:**

- **coding** ("true" coding, and text-to-query),
- **general conversational assistant** (common facts, web search),
- **creativity**,
- **learning**.

**Tomorrow:** democratisation of existing agents — e.g. the **Google Workspace
Studio** launch (December 2025), *"Automate everyday work with AI agents"*.

**Near term:** the **browser** — e.g. OpenAI's **Atlas** launch (October 2025).
Looking further: an **OS-level LLM**?

**Long term:** *"truly" autonomous agents with vast responsibilities*?

**Impossible?** *Actually useful customer service (finally?)* — the instructors'
joke, and a fair one.

### Ongoing challenges

Top of mind:

- **fixed weights, not updated** (≠ continuous learning),
- **"hallucinations"**,
- **personalisation**,
- **interpretability**,
- **safety**.

> **Intuition — why "fixed weights" is listed first, and why it is the deepest
> item on the list.** Everything else in this course works *around* the fact that a
> model's parameters are frozen at the end of training. RAG works around it by
> injecting knowledge at inference. Tool calling works around it by fetching live
> information. Long contexts work around it by holding more in working memory. But
> none of these is *learning*: nothing the model encounters in a conversation
> changes what it knows tomorrow. A human colleague who forgot everything at the
> end of each day, no matter how capable, would be limited in exactly this way.
> Genuine continuous learning would collapse the distinction between the
> pretraining/finetuning/inference stages that this entire course is organised
> around — which is why it is both the most consequential open problem and the
> hardest, since updating weights on the fly risks catastrophic forgetting and
> opens a large safety surface.

---

## 9.8 How to stay up to date

**Papers:**

- **arXiv > Computer Science > Computation and Language** (cs.CL),
- general ML venues (**NeurIPS, ICML, ICLR**) and NLP venues (**ACL, EMNLP**).

**Code:**

- authors' **GitHub** repositories, linked from their papers,
- **Hugging Face "trending papers"** — for browsing state-of-the-art datasets and
  methods.

**Miscellaneous:**

- **Twitter/X** — academics and industry leaders,
- **YouTube** — theoretical (Two Minute Papers, Yannic Kilcher) and practical
  (Google Developers, HuggingFace, Andrej Karpathy),

- **company/academia technical papers and blogs** — Amazon Science, Anthropic,
  Apple ML, Google DeepMind, Meta AI, Microsoft AI, OpenAI, Stanford NLP, and
  others.

### Course materials worth keeping

- **VIP Cheatsheet** —
  `github.com/afshinea/stanford-cme-295-transformers-large-language-models`,
  translated into 11 languages (Čeština, English, Español, فارسی, Français,
  Italiano, 日本語, 한국어, ไทย, Türkçe, 中文).

- **Super Study Guide: Transformers & Large Language Models** — the course
  textbook, `superstudy.guide`, which is the source of many of the figures used
  throughout the lectures.

---

## Closing synthesis

If you want a single thread through the nine lectures, it is this. The
Transformer replaced recurrence with attention because attention parallelises
(L1), and everything since has been the consequences of that choice: engineering
around its `O(n²)` cost and its position-blindness (L2), scaling it into a
language model and making inference affordable (L3), training it at a scale where
memory and precision dominate the design (L4), then discovering that a model
which merely predicts text is not a model that behaves well — so aligning it to
preferences (L5) and to verifiable correctness (L6) — then discovering that even
an aligned model is trapped inside its own weights, so connecting it to knowledge
and to actions (L7) — and then discovering that none of it can be improved
without a way to measure it (L8). Lecture 9's point is that each of these
decisions is still open, including the very first one.
