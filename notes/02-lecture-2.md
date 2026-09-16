# Lecture 2 — Transformer-based Models & Tricks

**Video length: 1h47.** Covered here: why position information needs a better
treatment than "add a vector at the input", learned and sinusoidal encodings,
relative-position biases (T5, ALiBi), RoPE in detail, layer normalisation and
Pre-Norm vs Post-Norm vs RMSNorm, attention approximations (Longformer, sliding
windows, MQA/GQA), the three families of Transformer-based model, and a deep
dive on BERT including distillation, DistilBERT and RoBERTa.

The lecture opens by revisiting an attention map from the original Transformer
paper: two heads in layer 5 of 6 that appear to be doing **anaphora
resolution**, showing the attention emitted by the single word *its*. It is a
concrete demonstration that heads specialise, which was the claim made in
Lecture 1.

---

## 2.1 Position information, done properly

### The problem restated

Attention creates *direct links* between all pairs of tokens, and in doing so it
**loses position information entirely**. `bear teddy cute a` and `a cute teddy
bear` produce the same set of pairwise interactions.

### Attempt 1: learned absolute position embeddings

Add a learned, position-specific vector to each token embedding. Position 0 has
its own trainable vector, position 1 has another, and so on.

**Limitation:** you must fix a maximum length up front, and you **need to retrain
for longer sequences**. Position 4096 has no embedding if you only ever trained
up to 2048, and there is no principled way to invent one.

### Attempt 2: hardcoded sinusoidal position embeddings

Instead of learning the vectors, compute them:

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

Even dimensions get sines, odd dimensions get cosines, and the wavelength grows
geometrically with the dimension index `i` — fast oscillation in early
dimensions, very slow oscillation in later ones.

The lecture shows two plots: the **values of the embeddings** as a heat map over
(position × dimension), and the **similarity between positions**, i.e. the dot
product `PE(pos_a) · PE(pos_b)`, which peaks on the diagonal and decays smoothly
away from it.

**Benefit:** it **extends to any sequence length** — the formula is defined for
every `pos`, so nothing needs retraining.

> **Intuition — why sinusoids of many frequencies?** Think of it as a binary
> counter made continuous. In binary, the lowest bit flips every step, the next
> every two steps, the next every four; together, a handful of bits uniquely
> identify a large range of integers. The sinusoidal encoding does the same with
> continuous-valued "bits" at geometrically spaced frequencies. Two properties
> follow. First, uniqueness: no two positions within a huge range share an
> encoding. Second — and this is the important one — `PE(pos + k)` is a *fixed
> linear function* of `PE(pos)` for any offset `k`, because shifting a sinusoid
> is a rotation in its (sin, cos) plane. So a linear layer *can* in principle
> learn to compute relative offsets from these absolute encodings. The slow
> smooth decay of the similarity plot is what lets the model perceive
> "nearby" versus "far".

### The conceptual shift: we actually want *relative* position

The lecture then makes the key argument. Language cares about relative
positions, not absolute ones: the relationship between an adjective and the noun
it modifies is "immediately before", regardless of whether that happens at
position 3 or position 3,000. And since the place where positions actually
interact is the attention layer, the natural conclusion is:
**let's change the attention layer instead** of doctoring the inputs.

### Approach A: additive bias on the query–key scores

Add a bias term to the score matrix before the softmax:

```
score_ij = (q_i · k_j)/√d_k  +  b(i − j)
```

Two well-known instances:

- **T5 bias** (Raffel et al.) — the bias is **learned per head**, with relative
  distances bucketed (so that distances 1, 2, 3 get their own buckets while
  distances 100–127 share one). Each head learns its own preferred distance
  profile.
- **ALiBi** (Press et al., 2021, *Train Short, Test Long*) — the bias is
  **linear, deterministic, and not bounded**: `b(i − j) = −m·|i − j|`, with a
  different slope `m` per head. Nothing is learned; a token is penalised in
  proportion to how far away it is, and different heads have different
  "attention spans" set by their slopes.

> **Intuition.** ALiBi is essentially a learned-free, hard-coded recency prior.
> The title of the paper is the selling point: because the penalty is a simple
> function of distance with no parameters and no ceiling, a model trained on
> length 1024 still behaves sensibly at length 4096 — the bias just keeps getting
> more negative, which is exactly what you want. Steep-slope heads become local
> pattern detectors, shallow-slope heads keep a global view. Contrast with T5's
> learned buckets, which have nothing sensible to say about distances larger than
> the largest bucket seen in training.

### Approach B (the modern default): RoPE — Rotary Position Embeddings

**Su et al., 2021 (RoFormer).** The idea: instead of *adding* anything, **rotate
the query and key vectors** by an angle proportional to their position.

In two dimensions, rotating a vector by angle `mθ` (where `m` is the position):

```
R(mθ) = [  cos(mθ)   −sin(mθ) ]
        [  sin(mθ)    cos(mθ) ]
```

and we set `q_m ← R(mθ)·q_m`, `k_n ← R(nθ)·k_n`.

**Extension to `d > 2`:** split the `d`-dimensional vector into `d/2` consecutive
pairs of coordinates and **rotate every block of dimension 2**, each block with
its own frequency `θ_i` (again geometrically spaced, as in the sinusoidal
scheme).

**Why this is elegant — the relative-distance property.** Because rotations
compose by adding angles, and because a rotation preserves dot products:

```
(R(mθ)·q)ᵀ · (R(nθ)·k)  =  qᵀ · R((n − m)θ) · k
```

The attention score between positions `m` and `n` depends **only on `n − m`**.
You inject absolute positions into the vectors, and the attention mechanism
automatically sees only relative distance. No bias table, no extra parameters,
nothing added to the residual stream.

**Long-term decay.** The lecture also shows that the relative upper bound on the
attention weight *decays* as `|n − m|` grows: as the many frequency components
drift out of phase with each other, distant pairs tend towards lower scores.
So RoPE bakes in a mild, automatic recency prior too.

> **Intuition — the clock analogy.** Give every position a set of clock hands
> spinning at different speeds, one pair of hands per pair of dimensions. A query
> and a key "match" strongly when their hands line up. Since each hand's angle
> is `position × its own speed`, the alignment between two tokens depends only on
> how many steps apart they are — like reading the phase difference between two
> clocks rather than the absolute time on either. The reason RoPE is the default
> in essentially every modern open LLM (LLaMA, Mistral, Qwen, Gemma, DeepSeek) is
> that it adds zero parameters, gives exact relative behaviour, does not touch
> the residual stream, and — importantly for Lecture 3 — is compatible with KV
> caching, because you rotate each key once when you compute it and it stays
> valid forever.

> **Watch out — RoPE is applied to `q` and `k` only, never to `v`.** The
> position information belongs in the *matching* computation, not in the content
> being copied. If you rotated the values you would be corrupting the payload.

---

## 2.2 Layer normalisation

The `Add & Norm` boxes in the architecture diagram are `LN`, layer
normalisation (Ba et al., 2016). For a vector `x` of dimension `d`:

```
μ  = (1/d) · Σ_i x_i
σ² = (1/d) · Σ_i (x_i − μ)²
LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β
```

where `γ` and `β` are learned per-dimension scale and shift, and `ε` is a small
constant for numerical safety. **Benefit: helps training stability and
convergence.**

> **Intuition.** Note what is being normalised: for *each token independently*,
> across its own feature dimensions. This is different from batch normalisation,
> which normalises each feature across the batch — a poor fit for text, where
> sequences have different lengths and batch statistics are noisy. LayerNorm has
> no dependence on batch size at all, which is why it is the choice here.
> Mechanically, it keeps the activations entering each sub-layer at a consistent
> scale, so no layer receives inputs whose magnitude has drifted; that
> consistency is what keeps gradients well-behaved in deep stacks.

### Post-Norm vs Pre-Norm

Where you put the normalisation relative to the residual connection matters
enormously (Xiong et al., 2020).

**Post-Norm** — the original 2017 arrangement, normalisation *after* the
residual addition:

```
x ← LN( x + Sublayer(x) )
```

**Pre-Norm** — normalisation *before* the sub-layer, with the residual added
afterwards, unnormalised:

```
x ← x + Sublayer( LN(x) )
```

**Nowadays: Pre-Norm.** And, additionally, **RMSNorm** (Zhang et al., 2019)
instead of full LayerNorm:

```
RMS(x)     = √( (1/d)·Σ_i x_i² )
RMSNorm(x) = γ ⊙ x / RMS(x)
```

i.e. drop the mean subtraction and the shift `β`; only rescale by the root mean
square. So the modern default is **Pre-Norm + RMSNorm**.

> **Intuition — why Pre-Norm won.** In the Post-Norm arrangement, every residual
> stream passes through a normalisation on its way up the stack, so the "clean"
> identity path from input to output is repeatedly rescaled. Gradients flowing
> back get multiplied by the Jacobian of each LayerNorm, and in deep models this
> makes the early layers receive badly-scaled gradients. The practical
> consequence is that Post-Norm Transformers need a careful **learning-rate
> warmup** to train at all. Pre-Norm leaves an unobstructed additive highway from
> input to output — every sub-layer's contribution is added to a stream nothing
> ever renormalises — so gradients reach the bottom intact and warmup becomes
> optional. The cost is that the residual stream's magnitude grows with depth,
> which is why a final normalisation is added before the output head.
>
> RMSNorm is a pure efficiency win: subtracting the mean turns out to contribute
> almost nothing to the benefit, while costing an extra pass over the vector and
> extra parameters. Removing it is measurably faster at essentially no quality
> cost — the kind of small win that matters when you multiply it by every layer
> of every forward pass of a trillion-token training run.

---

## 2.3 Making attention cheaper

All of this exists because of the `O(n²)` cost established in Lecture 1.

### Sparse attention: Longformer (Beltagy et al., 2020)

Full attention is an `n × n` grid where every cell is computed. Longformer's
observation is that most of those cells are near-worthless: language is mostly
local. So restrict which cells you compute:

- **Sliding-window attention (SWA):** each token attends only to a window of `w`
  neighbours on each side. Cost drops from `O(n²)` to `O(n·w)`, linear in `n`.
- **Global attention on selected tokens:** a small number of designated tokens
  (`[CLS]`, or task-specific ones such as the question tokens in QA) attend to
  everything *and* are attended to by everything.

**Variations include interleaving local and global attention layers** — some
layers local, some global, rather than mixing both patterns in every layer.

**Mistral 7B (2023)** used sliding-window attention in production, and the
lecture makes the illuminating comparison to the **receptive field** in
convolutional networks.

> **Intuition — the receptive-field argument, which is the crux.** A single
> sliding-window layer with window `w` lets a token see `w` neighbours. Stack two
> such layers and its effective reach is `2w`, since its neighbours have
> themselves already gathered from *their* neighbours. After `L` layers the
> receptive field is `L·w`. With `w = 4096` and 32 layers, information can travel
> over 100k tokens — while every individual attention computation stays cheap and
> linear. This is exactly how CNNs see whole images through small 3×3 filters. The
> trade-off is real, though: information from far away arrives *compressed*,
> having been repeatedly averaged, whereas full attention gives every token a
> direct, uncompressed link to every other. That is why the `[CLS]`-style global
> tokens are kept — they act as an express lane through the network.

### Sharing attention heads: MHA → GQA → MQA

A different axis of saving. In vanilla **multi-head attention (MHA)**:

```
#query heads = #key heads = #value heads = h
```

The idea is to **share key/value heads within groups of queries**:

- **MHA** — `h` query heads, `h` key heads, `h` value heads. Maximum expressive
  power, maximum memory.
- **GQA (Grouped-Query Attention)** — `h` query heads, but only `G` key heads
  and `G` value heads, with `G < h`. Query heads are partitioned into `G` groups;
  all queries in a group share one K/V pair.
- **MQA (Multi-Query Attention)** — the extreme: `h` query heads, **1** key head
  and **1** value head. Every query head shares the same keys and values.

So `#query = h`, `#key = #value = G`, with `G = h` recovering MHA and `G = 1`
recovering MQA.

> **Intuition — why this is worth doing.** The saving is not really in FLOPs, it
> is in the **KV cache** (Lecture 3). During generation you must keep, for every
> layer and every head, the keys and values of every token generated so far. That
> cache scales as `2 · n · h · d_head · N_layers`, and for long conversations it
> becomes the single largest consumer of GPU memory — often larger than the model
> weights. Cutting the key/value head count by 8× cuts the cache by 8×, which
> means you can serve 8× more concurrent users on the same hardware, or support
> 8× longer contexts. MQA is the cheapest but measurably degrades quality; GQA
> with `G = 8` was found to keep essentially all the quality at most of the
> saving, which is why it is the standard choice today (LLaMA 2/3, Mistral).
> This material is repeated in Lecture 3 under inference optimisation, which
> tells you how central it is.

---

## 2.4 The three families of Transformer-based model

This taxonomy is the organising idea of the rest of the course.

| Family | What it does | Examples | Era |
|---|---|---|---|
| **Encoder–decoder** | text to text | T5, mT5, ByT5 | — |
| **Encoder-only** | project the embedding for class prediction (e.g. sentiment extraction) | BERT, DistilBERT, RoBERTa | popular ~2018–2022 |
| **Decoder-only** | text to text | GPT series | **popular now** |

Encoder-only means: keep the left half of the diagram, and read out the encoded
representations. Decoder-only means: keep the right half, drop cross-attention
(there is no encoder to attend to), and just do causal language modelling.

> **Intuition — why decoder-only won, given that encoder–decoder came first.**
> Three reasons, and they build on each other.
>
> First, *unification*. A decoder-only model has exactly one interface —
> "continue this text" — and every task can be expressed in it. Translation is
> "English: ... French: ...". Classification is "Review: ... Sentiment: ...".
> With an encoder-only model you must attach and finetune a task-specific head
> for every new task.
>
> Second, *training efficiency per token*. Causal language modelling extracts a
> prediction target from **every** position of every training sequence. BERT's
> masked language modelling only gets signal from the ~15% of positions it masks.
> When the bottleneck is how much you can learn per unit of compute, that is a
> large constant factor.
>
> Third, *scaling and emergence*. Simply making causal LMs bigger produced
> in-context learning (Lecture 3) — the ability to pick up a task from examples
> in the prompt, with no gradient updates at all. That capability made the
> finetune-per-task workflow largely unnecessary.
>
> Encoder-only models are not obsolete, though. They remain the right tool
> whenever you need a *vector*, not text: retrieval embeddings, reranking, and
> classification at high volume and low latency. Lecture 7's RAG pipeline runs on
> BERT-family bi-encoders and cross-encoders, and Lecture 5's reward models can
> be built the same way. That is precisely why BERT gets a deep dive here.

---

## 2.5 BERT deep dive

**BERT = Bidirectional Encoder Representations from Transformers**
(Devlin et al., 2018).

### Why "bidirectional" is the operative word

An encoder's self-attention is unmasked: every token sees every other token,
both to its left and to its right. That is **bidirectional**. A decoder's
self-attention is causal-masked: a token sees only its past. That is **not
bidirectional**, and the lecture flags it explicitly with a warning marker.

The naming is a deliberate jab at **ELMo** (Peters et al., Feb 2018), which
achieved "bidirectional" context by running two *separate* unidirectional LSTMs
— one left-to-right, one right-to-left — and concatenating them. Each direction
never saw the other while computing. BERT's attention is *jointly* bidirectional
in a single pass. (And yes, ELMo and BERT are both Sesame Street characters;
BERT's paper came out in October 2018, eight months after ELMo's, and the
naming was not an accident.)

### The two-step strategy

- **Step 1 — Pretraining** with proxy tasks: MLM and NSP, on unlabelled text.
- **Step 2 — Finetuning** for a given end task, on a small labelled dataset.

| Pros | Cons |
|---|---|
| finetuning does not need much data | not suited to a range of tasks (e.g. text generation) |
| good performance | finetuning is a **required** step |

**Variants:** RoBERTa, DistilBERT, ALBERT.

### Architecture, versus the original Transformer

BERT is the encoder stack only. No decoder, no cross-attention, no causal mask.
On top of the final encoded embeddings you attach whatever head your task needs.

### Input processing

**WordPiece tokenisation.** The tokeniser is trained on a corpus beforehand;
vocabulary size around **30,000**; it is *great at detecting common particles*
(the frequent prefixes and suffixes get their own tokens).

**Special tokens, for the NSP/MLM setup:**
- `[CLS]` at the very beginning of the input.
- `[SEP]` separating consecutive segments, plus one at the end.
- `[MASK]` to hide inputs during MLM.
- `[PAD]` to fill a batch to uniform length.

### Input embeddings — three of them, summed

1. **Token embedding** — a gigantic lookup table, one learned vector per
   vocabulary entry.
2. **Positional encoding** — learned or fixed sines/cosines, as in Lecture 1.
3. **Segment encoding (new in BERT!)** — a shared embedding identifying which
   segment a token belongs to: all of segment A gets one vector, all of segment
   B gets another.

The three are added elementwise to form the input to the encoder stack.

> **Intuition.** Segment embeddings exist because BERT is frequently fed *pairs*
> of texts — two sentences for NSP, a question and a passage for QA, a premise
> and a hypothesis for entailment. `[SEP]` marks the boundary, but a single
> boundary token is a weak signal for a model that mixes everything through
> attention; giving every token an explicit "I am in the second text" vector
> makes the distinction available everywhere at once.

### Proxy task 1: Masked Language Modelling (MLM)

**15% of input tokens are selected for prediction.** Of those selected:

- **80%** are replaced with `[MASK]`,
- **10%** are replaced with a **random word**,
- **10%** are **left unchanged**.

The model must predict the original token at every selected position.

**Benefits:** the network learns language modelling from *contextual*
information (both directions), and the randomisation acts as regularisation that
**reflects the probabilistic nature of language**.

> **Intuition — the famous 80/10/10 split, which confuses everyone.** The
> problem being solved is a train/inference mismatch. If masked positions were
> always `[MASK]`, the model would learn "predict a word only when you see the
> `[MASK]` token" — but at finetuning and inference time `[MASK]` never appears,
> so the representations of ordinary tokens would be under-trained. The 10%
> random-word substitution forces the model to *always* keep a prediction of
> what each position should contain, since any token might secretly be corrupted;
> this is what makes it robust rather than lazily copying. The 10% unchanged case
> forces it to keep predicting even when the input looks perfectly fine —
> otherwise it could learn "if it isn't `[MASK]` or obviously wrong, just copy
> the input". Together, the three cases mean every position's representation must
> encode a genuine contextual prediction.

### Proxy task 2: Next Sentence Prediction (NSP)

Pick two sentences from the corpus. **50% of the time they genuinely follow each
other; 50% of the time they do not.** Task: binary classification — do they
follow?

**Benefits:** the network implicitly learns to detect useful contextual
information across a sentence boundary, and it is an **easy classification task
that requires no labels** (the labels are free, from the corpus structure).

The prediction is read off the `[CLS]` token's final representation.

### Hyperparameters and model sizes

**Model:**
- `L` — number of layers,
- `H` — hidden layer size, i.e. the embedding dimension,
- `A` — number of attention heads operating in parallel.

**Data:**
- *language-specific vs multilingual* — which languages it was trained on,
- *cased vs uncased* — whether inputs are lowercased.

| Model | L | H | A | Parameters |
|---|---|---|---|---|
| BERT-Tiny | 2 | 128 | 2 | 4M |
| BERT-Mini | 4 | 256 | 4 | 11M |
| BERT-Small | 4 | 512 | 8 | 30M |
| BERT-Medium | 8 | 512 | 8 | 42M |
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

> **Watch out.** Compare 340M for BERT-Large against the hundreds of billions in
> Lecture 3. In 2018 BERT-Large was considered enormous. Keeping this table in
> mind gives you a visceral sense of how fast the scale moved.

### Finetuning

**Goal:** leverage the embeddings BERT learned for a "sister" task.

**Tricks:**
- start from the weights of the massively pretrained model,
- **freezing early layers** sometimes gives a better complexity/performance
  trade-off,
- great results are possible with **minimal labelled data**, depending on how
  complex the task is and how close it is to the pretraining data distribution
  and objectives.

**Use cases:**
- *sequence classification* — one label for the whole input, e.g. sentiment
  extraction,
- *token classification* — one label per token, e.g. extractive question
  answering (predicting answer start and end positions).

### Worked finetuning example: sentiment extraction

Input: `This teddy bear is SO CUTE!`

1. **Lowercase** (uncased model): `this teddy bear is so cute!`
2. **Tokenise** with WordPiece: `this | teddy | bear | is | so | cute | !`
3. **Prepend `[CLS]`** — the lecture's phrasing is that it acts as *a
   placeholder for the sentiment*.
4. **Append `[SEP]`**, then `[PAD]` out to the fixed sequence length:
   `[CLS] this teddy bear is so cute ! [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]`
5. **Add the token embedding**, then the **position embedding** (indices
   0,1,2,...,14), then the **segment embedding** (here segment A for the real
   text and B for the padding region).
6. Feed the resulting matrix through the **pretrained BERT encoder**.
7. Take the final-layer vector at the `[CLS]` position, pass it through a small
   **feed-forward network**, and read off the sentiment class.

> **Intuition — why `[CLS]` works as a sentence summary.** `[CLS]` carries no
> lexical meaning of its own, so its representation is free to become whatever
> the pooling operation needs to be. Through unmasked self-attention it can
> gather from every token in the sequence, and because NSP pretraining already
> forced it to encode a whole-sequence property, it starts finetuning as a
> reasonable summary vector rather than from scratch. It is a learned pooling
> token: strictly more flexible than mean-pooling, because attention weights
> which tokens matter.

### Takeaways and shortcomings

**Benefits:** state-of-the-art results; approximately *true contextual
representation* of words (the same token gets different vectors in different
sentences — the thing Word2vec could not do); adaptable to many classification
tasks. **Widely used in industry for anything related to encoding.**

**Limitations:**
- context window size is limited,
- computationally expensive — a hard sell for low-latency or cost-sensitive
  applications,
- the training paradigm is complex: MLM + NSP pretraining *and then* finetuning.

---

## 2.6 Distillation

**"The soft targets contain almost all the knowledge."** — Hinton et al., 2014
("Dark knowledge").

The setup: a large, already-trained **teacher** model and a smaller **student**.
Rather than training the student on the hard one-hot labels, train it to match
the teacher's full output *probability distribution*.

> **Intuition — why soft targets carry more information.** Suppose the correct
> label is "teddy bear". A hard label tells the student one bit of structure:
> class 7 is right. The teacher's soft distribution says something much richer —
> "85% teddy bear, 10% stuffed animal, 4% toy, 0.001% bulldozer". That encodes
> the teacher's entire learned similarity structure over the label space, in
> every single training example. The student learns not just the answer but the
> *shape of the confusion*, which is why distillation can transfer performance
> that the student would never reach training on labels alone. This is the "dark
> knowledge": information present in the teacher's function that the training
> labels never contained. Lecture 6 will use the word "distillation" for
> something related but different — training a small model on a big model's
> generated reasoning traces — and the lecture explicitly contrasts the two.

### DistilBERT (Sanh et al., 2019)

Distil BERT-Base (`N = 12` layers) down to **`N = 6` layers**. Result:
**~1.6× faster** while retaining **~97% of the performance**.

### RoBERTa (Liu et al., 2019)

**Goal:** a thorough analysis of the available directions of optimisation —
what in BERT's recipe actually mattered?

**Modelling findings:**
- **Removing NSP and segment encodings: approximately no effect!** (DistilBERT
  had already dropped it.)
- **Static → dynamic masking** across epochs: rather than masking each sequence
  once during preprocessing and reusing that mask every epoch, generate a fresh
  mask each time the example is seen.

**Data findings:**
- **richer**: pretraining corpus from **16 GB → 160 GB**,
- **for longer**: 1M steps at batch size 256, versus 500k steps at batch size 8k
  (a much larger total token count).

**Result: +4% across benchmarks with the same architecture.**

> **Intuition — the real lesson of RoBERTa, and it is a big one.** BERT's paper
> presented MLM+NSP as an integrated design, and the community assumed both were
> load-bearing. RoBERTa shows one of them was not, and that most of BERT's
> headroom was *undertraining*: same architecture, more data, longer schedule,
> +4%. This is the first appearance in the course of a theme that dominates
> Lecture 4 — that for a given architecture, the amount of data and compute you
> pour in often matters more than the architectural cleverness. The Chinchilla
> result is exactly this observation, made quantitative.
>
> Why did NSP not matter? Plausibly because predicting whether two sentences are
> adjacent is *too easy* — the model can often decide from topic overlap alone,
> without learning anything about discourse structure. An objective that provides
> no gradient signal after the first few thousand steps contributes nothing. This
> is worth remembering when you design your own proxy tasks: a task must be hard
> enough to force the representation you want.
