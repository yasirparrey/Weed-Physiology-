# Lecture 7 — Agentic LLMs

**Video length: 1h49.** Covered here: why retrieval is needed, the RAG pipeline
end to end, building a knowledge base, two-stage retrieval (candidate generation
then reranking), semantic vs BM25 vs hybrid search, HyDE and contextual
retrieval, retrieval metrics (NDCG@k, RR@k, recall@k, precision@k), tool/function
calling, how to teach a model to use tools, tool selection and routing, MCP,
agents and the ReAct loop, A2A, and safety.

This lecture addresses two of the four weaknesses from Lecture 6's list:
**knowledge is static** and **cannot perform actions**.

---

## 7.1 Why retrieval

Four motivations, each independent:

**1. The knowledge of an LLM is constrained to its pretraining data.** The
lecture shows the knowledge cutoff on GPT-5's model card. Anything after that
date does not exist as far as the model is concerned, and your own private
documents never existed at all.

**2. Limited context size.** Even a very large window cannot hold a company's
entire document store.

**3. The LLM gets distracted by useless information.** The lecture references the
**"Needle in a Haystack"** pressure tests (Kamradt, 2023), which hide a specific
fact in a long context and measure whether the model can retrieve it. Performance
depends on both the context length and *where* in the context the fact sits.

**4. Pricing is per input/output token.** Every irrelevant token you include is
money spent, latency added, and attention diluted.

> **Intuition — points 3 and 4 together are the real argument, and they are
> counterintuitive.** You might think that with a one-million-token window the
> answer to "how do I give the model knowledge?" is simply "put everything in the
> prompt". It is not, for two reasons. It costs a fortune per call. And — more
> importantly — it makes the model **worse**, because attention is a
> zero-sum-ish resource: softmax mass spent on 900,000 irrelevant tokens is mass
> not spent on the 100 that mattered. Retrieval is not a workaround for small
> context windows. It is a way of making the model's attention land on the right
> place, and it remains valuable no matter how big windows get.

---

## 7.2 RAG

**RAG = Retrieval-Augmented Generation** (Lewis et al., 2020).

**Idea: augment the prompt with relevant pieces of information.**

The three steps, in the order the acronym does *not* suggest:

1. **Retrieve** the relevant document via a similarity operation across the
   knowledge base.

2. **Augment** the prompt with the retrieved information.
3. **Generate** the response with the LLM, given the augmented prompt.

```
user prompt ──→ [retrieve] ──→ retrieved info + user prompt ──→ LLM ──→ response
```

The bulk of the lecture then focuses on **the retrieval stage**, because that is
where all the engineering difficulty lives.

### Prerequisite: creating the knowledge base

Three operations, done offline, once:

**Collect** — gather Document 1, Document 2, ... whatever your corpus is.

**Divide** — split each document into **chunks**.

**Embed** — turn each chunk into a vector, e.g. `[0.48, 0.33, ..., −0.51]`, and
store it in a vector index.

**Hyperparameters: embedding size, chunk size, overlap between chunks.**

> **Intuition — chunking is the most underrated decision in the pipeline.** Chunk
> too **small** and each chunk lacks the context needed to be interpretable: a
> sentence saying "he was in a cuddly mood" is useless if the paragraph that
> established who "he" is landed in a different chunk. Chunk too **large** and the
> embedding becomes a blurry average of several topics, matching everything
> weakly and nothing strongly — and you waste context on the irrelevant parts you
> retrieve along with the relevant sentence. **Overlap** exists so that a fact
> sitting on a boundary is not cut in half; a typical setting is 10–20% overlap.
> Practitioners consistently find that fixing bad chunking improves a RAG system
> more than swapping the LLM. Section 7.2's "contextualise document chunks" is a
> direct attack on the small-chunk problem.

### Retrieval: two stages

**Step 1 — Candidate retrieval.** Select potentially-relevant candidates.

- **Maximise recall.**
- Semantic embeddings, and optionally keyword-based methods.

**Step 2 — Ranking.** Give the final relevance score.

- **Maximise precision.**
- Re-rank on the smaller set of candidates.

> **Intuition — why two stages instead of one good one?** Cost, and the shape of
> the cost. Step 1 must touch **every** chunk in the knowledge base, potentially
> millions, so it must be extremely cheap per item — which means precomputed
> embeddings and an approximate nearest-neighbour search. Step 2 touches maybe 50
> candidates, so it can afford a model that is a thousand times more expensive per
> item. The division of labour is: cast a wide, cheap, slightly sloppy net that is
> unlikely to miss the right answer (recall), then apply an expensive, accurate
> judgement to what you caught (precision). This is the standard architecture of
> every search and recommendation system ever built.

### Step 1, method 1: semantic search with embeddings

Encode the **query** and each **chunk** into vectors and compute a **similarity
score** — typically cosine similarity or dot product.

Critically, the query encoder and the chunk encoder produce vectors in the
**same** space, and each is encoded **independently**. This arrangement is called
a **bi-encoder** (Sentence-BERT, Reimers et al., 2019).

**Worked example.** Query: *"Where is Cuddly?"* Candidate chunks:

- *"[...] Huggy likes to work downstairs [...]"*
- *"[...] Where is Paris located? [...]"*
- *"[...] he was in a cuddly mood [...]"*
- *"[...] Cuddly spends most days surrounded by books [...]"*

Semantic search should rank the last one highest — it is genuinely about the
entity Cuddly and its location — but it is *also* prone to being pulled towards
the second (structurally similar question, "Where is ... ?") and the third
(the word "cuddly", though used as an adjective).

> **Intuition — why the bi-encoder architecture is what makes this scalable.**
> Because the chunk embedding does not depend on the query, you can compute all of
> them **once, offline**, and store them. At query time you embed only the query
> and do a vector search. If the encoder saw query and chunk *together* you would
> have to run the model once per (query, chunk) pair — millions of forward passes
> per query, which is impossible. That is exactly the trade being made: the
> bi-encoder is fast because it never lets the query and the document interact,
> and it is less accurate for exactly the same reason. Step 2 pays for the
> interaction on a short list.

### Step 1, method 2: keyword matching with BM25

**BM25** is a classical lexical retrieval function — a refined TF-IDF. It scores a
chunk by how many of the query's terms it contains, weighting rare terms more
heavily, with saturation on repeated terms and a normalisation for document
length.

On the same example, BM25 latches onto the literal token *Cuddly*, so it ranks
both the "cuddly mood" chunk and the "Cuddly spends most days" chunk highly, and
correctly ignores the Paris chunk (no shared rare terms).

### Step 1, method 3: hybrid

**Search based on a hybrid combination of semantics and BM25** — run both, merge
the ranked lists (e.g. by reciprocal rank fusion), and pass the union to the
reranker.

> **Intuition — why hybrid is the practical default, framed as complementary
> failure modes.** Embeddings capture *meaning* and are robust to paraphrase, so
> they find "canine companion" when you searched "dog". But they are unreliable on
> exactly the things where precision matters most: rare proper nouns, product
> SKUs, error codes, function names, version numbers. A semantic encoder has no
> particular representation for `ERR_4471B` and will happily return `ERR_4472B`.
> BM25 is the opposite: it is exact on rare tokens and completely blind to
> synonymy. Their errors are close to uncorrelated, which is the ideal condition
> for combining two systems. Hybrid retrieval reliably beats either alone, which
> is why "just use a vector database" is incomplete advice.

### Extensions that help initial retrieval

**Mitigate the discrepancy in nature of embeddings — HyDE** (*Precise Zero-Shot
Dense Retrieval without Relevance Labels*, Gao et al., 2022).

The problem: a **query** and a **document** are different kinds of text. "Where is
Cuddly?" is a short interrogative; the answer is a long declarative passage.
Embedding them into the same space and comparing is comparing unlike things.

The fix: have the LLM write a **fake document** that answers the query —
*"Cuddly is in ..."* — and embed **that** instead of the query. Now you are
matching document-shaped text against document-shaped text.

> **Intuition.** The hypothetical document does not need to be *factually*
> correct — it will be a hallucination, and that is fine. It only needs to be
> *stylistically and topically* correct, because its job is to land in the right
> neighbourhood of the embedding space. You are using the LLM's generative fluency
> as a query-expansion device.

**Contextualise document chunks — Contextual Retrieval** (Anthropic, 2024).

The problem: a chunk pulled out of a document loses the context that made it
meaningful (the pronoun problem from above).

The fix: before embedding, prepend a short generated context to each chunk. The
exact prompt used:

```
<document>
{WHOLE_DOCUMENT}
</document>
Here is the chunk we want to situate within the whole document:
{CHUNK_CONTENT}
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk. Answer
only with the succinct context and nothing else.
```

Each chunk `i` becomes `Context i + Chunk i`, and that is what gets embedded.

**The cost problem, and the fix: prompt caching.** This prompt includes the
**whole document** once per chunk, so a 100-chunk document means passing the full
document 100 times — prohibitively expensive at face value. But because the long
prefix is *identical* across all those calls, **prompt caching** applies: the
provider stores the computed KV cache for the shared prefix and charges a small
fraction for cache reads. The lecture shows the pricing pages where cached input
tokens cost roughly a tenth of fresh ones.

> **Intuition — this is where Lecture 3's PagedAttention pays off commercially.**
> Prompt caching is only possible because KV cache blocks can be shared across
> requests. Contextual retrieval is a nice illustration of a technique that is
> obvious in principle, absurd in cost under naive accounting, and entirely
> practical once you understand the serving stack. Anthropic reported it cutting
> retrieval failure rates by around a third, and by around a half when combined
> with BM25 and reranking.

### Step 2: ranking with a cross-encoder

The reranker encodes the **query and the chunk together** in a single forward
pass and outputs a **relevance score** directly. This is a **cross-encoder**
(suggested reading: "Cross-Encoders", SBERT.net).

```
Re-Ranker(user prompt, {chunk d, chunk b, chunk a, chunk c})
       → chunk a (1), chunk b (2), chunk c (3), chunk d (4)
```

> **Intuition — what the cross-encoder can do that the bi-encoder cannot.**
> Because query and chunk are concatenated and passed through the same attention
> stack, every query token can attend directly to every chunk token. The model can
> therefore check *term by term* whether the chunk answers the specific question,
> including negation, qualifiers and exact entity matching — things a fixed-length
> vector inevitably smooths away. The price is that you must run the model once
> per candidate, and you cannot precompute anything, which is exactly why it is
> confined to the short list. Adding a reranker is typically the single highest
> return-on-effort improvement available to a mediocre RAG system.

### Quantifying retrieval performance

**Setup: evaluate whether the retrieved chunks are relevant.** Some chunks are
labelled relevant; a ranking is produced; you look at the top `k`.

**Normalised Discounted Cumulative Gain at k (NDCG@k):**

```
DCG@k  = Σ_{i=1..k}  rel_i / log₂(i + 1)
IDCG@k = the same quantity if the ranking were perfect
NDCG@k = DCG@k / IDCG@k
```

**Reciprocal Rank at k (RR@k):**

```
RR@k = 1 / (rank of the first relevant chunk)     [0 if none in the top k]
```

**Recall at k:** of all the relevant chunks that exist, what fraction appear in
the top `k`?

**Precision at k:** of the `k` chunks retrieved, what fraction are relevant?

> **Intuition — which metric to use when, because they answer different
> questions.** *Recall@k* is the metric for **step 1**: the only unforgivable
> failure at the candidate stage is that the right chunk is not in the list at
> all, since nothing downstream can recover from that. *NDCG@k* is the metric for
> **step 2**: it is position-sensitive (the `1/log₂(i+1)` discount means rank 1 is
> worth much more than rank 10) and handles graded relevance, so it captures
> whether the reranker put the best material first — which matters because of the
> lost-in-the-middle effect. *RR@k* is the right metric when there is exactly one
> correct answer and you only care how quickly it appears. *Precision@k* matters
> because irrelevant chunks are not free: they cost tokens and dilute attention.

---

## 7.3 Tool calling

### Motivation

RAG solved **unstructured** knowledge — documents you can chunk and embed. But a
great deal of what you need is **structured**: rows in a database, live prices, a
calculation, the current time.

You cannot embed a database usefully. What you *can* do is call a function:

```python
def get_data(id, field, ...):
    # Logic.
    return result
```

### Definition

> *"Tool calling [...] allows autonomous systems to complete complex tasks by
> dynamically accessing and [may act] upon external resources."* — IBM

The lecture emphasises two phrases in that definition: **autonomous systems** —
the model decides, it is not scripted — and **may act upon** — this is not only
reading, it includes changing the world.

### The example

**Without tools:**

> *"Find a bear near me!"* → *"Sorry, I don't know which bears are near you."*

**With tools:** the model is given a function API and can call it.

The function, `find_teddy_bear.py`, and the three properties the lecture
highlights about it:

```python
from dataclasses import dataclass
from geopy.distance import geodesic
import requests

@dataclass
class TeddyBearInfo:
    name: str
    distance_meters: float
    mood: str
    message: str

def find_teddy_bear(location: tuple[float, float]) -> TeddyBearInfo:
    """
    Finds the nearest teddy bear to the given GPS coordinates.

    Parameters:
        location: A (latitude, longitude) pair representing the user's
            current location.
    Returns:
        TeddyBearInfo: Information about the nearest teddy bear found.
    """
    user_lat, user_lon = location
    api_url = "https://api.to.teddy.bears.com/v1/closest"
    try:
        response = requests.get(
            api_url,
            params={"latitude": user_lat, "longitude": user_lon},
            timeout=5,
        )
        response.raise_for_status()
        closest_teddy_bear = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch teddy bear data from API: {e}")

    bear_lat, bear_lon = closest_teddy_bear["coords"]
    distance = geodesic((user_lat, user_lon), (bear_lat, bear_lon)).meters
    return TeddyBearInfo(
        name=closest_teddy_bear["name"],
        distance_meters=round(distance, 2),
        mood=closest_teddy_bear["mood"],
        message=f"{closest_teddy_bear['name']} is "
                f"{closest_teddy_bear['mood']} and only "
                f"{round(distance, 2)} meters away!",
    )
```

1. It **has a descriptive, well-documented API** — meaningful name, typed
   signature, and a docstring explaining parameters and return value.

2. It **(optionally) has some backend call** — it reaches out to a real service.
3. It **returns some info** — a structured object.

> **Intuition — point 1 is the whole ballgame, and it is easy to underestimate.**
> The model never sees the function *body*. All it sees is the name, the type
> signature, and the docstring. Those three things are the entire specification
> from which it must decide *whether* to call the tool and *what* to pass. Which
> means the docstring is not documentation for humans — **it is a prompt**, and
> it should be written with the same care you would give any other prompt. A
> function called `proc_dat(x, y)` with no docstring is unusable by an LLM no
> matter how good the model is. This reframing — "your API surface is prompt
> engineering" — is the single most actionable idea in the section, and Lecture 8
> shows the failure modes that follow from getting it wrong.

### How it works — three steps

1. **Let the LLM find the argument for the relevant function call.** Given the
   available function APIs and the user's message, the model emits something
   equivalent to: `location = (37.42, −122.17)` with `find_teddy_bear()`.

2. **Make the function call.** *Your code*, not the model, executes
   `find_teddy_bear(location)` against the backend, which returns
   `{"name": "Teddy", ...}`.

3. **Let the LLM deduce a conclusion based on the results.** The JSON is fed back
   into the model, which writes the natural-language response.

> **Watch out — the model does not execute anything.** This is the most commonly
> misunderstood point about tool calling. The LLM only *emits a structured request*
> — a function name and arguments, typically as JSON, and often produced with
> guided decoding (Lecture 3) to guarantee it parses. Your application decides
> whether to honour that request, executes it, and passes the result back. That
> boundary is where all of your authorisation, validation, rate-limiting and
> sandboxing must live, because the model's output is untrusted input.

### Teaching a model to use a tool

**Method 1: via training.** Build SFT data of two kinds:

- **Tool prediction** — input: the function API plus the conversation history
  ("Find a bear near me!"); target: the correct call,
  `location = (37.42, −122.17) with find_teddy_bear()`.

- **Response generation** — input: the API, the request, the emitted call *and*
  the backend's response; target: the final natural-language answer.

The lecture shows this generalising across examples: *"Find a bear in Paris!"* →
`location = (48.86, 2.35) with find_teddy_bear()`.

**Method 2: via prompting.** Provide `<function API> + <detailed explanation on
how to use it>` in the prompt. Which raises the question: **how do you write such
a description?**

**One way given in the lecture:** *use SFT pairs as evaluation, and use a
powerful reasoning model to write the description for you.* That is, treat the
tool description as an artifact to be optimised — have a strong model draft it,
score the draft against a held-out set of (request → correct call) pairs, and
iterate.

> **Intuition.** This is prompt optimisation with a proper objective function,
> and it is a genuinely good pattern. Instead of a human guessing at wording, you
> have a measurable target (does the model pick the right tool with the right
> arguments on these 200 cases?) and a capable writer (a reasoning model) doing the
> drafting. Note that modern frontier models are already trained for tool use, so
> Method 2 is what you will actually do; Method 1 matters when you are adapting an
> open model or have unusual tools.

### Common use cases

- **Information** — web/database search; weather, stocks and other trackers;
  codebase search.

- **Computation** — calculator; code execution, often Python.
- **Action** — send emails/messages and other in-computer actions; anything else
  within an assistant's domain.

### Tools summary

**Benefits:**

- LLMs just became **way more useful**,
- they can **interact with the real world**,
- it **overcomes the "knowledge cutoff"** limitation.

**Challenges** (*Automatic Tool Selection to Reduce Large Language Model
Latency*, Robert et al., 2024):

- **more tools = decreased performance**,
- **finite context length: not scalable** — every tool definition consumes prompt
  tokens,

- **many tools to define. Lots of work.**

### Tool selection / routing

**Goal: both reduce latency and improve performance.**

Insert a **router** before the LLM:

```
"Find a bear near me!" ──→ Router ──→ list of selected tools
                                          ↓
                       <selected function APIs> + prompt ──→ LLM ──→ ...
```

> **Intuition — why more tools makes a model worse, and why routing fixes it.**
> With fifty tool definitions in the prompt you have (a) consumed thousands of
> tokens before the user's question even appears, (b) created many near-duplicate
> options between which the model must discriminate, and (c) diluted attention
> across a large amount of boilerplate. Performance degrades for the same reason
> that a bloated RAG context degrades performance. A router — often a small
> classifier or an embedding search over tool descriptions — narrows fifty tools
> to three before the expensive model ever sees them. Note that this makes the
> router a new single point of failure, which is exactly the first failure mode
> Lecture 8 enumerates.

### Standardisation: MCP

**Motivation: avoid duplication of tool implementations.** With `m` LLM
applications and `n` tools you have `m × n` bespoke integrations to write and
maintain.

**MCP = Model Context Protocol** (Anthropic, 2024). **Idea: connect tools and
data to LLMs in a standard way** — turning `m × n` into `m + n`.

**Architecture:**

- **MCP host** — the application the user interacts with (e.g. Claude Desktop),
  containing an **MCP client**,

- **MCP server** — exposes **tools**, **prompts** and **resources**,
- the client and server speak the protocol.

**The lecture's example.** Host: Claude Desktop. User prompt: *"Recommend a new
poetry book to my teddy bear."* A **book provider MCP server** exposes a `Find`
capability backed by `find_title` over a *personal collection* resource, and a
`Recommend` capability backed by `recommend_taste` over a *top books* resource.
Claude Desktop discovers and calls these through the MCP client.

> **Intuition — this is USB for LLM tooling, and the analogy is precise.** Before
> USB, every peripheral needed its own port and its own driver per operating
> system. MCP standardises three things: how a client **discovers** what a server
> offers, how it **invokes** a capability, and how results come back. Once a tool
> speaks MCP, every MCP-capable host can use it with no bespoke work — which is
> why adoption was rapid across the industry. Note that MCP covers more than
> tools: **resources** (data the model can read) and **prompts** (reusable
> templates the server provides) are first-class, so a server can ship an entire
> curated way of working with a system rather than just a function list.

---

## 7.4 Agents

### Definition

> *"An agent is a system that autonomously pursues goals and completes tasks on a
> user's behalf."*

### The progression

```
Traditional:   Question ──→ LLM ──→ Answer

Reasoning:     Question ──→ LLM ──→ Reasoning ──→ Answer

Agent:         Question ──→ LLM ──→ Calls ──→ LLM ──→ ... ──→ Answer
```

The defining feature of the third row is **the loop**: the number of LLM calls is
not fixed in advance. The system decides when it is done.

### ReAct = Reason + Act

**Yao et al., 2022.** The loop has four positions plus an exit:

```
        Input
          ↓
      Observe ──→ Plan
          ↑         ↓
        Act ←───────┘
          ↓
       Output
```

Walking the lecture's worked example in full:

**Input.** *"My teddy bear is cold. Please do something."*
Inputs can be **manually entered** (a user question) or come from an **external
event** (a metric crossing a threshold).

**Observe.** *"The user's teddy bear is cold, which may be due to the current
temperature of the room, which is currently unknown."*
This step **synthesises previous actions and explicitly states what is currently
known**, including the model's own knowledge. It is a
**reasoning-heavy step to figure out what is needed** — note that it identifies
the *gap*, which is what makes the next step possible.

**Plan.** *"Determine the temperature of the room."*
**Detail what tasks need to be accomplished and what tools to call.**

**Act.** `get_current_room_temperature()`
Either **perform an action via an API** or **look for info in a database of
documents** (i.e. RAG is just one kind of act).

**→ back to Observe.** *"The temperature in the room is currently 65F. This is
about 5F less than an average temperature. We need to increase the room
temperature."*

**Plan.** *"Increase the temperature by 5F."*

**Act.** `increase_temperature(value=5)`

**→ Observe.** *"The thermostat is now set to 70F. This should be warm enough."*

**Output.** *"The thermostat is now set to 70F. Your teddy bear will soon feel
warmer."*

From the outside, all of that internal machinery is invisible — the user sees a
request go in and a result come out. That packaging is the **"agentic" view**: a
**thermostat agent**.

> **Intuition — why the loop must include Observe, and what actually makes this
> hard.** The naive version of tool use is one-shot: decide a call, make it,
> answer. That fails the moment a task needs more than one step, or the moment a
> tool returns something unexpected. The Observe step is where the agent
> **incorporates reality** — including reality disagreeing with the plan. It is
> what allows recovery from a failed call, refinement after partial information,
> and knowing when to stop.
>
> The hard engineering problems are all consequences of the loop. **When to
> terminate**, since a model that never concludes will loop forever and burn
> money. **Context growth**, since every observation is appended, so long tasks
> hit the window and suffer context rot. **Error compounding**, since a 95%
> reliable step run ten times gives you a 60% reliable task. Lecture 8's entire
> failure-mode taxonomy exists because of these.

### Multi-agent systems and A2A

There can be agents for many things: a **thermostat agent**, an **occupancy
agent**, an **air quality agent**, an **energy management agent**. Naturally the
question arises: **how do agents communicate with each other?**

**A2A = Agent2Agent** (Google, 2025) — a protocol for exactly that. Its core
objects:

- **AgentSkill** — one capability, with an `id`, a `description`, and
  `examples`. E.g. `id='maintain_comfort'`, description *'Maintain a comfortable
  T'*, examples `['Set to focus mode while I'm working']`; or
  `id='prepare_home'`, *'Sets T to target by arrival'*, examples
  `['Warm up the house to 70F when I am back']`; or `id='optimize_energy'`,
  *'Reduces energy consumption'*, examples `['Save energy overnight']`.

- **AgentCard** — the agent's public advertisement: `name='thermostat agent'`,
  `url='http://path.to.agent'`, `version='1.0.0'`, `skills=[...]`.

- **AgentExecutor** — the implementation side, with `async def execute(...)` and
  `async def cancel(...)`.

> **Intuition — MCP and A2A are not competitors, they are different layers.**
> MCP connects an agent **downwards** to tools and data. A2A connects agents
> **sideways** to each other. The AgentCard is the interesting design choice: it
> is a *discovery* document, so an agent can find out at runtime what another
> agent is able to do, in natural language, and delegate to it. Which means the
> "tool" an agent calls may itself be another agent that plans and loops. Note the
> presence of `cancel` — long-running autonomous tasks need to be interruptible,
> a requirement that does not arise for simple function calls.

### Safety

**Risks:**

- **potential for harm in the real world**,
- **example: data exfiltration** (*ToolSword*, Ye et al., 2024).

**Remediations:**

- **training steps** (*Towards Tool Use Alignment of LLMs*, Chen et al., 2024),
- **inference safeguards**,
- **benchmarks**, e.g. Agent-SafetyBench.

**...a very important topic!** And the lecture makes the point vivid by noting
that *just yesterday in the news*: Anthropic's report on **disrupting the first
reported AI-orchestrated cyber espionage campaign** (2025).

> **Intuition — why agents are a categorically different security problem.** A
> chatbot that is manipulated produces bad *text*. An agent that is manipulated
> produces bad *actions* — sends the email, executes the transfer, deletes the
> records. And the attack surface is unusual: **prompt injection** means that any
> content the agent reads is potential instruction. A malicious sentence hidden in
> a retrieved document, a web page, or an issue comment can redirect the agent,
> because the model has no reliable way to distinguish "data I was asked to
> process" from "instructions I was given". Data exfiltration is the canonical
> version: text in a document tells the agent to look up something private and
> include it in a URL it fetches. The structural mitigation is not smarter models
> but **least privilege** at the execution boundary — the agent should not *have*
> the ability to do the harmful thing.

### Closing thoughts of the lecture

- **Hallucination is a (big) problem.**
- **Reasoning abilities are a bottleneck** — finetuning helps, but it is hard;
  new capabilities are very welcome.

- **Evaluation is challenging.**
- **Good to start simple, then iterate and progressively scale up.**
- **Good to start with capable models, optimise on size later.**
- **Transparency / observability helps with user trust and debuggability.**

And a personal note from the instructors: their favourite use case for AI agents
in daily life is **coding**.

> **Intuition — the last three deserve to be read as engineering advice, because
> they contradict what people instinctively do.** "Start simple" means: try a
> plain prompt before RAG, try RAG before tools, try tools before an agent loop.
> Each step up adds failure modes multiplicatively, and most problems do not need
> the top of the ladder. "Start with capable models, optimise size later" means:
> prove the task is achievable with the best model available, *then* work
> downwards to something cheaper — because if the strongest model cannot do it,
> your prompt or your task decomposition is the problem, and you will waste days
> debugging a weak model. "Observability" means: log every step of every loop.
> When an agent fails, the question is *which* of eight steps went wrong, and
> without traces that is unanswerable — which is precisely the taxonomy Lecture 8
> builds.
