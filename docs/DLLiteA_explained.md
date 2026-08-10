# `DLLiteA.v` explained line-by-line (Coq for absolute beginners)

> This document is a scratch/teaching file. It explains **every line** of
> `DLLiteA.v` and teaches the Coq syntax as we go. It is meant to be deleted
> later once you've read it.

---

## Part 0 — What is Coq, and what is this file?

**Coq** is a *proof assistant* and a *dependently-typed programming language*.
You use it for two things at once:

1. **Define data and functions** (like ordinary programming: types, records, functions).
2. **State and prove theorems** about them, where the computer *checks* that your
   proof is 100% correct. Coq will reject a wrong proof.

This particular file, `DLLiteA.v`, does **only the first kind of work plus a few
tiny proofs**. It sets up the *syntax* and *meaning* (semantics) of a small
logic called **DL-Lite_A**.

**DL-Lite_A** is a "Description Logic" — a formal language used to describe
knowledge, similar to what powers ontologies / the semantic web (OWL). The
building blocks are:

- **Concepts** = sets of things (e.g. `Student`, `Person`). Think "nouns / categories".
- **Roles** = relationships between things (e.g. `teaches`, `works_at`). Think "verbs / links".
- **Individuals** = specific things (e.g. `alice`, `bob`).
- A **TBox** = the "schema" / general rules (e.g. "every Student is a Person").
- An **ABox** = the "data" / facts about individuals (e.g. "alice is a Student").
- A **Knowledge Base (KB)** = TBox + ABox together.

So the whole file is: *define what these things look like (syntax), define what
they mean mathematically (semantics), and give a small worked example.*

There are **no reasoning algorithms** in this file (the comment says so). The
companion file (`DLLiteA_TBox.v`, from your earlier message) starts adding the
reasoning. This doc focuses on `DLLiteA.v`, with a short appendix on the TBox
file at the end.

---

## Part 1 — A crash course in the Coq syntax you'll see here

Read this once; then the line-by-line part will make sense. Don't worry about
memorizing — refer back as needed.

### 1.1 Commands end with a period `.`
Every top-level *command* (sentence) in Coq ends with a period, exactly like an
English sentence. `Require Import Foo.` is one command. Whitespace/newlines don't
matter; the `.` is what ends the statement.

### 1.2 `Require`, `Import`, `Export` — using other files/libraries
- `Require Foo.` loads a compiled module named `Foo` so you can use it.
- `Import Foo.` also brings `Foo`'s names into scope so you can write `bar`
  instead of `Foo.bar`.
- `Require Import Foo.` does both at once.
- `Require Export Foo.` does both **and** re-exports: any file that later imports
  *this* file automatically gets `Foo` too. (It's "pass it along".)

### 1.3 `Inductive` — defining a new data type by listing its cases
This is the single most important construct in the file. It defines a brand-new
type by listing every way ("constructor") to build a value of that type.

```coq
Inductive Color :=
| Red
| Green
| Blue.
```

This says: a `Color` is `Red`, `Green`, or `Blue` — nothing else. Each `|`
introduces one **constructor** (a way to build the value). Constructors can carry
data:

```coq
Inductive Shape :=
| Circle (radius : nat)
| Rectangle (width : nat) (height : nat).
```

Here `Circle 5` is a `Shape`, and `Rectangle 3 4` is a `Shape`. The names in
parentheses (`radius`, `width`, `height`) are just labels for the arguments and
their types. This is exactly the pattern used all over `DLLiteA.v`
(`Direct (R : Role_Name.(T))`, etc.).

### 1.4 `Definition` — naming a value or a function
```coq
Definition five : nat := 5.
Definition add_one (n : nat) : nat := n + 1.
```
Pattern: `Definition NAME (arg : type) ... : return_type := body.`
- The `(arg : type)` parts are the inputs (optional).
- The `: return_type` is what it produces (often optional — Coq can infer it).
- Everything after `:=` is the body/value.

### 1.5 `Record` — a struct / bundle of named fields
```coq
Record Point := {
  x : nat;
  y : nat
}.
```
A `Record` is like a `struct`/`class` with fields. To read a field you write
`p.(x)` (see 1.8). To build one you use `{| x := 1; y := 2 |}` (see 1.9).
A record can also have fields that are *proofs* (Props), which this file uses to
attach "validity" guarantees to a value.

### 1.6 `match ... with ... end` — pattern matching (a `switch` on constructors)
```coq
Definition color_code (c : Color) : nat :=
  match c with
  | Red   => 0
  | Green => 1
  | Blue  => 2
  end.
```
`match` looks at which constructor built the value and picks the matching branch.
`=>` separates the pattern from the result. `end` closes the `match`. You can bind
the constructor's arguments: `match s with | Circle r => r | Rectangle w h => w*h end`.

### 1.7 `Prop` vs `Type`, and `Set_`
- `Type` is the type of ordinary data types (`nat : Type`, `Color : Type`).
- `Prop` is the type of **logical statements/propositions** (things that are
  either provable or not, e.g. `2 = 2`, `x < y`). `True` is the trivially-true
  proposition.
- `Set_ Individual` in this file means "a set of `Individual`s". Under the hood a
  set is represented as a *predicate*: a function `Individual -> Prop` that
  answers "is this element in the set?". This comes from the project's `Sets`
  library. So `I_C : Concept_Name.(T) -> Set_ Individual` means "given a concept
  name, give me the set of individuals in it."

### 1.8 Projection / field access with `.(field)`
`x.(f)` means "apply projection `f` to `x`". Two flavors appear:
- **Record field read:** if `p : Point`, then `p.(x)` reads its `x` field.
- **Typeclass/record accessor:** `Role_Name.(T)` reads the `T` field/component out
  of the thing called `Role_Name`. In this file, `Role_Name`, `Concept_Name`, and
  `Individual_Name` are pieces provided by the `Logic_Primitives` structure
  (see 1.11), and `.(T)` grabs "the underlying carrier type" from each. So
  `Role_Name.(T)` is simply "the type of role names".

### 1.9 Building records: `{| ... |}` and `exist`
- `{| field := value; ... |}` builds a `Record` value.
- `exist _ x proof` builds a value of a **subset type** (a "sigma"/dependent pair),
  explained in 1.10.

### 1.10 Subset types `{ x : A | P x }` and `exist`
```coq
{ n : nat | n > 0 }
```
reads as "the type of natural numbers `n` **together with a proof** that `n > 0`".
It's a *pair*: a value plus evidence it satisfies a property. To build one you use
the constructor `exist`:

```coq
exist _ 5 proof_that_5_gt_0
```
- The `_` is the property (Coq figures it out from context).
- `5` is the actual value.
- `proof_that_5_gt_0` is the proof term.

This file uses this to define `TBox` and `ABox` as "a finite set **plus a proof**
that every element is well-formed."

### 1.11 `Section`, `Context`, and typeclass variables
```coq
Section Foo.
Context {lp : Logic_Primitives}.
  ... (* stuff that may use lp *)
End Foo.
```
- `Section Foo. ... End Foo.` groups definitions and lets them share common
  parameters. When the section closes, any parameter that was actually used gets
  automatically added as an argument to those definitions. It's a scoping tool.
- `Context {lp : Logic_Primitives}.` declares a variable `lp` available to
  everything in the section. `Logic_Primitives` is a *typeclass* (a bundle of
  assumptions) defined elsewhere in the project. The **curly braces** `{ }` mean
  it's an **implicit argument**: you almost never type it by hand; Coq infers it.
  Practically, it silently supplies "what a role name is", "what a concept name
  is", "what an individual name is", etc., to the whole section.

### 1.12 Implicit arguments and the `Set` options at the top
- `Set Implicit Arguments.` tells Coq to automatically figure out ("implicit")
  arguments it can infer, so you don't have to write them.
- `Set Maximal Implicit Insertion.` makes that inference as aggressive as possible.
- `Open Scope type_scope.` selects a *notation scope*: it makes symbols like `*`
  mean **type-level pairing** (`A * B` = "pairs of an A and a B") rather than
  multiplication. That's why `Individual * Individual` below means "pairs of
  individuals" (used for role/edges).
- These three lines are configuration knobs; they change *how much you must write*,
  not *what the definitions mean*.

### 1.13 Curly `{}` vs round `()` arguments in definitions
```coq
Definition I_role {Individual} {Domain}
  (I_base : @BaseInterpretation Individual Domain) (R : Role) : ... := ...
```
- `{Individual}` and `{Domain}` in braces = **implicit** args (Coq infers them).
- `(I_base : ...)` and `(R : Role)` in parens = **explicit** args (you pass them).
- The leading `@` in `@BaseInterpretation Individual Domain` means "turn OFF the
  implicit-argument magic here and let me pass **all** arguments explicitly."
  It's used when you need to name arguments that would normally be inferred.

### 1.14 `forall`, `exists`, `/\`, connectives (logic)
Inside `Prop`s you'll see logic notation:
- `forall x, P x` — "for all x, P holds". (Universal quantifier.)
- `exists y, P y` — "there exists a y such that P holds".
- `/\` — logical **and**. `\/` — logical **or**. `->` — "implies".
- These are how mathematical statements are written; they read like English math.

### 1.15 Lemmas and proofs: `Lemma ... Proof. ... Qed.`
```coq
Lemma name : statement.
Proof.
  tactic1.
  tactic2.
Qed.
```
- `Lemma name : statement.` states something to prove (a `Theorem`/`Lemma`/`Fact`
  are synonyms).
- `Proof.` begins the proof.
- The indented words (`repeat`, `constructor`, `simpl`, `auto`, `left`, `right`,
  `reflexivity`, `apply`, `unfold`, ...) are **tactics** — commands that transform
  the current goal until it's solved. (More on the specific ones as they appear.)
- `Qed.` ends and *checks* the proof; Coq accepts it only if the goal is fully
  closed. `Admitted.` is the escape hatch: "trust me, assume it's true, I'll prove
  it later." Anything `Admitted` is an unproven hole.

### 1.16 Special project notations `!`, `!!`, `T`
These are **not** built-in Coq; they come from the project's libraries
(`Sets`, `Mem`, `TotalMem`, `ListUtils`). Based on usage:
- `X.(T)` — the carrier type of a named-type primitive (see 1.8).
- `S!` (postfix bang, as in `Domain!`, `T!`, `A!`) — "the underlying set/carrier
  of the finite/nonempty set `S`". It converts the wrapped structure into the raw
  set/predicate you can test membership against.
- `kb.(tbox)!!` (double bang after a finite set) — "the underlying **list** of
  elements". `!!` turns a `Finite_Set` into a plain `list` so you can talk about
  membership with `LIn` (list-membership). *(The exact definitions live in the
  project libraries we don't have here, but this is how they're used.)*

That's the whole syntax toolkit. Now the file itself.

---

## Part 2 — The file, block by block

### 2.1 The header comment
```coq
(**
    DL-Lite_A syntax and model-theoretic semantics.
    ...
**)
```
- `(* ... *)` is a comment in Coq. `(** ... **)` is a *documentation* comment
  (special comments that documentation tools can extract). It has **no effect** on
  behavior; it's just describing the file's purpose: it defines the *syntax* and
  the *model-theoretic semantics* (the mathematical meaning) of DL-Lite_A, mirrors
  an OCaml prototype, follows patterns from another file `EL.v`, and contains no
  reasoning algorithms yet.

### 2.2 Imports and global options
```coq
Require Import Mem TotalMem ListUtils Sets Lia List.
Require Export ConcreteDomain.
```
- Loads and imports the project libraries `Mem`, `TotalMem`, `ListUtils`, `Sets`,
  the standard tactic library `Lia` (solves linear arithmetic goals), and the
  standard `List` library (lists, `nil`, `::`, `Forall`, `NoDup`, ...).
- `Require Export ConcreteDomain.` loads `ConcreteDomain` **and** re-exports it, so
  files that import `DLLiteA` also get `ConcreteDomain`. (`ConcreteDomain` likely
  provides datatypes for concrete values/datatypes in DL-Lite_A.)

```coq
Open Scope type_scope.
Set Implicit Arguments.
Set Maximal Implicit Insertion.
```
- See 1.12. In short: make `*` mean type-pairing, and let Coq infer as many
  arguments as possible so the code is less verbose.

### 2.3 Section and context
```coq
Section DLLiteA.
Context {lp : Logic_Primitives}.
```
- Opens the section `DLLiteA` (closed at the very end with `End DLLiteA.`).
- Declares an **implicit** parameter `lp` of type `Logic_Primitives`. See 1.11.
  This is what makes `Role_Name`, `Concept_Name`, `Individual_Name`, and their
  `.(T)` carrier types available throughout. Every definition below implicitly
  depends on this `lp`.

### 2.4 `Role` — role expressions
```coq
Inductive Role :=
| Direct (R : Role_Name.(T))
| Inverse (R : Role_Name.(T)).
```
- Defines the type `Role`. A value is built **either** by:
  - `Direct r` — a plain role named `r` (where `r : Role_Name.(T)`, i.e. `r` is a
    role name), representing the relation "as-is" (e.g. `teaches`), **or**
  - `Inverse r` — the *inverse* of role `r` (e.g. `teaches⁻` = "is taught by").
- `Role_Name.(T)` is "the type of role names" (see 1.8). So a `Role` wraps a role
  name and remembers whether we mean it forwards or backwards.

### 2.5 `Basic_Concept` — basic concepts B
```coq
Inductive Basic_Concept :=
| Atomic (A : Concept_Name.(T))
| Top
| Bottom
| Exists (R : Role).
```
A basic concept is one of four things:
- `Atomic A` — a named concept `A` (e.g. `Student`). `A : Concept_Name.(T)`.
- `Top` — the concept containing **everything** in the domain (written ⊤). Always true.
- `Bottom` — the **empty** concept (written ⊥). Contains nothing.
- `Exists R` — "has at least one `R`-successor", written ∃R. This is the set of
  things `x` such that there's some `y` with the relation `R` from `x` to `y`
  (e.g. `Exists (Direct teaches)` = "things that teach something / teachers").

### 2.6 `General_Concept` — B or its negation
```coq
Inductive General_Concept :=
| Pos (B : Basic_Concept)
| Neg (B : Basic_Concept).
```
- `Pos B` — the concept `B` itself (positive).
- `Neg B` — the **negation** ("not B", written ¬B): everything *not* in `B`.
- The comment notes the core thesis work only uses `Pos`; `Neg` is kept so the
  syntax is the complete DL-Lite_A and for later "stretch" work.

### 2.7 `TBox_axiom` — a schema rule
```coq
Inductive TBox_axiom :=
| ConceptInclusion (B : Basic_Concept) (C : General_Concept)
| RoleInclusion (R R' : Role).
```
A TBox axiom (a general rule) is one of:
- `ConceptInclusion B C` — "B ⊑ C", read "every B is a C" (concept inclusion /
  subsumption). Note the left side is a *basic* concept, the right side a *general*
  concept (so it can be a negation).
- `RoleInclusion R R'` — "R ⊑ R'", read "relation R implies relation R'". Note
  `(R R' : Role)` is shorthand for two arguments `R` and `R'`, both of type `Role`.

### 2.8 `Assertion` — a data fact (ABox)
```coq
Inductive Assertion :=
| ConceptAssertion (a : Individual_Name.(T)) (B : Basic_Concept)
| RoleAssertion (a : Individual_Name.(T)) (R : Role) (b : Individual_Name.(T)).
```
An assertion is a concrete fact:
- `ConceptAssertion a B` — "individual `a` is a `B`" (e.g. "alice is a Student").
- `RoleAssertion a R b` — "individual `a` is related to individual `b` via `R`"
  (e.g. "bob teaches alice"). `a` and `b` are individual names.

### 2.9 Well-formedness predicates (currently trivial)
```coq
Definition TBox_axiom_wellformed (_ : TBox_axiom) : Prop := True.
Definition Assertion_wellformed (_ : Assertion) : Prop := True.
```
- Each takes one argument (an axiom / an assertion) and returns a `Prop`.
- The argument name is `_` (underscore) meaning "I don't use this argument".
- Both return `True` — the always-true proposition. So *right now every axiom and
  every assertion counts as well-formed*. The comments say these are placeholders
  to be tightened later (e.g. to forbid certain shapes). They exist so the `TBox`
  and `ABox` types below can carry a well-formedness proof.

### 2.10 `TBox` and `ABox` — finite sets with a proof
```coq
Definition TBox :=
  {T : Finite_Set TBox_axiom | Forall TBox_axiom_wellformed T!}.

Definition ABox :=
  {A : Finite_Set Assertion | Forall Assertion_wellformed A!}.
```
- `Finite_Set X` is a project type: a finite set of `X`s.
- `{ T : Finite_Set TBox_axiom | ... }` is a **subset type** (see 1.10): "a finite
  set `T` of axioms **together with a proof** of the property after the `|`".
- `Forall TBox_axiom_wellformed T!` — `Forall P xs` (from the `List` library) means
  "property `P` holds for every element of the list `xs`". Here `T!` is the
  underlying list/collection of the finite set `T` (postfix `!`, see 1.16). So the
  property is: "every axiom in `T` is well-formed."
- Net meaning: **a `TBox` is a finite set of axioms in which every axiom is
  well-formed.** `ABox` is the same idea for assertions.

### 2.11 `KB` — knowledge base record
```coq
Record KB := {
  tbox : TBox;
  abox : ABox
}.
```
- A record with two fields: `tbox` (a `TBox`) and `abox` (an `ABox`).
- Given `kb : KB`, you read them with `kb.(tbox)` and `kb.(abox)`.
- So a knowledge base = the schema plus the data, bundled together.

### 2.12 `BaseInterpretation` — the "meaning" structure (a model's core)
```coq
Record BaseInterpretation (Individual : Type) (Domain : Nonempty_Set Individual) := {
  I_I : Individual_Name.(T) -> Individual;
  I_C : Concept_Name.(T) -> Set_ Individual;
  I_R : Role_Name.(T) -> Set_ (Individual * Individual);

  I_I_valid : forall i, In _ Domain! (I_I i);
  I_C_valid : forall A, Included _ (I_C A) Domain!;
  I_R_valid : forall R ee, In _ (I_R R) ee ->
    In _ Domain! (fst ee) /\ In _ Domain! (snd ee)
}.
```
This is the heart of the *semantics*. An interpretation says what each name
"actually means" in some concrete world.
- The record is **parameterized** by two things (the `( ... )` after the name):
  - `Individual : Type` — the type of things in the world.
  - `Domain : Nonempty_Set Individual` — the (nonempty) universe of things.
- **Data fields** (what the names map to):
  - `I_I : Individual_Name.(T) -> Individual` — maps each individual *name* to an
    actual individual. (E.g. the name "alice" ↦ some element of the domain.)
  - `I_C : Concept_Name.(T) -> Set_ Individual` — maps each concept name to the
    **set** of individuals in it. (E.g. "Student" ↦ {all students}.)
  - `I_R : Role_Name.(T) -> Set_ (Individual * Individual)` — maps each role name
    to a set of **pairs** (the relation's edges). (E.g. "teaches" ↦ {(bob, alice), ...}.)
- **Proof fields** (validity constraints — this is why it's a record with Props):
  - `I_I_valid : forall i, In _ Domain! (I_I i)` — for every name `i`, its
    denotation lands *inside* the domain. `In _ Domain! x` reads "x is a member of
    the set `Domain!`". (The `_` is an implicit type argument Coq fills in.)
  - `I_C_valid : forall A, Included _ (I_C A) Domain!` — every concept's set is a
    **subset** of the domain. `Included _ S1 S2` = "S1 ⊆ S2".
  - `I_R_valid : forall R ee, In _ (I_R R) ee -> In _ Domain! (fst ee) /\ In _ Domain! (snd ee)`
    — for every role `R` and every edge `ee` in it, both endpoints are in the
    domain. `ee` is a pair; `fst ee`/`snd ee` are its first/second components; `/\`
    is "and". Read: "if the pair `ee` is in relation `R`, then its first and second
    elements are both in the domain."
- Together: an interpretation places every name somewhere sensible **inside** a
  fixed domain, with proofs guaranteeing nothing escapes the domain.

### 2.13 `I_role` — meaning of a role expression
```coq
Definition I_role {Individual} {Domain}
  (I_base : @BaseInterpretation Individual Domain) (R : Role)
  : Set_ (Individual * Individual) :=
  match R with
  | Direct Rn => I_base.(I_R) Rn
  | Inverse Rn => fun ee => In _ (I_base.(I_R) Rn) (snd ee, fst ee)
  end.
```
- Inputs: implicit `Individual`, `Domain`; explicit `I_base` (an interpretation)
  and `R` (a role). Output: a set of pairs (the relation's edges).
- `match R with ... end` splits on how the role was built:
  - `Direct Rn => I_base.(I_R) Rn` — a direct role just uses the interpretation's
    relation for that name.
  - `Inverse Rn => fun ee => In _ (I_base.(I_R) Rn) (snd ee, fst ee)` — the inverse
    role. Here `fun ee => ...` is an **anonymous function** ("lambda"): given a pair
    `ee`, it's in the inverse iff the *swapped* pair `(snd ee, fst ee)` is in the
    original relation. So it flips every edge's direction. (Remember: a `Set_` is a
    membership predicate, so we define the set by giving that predicate.)

### 2.14 `I_basic` — meaning of a basic concept
```coq
Definition I_basic {Individual} {Domain}
  (I_base : @BaseInterpretation Individual Domain)
  (B : Basic_Concept) : Set_ Individual :=
  match B with
  | Atomic A => I_base.(I_C) A
  | Top => Domain!
  | Bottom => Empty_set
  | Exists R => fun x => In _ Domain! x /\
      exists y, In _ Domain! y /\ In _ (I_role I_base R) (x, y)
  end.
```
Returns the set of individuals that a basic concept denotes:
- `Atomic A => I_base.(I_C) A` — a named concept means whatever the interpretation
  assigns to that name.
- `Top => Domain!` — ⊤ is the whole domain (everything).
- `Bottom => Empty_set` — ⊥ is the empty set (nothing). `Empty_set` is the set with
  no members.
- `Exists R => fun x => In _ Domain! x /\ exists y, In _ Domain! y /\ In _ (I_role I_base R) (x, y)`
  — ∃R is defined as an anonymous membership predicate over `x`: "`x` is in the
  domain **and** there exists some `y` in the domain such that the pair `(x, y)` is
  in the (interpreted) relation `R`." In words: *x has at least one R-successor.*
  This is the classic meaning of ∃R.

### 2.15 `I_general` — meaning of a general concept
```coq
Definition I_general {Individual} {Domain}
  (I_base : @BaseInterpretation Individual Domain)
  (C : General_Concept) : Set_ Individual :=
  match C with
  | Pos B => I_basic I_base B
  | Neg B => Complement _ (I_basic I_base B)
  end.
```
- `Pos B => I_basic I_base B` — a positive general concept is just the basic
  concept's meaning.
- `Neg B => Complement _ (I_basic I_base B)` — a negation is the **complement** of
  the basic concept's set: everything *not* in `B`. `Complement _ S` = "the set of
  all elements not in `S`" (the `_` is the inferred element type).

### 2.16 `is_a_model` — when does an interpretation satisfy a KB?
```coq
Definition is_a_model {Individual} {Domain}
  (I_base : @BaseInterpretation Individual Domain) (kb : KB) : Prop :=
  (forall ax, LIn kb.(tbox)!! ax ->
     match ax with
     | ConceptInclusion B C =>
         Included _ (I_basic I_base B) (I_general I_base C)
     | RoleInclusion R R' =>
         Included _ (I_role I_base R) (I_role I_base R')
     end) /\
  (forall ax, LIn kb.(abox)!! ax ->
     match ax with
     | ConceptAssertion a B =>
         In _ (I_basic I_base B) (I_base.(I_I) a)
     | RoleAssertion a R b =>
         In _ (I_role I_base R) (I_base.(I_I) a, I_base.(I_I) b)
     end).
```
Returns a `Prop`: the statement "this interpretation is a model of `kb`". It is a
conjunction (`/\`) of two universally-quantified conditions:

**Part A — every TBox axiom is respected.**
`forall ax, LIn kb.(tbox)!! ax -> match ...` reads: "for every axiom `ax`, **if**
`ax` is in the TBox's list (`LIn ... ax` = list-membership; `kb.(tbox)!!` is the
TBox as a list), **then** the following holds":
- If `ax` is `ConceptInclusion B C`: `Included _ (I_basic I_base B) (I_general I_base C)`
  — the set of `B` is a subset of the set of `C`. I.e. "every B is a C" actually
  holds in this interpretation.
- If `ax` is `RoleInclusion R R'`: `Included _ (I_role I_base R) (I_role I_base R')`
  — every edge of `R` is also an edge of `R'`.

**Part B — every ABox assertion is true.**
`forall ax, LIn kb.(abox)!! ax -> match ...`:
- `ConceptAssertion a B`: `In _ (I_basic I_base B) (I_base.(I_I) a)` — the
  individual named `a` (i.e. `I_I a`) is a member of the set `B`.
- `RoleAssertion a R b`: `In _ (I_role I_base R) (I_base.(I_I) a, I_base.(I_I) b)`
  — the pair (individual `a`, individual `b`) is an edge of relation `R`.

So `is_a_model I_base kb` is true exactly when the interpretation makes **all** the
schema rules hold **and** all the data facts true. This is the textbook definition
of "model" in logic.

### 2.17 Example names — `Parameter`
```coq
Parameter example_student example_person example_professor : Concept_Name.(T).
Parameter example_teaches example_works_at : Role_Name.(T).
Parameter example_alice example_bob : Individual_Name.(T).
```
- `Parameter x : T.` declares `x` to *exist* with type `T` **without giving a
  definition**. It's an assumption/axiom: "assume there is some concept name called
  `example_student`", etc. (Handy for examples; the comment says they become
  concrete at extraction time, i.e. when generating OCaml code.)
- Multiple names can be declared at once, sharing the type after the colon.
- So we now have three concept names, two role names, and two individual names to
  play with.

### 2.18 The example TBox axioms
```coq
Definition example_tbox_axioms : list TBox_axiom :=
  ConceptInclusion (Atomic example_student) (Pos (Atomic example_person)) ::
  ConceptInclusion (Atomic example_professor) (Pos (Atomic example_person)) ::
  RoleInclusion (Direct example_teaches) (Direct example_works_at) ::
  nil.
```
- Type is `list TBox_axiom` — an ordinary list of axioms.
- `::` (pronounced "cons") prepends an element to a list; `nil` is the empty list.
  So `a :: b :: c :: nil` is the three-element list `[a; b; c]`.
- The three axioms mean:
  1. `Student ⊑ Person` (every student is a person),
  2. `Professor ⊑ Person` (every professor is a person),
  3. `teaches ⊑ works_at` (teaching someone implies working at / with them).
- **Order matters here** because later proofs navigate the list positionally. This
  is exactly why, in your `DLLiteA_TBox.v`, the proof used `do 2 right; left` to
  reach the *third* element (the role inclusion): it's 3rd in this list.

### 2.19 The example ABox assertions
```coq
Definition example_abox_assertions : list Assertion :=
  ConceptAssertion example_alice (Atomic example_student) ::
  ConceptAssertion example_bob (Atomic example_professor) ::
  RoleAssertion example_bob (Direct example_teaches) example_alice ::
  nil.
```
Three facts:
1. `alice : Student` (alice is a student),
2. `bob : Professor` (bob is a professor),
3. `bob teaches alice`.

### 2.20 Well-formedness proofs for the examples
```coq
Lemma example_tbox_wellformed :
  Forall TBox_axiom_wellformed example_tbox_axioms.
Proof.
  repeat constructor; simpl; auto.
Qed.
```
- **Statement:** every axiom in `example_tbox_axioms` is well-formed.
- **Proof tactics:**
  - `repeat constructor` — `constructor` proves a goal by applying the appropriate
    constructor of the goal's type. Here the goal is a `Forall`, whose proof is
    built from `Forall_nil` (empty case) and `Forall_cons` (add one element).
    `repeat` applies it over and over, peeling the list element by element until
    only the tiny per-element goals remain.
  - `simpl` — simplifies/computes. Here it unfolds `TBox_axiom_wellformed _` to
    `True`.
  - `auto` — a small automation tactic that closes easy goals (like proving
    `True`). Semicolons `;` chain tactics: "do `repeat constructor`, then on every
    resulting subgoal do `simpl`, then `auto`."
- Because `TBox_axiom_wellformed` is defined as `True`, every element is trivially
  well-formed, so this closes. `Qed.` checks and seals it.

```coq
Lemma example_abox_wellformed :
  Forall Assertion_wellformed example_abox_assertions.
Proof.
  repeat constructor; simpl; auto.
Qed.
```
- Same idea for the ABox. Trivially true because `Assertion_wellformed` is `True`.

### 2.21 The `NoDup` obligations — left as holes
```coq
Lemma example_tbox_nodup : NoDup example_tbox_axioms.
Proof. Admitted.

Lemma example_abox_nodup : NoDup example_abox_assertions.
Proof. Admitted.
```
- `NoDup xs` (from `List`) means "the list `xs` has **no duplicate** elements".
  A `Finite_Set` likely requires this (a set shouldn't repeat elements).
- `Proof. Admitted.` — **not actually proven**. `Admitted` accepts the statement as
  an assumption so the file compiles, but flags it as an unfinished obligation.
  These are real "to-do" holes: someone still needs to prove the example lists have
  no duplicates. (They obviously don't, but Coq wants a formal proof.)
- Beginner takeaway: `Admitted` = "I promise this is true; check later." A file
  with `Admitted` lemmas is *incomplete* even though it compiles.

### 2.22 Packaging the finite sets: `example_tbox`, `example_abox`
```coq
Definition example_tbox : TBox :=
  exist _ (exist _ example_tbox_axioms example_tbox_nodup)
    example_tbox_wellformed.
```
Recall (2.10) `TBox = { T : Finite_Set TBox_axiom | Forall ... T! }`, and a
`Finite_Set` is itself a subset type "a list + a `NoDup` proof". So building a
`TBox` value is **two nested `exist`s**:
- **Inner** `exist _ example_tbox_axioms example_tbox_nodup` builds the
  `Finite_Set TBox_axiom`: the list plus its no-duplicates proof.
- **Outer** `exist _ (that finite set) example_tbox_wellformed` builds the `TBox`:
  the finite set plus the "all well-formed" proof.
- Each `_` is the property Coq infers. Read it as: "here is the data, and here is
  the proof it satisfies the required property," done twice (once per subset layer).

```coq
Definition example_abox : ABox :=
  exist _ (exist _ example_abox_assertions example_abox_nodup)
    example_abox_wellformed.
```
- Identical construction for the ABox.

### 2.23 The example knowledge base
```coq
Definition example_kb : KB := {|
  tbox := example_tbox;
  abox := example_abox;
|}.
```
- Builds a `KB` record value (see 1.9) with `{| field := value; ... |}` syntax,
  filling `tbox` with `example_tbox` and `abox` with `example_abox`. The trailing
  `;` before `|}` is allowed. This is the complete little world: schema + data.

### 2.24 Closing the section
```coq
End DLLiteA.
```
- Closes `Section DLLiteA`. At this point, every definition that used the implicit
  `lp : Logic_Primitives` gets it added as an argument automatically (see 1.11).
  Outside the section, these names are available (e.g. `Role`, `TBox`, `is_a_model`,
  `example_kb`), which is exactly why your `DLLiteA_TBox.v` can `Require Import
  DLLiteA` and use `example_tbox`, `Basic_Concept`, `Role`, etc.

---

## Part 3 — Big picture recap

The file builds up in layers, each using the one before:

1. **Vocabulary types** (`Role`, `Basic_Concept`, `General_Concept`) — the pieces
   of the language.
2. **Statement types** (`TBox_axiom`, `Assertion`) — rules and facts built from the
   vocabulary.
3. **Well-formedness** (`*_wellformed`) — placeholder "is this legal?" checks
   (currently always yes).
4. **Containers** (`TBox`, `ABox`, `KB`) — finite, well-formed collections bundled
   into a knowledge base.
5. **Semantics** (`BaseInterpretation`, `I_role`, `I_basic`, `I_general`) — how to
   turn syntax into actual sets/relations inside a domain.
6. **Satisfaction** (`is_a_model`) — the precise condition for an interpretation to
   "make the knowledge base true".
7. **A worked example** (`example_*`) — concrete students/professors/teaching, with
   two genuine to-do proof holes (`NoDup`, marked `Admitted`).

If you understand this file, the companion `DLLiteA_TBox.v` is the next step: it
adds *reasoning* — rules to **derive** new inclusions from the TBox (its
`closes_concept` / `closes_role` inductive relations), rather than just defining
what a model is.

---

## Appendix — quick pointers on the companion `DLLiteA_TBox.v`

A few constructs there that build directly on the above:

- `Inductive closes_concept (T : TBox) : Basic_Concept -> Basic_Concept -> Prop`
  — an **inductive relation** (not a data type): it defines *when* `T` derives
  `B1 ⊑ B2`, via constructors that are inference rules (`cc_axiom` = it's literally
  in the TBox, `cc_refl` = everything includes itself, `cc_trans` = chain two
  inclusions). This is how you define provability/derivation in Coq.
- The proof `do 2 right; left; reflexivity` navigates the disjunction produced by
  list membership `LIn` on the 3-element `example_tbox_axioms`: `right` skips an
  element, `left` selects the current one, `reflexivity` proves the chosen axiom
  equals the target. Since the role inclusion is the **3rd** item, you skip twice
  (`do 2 right`) then take it (`left`). This ties directly to the list order in
  2.18.
- `role_inverse_involutive` proven by `intros [Rn | Rn]; reflexivity` — `intros`
  introduces the `forall`-bound `R`, and `[Rn | Rn]` immediately **case-splits** it
  into its two constructors (`Direct`/`Inverse`); each case computes to itself, so
  `reflexivity` (proving `x = x`) closes both.

Delete this file whenever you're done reading.
