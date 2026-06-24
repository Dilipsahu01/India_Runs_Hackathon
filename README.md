# 🏃 India Runs Hackathon

## *How do you find 100 perfect AI Engineers among 100,000 candidates... on a CPU... in under 5 minutes?*

Most people attacked this challenge with **more AI**.

We attacked it with **less AI**.

And that changed everything.

---

# 🤔 The Problem Nobody Talks About

Imagine opening a folder containing **100,000 candidate profiles**.

Nearly **500MB of dense JSON**.

No GPUs.

No internet.

No OpenAI APIs.

No Pinecone.

No cloud vector databases.

Just:

* ⏱️ 5 minutes
* 🖥️ CPU only
* 💾 16GB RAM
* 🚫 Zero network access

And one goal:

> Find the **Top 100 Senior AI Engineers**.

Sounds straightforward.

Until you realize that running semantic embeddings on 100,000 resumes would take **30–60 minutes**.

Which means...

**Traditional AI loses before the race even starts.**

---

# 💡 The Moment Everything Changed

Most solutions ask:

> "How do we understand 100,000 people?"

We asked:

> "Why understand all 100,000 in the first place?"

That single question led to a completely different architecture.

Instead of solving an AI problem...

we solved a **data funnel optimization problem**.

---

# ⚡ The Philosophy

Spend expensive AI compute only where it matters.

Not on 100,000 candidates.

Not on 10,000 candidates.

Only on the final few hundred.

Everything else?

Eliminate using cheap mathematics.

---

# 🎯 The 3-Stage Funnel

```text
100,000 Candidates
        │
        ▼
┌───────────────────┐
│ Stage 1           │
│ Heuristic Filters │
└───────────────────┘
        │
        ▼
~27,000 Candidates
        │
        ▼
┌───────────────────┐
│ Stage 2           │
│ BM25 Search       │
└───────────────────┘
        │
        ▼
Top 500 Candidates
        │
        ▼
┌───────────────────┐
│ Stage 3           │
│ Semantic Ranking  │
└───────────────────┘
        │
        ▼
Top 100 Candidates
```

Simple.

Fast.

Deterministic.

Production-grade.

---

# 🪤 Stage 1 — The Honeypot Nuke

Before AI touches anything...

we ask a brutal question:

### "Does this candidate even deserve expensive compute?"

Because the dataset wasn't innocent.

It was adversarial.

---

## The Trap

Some profiles looked like this:

**Current Role:** Marketing Manager

**Summary:**

> "I work with ChatGPT, RAG, embeddings, vector databases and AI systems..."

A naïve semantic model sees:

```
ChatGPT ✅
RAG ✅
Embeddings ✅
Vector DB ✅
```

and screams:

> "99% Match!"

Which is exactly what the organizers wanted.

To trap bad pipelines.

---

## Our Counterattack

Before reading fancy summaries...

we inspect reality.

If your current title says:

* Marketing Manager
* Accountant
* Graphic Designer
* Sales Executive

then...

**Goodbye.**

No embeddings.

No cosine similarity.

No wasted CPU cycles.

---

### Result

> 72,689 candidates eliminated instantly.

The AI never even sees them.

---

# ⏳ Stage 1.5 — Time Travelers Don't Exist

Another trap.

Some candidates claimed:

```
Experience : 12 years
Graduation : 2022
```

Interesting.

Apparently they started coding professionally in middle school.

---

Our vectorized Pandas checks compare:

```python
Max Graduation Year
vs
Claimed Experience
```

Impossible combinations?

Eliminated.

---

# 🚫 The Service-Firm Filter

The JD wanted:

> Product Engineers

Not career consultants.

If a candidate's entire history consisted only of:

* TCS
* Infosys
* Wipro
* HCL

they were automatically filtered.

Because matching the job description means understanding intent—

not just keywords.

---

# ⚡ Stage 2 — The Compute Neutralizer

Here's a question:

Why didn't we use Pinecone?

Or ChromaDB?

Or Qdrant?

Because they were unnecessary.

And expensive.

---

Instead, we used something surprisingly old.

Something that powers search engines.

### BM25 (Okapi)

No vectors.

No GPUs.

Just mathematics.

---

BM25 indexed roughly **27,000 resumes** and sliced them down to:

# Top 500

in only a few seconds.

---

### Query Keywords

```text
python
embedding
retrieval
rag
sentencetransformers
pinecone
qdrant
vector databases
llm
```

Cheap compute.

Massive impact.

---

# 🧠 Stage 3 — Where AI Finally Enters

Only now...

after eliminating 99.5% of the dataset...

do we spend semantic compute.

Using:

```python
all-MiniLM-L6-v2
```

CPU optimized.

Fast.

Reliable.

---

## Our Scoring Matrix

Real recruiters don't hire using cosine similarity alone.

Neither do we.

---

### 40% Semantic Match

How closely does the profile resemble the ideal candidate?

---

### 30% Experience Match

The sweet spot:

```
5–9 Years
```

Outside that range?

Heavy penalties.

---

### 30% Behavioral Reality

A perfect engineer who never replies isn't useful.

So we incorporated:

* Recruiter response rate
* Login recency
* Activity score

Because hiring isn't just about skill.

It's about availability.

---

# 🐛 The Bug Nobody Notices

Imagine two candidates getting:

```text
0.8551
0.8551
```

Pandas may reorder them differently across runs.

Which means:

Today's Top-100 ≠ Tomorrow's Top-100

Chaos.

---

## The Fix

We enforced deterministic ranking:

1. Round scores to 4 decimal places.
2. Secondary sort by candidate ID.

```python
(score DESC, candidate_id ASC)
```

Result?

Every run produces identical output.

Always.

Production engineering matters.

---

# 🏗 Repository Structure

```bash
.
├── stage1_filter.py
├── stage2_bm25.py
├── stage3_semantic_ranker.py
├── validate_submission.py
├── app.py
├── requirements.txt
└── candidates.jsonl
```

---

# 🚀 Running the Pipeline

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Stage 1

```bash
python3 stage1_filter.py candidates.jsonl stage1_passed.jsonl
```

---

## Stage 2

```bash
python3 stage2_bm25.py stage1_passed.jsonl stage2_passed.jsonl
```

---

## Stage 3

```bash
python3 stage3_semantic_ranker.py stage2_passed.jsonl team_dilip.csv
```

---

## Validate

```bash
python3 validate_submission.py team_dilip.csv
```

---

# 🎮 Sandbox App

A lightweight Gradio application is included.

It reproduces the exact ranking pipeline on a smaller sample dataset and is suitable for deployment on Hugging Face Spaces.

```bash
python3 app.py
```

---

# 📌 Tech Stack

* Python
* Pandas
* rank_bm25
* Sentence Transformers
* Gradio
* NumPy

---

# What Makes This Interesting?

This project wasn't about building the biggest AI system.

It was about knowing **when not to use AI.**

The smartest part of the pipeline isn't Stage 3.

It's Stage 1.

Because intelligence isn't always about understanding everything.

Sometimes...

it's about knowing what to ignore.

---

# Final Numbers

```text
100,000 Candidates
↓
72,689 eliminated by heuristics
↓
27,000 survivors
↓
BM25 extracts Top 500
↓
Semantic ranking
↓
Top 100 finalists
```

---

# 🏁 Conclusion

Anyone can throw embeddings at a problem.

But engineering under constraints is different.

This project demonstrates that with the right architecture:

> Cheap compute + deterministic filters + selective AI

can outperform brute force.

And perhaps that's the most interesting lesson of all.

---

### Built for the Redrob India Runs Hackathon 🚀
