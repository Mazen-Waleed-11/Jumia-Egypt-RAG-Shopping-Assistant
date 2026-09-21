# Jumia Egypt RAG Shopping Assistant

A natural-language product recommender over the Jumia Egypt catalog (982 products,
phones / tablets / laptops / accessories). 100% local — no API costs.

You can:
- Ask for recommendations: *"I want a Samsung phone under 10000 EGP"*
- Drill into details: *"tell me about #2"* or *"the second one"*
- Refine with follow-ups: *"what about cheaper ones?"*, *"any HP laptops around 25000?"*
- The assistant remembers the conversation and uses it to interpret follow-ups.

## Pipeline

```
User question
    |
    v
1. Reference resolution        --> "#2", "the second one"  -> use cached product
    |                                (skip retrieval)
    v
2. LLM query rewrite           --> "what about cheaper ones?"  with history
    |                                "Samsung phone under 10k"
    |                                -> "samsung phones under 5000 egp"
    v
3. Price + sub-category parse  --> (None, 5000), subcategory="phone"
    |
    v
4. FAISS retrieval (top-K)     --> all-MiniLM-L6-v2 + cosine similarity
    |                                soft sub-category filter, hard price filter
    v
5. Numbered candidates shown   --> user can refer to "#2" next turn
    |
    v
6. Ollama LLM final answer     --> history (intent only) + candidates (facts)
```

## Stack

| Component         | Choice                                 | Why                          |
|-------------------|----------------------------------------|------------------------------|
| Embedding model   | `all-MiniLM-L6-v2`                     | 80 MB, CPU-fast              |
| Vector store      | FAISS (`IndexFlatIP`, in-memory)       | exact cosine over 982 items  |
| LLM               | Ollama (`phi3:latest` default)         | local, free, follows rules   |
| Chat history      | in-memory list of {user, assistant}    | last 4 turns kept            |

## Setup (one time)

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Install Ollama from https://ollama.com/download , then pull a model:
ollama pull phi3            # 3.8B, default, better instruction-following
# or for a smaller/faster model:
ollama pull gemma2:2b

# 3. Build the FAISS index from the CSV (only once)
python build_index.py
```

## Run

```bash
# Make sure Ollama is running (auto-starts on Windows; otherwise: `ollama serve`)
python rag.py
```

Then chat. Example session:

```
You: I want a Samsung phone under 10000 EGP
[Subcategory: phone]
Top 5 matches:
  #1 Samsung Galaxy A07 - 9,790 EGP
  #2 Samsung Galaxy A05s - 7,059 EGP

Assistant: Here are some options under 10,000 EGP:
  - [#2] Samsung Galaxy A05s - 7,059 EGP
  - [#5] Samsung Galaxy A07 Dual SIM - 7,457 EGP

You: what about cheaper ones?
[Rewritten -> samsung phones under 5000 egp]
Top 4 matches:
  #1 Nokia 150 Dual SIM - 1,100 EGP
  ...
Assistant: Here are cheaper options:
  - [#1] Nokia 150 Dual SIM - 1,100 EGP

You: tell me about #2
[Resolved reference -> Nokia 5310 ...]
Assistant: The Nokia 5310 is a 2.4-inch dual-SIM phone, 999 EGP.
```

## Files

| File                          | Purpose                                |
|-------------------------------|----------------------------------------|
| `rag.py`                      | Chat REPL + retrieval pipeline         |
| `build_index.py`              | One-shot CSV -> FAISS index            |
| `test_smoke.py`               | Non-interactive end-to-end smoke test  |
| `data/Jumia_Real_Final_Clean.csv` | Source catalog (982 products)      |
| `data/jumia.index`            | FAISS vectors (built once)             |
| `data/jumia_meta.pkl`         | Product metadata aligned to vectors    |

## Configuration

Override defaults via env vars:

```bash
OLLAMA_MODEL=gemma2:2b python rag.py        # use the smaller 2B model
OLLAMA_URL=http://other:11434/api/generate python rag.py
```

Inside `rag.py`:

| Constant            | Default | Meaning                                    |
|---------------------|---------|--------------------------------------------|
| `TOP_K`             | 5       | Candidates shown per turn                  |
| `FETCH_K_MULTIPLIER`| 8       | Over-fetch factor before filtering         |
| `MAX_HISTORY_TURNS` | 4       | Recent turns sent to LLM and used in rewrite |

## Smoke test

A non-interactive smoke test exercising the whole pipeline:

```bash
python test_smoke.py
# or with the smaller LLM:
OLLAMA_MODEL=gemma2:2b python test_smoke.py
```

It verifies sub-category detection, price parsing, reference resolution,
follow-up classification, and runs a 3-turn conversation through Ollama.

## What changed vs the original `rag.py`

Bug fixes:
- Strict category filter rejected ALL Samsung phones because the CSV category
  string `Phones and Tablets` contains the word `tablets`, which collided with
  the tablet keyword set. Replaced with proper sub-category rules.
- Price parser broke on numbers with commas (e.g. `10,000`).
- Follow-up regex missed `cheaper`, `smaller`, `and a ...`, etc.

New features:
- LLM-based query rewriting using chat history -> follow-ups now actually
  retrieve the right things (e.g. "cheaper ones" -> halves the previous price).
- Numbered candidate list shown per turn so the user can refer back: `#2`,
  `the second one`, `option 3`, `the last`, etc. -- all resolved before retrieval.
- Single-product detail prompt prevents small LLMs from hallucinating facts
  out of conversation history when the user asks "tell me about #N".
- Soft fallback: if sub-category filter would empty the result set, items
  matching the price band but not the sub-category are returned instead.

## Limitations

- Ollama on CPU is slow with phi3 (~30-60 s per turn). Use `gemma2:2b` for ~5x
  speedup at the cost of weaker instruction-following.
- Catalog is small (982 products) and skewed (only Computing + Phones/Tablets).
- Chat history is in-memory only -- not persisted across `rag.py` restarts.
