# Jumia Egypt RAG Shopping Assistant

A natural-language product recommender over the Jumia Egypt catalog
(982 products: 492 Phones and Tablets + 490 Computing). 100% local — no API
costs. Embeddings run on CPU; the LLM runs through a local Ollama instance.

You can:
- Ask for recommendations: *"I want a Samsung phone under 10000 EGP"*
- Drill into a specific result: *"tell me about #2"* or *"the second one"*
- Refine with follow-ups: *"what about cheaper ones?"*, *"any HP laptops around 25000?"*
- The assistant keeps the last few turns and uses them to interpret follow-ups.

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

| Component       | Choice                                     | Why                         |
|-----------------|--------------------------------------------|-----------------------------|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | ~80 MB, CPU-fast            |
| Vector store    | FAISS (`IndexFlatIP`, in-memory)           | exact cosine over 982 items |
| LLM             | Ollama (`gemma2:2b` default)               | local, free, ~5 s per turn  |
| Chat history    | in-memory list of `{user, assistant}`      | last 4 turns kept           |

## Prerequisites

- Python 3.11 (the bundled `venv/` was created with 3.11.7)
- [Ollama](https://ollama.com/download) installed and running locally on port 11434
- ~3 GB free disk for the embedding model + Ollama models

## Quick start (Windows, PowerShell)

The repo already ships with a working `venv/` and a prebuilt FAISS index
(`data/jumia.index`, `data/jumia_meta.pkl`), so steps 1 and 4 are optional on
first run.

```powershell
# 1) (Optional, only if venv/ is missing) create and populate the venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Activate the existing venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# 3) Pull a local LLM (gemma2:2b is the default the code expects)
ollama pull gemma2:2b
# Optional: slower but more articulate
ollama pull phi3

# 4) (Optional) rebuild the FAISS index from the CSV
python build_index.py
```


## Run

Make sure Ollama is running (`ollama serve` in another terminal if it isn't
already), then start the chat REPL:

```powershell
python rag.py
```

REPL commands:

| Input            | Effect                       |
|------------------|------------------------------|
| any text         | one chat turn                |
| `reset`          | clear conversation history   |
| `exit` / `quit`  | leave the REPL               |
| Ctrl-C / Ctrl-D  | leave the REPL               |

## Example session

```
You: I want a Samsung phone under 10000 EGP
[Subcategory: phone]
Top 5 matches:
  #1 Samsung Galaxy A07 - 9,790 EGP
  #2 Samsung Galaxy A05s - 7,059 EGP
  #3 HONOR X7c - 9,389 EGP
  #4 Samsung Galaxy A07 Dual SIM - 7,457 EGP
  #5 Samsung Galaxy A06 - 6,490 EGP

Assistant: Here are some Samsung phones under 10,000 EGP:
  - [#1] Samsung Galaxy A07 - 9,790 EGP
  - [#2] Samsung Galaxy A05s - 7,059 EGP
  - [#5] Samsung Galaxy A06 - 6,490 EGP
Ask about any number for more details.

You: what about cheaper ones?
[Rewritten -> samsung phones under 5000 egp]
...
```

## Files

| Path                                  | Purpose                                            |
|---------------------------------------|----------------------------------------------------|
| `rag.py`                              | Chat REPL + retrieval pipeline                     |
| `build_index.py`                      | One-shot CSV -> FAISS index builder                |
| `test_smoke.py`                       | Non-interactive end-to-end smoke test              |
| `requirements.txt`                    | Pinned Python dependencies                         |
| `data/Jumia_Real_Final_Clean.csv`     | Source catalog (982 products)                      |
| `data/jumia.index`                    | Prebuilt FAISS vectors                             |
| `data/jumia_meta.pkl`                 | Product metadata aligned to vectors                |
| `data/Jumia_Data_Cleanup_and_EDA.ipynb` | EDA notebook that produced the cleaned CSV       |
| `data/Data_IR.pdf`                    | Project report                                     |
| `data/data_IR.pbix`                   | Power BI dashboard                                 |
| `venv/`                               | Pre-populated Python 3.11 virtual environment      |

## Configuration

Override defaults via environment variables (no code changes needed):

```powershell
$env:OLLAMA_MODEL = "phi3"; python rag.py        # use phi3 instead of gemma2:2b
$env:OLLAMA_URL   = "http://other:11434/api/generate"; python rag.py
```

Tunable constants live at the top of `rag.py`:

| Constant            | Default | Meaning                                      |
|---------------------|---------|----------------------------------------------|
| `TOP_K`             | 5       | Candidates shown per turn                    |
| `FETCH_K_MULTIPLIER`| 8       | Over-fetch factor before filtering           |
| `MAX_HISTORY_TURNS` | 4       | Recent turns sent to LLM and used in rewrite |

## Smoke test

A non-interactive run that exercises the whole pipeline:

```powershell
python test_smoke.py
# or with a different LLM:
$env:OLLAMA_MODEL = "phi3"; python test_smoke.py
```

It verifies sub-category detection, price parsing, reference resolution,
follow-up classification, and runs a 3-turn conversation through Ollama.

## Design notes

- **Sub-category routing instead of strict CSV-category filtering.** The CSV
  category `Phones and Tablets` contains the substring `tablets`, which would
  reject every phone if used as a hard filter. `rag.py` uses keyword rules on
  the product *name* instead, with a soft fallback so the result set is never
  empty when there are products in the requested price band.
- **LLM query rewriting.** Follow-ups like *"cheaper ones"* are rewritten by
  the LLM using the last few turns before retrieval, so the embedding query
  carries real signal (e.g. *"samsung phones under 5000 egp"*).
- **Numbered candidates + reference resolution.** Each turn lists up to 5
  numbered candidates, and `#2` / `the second one` / `option 3` / `the last`
  are all resolved before retrieval so detail questions short-circuit the
  pipeline and use a focused single-product prompt.
- **Comma-tolerant price parser.** Handles `10,000 EGP`, `5k`, `between 5000
  and 9000`, `around 8000`, etc.

## Limitations

- Ollama on CPU: `gemma2:2b` is the default and runs ~5 s per turn. `phi3`
  (3.8B) is more articulate but ~3-5x slower on CPU.
- Catalog is small (982 products) and skewed: only Computing and Phones &
  Tablets, mostly low-to-mid price.
- Chat history is in-memory only — not persisted across `rag.py` restarts.
- All prices are in EGP; the parser does not convert currencies.
