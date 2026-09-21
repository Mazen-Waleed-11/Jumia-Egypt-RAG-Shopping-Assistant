"""
Jumia Egypt RAG Shopping Assistant
──────────────────────────────────
Conversational product recommender over the Jumia Egypt catalog.

Pipeline per turn:
    1. Resolve references (#1, "the second one") against last shown products
    2. LLM-rewrite the user query using chat history (gives retrieval real signal)
    3. Parse explicit price filter from rewritten query
    4. FAISS retrieve top-K candidates, soft-filter by detected sub-category
    5. Show numbered candidates to user
    6. LLM generates the final answer with history + candidates as context
    7. Append turn to history, cache last candidates for follow-ups

Local-only — no external API costs. Requires Ollama running at localhost:11434.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
import requests
from colorama import Fore, Style, init
from sentence_transformers import SentenceTransformer

init(autoreset=True)

# ── Config ────────────────────────────────────────────────────────────────
_RAG_DIR      = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH    = os.path.join(_RAG_DIR, "data", "jumia.index")
METADATA_PATH = os.path.join(_RAG_DIR, "data", "jumia_meta.pkl")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# gemma2:2b is fast on CPU and accurate enough with the prompts in this file.
# Set OLLAMA_MODEL=phi3:latest if you have GPU/RAM headroom for slower but
# slightly more articulate answers.
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

TOP_K              = 5    # candidates shown to user / sent to LLM
FETCH_K_MULTIPLIER = 8    # over-fetch before filtering
MAX_HISTORY_TURNS  = 4    # turns kept in prompt + used for query rewriting
# ─────────────────────────────────────────────────────────────────────────


# ── Sub-category routing ─────────────────────────────────────────────────
# Maps a "user intent" sub-category to (must-have keywords, must-NOT-have keywords).
# The product name + CSV Category column are searched. None means no filter.
_PHONE_ACCESSORY_EXCLUSIONS = [
    "case", "cover", "shell", "skin", "pouch", "sleeve",
    "screen protector", "tempered glass", "glass protector",
    "charger", "charging cable", "data cable", "usb cable",
    "cable", "adapter", "dock", "stand", "holder", "mount",
    "earphone", "earbud", "headphone", "headset",
    "power bank", "powerbank",
    "memory card", "microsd", "sim tray", "selfie stick",
    "watch", "smartwatch", "tracker", "fitness band",
]

_LAPTOP_ACCESSORY_EXCLUSIONS = [
    "bag", "backpack", "satchel", "carrier",
    "sleeve", "case", "cover", "skin",
    "stand", "holder", "mount", "tray", "table", "desk", "bed",
    "cooler", "cooling pad", "fan",
    "lock", "cleaner", "wipe",
    "lapdesk", "lap desk",
]


SUBCATEGORY_RULES: dict[str, dict] = {
    "phone": {
        # Permissive: match if any common phone signal is present (incl.
        # 'dual sim' for Nokia-like names that omit the word "phone").
        "any": ["phone", "smartphone", "iphone", "galaxy", "redmi", "tecno",
                "infinix", "honor", "android", "5g mobile", "4g mobile",
                "dual sim", "dual-sim", "nokia", "oppo", "realme", "vivo",
                "itel", "huawei", "samsung galaxy"],
        "none": ["tablet", "ipad", "laptop", "notebook"] + _PHONE_ACCESSORY_EXCLUSIONS,
        "csv_categories": ["Phones and Tablets"],
    },
    "tablet": {
        "any": ["tablet", "ipad"],
        "none": ["laptop", "phone"] + _PHONE_ACCESSORY_EXCLUSIONS,
        "csv_categories": ["Phones and Tablets"],
    },
    "laptop": {
        "any": ["laptop", "notebook", "macbook", "chromebook"],
        "none": _LAPTOP_ACCESSORY_EXCLUSIONS,
        "csv_categories": ["Computing"],
    },
    "monitor": {
        "any": ["monitor", "display"],
        "none": ["stand"],
        "csv_categories": ["Computing"],
    },
    "headphones": {
        "any": ["headphone", "headset", "earphone", "earbud", "airpod", "tws"],
        "none": [],
        "csv_categories": None,  # can be in either
    },
    "powerbank": {
        "any": ["power bank", "powerbank"],
        "none": [],
        "csv_categories": None,
    },
    "charger": {
        "any": ["charger", "charging cable", "fast charge"],
        "none": ["power bank"],
        "csv_categories": None,
    },
    "storage": {
        "any": ["ssd", "hard drive", "hard disk", "flash drive", "usb stick",
                "memory card", "microsd"],
        "none": [],
        "csv_categories": ["Computing"],
    },
    "mouse": {
        "any": ["mouse"],
        "none": ["mousepad"],
        "csv_categories": ["Computing"],
    },
    "keyboard": {
        "any": ["keyboard"],
        "none": [],
        "csv_categories": ["Computing"],
    },
}

# User-text triggers that decide which sub-category we lock
SUBCATEGORY_TRIGGERS = {
    "phone":      ["phone", "mobile", "smartphone", "iphone", "galaxy",
                   "redmi", "xiaomi phone", "samsung phone"],
    "tablet":     ["tablet", "ipad"],
    "laptop":     ["laptop", "notebook", "macbook", "chromebook"],
    "monitor":    ["monitor", "screen", "display"],
    "headphones": ["headphone", "headset", "earphone", "earbud", "airpod", "tws"],
    "powerbank":  ["power bank", "powerbank"],
    "charger":    ["charger"],
    "storage":    ["ssd", "hard drive", "hard disk", "flash drive",
                   "memory card", "microsd"],
    "mouse":      ["mouse"],
    "keyboard":   ["keyboard"],
}


def detect_subcategory(text: str) -> Optional[str]:
    text = text.lower()
    # Longest match wins (so "power bank" beats "phone case" etc.)
    best: tuple[int, Optional[str]] = (0, None)
    for sub, triggers in SUBCATEGORY_TRIGGERS.items():
        for t in triggers:
            if t in text and len(t) > best[0]:
                best = (len(t), sub)
    return best[1]


def item_matches_subcategory(item: dict, sub: str) -> bool:
    rules = SUBCATEGORY_RULES.get(sub)
    if not rules:
        return True
    # Use ONLY the product Name for keyword checks. Including the CSV Category
    # ("Phones and Tablets") would inject the word "tablets" into every phone
    # row and trip the must-NOT-contain filter.
    name = (item.get("Name") or "").lower()
    csv_cats = rules.get("csv_categories")
    if csv_cats and item.get("Category") not in csv_cats:
        return False
    must_any = rules.get("any") or []
    if must_any and not any(kw in name for kw in must_any):
        return False
    must_none = rules.get("none") or []
    if any(kw in name for kw in must_none):
        return False
    return True


# ── Price parsing ─────────────────────────────────────────────────────────
def _to_num(x: str) -> float:
    x = x.replace(",", "").strip().lower()
    if x.endswith("k"):
        return float(x[:-1]) * 1000
    return float(x)


def parse_price_filter(text: str) -> tuple[Optional[float], Optional[float]]:
    """Returns (min, max). Either may be None when not specified."""
    t = text.lower().replace("egp", "").replace("le", "")
    # Strip commas inside numbers ("10,000" -> "10000") so the regex matches.
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"(\d),(\d)", r"\1\2", t)  # repeat for "1,000,000"

    # range: "between 5000 and 10000", "5k - 10k", "5000 to 10000"
    m = re.search(
        r"(\d+\.?\d*\s*k?)\s*(?:to|-|and|–)\s*(\d+\.?\d*\s*k?)", t
    )
    if m:
        return _to_num(m.group(1)), _to_num(m.group(2))

    # max: "under 10k", "below 5000", "less than 8000", "max 6000"
    m = re.search(
        r"(?:under|below|less\s+than|max(?:imum)?|cheaper\s+than|up\s+to|≤|<=?)"
        r"\s+(\d+\.?\d*\s*k?)", t
    )
    if m:
        return None, _to_num(m.group(1))

    # min: "above 5000", "more than 5k", "at least 3000"
    m = re.search(
        r"(?:above|over|more\s+than|at\s+least|≥|>=?)\s+(\d+\.?\d*\s*k?)", t
    )
    if m:
        return _to_num(m.group(1)), None

    # bare "around 8000"
    m = re.search(r"(?:around|about|near)\s+(\d+\.?\d*\s*k?)", t)
    if m:
        target = _to_num(m.group(1))
        return target * 0.8, target * 1.2

    return None, None


# ── Reference resolution ──────────────────────────────────────────────────
ORDINAL_WORDS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "last": -1,
}

REFERENCE_RE = re.compile(
    r"(?:#\s*(\d+)|"
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)"
    r"(?:\s+one)?\b|"
    r"\b(?:option|product|item|number)\s+(\d+)\b)",
    re.IGNORECASE,
)


def resolve_reference(query: str, cached: list[dict]) -> Optional[dict]:
    """If the user is referring to a specific cached product, return it."""
    if not cached:
        return None
    m = REFERENCE_RE.search(query)
    if not m:
        return None
    if m.group(1):
        idx = int(m.group(1))
    elif m.group(2):
        word = m.group(2).lower()
        idx = ORDINAL_WORDS.get(word)
        if idx is None:
            return None
    else:
        idx = int(m.group(3))

    if idx == -1:
        return cached[-1]
    if 1 <= idx <= len(cached):
        return cached[idx - 1]
    return None


# ── Follow-up classification (broader than before) ───────────────────────
FOLLOWUP_RE = re.compile(
    r"\b(price|cost|how\s+much|spec(?:s|ification)?s?|tell\s+me\s+more|"
    r"details?|features?|compare|vs\.?|versus|which\s+is\s+better|"
    r"cheaper|smaller|bigger|better|the\s+(?:first|second|third|last|other)|"
    r"that\s+one|this\s+one|it|them|those|these|"
    r"and\s+a|what\s+about|how\s+about|any\s+other)\b",
    re.IGNORECASE,
)


def is_followup(query: str) -> bool:
    return bool(FOLLOWUP_RE.search(query))


# ── Resource loading ─────────────────────────────────────────────────────
@dataclass
class Resources:
    index: faiss.Index
    metadata: list[dict]
    embed_model: SentenceTransformer


def load_resources() -> Resources:
    print(Fore.CYAN + "Loading FAISS index ... ", end="", flush=True)
    index = faiss.read_index(INDEX_PATH)
    print(Fore.GREEN + f"ok ({index.ntotal} vectors)")

    print(Fore.CYAN + "Loading metadata    ... ", end="", flush=True)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(Fore.GREEN + f"ok ({len(metadata)} products)")

    print(Fore.CYAN + "Loading embed model ... ", end="", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    print(Fore.GREEN + "ok\n")

    return Resources(index, metadata, embed_model)


# ── Retrieval ─────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    res: Resources,
    subcategory: Optional[str] = None,
    price_filter: Optional[tuple[Optional[float], Optional[float]]] = None,
    k: int = TOP_K,
) -> list[dict]:
    min_p, max_p = price_filter or (None, None)

    vec = res.embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    fetch_k = min(k * FETCH_K_MULTIPLIER, res.index.ntotal)
    scores, idxs = res.index.search(vec, fetch_k)

    primary: list[dict] = []   # passes both subcategory + price
    secondary: list[dict] = [] # passes price only (fallback if primary empty)

    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        item = dict(res.metadata[idx])  # copy so we can attach score
        item["_score"] = float(score)

        # Skip broken / empty rows (e.g. Excel "#NAME?" artefacts)
        name = str(item.get("Name") or "").strip()
        if not name or name.startswith("#NAME") or name.lower() == "nan":
            continue

        price = float(item.get("Price", 0))
        if min_p is not None and price < min_p:
            continue
        if max_p is not None and price > max_p:
            continue

        if subcategory and not item_matches_subcategory(item, subcategory):
            secondary.append(item)
            continue

        primary.append(item)
        if len(primary) >= k:
            break

    if primary:
        return primary
    # fallback so we never return zero when products exist in price band
    return secondary[:k]


# ── LLM helpers ───────────────────────────────────────────────────────────
def _ollama(prompt: str, *, stream: bool, num_ctx: int = 2048,
            temperature: float = 0.4) -> str:
    """Call Ollama. If stream=True, prints tokens to stdout as they arrive."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_ctx": num_ctx,
                    "temperature": temperature,
                },
            },
            stream=stream,
            timeout=180,
        )
        if not stream:
            data = r.json()
            return (data.get("response") or "").strip()
        full = []
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            tok = data.get("response", "")
            print(tok, end="", flush=True)
            full.append(tok)
        print()
        return "".join(full).strip()
    except Exception as e:  # noqa: BLE001
        msg = f"[LLM error: {e}]"
        if stream:
            print(msg)
        return msg


# ── Query rewriting ───────────────────────────────────────────────────────
REWRITE_PROMPT = """You rewrite a follow-up shopping question into a single self-contained search query for a product retrieval system.

Rules:
- Keep the same PRODUCT TYPE from earlier turns (phone, laptop, etc.).
- Keep the same BRAND if mentioned earlier (Samsung, HP, etc.).
- DO NOT include specific model numbers (A07, A05s, M55, etc.) in the rewritten
  query -- the user is asking about the broader brand/category, not that one model.
- If the user says "cheaper" or "less expensive", LOWER the previous max price
  to about half of it (e.g. "under 10k" -> "under 5k").
- If the user says "more expensive" or "premium", RAISE the previous max price.
- Output ONLY ONE rewritten query as a short plain-text line.
- No quotes, no explanation, no leading verbs like "Find" -- just the query.
- If already self-contained, output it unchanged.

EXAMPLES:
History: "Samsung phone under 10000 EGP. Showed Galaxy A07 and A05s"
User: "what about cheaper ones?"
-> samsung phone under 5000 egp

History: "HP laptop around 25000. Showed HP Pavilion and Victus."
User: "any with better GPU?"
-> hp laptop with good gpu around 25000 egp

CONVERSATION:
{history}

USER (just now): {query}

REWRITTEN QUERY:"""


def rewrite_query(query: str, history: list[dict]) -> str:
    if not history:
        return query
    hist_text = "\n".join(
        f"User: {t['user']}\nAssistant: {t['assistant'][:200]}"
        for t in history[-MAX_HISTORY_TURNS:]
    )
    prompt = REWRITE_PROMPT.format(history=hist_text, query=query)
    rewritten = _ollama(prompt, stream=False, num_ctx=1024, temperature=0.0)
    rewritten = rewritten.splitlines()[0].strip().strip('"').strip("'")
    # Strip common LLM filler verbs and trailing punctuation
    rewritten = re.sub(r"^(?:find|show\s+me|search\s+for|i\s+want)\s+",
                       "", rewritten, flags=re.IGNORECASE)
    rewritten = rewritten.rstrip(".!?,").strip()
    if not rewritten or rewritten.startswith("[LLM error"):
        return query
    return rewritten


# ── Final answer prompt ──────────────────────────────────────────────────
def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def format_candidates(products: list[dict]) -> str:
    parts = []
    for i, p in enumerate(products, 1):
        feat = (p.get("Key_Features") or "")
        if feat in ("Not Available", None, ""):
            feat = "(no features listed)"
        feat = feat[:200]
        parts.append(
            f"[#{i}] {p.get('Name')}\n"
            f"     Brand: {p.get('Brand')}  |  "
            f"Category: {p.get('Category')}  |  "
            f"Price: {float(p.get('Price', 0)):,.0f} EGP\n"
            f"     Features: {feat}"
        )
    return "\n\n".join(parts)


ANSWER_PROMPT = """You are a Jumia Egypt shopping assistant.

GROUND TRUTH:
The AVAILABLE PRODUCTS section below is the ONLY source of product facts.
The conversation history is for understanding the user's intent ONLY -- do NOT
quote facts (names, prices, specs) from history; all facts must come from
AVAILABLE PRODUCTS.

RULES:
1. Recommend or describe ONLY products in AVAILABLE PRODUCTS. Never invent.
2. Always refer to products by their #-number and short name.
3. Always include the price in EGP.
4. If AVAILABLE PRODUCTS contains exactly one item, the user's question is
   about THAT item -- describe THAT item, NOT anything from history.
5. If the user asked for a constraint (brand, feature) and AVAILABLE PRODUCTS
   does not satisfy it, say so explicitly: e.g. "I couldn't find a Samsung at
   that price; here are similar alternatives". Do NOT silently switch brands.
6. Be concise. Use bullet points only when listing 2+ products.
7. End with a short invitation for follow-up (e.g. "ask about any number for
   more details").

CONVERSATION HISTORY (intent only, NOT facts):
{history}

AVAILABLE PRODUCTS (the only source of truth):
{products}

USER QUESTION:
{query}

ASSISTANT:"""


DETAIL_PROMPT = """You are a Jumia Egypt shopping assistant.

The user is asking about ONE specific product. Describe ONLY this product.
DO NOT mention any product names, brands, or prices that are not in PRODUCT below.
Ignore conversation history when listing facts -- use ONLY PRODUCT below.

PRODUCT:
{product}

USER QUESTION:
{query}

Write a short, factual paragraph about this product (name, price in EGP,
brand, key features) and end with one short follow-up suggestion.

ANSWER:"""


def build_answer_prompt(query: str, products: list[dict],
                        history: list[dict]) -> str:
    # Special path: when retrieval/reference returned exactly one product,
    # use a focused single-product prompt that small LLMs handle reliably.
    if len(products) == 1:
        return DETAIL_PROMPT.format(
            product=format_candidates(products),
            query=query,
        )
    return ANSWER_PROMPT.format(
        history=format_history(history) or "(none)",
        products=format_candidates(products) or "(none)",
        query=query,
    )


# ── Session state ────────────────────────────────────────────────────────
@dataclass
class Session:
    history: list[dict] = field(default_factory=list)
    cached: list[dict] = field(default_factory=list)
    subcategory: Optional[str] = None

    def remember(self, user: str, assistant: str) -> None:
        self.history.append({"user": user, "assistant": assistant})
        if len(self.history) > MAX_HISTORY_TURNS * 3:
            self.history = self.history[-MAX_HISTORY_TURNS * 2:]


# ── Single-turn handler ──────────────────────────────────────────────────
def handle_turn(user_query: str, res: Resources, sess: Session) -> str:
    # 1) Direct reference to a previously shown product?
    ref = resolve_reference(user_query, sess.cached)
    if ref is not None:
        print(Fore.MAGENTA + f"[Resolved reference -> {ref['Name'][:60]}]")
        products = [ref]
        rewritten = user_query
    else:
        # 2) Use history to rewrite the query (only if there is history)
        rewritten = rewrite_query(user_query, sess.history)
        if rewritten.lower() != user_query.lower():
            print(Fore.MAGENTA + f"[Rewritten -> {rewritten}]")

        # 3) Lock subcategory the first time we detect one; allow override
        detected = detect_subcategory(rewritten) or detect_subcategory(user_query)
        if detected:
            if sess.subcategory != detected:
                print(Fore.MAGENTA + f"[Subcategory: {detected}]")
            sess.subcategory = detected

        # 4) Price filter from this turn (don't carry across turns silently)
        price_filter = parse_price_filter(rewritten)

        # 5) Retrieve
        products = retrieve(
            rewritten, res,
            subcategory=sess.subcategory,
            price_filter=price_filter,
        )
        if products:
            sess.cached = products

    if not products:
        msg = "No matching products found in the catalog. Try relaxing the price or brand."
        print(Fore.RED + msg)
        return msg

    # 6) Show candidates to the user (numbered, so they can reference)
    print(Fore.YELLOW + f"\nTop {len(products)} matches:")
    for i, p in enumerate(products, 1):
        name = p.get("Name", "")[:75]
        print(Fore.YELLOW + f"  #{i} {name} - {float(p.get('Price', 0)):,.0f} EGP "
                            f"(score {p.get('_score', 0):.2f})")

    # 7) LLM answer with history + numbered candidates
    prompt = build_answer_prompt(user_query, products, sess.history)
    print()
    print(Fore.CYAN + "Assistant:" + Style.RESET_ALL)
    answer = _ollama(prompt, stream=True, num_ctx=2048, temperature=0.4)
    print()
    return answer


# ── Main loop ────────────────────────────────────────────────────────────
def chat() -> None:
    print(Fore.YELLOW + Style.BRIGHT + "\nJumia Egypt RAG Shopping Assistant")
    print(Fore.YELLOW + "Type 'exit' to quit, 'reset' to clear session.\n")

    res = load_resources()
    sess = Session()

    while True:
        try:
            q = input(Fore.GREEN + "You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            sess = Session()
            print(Fore.MAGENTA + "[Session cleared]\n")
            continue

        answer = handle_turn(q, res, sess)
        sess.remember(q, answer)
        print()


if __name__ == "__main__":
    chat()
