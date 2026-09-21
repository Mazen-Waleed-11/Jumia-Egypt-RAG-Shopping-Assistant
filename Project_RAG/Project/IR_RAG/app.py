import os
import sys
import inspect
from pathlib import Path

# Keep the backend bootstrap at the very top of the file.
current_directory = os.path.dirname(os.path.abspath(__file__))
ir_rag_directory = os.path.join(current_directory, "IR_RAG")
if ir_rag_directory not in sys.path:
    sys.path.insert(0, ir_rag_directory)

import rag

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Jumia Egypt Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE_NAME = "Jumia_Real_Final_Clean.csv"
REQUIRED_COLUMNS = {"Category", "Name", "Brand", "Price", "Key_Features"}
PRICE_TIER_ORDER = [
    "Entry Saver",
    "Smart Value",
    "Premium Pick",
    "Flagship Range",
]
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ─────────────────────────────────────────────
           GOOGLE FONTS — Refined pairing
        ───────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&family=DM+Serif+Display:ital@0;1&display=swap');

        /* ─────────────────────────────────────────────
           DESIGN TOKENS
        ───────────────────────────────────────────── */
        :root {
            /* Backgrounds */
            --bg-canvas:        #eef4f0;
            --bg-orb-1:         rgba(16, 185, 129, 0.18);
            --bg-orb-2:         rgba(99, 179, 237, 0.14);
            --bg-orb-3:         rgba(251, 191, 36, 0.10);

            /* Glass surfaces */
            --glass-fill:       rgba(255, 255, 255, 0.62);
            --glass-fill-heavy: rgba(255, 255, 255, 0.80);
            --glass-border:     rgba(255, 255, 255, 0.80);
            --glass-border-dim: rgba(200, 230, 215, 0.55);
            --glass-blur:       blur(18px) saturate(170%);
            --glass-blur-sm:    blur(10px) saturate(150%);

            /* Shadows */
            --shadow-xs:  0 2px  8px rgba(15, 40, 30, 0.06);
            --shadow-sm:  0 4px 16px rgba(15, 40, 30, 0.08);
            --shadow-md:  0 8px 32px rgba(15, 40, 30, 0.10);
            --shadow-lg:  0 16px 48px rgba(15, 40, 30, 0.12);

            /* Typography */
            --font-body:    'DM Sans', sans-serif;
            --font-display: 'DM Serif Display', serif;

            /* Palette */
            --ink-900:    #0f1f17;
            --ink-700:    #1e3a2f;
            --ink-500:    #3d5a4a;
            --ink-300:    #7a9e8e;
            --ink-100:    #c8ddd5;

            --mint-600:   #059669;
            --mint-500:   #10b981;
            --mint-400:   #34d399;
            --mint-100:   #d1fae5;
            --mint-50:    #f0fdf8;

            --slate-600:  #475569;
            --slate-400:  #94a3b8;
            --slate-100:  #f1f5f9;
        }

        /* ─────────────────────────────────────────────
           GLOBAL RESET & BASE
        ───────────────────────────────────────────── */
        *, *::before, *::after { box-sizing: border-box; }

        html, body, .stApp {
            font-family: var(--font-body) !important;
            color: var(--ink-700) !important;
        }

        /* ─────────────────────────────────────────────
           ATMOSPHERIC BACKGROUND
           Three soft bokeh orbs + fine dot grid
        ───────────────────────────────────────────── */
        .stApp {
            background-color: var(--bg-canvas) !important;
            background-image:
                /* dot grid */
                radial-gradient(circle, rgba(16, 185, 129, 0.18) 1px, transparent 1px),
                /* orb top-right */
                radial-gradient(ellipse 55% 45% at 90% 5%,  var(--bg-orb-1) 0%, transparent 70%),
                /* orb bottom-left */
                radial-gradient(ellipse 45% 55% at 5% 95%,  var(--bg-orb-2) 0%, transparent 70%),
                /* orb centre */
                radial-gradient(ellipse 60% 40% at 50% 50%, var(--bg-orb-3) 0%, transparent 60%) !important;
            background-size: 28px 28px, 100% 100%, 100% 100%, 100% 100% !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 3.5rem;
            max-width: 1280px;
        }

        /* ─────────────────────────────────────────────
           TYPOGRAPHY DEFAULTS
        ───────────────────────────────────────────── */
        h1, h2, h3, h4 {
            font-family: var(--font-display) !important;
            color: var(--ink-900) !important;
            letter-spacing: -0.01em;
        }
        p, li, label, span, div {
            color: var(--ink-700) !important;
        }
        .stMarkdown p, .stMarkdown li {
            color: var(--ink-700) !important;
            font-size: 0.97rem;
            line-height: 1.7;
        }

        /* ─────────────────────────────────────────────
           HERO SECTION — Morph-Glass card
        ───────────────────────────────────────────── */
        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 3rem 3rem 2.75rem;
            border-radius: 28px;
            background: var(--glass-fill-heavy);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1.5px solid var(--glass-border);
            box-shadow: var(--shadow-lg),
                        inset 0 1px 0 rgba(255,255,255,0.9);
            margin-bottom: 2rem;
        }
        /* Decorative inner glow blob */
        .hero-shell::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 320px; height: 320px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(52, 211, 153, 0.22) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-shell::after {
            content: '';
            position: absolute;
            bottom: -40px; left: 20%;
            width: 200px; height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(99, 179, 237, 0.15) 0%, transparent 70%);
            pointer-events: none;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--mint-100), rgba(52,211,153,0.2));
            border: 1px solid rgba(16, 185, 129, 0.30);
            color: var(--mint-600) !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 1rem 0 0.6rem;
            font-size: clamp(2.2rem, 4.5vw, 3.6rem) !important;
            font-weight: 400 !important;   /* DM Serif Display looks best at 400 */
            line-height: 1.12;
            color: var(--ink-900) !important;
        }
        .hero-title em {
            font-style: italic;
            color: var(--mint-500) !important;
        }
        .hero-copy {
            color: var(--ink-500) !important;
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 620px;
        }

        /* ─────────────────────────────────────────────
           TABS
        ───────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.6rem;
            margin-bottom: 1.5rem;
            background: transparent !important;
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3.2rem;
            padding: 0 1.4rem;
            border-radius: 14px;
            background: var(--glass-fill) !important;
            backdrop-filter: var(--glass-blur-sm);
            -webkit-backdrop-filter: var(--glass-blur-sm);
            border: 1.5px solid var(--glass-border-dim) !important;
            color: var(--ink-500) !important;
            font-family: var(--font-body);
            font-weight: 600;
            font-size: 0.92rem;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-xs);
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.82) !important;
            border-color: rgba(16,185,129,0.30) !important;
            color: var(--mint-600) !important;
            box-shadow: var(--shadow-sm);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(255,255,255,0.95) !important;
            border-color: rgba(16,185,129,0.50) !important;
            color: var(--mint-600) !important;
            box-shadow: var(--shadow-md),
                        inset 0 1px 0 rgba(255,255,255,1);
        }
        /* Remove default underline indicator */
        .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

        /* ─────────────────────────────────────────────
           PANEL CONTAINERS  (smart_container → border=True)
        ───────────────────────────────────────────── */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--glass-fill) !important;
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1.5px solid var(--glass-border) !important;
            border-radius: 24px !important;
            box-shadow: var(--shadow-md),
                        inset 0 1px 0 rgba(255,255,255,0.85) !important;
            padding: 1.75rem !important;
            transition: box-shadow 0.25s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: var(--shadow-lg),
                        inset 0 1px 0 rgba(255,255,255,0.9) !important;
        }

        .panel-title {
            font-family: var(--font-display);
            font-size: 1.3rem;
            font-weight: 400;
            color: var(--ink-900) !important;
            margin-bottom: 0.25rem;
        }
        .panel-copy {
            color: var(--ink-300) !important;
            font-size: 0.9rem;
            margin-bottom: 1.4rem;
        }

        /* ─────────────────────────────────────────────
           CHAT MESSAGES — Morph-Glass bubbles
        ───────────────────────────────────────────── */
        div[data-testid="stChatMessage"] {
            background: var(--glass-fill-heavy) !important;
            backdrop-filter: var(--glass-blur-sm) !important;
            -webkit-backdrop-filter: var(--glass-blur-sm) !important;
            border: 1.5px solid var(--glass-border) !important;
            border-radius: 20px !important;
            padding: 1.1rem 1.3rem !important;
            box-shadow: var(--shadow-sm) !important;
            margin-bottom: 0.9rem !important;
            transition: box-shadow 0.2s ease;
        }
        div[data-testid="stChatMessage"]:hover {
            box-shadow: var(--shadow-md) !important;
        }
        /* User avatar */
        div[data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(135deg, var(--slate-100), #e2e8f0) !important;
            color: var(--slate-600) !important;
            border: 1.5px solid rgba(148,163,184,0.3) !important;
        }
        /* Assistant avatar */
        div[data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, var(--mint-100), rgba(52,211,153,0.25)) !important;
            color: var(--mint-600) !important;
            border: 1.5px solid rgba(16,185,129,0.25) !important;
        }
        /* Chat text */
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span {
            color: var(--ink-700) !important;
            font-size: 0.96rem;
            line-height: 1.7;
        }
        div[data-testid="stChatMessage"] strong {
            color: var(--ink-900) !important;
        }

        /* ─────────────────────────────────────────────
           CHAT INPUT — Nuclear light-theme override
           Streamlit injects a dark color-scheme on the
           textarea via the browser's prefers-color-scheme.
           We have to override every layer aggressively.
        ───────────────────────────────────────────── */

        /* Outer wrapper */
        div[data-testid="stChatInput"],
        div[data-testid="stChatInputContainer"] {
            background: transparent !important;
            color-scheme: light !important;
        }

        /* The actual styled box Streamlit renders */
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] > div > div,
        div[data-testid="stChatInput"] > div > div > div,
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottom"] > div > div {
            background-color: #ffffff !important;
            background:       #ffffff !important;
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1.5px solid rgba(16, 185, 129, 0.40) !important;
            border-radius: 18px !important;
            box-shadow: var(--shadow-sm),
                        inset 0 1px 0 rgba(255,255,255,0.95) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            color-scheme: light !important;
        }

        /* Focus ring */
        div[data-testid="stChatInput"] > div:focus-within,
        div[data-testid="stChatInput"] > div > div:focus-within {
            border-color: var(--mint-500) !important;
            box-shadow: var(--shadow-md),
                        0 0 0 3px rgba(16,185,129,0.14),
                        inset 0 1px 0 rgba(255,255,255,1) !important;
        }

        /* The textarea itself — hit every possible selector */
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInputTextArea"] textarea,
        textarea[data-testid="stChatInputTextArea"],
        [data-baseweb="textarea"] textarea,
        .stChatInput textarea {
            background-color: #ffffff !important;
            background:       #ffffff !important;
            color:            #0f1f17 !important;
            -webkit-text-fill-color: #0f1f17 !important;
            font-family: var(--font-body) !important;
            font-size: 0.97rem !important;
            caret-color: var(--mint-500) !important;
            color-scheme: light !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Placeholder text */
        div[data-testid="stChatInput"] textarea::placeholder,
        div[data-testid="stChatInputTextArea"] textarea::placeholder {
            color: #7a9e8e !important;
            -webkit-text-fill-color: #7a9e8e !important;
            opacity: 1 !important;
        }

        /* Send button icon */
        div[data-testid="stChatInput"] svg,
        div[data-testid="stChatInput"] [data-testid="chatInputSubmitButton"] svg {
            fill:  var(--mint-500) !important;
            color: var(--mint-500) !important;
        }
        div[data-testid="stChatInput"] button,
        div[data-testid="stChatInput"] [data-testid="chatInputSubmitButton"] {
            background: transparent !important;
            border-radius: 12px !important;
            transition: background 0.15s ease;
        }
        div[data-testid="stChatInput"] button:hover {
            background: var(--mint-50) !important;
        }

        /* Bottom sticky bar Streamlit wraps the input in */
        div[data-testid="stBottom"] {
            background: transparent !important;
            color-scheme: light !important;
        }

        /* ─────────────────────────────────────────────
           METRIC CARDS — Elevated glass panels
        ───────────────────────────────────────────── */
        .metric-card {
            position: relative;
            overflow: hidden;
            background: var(--glass-fill-heavy);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            padding: 1.75rem 1.75rem 1.5rem;
            border-radius: 22px;
            border: 1.5px solid var(--glass-border);
            box-shadow: var(--shadow-md),
                        inset 0 1px 0 rgba(255,255,255,0.95);
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg),
                        inset 0 1px 0 rgba(255,255,255,1);
        }
        /* Decorative background glow per card */
        .metric-card::after {
            content: '';
            position: absolute;
            bottom: -30px; right: -30px;
            width: 120px; height: 120px;
            border-radius: 50%;
            opacity: 0.55;
            pointer-events: none;
        }
        .tone-a::after { background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%); }
        .tone-b::after { background: radial-gradient(circle, rgba(16,185,129,0.25) 0%, transparent 70%); }
        .tone-c::after { background: radial-gradient(circle, rgba(139,92,246,0.25) 0%, transparent 70%); }

        /* Top accent stripe */
        .tone-a { border-top: 3px solid #60a5fa; }
        .tone-b { border-top: 3px solid var(--mint-400); }
        .tone-c { border-top: 3px solid #a78bfa; }

        .metric-eyebrow {
            color: var(--ink-300) !important;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .metric-value {
            font-family: var(--font-display);
            font-size: 2.6rem;
            font-weight: 400;
            color: var(--ink-900) !important;
            margin: 0.5rem 0 0.15rem;
            line-height: 1;
        }
        .metric-title {
            font-size: 0.97rem;
            font-weight: 600;
            color: var(--ink-700) !important;
        }
        .metric-copy {
            color: var(--ink-300) !important;
            font-size: 0.85rem;
            margin-top: 0.4rem;
        }

        /* ─────────────────────────────────────────────
           SPINNER / STATUS TEXT
        ───────────────────────────────────────────── */
        div[data-testid="stStatusWidget"] span,
        .stSpinner span {
            color: var(--ink-500) !important;
        }

        /* ─────────────────────────────────────────────
           SCROLLBAR — Minimal & on-brand
        ───────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--ink-100);
            border-radius: 999px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--ink-300); }

        /* ─────────────────────────────────────────────
           SELECTBOX / MULTISELECT
        ───────────────────────────────────────────── */
        div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.80) !important;
            border-color: var(--glass-border-dim) !important;
            border-radius: 12px !important;
            color: var(--ink-700) !important;
        }

        /* ─────────────────────────────────────────────
           SIDEBAR (collapsed but styled just in case)
        ───────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--glass-fill-heavy) !important;
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border-right: 1.5px solid var(--glass-border) !important;
        }

        /* ─────────────────────────────────────────────
           ALERT / ERROR MESSAGES
        ───────────────────────────────────────────── */
        div[data-testid="stAlert"] {
            background: rgba(255,255,255,0.75) !important;
            backdrop-filter: var(--glass-blur-sm);
            -webkit-backdrop-filter: var(--glass-blur-sm);
            border-radius: 14px !important;
            border: 1px solid var(--glass-border-dim) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def smart_container():
    try: return st.container(border=True)
    except TypeError: return st.container()

def resolve_data_path() -> Path:
    candidate_paths = [
        Path(ir_rag_directory) / "data" / DATA_FILE_NAME,
        Path(current_directory) / "data" / DATA_FILE_NAME,
    ]
    for path in candidate_paths:
        if path.exists(): return path
    raise FileNotFoundError(f"Could not find {DATA_FILE_NAME}")

@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, str]:
    data_path = resolve_data_path()
    df = pd.read_csv(data_path)
    cleaned_df = df.copy()
    cleaned_df["Price"] = pd.to_numeric(cleaned_df["Price"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["Price"]).copy()
    cleaned_df["Price"] = cleaned_df["Price"].astype(float)
    cleaned_df["Brand"] = cleaned_df["Brand"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    cleaned_df["Category"] = cleaned_df["Category"].fillna("Uncategorized").astype(str).str.strip().replace({"": "Uncategorized", "nan": "Uncategorized", "None": "Uncategorized"})
    cleaned_df["Display_Category"] = cleaned_df["Category"].replace({"Phones and Tablets": "Phones & Tablets"})
    return cleaned_df, str(data_path)

@st.cache_resource(show_spinner=False)
def initialize_rag_resources():
    loader = getattr(rag, "load_resources", None)
    return loader() if callable(loader) else None

def initialize_rag_session():
    session_factory = getattr(rag, "Session", None)
    return session_factory() if callable(session_factory) else None

def call_rag_backend(user_query: str) -> str:
    resources = st.session_state.get("rag_resources")
    session = st.session_state.get("rag_session")
    for fname in ["handle_turn", "generate_response", "get_response", "answer_query", "chat_turn", "query", "query_rag"]:
        func = getattr(rag, fname, None)
        if callable(func):
            try:
                args = len(inspect.signature(func).parameters)
                if args >= 3: return str(func(user_query, resources, session))
                elif args == 2: return str(func(user_query, resources))
                else: return str(func(user_query))
            except Exception: pass
    raise AttributeError("No supported response function found.")

def remember_turn(user_query: str, response: str) -> None:
    if "rag_session" in st.session_state and st.session_state.rag_session:
        rem = getattr(st.session_state.rag_session, "remember", None)
        if callable(rem): rem(user_query, response)

def build_metric_card(label: str, value: str, title: str, copy_text: str, tone: str) -> str:
    return f"""
    <div class="metric-card {tone}">
        <div class="metric-eyebrow">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-title">{title}</div>
        <div class="metric-copy">{copy_text}</div>
    </div>
    """

# ── Shared chart style helpers ────────────────────────────────────────────────
_CHART_FONT = "DM Sans, sans-serif"
_AXIS_COLOR  = "#3d5a4a"
_GRID_COLOR  = "rgba(122,158,142,0.15)"

def base_figure_layout(title: str) -> dict:
    return {
        "title": {
            "text": title,
            "x": 0.02,
            "font": {"size": 17, "family": _CHART_FONT, "color": "#0f1f17", "weight": "bold"},
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "margin": {"l": 20, "r": 20, "t": 58, "b": 20},
        "font": {"family": _CHART_FONT, "color": _AXIS_COLOR},
    }

def build_price_tier_chart(df: pd.DataFrame) -> go.Figure:
    tier_df = df.copy()
    tier_df["Price_Tier"] = pd.cut(
        tier_df["Price"],
        bins=[0, 1000, 5000, 10000, float("inf")],
        labels=PRICE_TIER_ORDER,
        include_lowest=True,
    )
    tier_counts = (
        tier_df["Price_Tier"]
        .value_counts()
        .reindex(PRICE_TIER_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    tier_counts.columns = ["Tier", "Items"]

    # Soft emerald gradient palette
    colors = ["#a7f3d0", "#34d399", "#10b981", "#065f46"]

    fig = go.Figure(go.Bar(
        x=tier_counts["Tier"],
        y=tier_counts["Items"],
        marker_color=colors,
        marker_line_width=0,
        text=tier_counts["Items"],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Items: %{y}<extra></extra>",
    ))
    fig.update_layout(**base_figure_layout("Price Tiers Overview"))
    fig.update_xaxes(title_text="", showgrid=False, tickfont=dict(color=_AXIS_COLOR, size=12))
    fig.update_yaxes(title_text="", gridcolor=_GRID_COLOR, tickfont=dict(color=_AXIS_COLOR))
    fig.update_traces(textfont=dict(color="#0f1f17", size=13, weight="bold"))
    return fig

def build_brand_chart(df: pd.DataFrame) -> go.Figure:
    top_brands = (
        df.loc[df["Brand"].ne("Unknown"), "Brand"]
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
        .reset_index()
    )
    top_brands.columns = ["Brand", "Listings"]

    fig = go.Figure(go.Bar(
        x=top_brands["Listings"],
        y=top_brands["Brand"],
        orientation="h",
        marker={
            "color": top_brands["Listings"],
            "colorscale": [
                [0.0, "#d1fae5"],
                [0.4, "#34d399"],
                [0.7, "#10b981"],
                [1.0, "#065f46"],
            ],
            "line": {"width": 0},
        },
        text=top_brands["Listings"],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Listings: %{x}<extra></extra>",
    ))
    fig.update_layout(**base_figure_layout("Top 10 Dominant Brands"))
    fig.update_xaxes(title_text="", gridcolor=_GRID_COLOR, tickfont=dict(color=_AXIS_COLOR))
    fig.update_yaxes(title_text="", showgrid=False, tickfont=dict(color=_AXIS_COLOR, size=12))
    fig.update_traces(textfont=dict(color="#0f1f17", size=12, weight="bold"))
    return fig

def build_category_donut(df: pd.DataFrame) -> go.Figure:
    category_counts = (
        df["Display_Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Items")
    )
    colors = ["#34d399", "#60a5fa", "#fbbf24", "#a78bfa"]

    fig = go.Figure(go.Pie(
        labels=category_counts["Category"],
        values=category_counts["Items"],
        hole=0.62,
        sort=False,
        marker={
            "colors": colors,
            "line": {"color": "rgba(255,255,255,0.9)", "width": 2.5},
        },
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Items: %{value}<extra></extra>",
    ))
    fig.update_layout(**base_figure_layout("Category Breakdown"))
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": _AXIS_COLOR, "size": 12},
        },
    )
    fig.update_traces(textfont=dict(color="#0f1f17", size=13, weight="bold"))
    return fig

def build_price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Price", color="Display_Category", nbins=30, barmode="overlay",
        color_discrete_sequence=["#34d399", "#60a5fa", "#fbbf24", "#a78bfa"],
        opacity=0.72,
    )
    fig.update_layout(**base_figure_layout("Price Distribution"))
    fig.update_xaxes(
        title_text="Price (EGP)",
        gridcolor=_GRID_COLOR,
        tickfont=dict(color=_AXIS_COLOR),
    )
    fig.update_yaxes(
        title_text="Count",
        gridcolor=_GRID_COLOR,
        tickfont=dict(color=_AXIS_COLOR),
    )
    fig.update_layout(
        legend=dict(font=dict(color=_AXIS_COLOR, size=12)),
        bargap=0.04,
    )
    fig.update_traces(marker_line_width=0)
    return fig

# ── Page sections ─────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-badge">✦ Local RAG &nbsp;+&nbsp; Analytics</div>
            <h1 class="hero-title">Jumia Egypt<br><em>Shopping Assistant</em></h1>
            <p class="hero-copy">
                A premium interface for conversational product discovery and interactive
                market analytics — powered by a local RAG pipeline and live catalog data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_chat_tab() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_resources" not in st.session_state:
        st.session_state.rag_resources = initialize_rag_resources()
    if "rag_session" not in st.session_state:
        st.session_state.rag_session = initialize_rag_session()

    with smart_container():
        st.markdown(
            """
            <div class="panel-title">AI Shopping Assistant</div>
            <div class="panel-copy">Ask for recommendations, compare products, or specify a budget.</div>
            """,
            unsafe_allow_html=True,
        )

        chat_placeholder = st.container()
        user_query = st.chat_input("Ask about products, prices, or specs…")

        with chat_placeholder:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_query:
            with chat_placeholder:
                with st.chat_message("user"):
                    st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            with chat_placeholder:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing catalog…"):
                        try:
                            response = call_rag_backend(user_query)
                            remember_turn(user_query, response)
                        except Exception as exc:
                            response = f"❌ Backend Error: {exc}"
                        st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

def render_dashboard_tab() -> None:
    try:
        df, data_path = load_dashboard_data()
    except Exception as exc:
        st.error(f"Unable to load the dataset: {exc}")
        return

    metric_columns = st.columns(3)
    metric_columns[0].markdown(
        build_metric_card("Scale", f"{len(df):,}", "Total Items", "Full catalog indexed", "tone-a"),
        unsafe_allow_html=True,
    )
    metric_columns[1].markdown(
        build_metric_card("Focus", f"{int(df['Category'].eq('Computing').sum()):,}", "Computing", "Laptops & hardware", "tone-b"),
        unsafe_allow_html=True,
    )
    metric_columns[2].markdown(
        build_metric_card("Mobile", f"{int(df['Category'].eq('Phones and Tablets').sum()):,}", "Phones & Tablets", "Mobile devices", "tone-c"),
        unsafe_allow_html=True,
    )

    st.write("")  # Spacer

    col1, col2 = st.columns(2)
    with col1:
        with smart_container():
            st.markdown('<div class="panel-title">Price Tiers</div>', unsafe_allow_html=True)
            st.plotly_chart(build_price_tier_chart(df), use_container_width=True, config=PLOTLY_CONFIG)
    with col2:
        with smart_container():
            st.markdown('<div class="panel-title">Top Brands</div>', unsafe_allow_html=True)
            st.plotly_chart(build_brand_chart(df), use_container_width=True, config=PLOTLY_CONFIG)

    col3, col4 = st.columns(2)
    with col3:
        with smart_container():
            st.markdown('<div class="panel-title">Category Share</div>', unsafe_allow_html=True)
            st.plotly_chart(build_category_donut(df), use_container_width=True, config=PLOTLY_CONFIG)
    with col4:
        with smart_container():
            st.markdown('<div class="panel-title">Price Density</div>', unsafe_allow_html=True)
            st.plotly_chart(build_price_distribution(df), use_container_width=True, config=PLOTLY_CONFIG)

def main() -> None:
    inject_styles()
    render_header()
    tab1, tab2 = st.tabs(["💬  AI Assistant", "📊  Analytics Dashboard"])
    with tab1: render_chat_tab()
    with tab2: render_dashboard_tab()

if __name__ == "__main__":
    main()