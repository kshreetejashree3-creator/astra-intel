"""
ASTRA INTEL — Streamlit application.

Features:
- Official + user-added PDF/TXT/MD documents
- Multi-document grounded retrieval
- Gemini-backed answers with evidence citations
- Visible chat history + clear history
- Add/remove documents from the sidebar
"""
import hashlib
import html
import os
import time
from collections import Counter
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ASTRA_LOGO = BASE_DIR / "assets" / "astra-logo.png"

import streamlit as st

from astra.config import (
    ADDITIONAL_DOCS_DIR,
    ALL_DOCS_DIRS,
    COLOR_CHARCOAL,
    COLOR_SIGNAL_GOLD,
    COLOR_WHITE,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    MAX_UPLOAD_MB,
    MIN_BEST_SCORE,
    MIN_CHUNK_SCORE,
    PER_DOC_K,
    SUPPORTED_EXTENSIONS,
)
from astra.embed import get_model
from astra.index import get_or_build_index, get_document_paths, document_summary
from astra.llm import LLMError
from astra.rag import ask


st.set_page_config(
    page_title="ASTRA INTEL",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- visual design -------------------------

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

:root {{
  --charcoal:{COLOR_CHARCOAL};
  --charcoal2:#2A3036;
  --gold:{COLOR_SIGNAL_GOLD};
  --white:#fff;
  --bg:#ffffff;
  --muted:#6f7478;
  --neon:#00d9ff;
  --neon2:#39ff14;
}}

footer {{visibility:hidden}}
.block-container {{max-width:1100px;padding-top:1.5rem;padding-bottom:4rem}}

.astra-header {{
  background:linear-gradient(135deg,#1B1F23,#252C33);
  padding:1.45rem 2rem;
  border-radius:14px;
  border-bottom:3px solid var(--gold);
  margin-bottom:1.2rem;
  box-shadow:0 10px 30px rgba(0,0,0,.10);
}}
.astra-title {{
  font-family:'Space Grotesk',Arial,sans-serif;color:#fff;font-size:2.3rem;
  font-weight:700;letter-spacing:.16em;margin:0;
}}
.astra-subtitle {{color:#d6d6d6;font-size:1rem;margin-top:.35rem}}
.gold {{color:var(--gold)}}

.section-label {{
  font-family:'Space Grotesk',Arial,sans-serif;color:var(--gold);
  font-size:.74rem;font-weight:700;letter-spacing:.18em;
  text-transform:uppercase;margin:1.35rem 0 .5rem;
}}
.q-echo {{
  border-left:4px solid var(--gold);padding:.35rem 0 .35rem .9rem;
  margin:.4rem 0 .9rem;font-size:1.08rem;color:var(--charcoal)
}}
.q-echo small {{display:block;font-size:.68rem;letter-spacing:.16em;color:#777;margin-bottom:.1rem}}

.status-row {{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;margin-bottom:.6rem}}
.badge {{
  font-family:'Space Grotesk',Arial,sans-serif;font-weight:700;font-size:.75rem;
  letter-spacing:.12em;padding:.28rem .8rem;border-radius:999px
}}
.b-supported {{background:#e3f1e8;color:#176b3a;border:1.5px solid #176b3a}}
.b-partial {{background:#fbf0d8;color:#9a6700;border:1.5px solid #9a6700}}
.b-notfound {{background:#f8e3e4;color:#9b2226;border:1.5px solid #9b2226}}
.status-note {{color:#666;font-size:.88rem}}

.evidence-card {{
  border:1px solid #ddd;border-left:5px solid var(--gold);border-radius:9px;
  padding:.9rem 1rem;margin-bottom:.8rem;background:#fafafa
}}
.evidence-meta {{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin-bottom:.55rem}}
.evidence-source {{font-weight:700;color:var(--charcoal);font-size:.92rem;word-break:break-all}}
.chip {{font-size:.7rem;padding:.1rem .55rem;border-radius:999px;border:1px solid #c9c9c9;color:#555;white-space:nowrap}}
.chip-gold {{border-color:var(--gold);color:var(--gold);font-weight:700}}
.evidence-text {{color:#2b2f33;font-size:.92rem;line-height:1.55}}

.empty {{text-align:center;padding:1.6rem 0 .6rem}}
.empty-title {{font-family:'Space Grotesk',Arial,sans-serif;font-size:1.3rem;font-weight:700}}
.empty-sub {{color:#666;margin-top:.4rem;font-size:.95rem}}
.foot {{color:#888;font-size:.76rem;margin-top:2.5rem;text-align:center}}

.doc-card {{
  display:flex;align-items:center;gap:.45rem;padding:.4rem .55rem;
  margin:.25rem 0;background:#252B31;border:1px solid #394149;border-radius:7px;
  color:#eee;font-size:.78rem
}}
.doc-name {{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}}
.doc-count {{color:#9fa7ad;font-size:.68rem;white-space:nowrap}}

.tech {{
  background:#070b10;border:1px solid var(--neon);border-radius:10px;
  padding:1rem;color:#cfeff2;box-shadow:0 0 18px rgba(0,217,255,.14)
}}
.tech .metric {{
  background:#0c141c;border:1px solid rgba(57,255,20,.35);border-radius:8px;
  padding:.55rem .7rem
}}
.tech small {{color:var(--neon);letter-spacing:.1em;text-transform:uppercase}}
.tech b {{color:var(--neon2);font-family:'Space Grotesk',monospace}}

div[data-testid="stSidebar"] {{
  background:var(--charcoal);
}}
div[data-testid="stSidebar"] * {{color:#f4f4f4}}
div[data-testid="stSidebar"] .stCaption {{color:#aeb5ba}}
div[data-testid="stSidebar"] hr {{border-color:#394149}}
div[data-testid="stSidebar"] button {{
  border-color:#394149;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------- state -------------------------

st.session_state.setdefault("history", [])
st.session_state.setdefault("selected", None)
st.session_state.setdefault("pending", None)
st.session_state.setdefault("upload_version", 0)

# ------------------------- cached index -------------------------

@st.cache_resource(show_spinner="Loading ASTRA document index and retrieval model...")
def load_index():
    idx = get_or_build_index()
    get_model()
    return idx

try:
    index = load_index()
except Exception as e:
    st.error(f"Could not load the ASTRA INTEL index: {e}")
    st.stop()

summary = document_summary(index)
API_KEY_SET = bool(os.getenv("GEMINI_API_KEY"))

# ------------------------- callbacks/helpers -------------------------

STATUS = {
    "supported": ("✓ SUPPORTED", "b-supported", "Directly supported by the indexed documents."),
    "partial": ("◐ PARTIAL", "b-partial", "The documents answer only part of this question."),
    "not_found": ("✕ NOT FOUND", "b-notfound", "Not supported by the indexed documents."),
}
ICON = {"supported": "✓", "partial": "◐", "not_found": "✕"}

EXAMPLES = [
    "What are the main categories of UAVs?",
    "What are the three subdivisions of electronic warfare?",
    "Name one specific UGV program and its developer.",
    "What percentage of UAV missions are fully autonomous?",
]

def new_question():
    st.session_state.selected = None

def view_item(i):
    st.session_state.selected = i

def clear_history():
    st.session_state.history = []
    st.session_state.selected = None

def use_example(q):
    st.session_state.pending = q

def short(text, n=44):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n-1].rstrip() + "…"

def refresh_index():
    load_index.clear()
    st.session_state.selected = None
    st.rerun()

def file_key(uploaded):
    return hashlib.sha256(uploaded.getvalue()).hexdigest()

def add_documents(files):
    if not files:
        return

    ADDITIONAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name.lower() for p in get_document_paths()}
    added = 0

    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        safe_name = Path(uploaded.name).name

        if suffix not in SUPPORTED_EXTENSIONS:
            st.error(f"{safe_name}: only PDF, TXT and MD files are supported.")
            continue

        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"{safe_name}: file is larger than {MAX_UPLOAD_MB} MB.")
            continue

        if safe_name.lower() in existing:
            st.warning(f"{safe_name}: already exists, so it was not duplicated.")
            continue

        target = ADDITIONAL_DOCS_DIR / safe_name
        target.write_bytes(uploaded.getvalue())
        existing.add(safe_name.lower())
        added += 1

    if added:
        st.success(f"Added {added} document(s). Rebuilding the search index…")
        load_index.clear()
        st.session_state.upload_version += 1
        st.rerun()

def remove_document(name):
    path = ADDITIONAL_DOCS_DIR / Path(name).name
    if path.exists() and path.is_file():
        path.unlink()
        load_index.clear()
        st.rerun()

def run_query(question):
    question = (question or "").strip()
    if not question:
        st.warning("Please enter a question.")
        return

    start = time.time()
    try:
        with st.spinner("Retrieving evidence and checking it against the documents…"):
            result = ask(question, index)
    except LLMError as e:
        st.error(f"ASTRA could not generate an answer: {e}")
        return
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return

    result["elapsed"] = time.time() - start
    st.session_state.history.append(result)
    st.session_state.selected = len(st.session_state.history) - 1

# ------------------------- rendering -------------------------

def render_result(result):
    status = result["status"]
    label, css_class, note = STATUS.get(status, STATUS["not_found"])

    st.markdown(
        f'<div class="q-echo"><small>QUESTION</small>{html.escape(result["question"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="status-row"><span class="badge {css_class}">{label}</span>'
        f'<span class="status-note">{note}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="answer-box">{html.escape(result["answer"])}</div>',
                unsafe_allow_html=True)

    if status == "not_found":
        if result.get("refused_by") == "retrieval":
            why = "No passage in the indexed documents was similar enough to this question."
        elif result.get("downgraded"):
            why = "The model's answer cited no valid evidence, so it was rejected."
        else:
            why = "Related passages were retrieved, but they do not contain the requested information."
        st.caption(why + " ASTRA does not answer from outside knowledge.")

    evidence = result.get("evidence", [])
    if evidence:
        st.markdown(
            f'<div class="section-label">Evidence · {len(evidence)} passage(s)</div>',
            unsafe_allow_html=True,
        )
        for n, c in enumerate(evidence, start=1):
            passage = " ".join(c["text"].split())
            st.markdown(
                '<div class="evidence-card">'
                '<div class="evidence-meta">'
                f'<span class="evidence-source">[{n}] {html.escape(c["doc_name"])}</span>'
                f'<span class="chip chip-gold">Page/section {c["page"]}</span>'
                f'<span class="chip">Similarity {c["score"]:.3f}</span>'
                '</div>'
                f'<div class="evidence-text">…{html.escape(passage)}…</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    with st.expander("Technical details"):
        retrieved = result.get("retrieved", [])
        cols = st.columns(4)
        cols[0].metric("Retrieved", len(retrieved))
        cols[1].metric("Cited", len(evidence))
        cols[2].metric("Refused by", result.get("refused_by") or "none")
        cols[3].metric("Time", f"{result.get('elapsed', 0):.1f}s")
        cited = {c["chunk_id"] for c in evidence}
        rows = [{
            "Cited": "●" if c["chunk_id"] in cited else "",
            "Score": round(c["score"], 3),
            "Document": c["doc_name"],
            "Page/section": c["page"],
            "Preview": " ".join(c["text"].split())[:110],
        } for c in retrieved]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)

def render_empty_state():
    if summary:
        title = "Ask about your documents"
        sub = "Every answer comes only from the indexed files, with its source document, page/section and supporting passage."
    else:
        title = "Upload a document to begin"
        sub = "Use the Upload documents button in the sidebar to add PDF, TXT or MD files."
    st.markdown(
        f'<div class="empty"><div class="empty-title">{title}</div>'
        f'<div class="empty-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )
    if summary:
        st.caption("Try one of these:")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            cols[i % 2].button(q, key=f"ex_{i}", on_click=use_example, args=(q,))

# ------------------------- header -------------------------

st.markdown(
    '<div class="astra-header">'
    '<img src="data:image/png;base64,'
    + __import__("base64").b64encode(ASTRA_LOGO.read_bytes()).decode()
    + '" style="width:220px; margin-bottom:12px;">'
    '<div class="astra-subtitle">Evidence-grounded defence document analyst</div>'
    '</div>',
    unsafe_allow_html=True,
)

if not API_KEY_SET:
    st.warning("GEMINI_API_KEY is not configured. Add it to .env for question answering, then restart ASTRA.")

# ------------------------- main question UI -------------------------

with st.form("ask_form", clear_on_submit=True):
    typed = st.text_input(
        "Question",
        placeholder="Ask a question about the provided documents…",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("ASK ASTRA", type="primary", use_container_width=False)

queued = st.session_state.pending
st.session_state.pending = None

if submitted:
    run_query(typed)
elif queued:
    run_query(queued)

sel = st.session_state.selected
history = st.session_state.history
if sel is not None and 0 <= sel < len(history):
    render_result(history[sel])
else:
    render_empty_state()

st.markdown(
    '<div class="foot">ASTRA INTEL · Evidence-grounded document analysis · '
    'Source documents supplied with the challenge.</div>',
    unsafe_allow_html=True,
)

# ------------------------- sidebar -------------------------

with st.sidebar:
    st.markdown(
        '<div style="font-family:Space Grotesk,Arial;font-weight:700;letter-spacing:.16em;font-size:1.05rem">'
        '<span style="color:#B8860B">✦</span> ASTRA INTEL</div>',
        unsafe_allow_html=True,
    )
    st.caption("Evidence-grounded document analyst")

    st.button("＋  New question", on_click=new_question, use_container_width=True)

    st.markdown('<div class="section-label">HISTORY</div>', unsafe_allow_html=True)
    if not history:
        st.caption("No questions yet.")
    for i in reversed(range(len(history))):
        item = history[i]
        st.button(
            f"{ICON.get(item['status'], '✕')}  {short(item['question'])}",
            key=f"hist_{i}",
            on_click=view_item,
            args=(i,),
            type="primary" if i == sel else "secondary",
            use_container_width=True,
            help=item["question"],
        )
    st.button(
        "Clear history",
        on_click=clear_history,
        disabled=not history,
        use_container_width=True,
    )

    st.markdown('<div class="section-label">DOCUMENTS</div>', unsafe_allow_html=True)
    uploader_key = f"documents_{st.session_state.upload_version}"
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=uploader_key,
        label_visibility="collapsed",
        help=f"Add PDF, TXT or MD files up to {MAX_UPLOAD_MB} MB each.",
    )
    if uploaded:
        st.button(
            f"⬆ Add {len(uploaded)} document(s)",
            key="add_docs",
            type="primary",
            use_container_width=True,
            on_click=add_documents,
            args=(uploaded,),
        )

    if not summary:
        st.caption("No documents indexed.")
    else:
        for name, count in sorted(summary.items()):
            is_additional = (ADDITIONAL_DOCS_DIR / name).exists()
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f'<div class="doc-card"><span class="doc-name" title="{html.escape(name)}">'
                f'{"📄" if name.lower().endswith(".pdf") else "📝"} {html.escape(name)}</span>'
                f'<span class="doc-count">{count}</span></div>',
                unsafe_allow_html=True,
            )
            if is_additional:
                c2.button("×", key=f"del_{name}", help=f"Remove {name}",
                          on_click=remove_document, args=(name,))

    if any((ADDITIONAL_DOCS_DIR / n).exists() for n in summary):
        if st.button("Remove all added documents", use_container_width=True):
            for p in ADDITIONAL_DOCS_DIR.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    p.unlink()
            load_index.clear()
            st.rerun()

    st.markdown('<div class="section-label">SYSTEM</div>', unsafe_allow_html=True)
    with st.expander("Index & models"):
        st.write(f"**{len(summary)}** document(s)")
        st.write(f"**{len(index.chunks)}** indexed chunks")
        st.write(f"**Retrieval:** {EMBEDDING_MODEL.split('/')[-1]}")
        st.write(f"**Generation:** {GEMINI_MODEL}")
        st.write(f"**API key:** {'configured' if API_KEY_SET else 'missing'}")
        st.caption(
            f"Balanced retrieval: up to {PER_DOC_K} chunks per document · "
            f"noise floor {MIN_CHUNK_SCORE} · minimum best score {MIN_BEST_SCORE}"
        )
