"""
Connected Car Security Research Assistant
Full RAG + Summarization + Analysis UI
Powered by Ollama (local LLMs)
"""

import streamlit as st
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="AutoSec AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0a0e14; color: #e6edf3; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1f2937;
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.card {
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid #f97316; }
.card-blue   { border-left: 3px solid #3b82f6; }
.card-green  { border-left: 3px solid #22c55e; }
.card-purple { border-left: 3px solid #a855f7; }

.tag {
    display: inline-block;
    background: #1f2937;
    color: #9ca3af;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}
.tag-orange { background: #431407; color: #fb923c; }
.tag-blue   { background: #0c1a3a; color: #60a5fa; }

.score-bar-bg {
    background: #1f2937;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
.score-bar {
    background: linear-gradient(90deg, #f97316, #fb923c);
    border-radius: 4px;
    height: 6px;
}
.chunk-preview {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    background: #111827;
    border-radius: 4px;
    padding: 0.5rem;
    margin-top: 0.5rem;
    white-space: pre-wrap;
}
.stButton > button {
    background: #f97316 !important;
    color: #0a0e14 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
}
.stButton > button:hover { background: #ea6c0a !important; }

div[data-testid="stExpander"] {
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────
def init_state():
    defaults = {
        "pipeline_ready": False,
        "rag": None,
        "llm": None,
        "qa_chain": None,
        "sum_chain": None,
        "analysis_chain": None,
        "chat_history": [],
        "ollama_model": "llama3",
        "ollama_url": "http://localhost:11434",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Load pipeline (cached) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(ollama_url: str, ollama_model: str):
    from src.rag_pipeline import RAGPipeline
    from src.llm_interface import OllamaLLM
    from src.chains import QAChain, SummarizationChain, AnalysisChain
    from src.paper_data import PAPER_TEXT

    config = {
        "embedding_type": "sentence_transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 350,
        "chunk_overlap": 50,
        "store_path": "data/vector_store.pkl",
        "ollama_url": ollama_url,
        "llm_provider": "ollama",
        "llm_model": ollama_model,
    }

    rag = RAGPipeline(config)
    if len(rag.vector_store.documents) == 0:
        rag.ingest_text(PAPER_TEXT, source="connected_car_security_paper")

    llm = OllamaLLM(model=ollama_model, base_url=ollama_url)

    qa     = QAChain(rag, llm, top_k=5)
    summ   = SummarizationChain(rag, llm)
    analysis = AnalysisChain(rag, llm)

    return rag, llm, qa, summ, analysis


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AutoSec AI")
    st.markdown('<div class="tag tag-orange">LOCAL · OLLAMA</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Ollama Config")
    ollama_url   = st.text_input("Server URL", value="http://localhost:11434")
    ollama_model = st.selectbox("Model", [
        "llama3", "llama3.1", "llama3.2", "mistral",
        "mistral-nemo", "gemma2", "phi3", "qwen2.5",
        "deepseek-r1:7b", "codellama",
    ])

    if st.button("🔌 Connect & Load Pipeline"):
        with st.spinner("Loading embeddings + connecting to Ollama…"):
            try:
                rag, llm, qa, summ, analysis = load_pipeline(ollama_url, ollama_model)
                st.session_state.update({
                    "pipeline_ready": True,
                    "rag": rag, "llm": llm,
                    "qa_chain": qa,
                    "sum_chain": summ,
                    "analysis_chain": analysis,
                    "ollama_model": ollama_model,
                    "ollama_url": ollama_url,
                })
                st.success("✅ Pipeline ready!")
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("---")
    if st.session_state.pipeline_ready:
        stats = st.session_state.rag.get_stats()
        st.markdown("### 📊 Index Stats")
        st.markdown(f"""
<div class="card">
<span class="tag">Chunks</span> {stats['total_chunks']}<br>
<span class="tag">Embeddings</span> {stats['embedding_model']}<br>
<span class="tag">LLM</span> {st.session_state.ollama_model}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Upload Your PDF")
    uploaded = st.file_uploader("Add a PDF document", type=["pdf"])
    if uploaded and st.session_state.pipeline_ready:
        if st.button("Ingest PDF"):
            with st.spinner("Parsing & embedding…"):
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                n = st.session_state.rag.ingest_pdf(tmp_path)
                os.unlink(tmp_path)
                st.success(f"Ingested {n} chunks from {uploaded.name}")


# ── Main area ──────────────────────────────────────────────
st.markdown("# 🔐 Connected Car Security Research Assistant")
st.markdown(
    '<span class="tag tag-blue">RAG</span>'
    '<span class="tag tag-blue">Summarization</span>'
    '<span class="tag tag-blue">Vulnerability Analysis</span>'
    '<span class="tag tag-orange">Ollama · Local</span>',
    unsafe_allow_html=True
)
st.markdown("---")

if not st.session_state.pipeline_ready:
    st.markdown("""
<div class="card card-accent">
<h3>🚀 Getting Started</h3>
<ol>
<li>Install Ollama → <code>https://ollama.com</code></li>
<li>Pull a model → <code>ollama pull llama3</code></li>
<li>Install deps → <code>pip install -r requirements.txt</code></li>
<li>Configure server URL + model in the sidebar</li>
<li>Click <strong>Connect & Load Pipeline</strong></li>
</ol>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card card-blue">
<h3>📋 What this system does</h3>
<ul>
<li>🔍 <b>RAG Q&A</b> — ask anything about the paper with source citations</li>
<li>📝 <b>Summarization</b> — topic summaries & full executive summary</li>
<li>🔬 <b>Vulnerability Analysis</b> — deep-dive on specific attack surfaces</li>
<li>⚔️ <b>Concept Comparison</b> — compare two security concepts side-by-side</li>
<li>📊 <b>Key Findings</b> — extract all major findings automatically</li>
<li>📚 <b>Quiz Generator</b> — test your knowledge</li>
</ul>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Q&A Chat",
    "📝 Summarize",
    "🔬 Analyze",
    "⚔️ Compare",
    "📊 Key Findings",
    "📚 Quiz",
])


# ─────────────────────────────────────────────────────────
# TAB 1 — Q&A Chat
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 💬 Ask the Research Paper")
    st.caption("Answers are grounded in the paper via semantic retrieval.")

    # Suggested questions
    suggested = [
        "What are the main CAN bus attack types?",
        "How do relay attacks on keyless entry work?",
        "What is the Uptane framework?",
        "How can AI detect CAN bus intrusions?",
        "What are the open challenges in automotive cybersecurity?",
        "Explain the 2015 Jeep Cherokee hack",
    ]
    st.markdown("**Quick questions:**")
    cols = st.columns(3)
    for i, q in enumerate(suggested):
        if cols[i % 3].button(q, key=f"sq_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})

    st.markdown("---")

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"],
                             avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander(f"📎 {len(msg['sources'])} source chunks"):
                    for j, (src, score) in enumerate(
                            zip(msg["sources"], msg.get("scores", []))):
                        pct = int(score * 100)
                        st.markdown(
                            f'<span class="tag">{src.get("section","§")}</span>'
                            f' chunk #{src.get("chunk_index","?")} &nbsp;'
                            f'<b>{pct}%</b>'
                            f'<div class="score-bar-bg"><div class="score-bar" style="width:{pct}%"></div></div>',
                            unsafe_allow_html=True,
                        )

    # Input
    user_input = st.chat_input("Ask about connected car security…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Retrieving & generating…"):
                result = st.session_state.qa_chain.ask(user_input)
            st.markdown(result["answer"])
            with st.expander(f"📎 {result['retrieved_chunks']} source chunks"):
                for src, score in zip(result["sources"], result["scores"]):
                    pct = int(score * 100)
                    st.markdown(
                        f'<span class="tag">{src.get("section","§")}</span>'
                        f' chunk #{src.get("chunk_index","?")} &nbsp;<b>{pct}%</b>'
                        f'<div class="score-bar-bg"><div class="score-bar" style="width:{pct}%"></div></div>',
                        unsafe_allow_html=True,
                    )

        assistant_msg = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "scores": result["scores"],
        }
        st.session_state.chat_history.append(assistant_msg)

    if st.button("🗑️ Clear history"):
        st.session_state.chat_history = []
        st.session_state.qa_chain.clear_history()
        st.rerun()


# ─────────────────────────────────────────────────────────
# TAB 2 — Summarization
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📝 Summarization")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Topic Summary")
        topic_options = [
            "CAN bus vulnerabilities",
            "Remote keyless entry attacks",
            "Infotainment system security",
            "OTA update security",
            "V2X communication threats",
            "Intrusion detection systems",
            "Secure boot and HSM",
            "AI-based anomaly detection",
            "Regulatory standards ISO/SAE 21434",
        ]
        topic = st.selectbox("Choose a topic", topic_options)
        custom_topic = st.text_input("…or enter custom topic")
        final_topic = custom_topic.strip() if custom_topic.strip() else topic

        if st.button("📝 Summarize Topic"):
            with st.spinner(f"Summarizing: {final_topic}…"):
                result = st.session_state.sum_chain.summarize_topic(final_topic)
            st.markdown(f'<div class="card card-green">{result["summary"]}</div>',
                        unsafe_allow_html=True)
            st.caption(f"Sections used: {', '.join(set(result['sources']))}")

    with col_b:
        st.markdown("#### Executive Summary")
        st.caption("Full-paper summary across all sections.")
        if st.button("📄 Generate Executive Summary"):
            with st.spinner("Generating executive summary…"):
                result = st.session_state.sum_chain.summarize_full_paper()
            st.markdown(f'<div class="card card-blue">{result["summary"]}</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TAB 3 — Vulnerability Analysis
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🔬 Deep Vulnerability Analysis")

    vuln_options = [
        "CAN bus message injection",
        "Bus-off denial of service attack",
        "Relay attack on PKES keyless entry",
        "OBD-II port exploitation",
        "Infotainment system remote code execution",
        "Telematics Control Unit firmware attack",
        "OTA update supply chain compromise",
        "V2X Sybil attack",
        "GPS spoofing in V2X",
    ]

    selected_vuln = st.selectbox("Select vulnerability to analyze", vuln_options)
    custom_vuln   = st.text_input("…or enter custom vulnerability")
    final_vuln    = custom_vuln.strip() if custom_vuln.strip() else selected_vuln

    if st.button("🔬 Run Analysis"):
        with st.spinner(f"Analyzing: {final_vuln}…"):
            result = st.session_state.analysis_chain.analyze_vulnerability(final_vuln)

        st.markdown(f"""
<div class="card card-accent">
<h4>🛡️ Analysis: {result['vulnerability']}</h4>
{result['analysis']}
<br><span class="tag">Evidence chunks: {result['evidence_chunks']}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TAB 4 — Concept Comparison
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ⚔️ Compare Security Concepts")
    st.caption("Side-by-side comparison of two automotive security topics.")

    concepts = [
        "CAN bus message injection",
        "Bus-off attack",
        "Relay attack",
        "Replay attack",
        "OTA update attack",
        "V2X Sybil attack",
        "GPS spoofing",
        "Bluetooth pairing vulnerability",
        "Intrusion Detection System (IDS)",
        "Secure boot",
        "Hardware Security Module (HSM)",
        "Uptane framework",
        "Network segmentation",
        "UWB ranging for keyless entry",
    ]

    c1, c2 = st.columns(2)
    concept_a = c1.selectbox("Concept A", concepts, index=0)
    concept_b = c2.selectbox("Concept B", concepts, index=5)

    if st.button("⚔️ Compare"):
        if concept_a == concept_b:
            st.warning("Please select two different concepts.")
        else:
            with st.spinner("Comparing…"):
                result = st.session_state.analysis_chain.compare_concepts(concept_a, concept_b)
            st.markdown(f"""
<div class="card card-purple">
<h4>⚔️ {concept_a} vs {concept_b}</h4>
{result['comparison']}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TAB 5 — Key Findings
# ─────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 📊 Extract All Key Findings")
    st.caption("Automatically extracts findings across all major vulnerability categories.")

    if st.button("📊 Extract Key Findings"):
        with st.spinner("Extracting findings across 5 attack surfaces…"):
            result = st.session_state.analysis_chain.extract_key_findings()

        colors = ["card-accent", "card-blue", "card-green", "card-purple", "card-blue"]
        for i, (topic, finding) in enumerate(result["findings"].items()):
            color = colors[i % len(colors)]
            st.markdown(f"""
<div class="card {color}">
<h4>🔍 {topic.title()}</h4>
{finding}
</div>
""", unsafe_allow_html=True)

        st.success(f"✅ Extracted {result['topic_count']} finding categories")


# ─────────────────────────────────────────────────────────
# TAB 6 — Quiz
# ─────────────────────────────────────────────────────────
with tab6:
    st.markdown("### 📚 Knowledge Quiz Generator")
    st.caption("Auto-generated multiple choice questions from the research paper.")

    num_q = st.slider("Number of questions", min_value=3, max_value=10, value=5)

    if st.button("📚 Generate Quiz"):
        with st.spinner("Generating quiz questions…"):
            result = st.session_state.analysis_chain.generate_quiz(num_q)
        st.markdown(f"""
<div class="card card-blue">
<pre style="white-space:pre-wrap;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#e6edf3">
{result['quiz']}
</pre>
</div>
""", unsafe_allow_html=True)
