# 🚗 Connected Car Security Research Assistant

**Full RAG + Summarization + Analysis pipeline — 100% local via Ollama**

---

## What it does

| Feature | Description |
|---|---|
| 💬 RAG Q&A | Ask questions about the paper; answers grounded in retrieved chunks |
| 📝 Summarization | Topic summaries + full executive summary |
| 🔬 Vulnerability Analysis | Deep-dive on any attack vector |
| ⚔️ Concept Comparison | Side-by-side comparison of security concepts |
| 📊 Key Findings Extraction | Auto-extract findings from all 5 attack surfaces |
| 📚 Quiz Generator | Multiple-choice quiz from the paper |
| 📄 PDF Upload | Ingest any additional research PDFs |

---

## Setup

### 1. Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com
```

### 2. Pull a model

```bash
ollama pull llama3          # recommended (8B, ~4.7GB)
ollama pull mistral         # alternative (7B, ~4.1GB)
ollama pull phi3            # lightweight (3.8B, ~2.3GB)
ollama pull gemma2          # Google's model (9B)
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> First run downloads `all-MiniLM-L6-v2` (~90MB) for local embeddings.

---

## Run

### Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### CLI (terminal)

```bash
python cli.py

# Custom model/URL
OLLAMA_MODEL=mistral OLLAMA_URL=http://localhost:11434 python cli.py
```

---

## Project Structure

```
connected_car_llm/
├── app.py                  ← Streamlit web UI
├── cli.py                  ← Terminal REPL
├── requirements.txt
├── data/
│   └── vector_store.pkl    ← auto-created on first run
└── src/
    ├── rag_pipeline.py     ← Chunking · Embedding · VectorStore · Retrieval
    ├── llm_interface.py    ← Ollama · OpenAI · Anthropic · HuggingFace backends
    ├── chains.py           ← QA · Summarization · Analysis chains
    └── paper_data.py       ← Embedded paper text
```

---

## Switching LLM backend

Edit the config in `app.py` or set environment variables:

```python
# OpenAI
config["llm_provider"] = "openai"
config["llm_model"]    = "gpt-4o-mini"
config["openai_api_key"] = "sk-..."

# Anthropic
config["llm_provider"] = "anthropic"
config["llm_model"]    = "claude-3-haiku-20240307"

# HuggingFace (local)
config["llm_provider"] = "huggingface"
config["llm_model"]    = "mistralai/Mistral-7B-Instruct-v0.2"
```

---

## Example questions to ask

- *What are the main CAN bus attack types?*
- *How do relay attacks on passive keyless entry work?*
- *What is the Uptane framework and why does it matter?*
- *How can LSTM networks detect CAN bus intrusions?*
- *What open challenges does the paper identify?*
- *Explain the 2015 Jeep Cherokee hack step by step*
- *Compare CAN message injection vs bus-off attacks*
