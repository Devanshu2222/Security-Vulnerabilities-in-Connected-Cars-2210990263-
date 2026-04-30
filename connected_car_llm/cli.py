"""
CLI interface — run the full pipeline from the terminal.
Usage: python cli.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import RAGPipeline
from src.llm_interface import OllamaLLM
from src.chains import QAChain, SummarizationChain, AnalysisChain
from src.paper_data import PAPER_TEXT, PAPER_METADATA


BANNER = """
╔══════════════════════════════════════════════════════════╗
║     🚗  Connected Car Security Research Assistant        ║
║         RAG + Summarization + Analysis  ·  Ollama        ║
╚══════════════════════════════════════════════════════════╝
"""

MENU = """
╭─────────────────────────────────────────╮
│  1. Ask a question (RAG Q&A)            │
│  2. Summarize a topic                   │
│  3. Analyze a vulnerability             │
│  4. Compare two concepts                │
│  5. Extract all key findings            │
│  6. Generate a quiz                     │
│  7. Executive summary                   │
│  8. Ingest a PDF                        │
│  9. Show index stats                    │
│  0. Exit                                │
╰─────────────────────────────────────────╯
"""


def print_result(text: str, title: str = ""):
    width = 72
    if title:
        print(f"\n{'─'*width}")
        print(f"  {title}")
    print(f"{'─'*width}")
    for line in text.split("\n"):
        print(f"  {line}")
    print(f"{'─'*width}\n")


def main():
    print(BANNER)

    # Config
    ollama_url   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    print(f"  Ollama URL  : {ollama_url}")
    print(f"  Ollama Model: {ollama_model}")
    print(f"  Paper       : {PAPER_METADATA['title']}\n")

    # Build pipeline
    print("  ⏳ Loading embeddings model (first run may download ~90MB)…")
    config = {
        "embedding_type": "sentence_transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 350,
        "chunk_overlap": 50,
        "store_path": "data/vector_store.pkl",
    }
    rag = RAGPipeline(config)

    if len(rag.vector_store.documents) == 0:
        print("  📄 Ingesting paper into vector store…")
        n = rag.ingest_text(PAPER_TEXT, source="connected_car_security_paper")
        print(f"  ✅ Ingested {n} chunks\n")
    else:
        stats = rag.get_stats()
        print(f"  ✅ Loaded {stats['total_chunks']} chunks from cache\n")

    print("  ⏳ Connecting to Ollama…")
    try:
        llm = OllamaLLM(model=ollama_model, base_url=ollama_url)
        print(f"  ✅ Connected to Ollama ({ollama_model})\n")
    except Exception as e:
        print(f"  ❌ Ollama connection failed: {e}")
        print("  Make sure Ollama is running: ollama serve")
        sys.exit(1)

    qa       = QAChain(rag, llm, top_k=5)
    summ     = SummarizationChain(rag, llm)
    analysis = AnalysisChain(rag, llm)

    # REPL
    while True:
        print(MENU)
        choice = input("  › ").strip()

        if choice == "0":
            print("\n  Goodbye! 🚗\n")
            break

        elif choice == "1":
            q = input("  Question: ").strip()
            if not q:
                continue
            print("  ⏳ Retrieving & generating…")
            result = qa.ask(q)
            print_result(result["answer"], f"Answer [{result['retrieved_chunks']} chunks]")
            print(f"  Sources: {[s.get('section','?') for s in result['sources'][:3]]}")

        elif choice == "2":
            topic = input("  Topic: ").strip()
            if not topic:
                continue
            print("  ⏳ Summarizing…")
            result = summ.summarize_topic(topic)
            print_result(result["summary"], f"Summary: {topic}")

        elif choice == "3":
            vuln = input("  Vulnerability: ").strip()
            if not vuln:
                continue
            print("  ⏳ Analyzing…")
            result = analysis.analyze_vulnerability(vuln)
            print_result(result["analysis"], f"Analysis: {vuln}")

        elif choice == "4":
            ca = input("  Concept A: ").strip()
            cb = input("  Concept B: ").strip()
            if not ca or not cb:
                continue
            print("  ⏳ Comparing…")
            result = analysis.compare_concepts(ca, cb)
            print_result(result["comparison"], f"{ca}  vs  {cb}")

        elif choice == "5":
            print("  ⏳ Extracting findings across all vulnerability categories…")
            result = analysis.extract_key_findings()
            for topic, finding in result["findings"].items():
                print_result(finding, topic.upper())

        elif choice == "6":
            n = input("  Number of questions [5]: ").strip()
            n = int(n) if n.isdigit() else 5
            print(f"  ⏳ Generating {n}-question quiz…")
            result = analysis.generate_quiz(n)
            print_result(result["quiz"], "QUIZ")

        elif choice == "7":
            print("  ⏳ Generating executive summary…")
            result = summ.summarize_full_paper()
            print_result(result["summary"], "Executive Summary")

        elif choice == "8":
            pdf_path = input("  PDF path: ").strip()
            if not os.path.exists(pdf_path):
                print(f"  ❌ File not found: {pdf_path}")
                continue
            print("  ⏳ Ingesting PDF…")
            n = rag.ingest_pdf(pdf_path)
            print(f"  ✅ Ingested {n} chunks from {pdf_path}")

        elif choice == "9":
            stats = rag.get_stats()
            print_result(
                "\n".join(f"  {k}: {v}" for k, v in stats.items()),
                "Index Stats"
            )

        else:
            print("  Invalid choice. Please enter 0–9.")


if __name__ == "__main__":
    main()
