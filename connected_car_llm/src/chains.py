"""
Question Answering and Summarization chains.
Uses RAG pipeline + LLM to answer domain-specific queries.
"""

from typing import Dict, List, Optional, Tuple
from src.rag_pipeline import RAGPipeline, Document
from src.llm_interface import BaseLLM


SYSTEM_PROMPT = """You are an expert automotive cybersecurity assistant specializing in connected car security, ECU vulnerabilities, CAN bus attacks, V2X communication, and related mitigation strategies. 

You answer questions accurately based on the provided research context. If information is not in the context, say so clearly. Use technical terminology appropriately. Always cite which part of the context supports your answer."""

QA_PROMPT_TEMPLATE = """Based on the following research context from a connected car security paper, answer the question thoroughly and accurately.

RESEARCH CONTEXT:
{context}

QUESTION: {question}

ANSWER (be specific, cite evidence from the context, and structure your response clearly):"""

SUMMARIZE_PROMPT_TEMPLATE = """Summarize the following section of an automotive cybersecurity research paper. 
Provide:
1. A 2-3 sentence executive summary
2. Key technical points (bullet list)
3. Security implications

SECTION CONTENT:
{content}

SUMMARY:"""

ANALYSIS_PROMPT_TEMPLATE = """As an automotive cybersecurity expert, analyze the following research content and provide:
1. Critical vulnerabilities identified
2. Effectiveness of proposed mitigations
3. Gaps or open questions
4. Practical recommendations for automotive engineers

CONTENT:
{content}

ANALYSIS:"""

COMPARE_PROMPT_TEMPLATE = """Compare the following two concepts from connected car security research:

CONCEPT A: {concept_a}
CONCEPT B: {concept_b}

CONTEXT:
{context}

Provide a structured comparison covering: definition, attack surface, severity, and mitigation effectiveness."""


class QAChain:
    """RAG-based question answering over research documents."""

    def __init__(self, rag_pipeline: RAGPipeline, llm: BaseLLM,
                 top_k: int = 5, min_score: float = 0.1):
        self.rag = rag_pipeline
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.chat_history: List[Dict] = []

    def ask(self, question: str, use_history: bool = True,
            stream: bool = False) -> Dict:
        """
        Answer a question using RAG.
        Returns dict with 'answer', 'sources', 'scores'.
        """
        # Retrieve relevant chunks
        results = self.rag.retrieve(question, top_k=self.top_k)
        filtered = [(doc, score) for doc, score in results if score >= self.min_score]

        if not filtered:
            return {
                "answer": "I couldn't find relevant information in the research paper to answer this question.",
                "sources": [],
                "scores": [],
                "retrieved_chunks": 0,
            }

        context = self.rag.format_context(filtered)
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        # Add history context if requested
        if use_history and self.chat_history:
            history_text = "\n".join([
                f"Q: {h['question']}\nA: {h['answer'][:200]}..."
                for h in self.chat_history[-3:]  # last 3 turns
            ])
            prompt = f"CONVERSATION HISTORY:\n{history_text}\n\n{prompt}"

        if stream:
            answer_gen = self.llm.stream(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
            return {
                "answer_stream": answer_gen,
                "sources": [doc.metadata for doc, _ in filtered],
                "scores": [score for _, score in filtered],
                "retrieved_chunks": len(filtered),
            }

        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)

        # Store in history
        self.chat_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "sources": [doc.metadata for doc, _ in filtered],
            "scores": [score for _, score in filtered],
            "retrieved_chunks": len(filtered),
            "context_preview": [doc.content[:200] for doc, _ in filtered[:2]],
        }

    def clear_history(self):
        self.chat_history = []

    def batch_ask(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions."""
        return [self.ask(q, use_history=False) for q in questions]


class SummarizationChain:
    """Summarizes document sections or full papers."""

    def __init__(self, rag_pipeline: RAGPipeline, llm: BaseLLM):
        self.rag = rag_pipeline
        self.llm = llm

    def summarize_topic(self, topic: str, top_k: int = 8) -> Dict:
        """Retrieve and summarize content about a specific topic."""
        results = self.rag.retrieve(topic, top_k=top_k)
        if not results:
            return {"summary": "No content found for this topic.", "sources": []}

        combined = "\n\n".join([doc.content for doc, _ in results])
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(content=combined[:4000])  # token guard

        summary = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=800)
        return {
            "summary": summary,
            "topic": topic,
            "sources": [doc.metadata.get("section", "Unknown") for doc, _ in results],
            "chunks_used": len(results),
        }

    def summarize_full_paper(self) -> Dict:
        """Generate an executive summary of the entire paper."""
        # Sample across the whole document
        all_docs = self.rag.vector_store.documents
        step = max(1, len(all_docs) // 15)
        sampled = all_docs[::step][:15]
        combined = "\n\n".join([d.content for d in sampled])

        prompt = f"""Provide a comprehensive executive summary of this automotive cybersecurity research paper.
Include: main thesis, key vulnerabilities analyzed, mitigation strategies proposed, and conclusions.

PAPER CONTENT (sampled):
{combined[:5000]}

EXECUTIVE SUMMARY:"""

        summary = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1200)
        return {"summary": summary, "type": "executive_summary"}


class AnalysisChain:
    """Deep analysis and comparison of security concepts."""

    def __init__(self, rag_pipeline: RAGPipeline, llm: BaseLLM):
        self.rag = rag_pipeline
        self.llm = llm

    def analyze_vulnerability(self, vulnerability: str) -> Dict:
        """Analyze a specific vulnerability in depth."""
        results = self.rag.retrieve(vulnerability, top_k=6)
        context = "\n\n".join([doc.content for doc, _ in results])
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(content=context[:4000])

        analysis = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
        return {
            "vulnerability": vulnerability,
            "analysis": analysis,
            "evidence_chunks": len(results),
        }

    def compare_concepts(self, concept_a: str, concept_b: str) -> Dict:
        """Compare two security concepts from the paper."""
        results_a = self.rag.retrieve(concept_a, top_k=3)
        results_b = self.rag.retrieve(concept_b, top_k=3)
        all_results = results_a + results_b
        context = "\n\n".join([doc.content for doc, _ in all_results[:8]])

        prompt = COMPARE_PROMPT_TEMPLATE.format(
            concept_a=concept_a,
            concept_b=concept_b,
            context=context[:4000],
        )
        comparison = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "comparison": comparison,
        }

    def extract_key_findings(self) -> Dict:
        """Extract all major security findings from the paper."""
        topics = [
            "CAN bus vulnerabilities and attacks",
            "remote keyless entry relay attacks",
            "infotainment system security flaws",
            "OTA update security risks",
            "V2X protocol vulnerabilities",
        ]
        findings = {}
        for topic in topics:
            results = self.rag.retrieve(topic, top_k=3)
            if results:
                context = results[0][0].content
                answer = self.llm.generate(
                    f"In one concise paragraph, what does the research say about: {topic}\n\nContext: {context}",
                    system=SYSTEM_PROMPT, max_tokens=256,
                )
                findings[topic] = answer

        return {"findings": findings, "topic_count": len(findings)}

    def generate_quiz(self, num_questions: int = 5) -> Dict:
        """Generate a quiz based on the research paper."""
        all_docs = self.rag.vector_store.documents
        step = max(1, len(all_docs) // num_questions)
        samples = [all_docs[i * step].content[:500] for i in range(min(num_questions, len(all_docs) // step))]
        combined = "\n\n".join(samples)

        prompt = f"""Based on this automotive cybersecurity research content, generate {num_questions} multiple-choice quiz questions.

Format each question as:
Q[N]: [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [Letter]
Explanation: [Brief explanation]

CONTENT:
{combined[:3000]}

QUIZ:"""

        quiz_text = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1500)
        return {"quiz": quiz_text, "num_questions": num_questions}
