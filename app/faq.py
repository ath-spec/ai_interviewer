"""
FAQ Q&A module.
Uses the FAQ markdown file as context for LLM answers.
"""

import os
from app import config
from app.llm import LLMClient

def _read_faq_text() -> str:
    """Load FAQ markdown; return plain text (empty string if missing)."""
    path = getattr(config, "FAQ_FILE", "faq.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                print(f"[WARN] {path} is empty; FAQ answers may be poor.")
            return text
    except FileNotFoundError:
        print(f"[WARN] {path} not found; FAQ answers may be poor.")
        return ""


def faq_loop(speak_fn=None):
    """
    Interactive loop where candidate can ask questions.
    Returns a list of {"question": ..., "answer": ...}.
    """
    faq_pairs = []
    faq_text = _read_faq_text()
    if not faq_text:
        faq_text = "(No FAQ context provided.)"

    client = LLMClient()

    print("\n--- FAQ Q&A ---")
    print("Ask your questions about the program. Type 'no' or 'done' to exit.\n")

    while True:
        q = input("You (FAQ): ").strip()
        if not q or q.lower() in ("no", "done", "nothing", "exit"):
            break

        try:
            resp = client.generate(
                f"You are an admissions assistant. Answer the user's question using ONLY this program FAQ:\n\n{faq_text}\n\nQuestion: {q}\n\nAnswer:"
            )
            answer = resp.strip()
        except Exception as e:
            answer = f"[Error getting answer: {e}]"

        print(f"Agent (FAQ): {answer}")
        if speak_fn:
            speak_fn(answer)

        faq_pairs.append({"question": q, "answer": answer})

    return faq_pairs


def render_faq_md(faq_pairs):
    """
    Render Q/A pairs into Markdown for summaries.
    """
    if not faq_pairs:
        return ""

    lines = ["\n### Candidate Q&A\n"]
    for p in faq_pairs:
        lines.append(f"- **Q:** {p['question']}\n  **A:** {p['answer']}")
    return "\n".join(lines)
