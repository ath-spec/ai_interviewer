"""
Summarization that outputs BOTH:
1) Reviewer-friendly Markdown notes
2) Structured JSON with raw answers, per-question notes, and an overall evaluation

Works in two modes:
- Template (no LLM): heuristic notes from answers
- LLM mode: structured JSON via LLM with robust parsing, token caps, and a small cooldown
"""

from __future__ import annotations
import json, re, time, os
from typing import Dict, Any, Tuple
from app.llm import LLMClient
from app import config

# ---------- Utilities ----------

def _extract_json_block(text: str) -> str | None:
    if not text: return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        try: json.loads(cand); return cand
        except: pass
    opens = [m.start() for m in re.finditer(r"\{", text)]
    closes = [m.start() for m in re.finditer(r"\}", text)]
    for i in range(len(opens)):
        for j in range(len(closes)-1, i-1, -1):
            s = text[opens[i]:closes[j]+1]
            try: json.loads(s); return s
            except: continue
    try:
        json.loads(text.strip()); return text.strip()
    except: return None

def _safe_json_parse(text: str) -> Dict[str, Any]:
    if not text: return {}
    try: return json.loads(text)
    except: 
        block = _extract_json_block(text)
        if block:
            try: return json.loads(block)
            except: return {}
    return {}

def _shorten(s: str, n: int=220) -> str:
    if not s: return ""
    s = s.strip()
    return s if len(s)<=n else s[:n].rstrip()+"…"

def _cap_text(s: str, max_len: int=800) -> str:
    if not s: return ""
    s = s.strip()
    return s if len(s)<=max_len else s[:max_len].rstrip()+"…"

def _read_principles() -> str:
    """
    Load principles markdown from config.PRINCIPLES_FILE.
    Returns plain text (empty string if missing).
    """
    path = getattr(config, "PRINCIPLES_FILE", "principles.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                print(f"[WARN] {path} is empty; principles alignment may be poor.")
            return text
    except FileNotFoundError:
        print(f"[WARN] {path} not found; principles alignment may be poor.")
        return ""


def _coerce_schema(raw: Dict[str, Any], answers: Dict[str,str]) -> Dict[str,Any]:
    notes = raw.get("notes") or {}

    def pick(*keys):
        for k in keys:
            v = answers.get(k)
            if v: return v
        return ""

    # readiness inference
    readiness_text = pick("readiness")
    readiness_flag = "unknown"
    if readiness_text:
        t = readiness_text.lower()
        if any(w in t for w in ["immediate","ready","now"]): readiness_flag="ready_now"
        elif any(w in t for w in ["week","month"," from ","start "]): readiness_flag="date"
        elif any(w in t for w in ["prep","support","refresh","not ready"]): readiness_flag="needs_prep"

    # suitability heuristic if LLM didn’t provide one
    suitability = raw.get("suitability")
    if suitability not in ("strong","average","weak"):
        mot = pick("why_lunartech","motivation")
        exp = pick("experience")
        if len(mot)>120 or len(exp)>120: suitability="strong"
        elif len(mot)>40 or len(exp)>40: suitability="average"
        else: suitability="weak"

    structured = {
        "raw_answers":{
            "background": pick("background","full_name_background"),
            "why_lunartech": pick("why_lunartech","motivation"),
            "experience": pick("experience"),
            "future_goals": pick("future_goals","goals"),
            "readiness": pick("readiness"),
        },
        "notes":{
            "background": notes.get("background") or _shorten(pick("background","full_name_background")),
            "why_lunartech": notes.get("why_lunartech") or _shorten(pick("why_lunartech","motivation")),
            "experience": notes.get("experience") or _shorten(pick("experience")),
            "future_goals": notes.get("future_goals") or _shorten(pick("future_goals","goals")),
            "readiness": notes.get("readiness") or _shorten(pick("readiness")),
        },
        "evaluation": raw.get("evaluation") or "Automated notes generated; consider human review for final decision.",
        "suitability": suitability,
        "readiness_flag": raw.get("readiness_flag") or readiness_flag,
    }

    # Principles alignment keys should mirror principles.md list items.
    principles_md = _read_principles()
    principle_names = []
    for line in (principles_md.splitlines() if principles_md else []):
        m = re.match(r"\s*(?:[-*]|\d+\.)\s*(.+)", line)
        if m: principle_names.append(m.group(1).strip())
    pa = raw.get("principles_alignment") or {}
    structured["principles_alignment"] = {name: pa.get(name,"") for name in principle_names}

    return structured

def _render_markdown(summary: Dict[str, Any]) -> str:
    n = summary.get("notes", {})
    eval_txt = summary.get("evaluation", "")
    suit = summary.get("suitability", "unknown")
    ready = summary.get("readiness_flag", "unknown")

    lines = [
        "### Candidate Review (Condensed)",
        f"- **Background:** {n.get('background','')}",
        f"- **Why LunarTech:** {n.get('why_lunartech','')}",
        f"- **Experience:** {n.get('experience','')}",
        f"- **Goals:** {n.get('future_goals','')}",
        f"- **Readiness:** {n.get('readiness','')}  (_flag_: {ready})",
        "",
        f"**Evaluation:** {eval_txt}",
        f"**Suitability:** {suit.upper()}",
        "",
        "_Full raw answers are stored in the session JSON under `summary.raw_answers`._",
    ]

    if summary.get("principles_alignment"):
        lines.append("\n### Principles Alignment")
        for k,v in summary["principles_alignment"].items():
            lines.append(f"- **{k}:** {v or '—'}")

    return "\n".join(lines)

# ---------- Template (no LLM) ----------

def _template_summary(session:Dict[str,Any])->Tuple[str,Dict[str,Any]]:
    answers = session.get("answers",{}) or {}
    structured=_coerce_schema({},answers)
    structured["evaluation"]="Template summary: Candidate provided responses. Enable LLM summaries for richer evaluation."
    md=_render_markdown(structured)
    return md,structured

# ---------- LLM path (tightened principles evidence) ----------

_SYSTEM = (
    "You are an admissions assistant for a data/AI bootcamp. "
    "Return STRICT JSON only. Do not include any prose outside JSON."
)

_USER_FMT = """
You will evaluate a candidate's interview **only** using the answers below and the provided principles.
Do **not** invent information. If there is **no clear supporting evidence** for a principle in the answers, leave that principle's value as an empty string.

Interview answers (JSON-like; keys may vary):
{answers_json}

Evaluation principles (markdown list items):
{principles_text}

Return a single JSON object with EXACT keys:
{{
  "raw_answers": {{
    "background": "",
    "why_lunartech": "",
    "experience": "",
    "future_goals": "",
    "readiness": ""
  }},
  "notes": {{
    "background": "1–2 sentence note based strictly on answers",
    "why_lunartech": "1–2 sentence note",
    "experience": "1–2 sentence note",
    "future_goals": "1–2 sentence note",
    "readiness": "1–2 sentence note"
  }},
  "evaluation": "Overall suitability notes (2–4 sentences, grounded in answers).",
  "suitability": "strong|average|weak",
  "readiness_flag": "ready_now|date|needs_prep|unknown",
  "principles_alignment": {{
    // keys must mirror each principle name exactly;
    // each value must be SHORT evidence from the interview (1 sentence max) or "" if not supported.
  }}
}}

Rules:
- Use **only** the provided answers for evidence; do not assume background knowledge.
- If a principle is **not** covered by any answer, set its value to "" (empty string).
- Keep notes concise and factual.
- Output **STRICT JSON only** (no markdown fences or extra text).
"""

def _llm_summary(session:Dict[str,Any])->Tuple[str,Dict[str,Any]]:
    time.sleep(float(getattr(config, "SUMMARY_COOLDOWN_SECONDS", 1.5)))
    answers = session.get("answers",{}) or {}
    cap = int(getattr(config, "SUMMARY_ANSWER_CAP", 800))
    answers_capped = {k:_cap_text(v, cap) for k,v in answers.items() if v}

    client=LLMClient()
    principles_text=_read_principles()
    prompt=_USER_FMT.format(
        answers_json=json.dumps(answers_capped,ensure_ascii=False),
        principles_text=principles_text or "(no principles provided)"
    )
    response=client.generate(f"{_SYSTEM}\n\n{prompt}")
    raw=_safe_json_parse(response)
    structured=_coerce_schema(raw,answers)
    md=_render_markdown(structured)
    return md,structured

# ---------- Public API ----------

def summarize_session(session:Dict[str,Any])->Tuple[str,Dict[str,Any]]:
    if not bool(getattr(config,"USE_LLM_SUMMARY",False)):
        return _template_summary(session)
    try:
        return _llm_summary(session)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return _template_summary(session)
