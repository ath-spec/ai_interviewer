import os
from dotenv import load_dotenv
load_dotenv()

# ---------------- Project ----------------
PROJECT_NAME = "AI Interview Agent"
VERSION = "0.1.0"

# ---------------- Summaries ----------------
USE_LLM_SUMMARY = True
SUMMARY_COOLDOWN_SECONDS = 1.5
SUMMARY_ANSWER_CAP = 800

# ---------------- Interview heuristics ----------------
MIN_ANSWER_CHARS = 5
UNCLEAR_TOKENS = ["idk", "don't know", "not sure", "n/a", "na"]

# ---------------- Files ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "data", "faq.md")
PRINCIPLES_FILE = os.path.join(BASE_DIR, "data","principles.md")
SESSION_DIR = "sessions"

# ---------------- Voice ----------------
AGENT_VOICE_OUTPUT = True
USER_VOICE_INPUT  = True
VOICE_SAMPLE_RATE = 16000
VOICE_MAX_RECORD_SECONDS = 180
VOICE_PAUSE_SECONDS = 5.0
VOICE_CALIBRATE_SECONDS = 1.0
VOICE_VAD_SILENCE_LEVEL = 500
MIC_DEVICE_INDEX = None

VOICE_INPUT_ENGINE  = "openai"
VOICE_OUTPUT_ENGINE = "free"

# ---------------- LLM ----------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "mistral").lower()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")

# Mistral settings
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai")

if LLM_BACKEND not in {"openai", "mistral"}:
    raise ValueError(f"LLM_BACKEND must be 'openai' or 'mistral', got {LLM_BACKEND}")
