import os, time, random, hashlib, requests
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from app import config

# Optional cache
ENABLE_CACHE = True
CACHE_DIR = ".llm_cache"
if ENABLE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(prompt: str, model: str) -> str:
    import hashlib
    h = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.txt")

def _cache_get(prompt: str, model: str) -> Optional[str]:
    if not ENABLE_CACHE: return None
    p = _cache_path(prompt, model)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f: return f.read()
        except: return None
    return None

def _cache_set(prompt: str, model: str, text: str):
    if not ENABLE_CACHE: return
    try:
        with open(_cache_path(prompt, model), "w", encoding="utf-8") as f:
            f.write(text)
    except: pass

class LLMClient:
    """
    Flexible LLM wrapper for OpenAI and Mistral APIs.
    """
    def __init__(self, timeout: float = 30.0):
        self.backend = config.LLM_BACKEND
        self.timeout = timeout

        if self.backend == "openai":
            from openai import OpenAI
            self.primary_model = config.OPENAI_MODEL
            self.fallback_model = config.OPENAI_FALLBACK_MODEL
            self.client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=timeout)

        elif self.backend == "mistral":
            self.api_key = config.MISTRAL_API_KEY
            if not self.api_key:
                raise RuntimeError("MISTRAL_API_KEY not set in env")
            self.model = config.MISTRAL_MODEL
            self.base_url = config.MISTRAL_API_URL

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # ---------- OpenAI ----------
    def _call_openai(self, model: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful admissions assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content

    # ---------- Mistral ----------
    def _call_mistral(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": False,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    # ---------- Public generate ----------
    def generate(self, prompt: str) -> str:
        model = self.primary_model if self.backend == "openai" else self.model

        # Cache
        cached = _cache_get(prompt, model)
        if cached is not None:
            return cached

        if self.backend == "openai":
            try:
                text = self._call_openai(self.primary_model, prompt)
            except Exception as e:
                print(f"[LLM WARN] OpenAI primary failed: {e}")
                if self.fallback_model:
                    text = self._call_openai(self.fallback_model, prompt)
                else:
                    raise
        elif self.backend == "mistral":
            text = self._call_mistral(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        _cache_set(prompt, model, text)
        return text
