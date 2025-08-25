"""
Central voice output wrapper (TTS).
Prefers config.VOICE_OUTPUT_ENGINE.
"""
from app import config

engine = getattr(config, "VOICE_OUTPUT_ENGINE", "free").lower()

if engine == "openai":
    from app.voice_openai import speak
elif engine == "free":
    from app.voice_free import speak
else:
    raise ValueError(f"Unknown VOICE_OUTPUT_ENGINE: {engine}")

__all__ = ["speak"]
