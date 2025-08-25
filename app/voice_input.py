"""
Central voice input wrapper (STT).
Prefers config.VOICE_INPUT_ENGINE.
"""
from app import config

engine = getattr(config, "VOICE_INPUT_ENGINE", "free").lower()

if engine == "openai":
    from app.voice_openai import record_audio, transcribe_audio
elif engine == "free":
    from app.voice_free import record_audio, transcribe_audio
else:
    raise ValueError(f"Unknown VOICE_INPUT_ENGINE: {engine}")

__all__ = ["record_audio", "transcribe_audio"]
