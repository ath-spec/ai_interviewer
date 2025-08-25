"""
Free/local voice backend.
- STT: SpeechRecognition + PocketSphinx (offline), fallback to Google Web Speech (online)
- TTS: pyttsx3 (offline, system voices)
- Recording stops after a long pause (config.VOICE_PAUSE_SECONDS) or max duration.
"""

from __future__ import annotations
import tempfile
import threading
from typing import Optional
import simpleaudio as sa
import numpy as np
import speech_recognition as sr
import pyttsx3

from app import config

_recognizer = sr.Recognizer()
_tts_lock = threading.Lock()  # ensure TTS calls never overlap

def play_beep(frequency=1000, duration=200, volume=0.5):
    """Play a short beep sound before recording."""
    fs = 44100  # sample rate
    t = np.linspace(0, duration / 1000, int(fs * duration / 1000), False)
    wave = np.sin(frequency * t * 2 * np.pi)
    audio = (wave * (32767 * volume)).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

def list_microphones() -> list[str]:
    """Return available microphone device names (useful for choosing MIC_DEVICE_INDEX)."""
    return sr.Microphone.list_microphone_names()


# -------- Recording (until silence) --------
def record_audio() -> str:
    """
    Record from the default / selected microphone until a long pause, then save to temp WAV.
    Returns the WAV file path.
    """
    sr_rate: int = int(getattr(config, "VOICE_SAMPLE_RATE", 16000))
    pause: float = float(getattr(config, "VOICE_PAUSE_SECONDS", 8.0))
    calibrate: float = float(getattr(config, "VOICE_CALIBRATE_SECONDS", 1.0))
    max_secs: float = float(getattr(config, "VOICE_MAX_RECORD_SECONDS", 180.0))
    mic_index: Optional[int] = getattr(config, "MIC_DEVICE_INDEX", None)

    # Configure SR silence behavior
    _recognizer.dynamic_energy_threshold = True
    _recognizer.pause_threshold = pause       
    _recognizer.non_speaking_duration = 0.6

    # Open mic and capture
    with sr.Microphone(sample_rate=sr_rate, device_index=mic_index) as source:
        print(f"🎙️ Calibrating mic ({calibrate}s)...")
        _recognizer.adjust_for_ambient_noise(source, duration=calibrate)
        print(f"🎙️ Recording... speak freely. (Stops after {pause}s of silence)")
        audio = _recognizer.listen(
            source,
            timeout=None,                     
            phrase_time_limit=max_secs        
        )

    # Save to temp WAV (so downstream is file-path based and consistent)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio.get_wav_data())
    tmp.flush()
    tmp.close()
    print("✅ Stopped (pause detected or max duration reached).")
    return tmp.name


# -------- Transcription (offline-first) --------
def transcribe_audio(path: str) -> str:
    """
    Transcribe audio file at `path`.
    1) Try PocketSphinx (offline)
    2) Fallback to Google Web Speech (online)
    Returns a plain text string ("" on failure).
    """
    with sr.AudioFile(path) as source:
        audio = _recognizer.record(source)

    # PocketSphinx
    try:
        text = _recognizer.recognize_sphinx(audio)
        return text.strip()
    except Exception as e1:
        print(f"[WARN] PocketSphinx failed: {e1}")

    # Google Web Speech
    try:
        text = _recognizer.recognize_google(audio, language="en-US")
        return text.strip()
    except Exception as e2:
        print(f"[ERROR] Google STT failed: {e2}")
        return ""


# -------- TTS (robust on Windows) --------
def speak(text: str):
    """
    Offline TTS via pyttsx3. Re-initialize per call to avoid 'only first utterance plays' bug on Windows.
    Blocking: returns when playback completes.
    """
    if not text or not text.strip():
        return
    with _tts_lock:
        try:
            engine = pyttsx3.init()       
            engine.say(text)
            engine.runAndWait()           
            engine.stop()
        except Exception as e:
            print(f"[TTS ERROR] {e}")