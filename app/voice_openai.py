"""
Voice backend using OpenAI.
- STT: Whisper (speech-to-text, via GPT-4o-transcribe)
- TTS: GPT-4o-mini-tts (text-to-speech)
- Recording stops after VOICE_PAUSE_SECONDS of silence or VOICE_MAX_RECORD_SECONDS cap.
"""

import tempfile
import threading
import wave
import simpleaudio as sa
import numpy as np
import speech_recognition as sr
from openai import OpenAI
from app import config

client = OpenAI()
_recognizer = sr.Recognizer()
_tts_lock = threading.Lock()


def play_beep(frequency=1000, duration=200, volume=0.5):
    """Play a short beep sound before recording."""
    fs = 44100
    t = np.linspace(0, duration / 1000, int(fs * duration / 1000), False)
    wave_ = np.sin(frequency * t * 2 * np.pi)
    audio = (wave_ * (32767 * volume)).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()


# --- Record user audio (mic -> temp WAV, silence-aware) ---
def record_audio() -> str:
    """
    Record from microphone until VOICE_PAUSE_SECONDS of silence or max duration.
    Returns path to WAV file.
    """
    sr_rate = int(getattr(config, "VOICE_SAMPLE_RATE", 16000))
    pause = float(getattr(config, "VOICE_PAUSE_SECONDS", 5.0))
    calibrate = float(getattr(config, "VOICE_CALIBRATE_SECONDS", 1.0))
    max_secs = float(getattr(config, "VOICE_MAX_RECORD_SECONDS", 180.0))
    mic_index = getattr(config, "MIC_DEVICE_INDEX", None)

    _recognizer.dynamic_energy_threshold = True
    _recognizer.pause_threshold = pause
    _recognizer.non_speaking_duration = 0.6

    with sr.Microphone(sample_rate=sr_rate, device_index=mic_index) as source:
        print(f"Calibrating mic ({calibrate}s)...")
        _recognizer.adjust_for_ambient_noise(source, duration=calibrate)
        print(f"Recording... speak freely. (Stops after {pause}s silence)")
        audio = _recognizer.listen(
            source,
            timeout=None,
            phrase_time_limit=max_secs
        )

    # Save to temp WAV for consistency
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio.get_wav_data())
    tmp.flush()
    tmp.close()
    print("Stopped (pause detected or max duration reached).")
    return tmp.name


# --- Transcribe recorded audio with Whisper ---
def transcribe_audio(path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper (gpt-4o-transcribe).
    Returns recognized text ("" if fails).
    """
    try:
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )
        return resp.text.strip()
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return ""


# --- Speak agent response ---
def speak(text: str, voice: str = "alloy"):
    """
    Text-to-speech via OpenAI API.
    Blocking: returns after playback finishes.
    """
    if not text or not text.strip():
        return

    with _tts_lock:
        try:
            resp = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                format="wav",
                input=text
            )

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(resp.content)
            tmp.flush()
            tmp.close()

            wave_obj = sa.WaveObject.from_wave_file(tmp.name)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"[TTS ERROR] {e}")
