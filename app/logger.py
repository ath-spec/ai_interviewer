import json
from datetime import datetime

class SessionLogger:
    def __init__(self):
        self.lines = []

    def log_turn(self, role: str, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lines.append(f"[{ts}] {role.upper()}: {text}")

    def save_transcript(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))

    def save_session_json(self, path: str, session_obj: dict, summary_json: dict):
        payload = {
            "session": session_obj,
            "summary": summary_json,
            "saved_at": datetime.now().isoformat()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
