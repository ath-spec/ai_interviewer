from app import config
from datetime import datetime

REQUIRED_QUESTIONS = [
    {"key": "background", "text": "Tell us your background and journey. What inspires you to pursue data science or AI?"},
    {"key": "why_company", "text": "What motivates you to learn at [Company Name] specifically?"},
    {"key": "experience", "text": "What hands-on projects or technical experiences have you completed so far?"},
    {"key": "future_goals", "text": "What are your goals after completing the bootcamp? How do you hope to make an impact?"},
    {"key": "readiness", "text": "Are you ready to start? If not, what support or prep would help you feel fully prepared?"},
]

class InterviewAgent:
    def __init__(self):
        self.questions = REQUIRED_QUESTIONS[:]
        self.answers = {}
        self.turns = []  # [(role, text, timestamp)]
        self.index = 0

    def start(self):
        self._log_turn("agent", "Hello! I’ll ask a few questions to learn about you.")

    def has_next(self):
        return self.index < len(self.questions)

    def next_question(self):
        q = self.questions[self.index]
        self._log_turn("agent", q["text"])
        self.index += 1
        return q

    # ---------- Core logic ----------

    def _too_short(self, text: str) -> bool:
        """
        Heuristic: consider 'too short' only if it's basically empty
        or trivially short. We use MIN_ANSWER_CHARS from config.
        """
        if text is None:
            return True
        t = text.strip()
        if len(t) == 0:
            return True
        return len(t) < getattr(config, "MIN_ANSWER_CHARS", 5)

    def record_answer(self, key: str, user_text: str):
        """
        Accept any non-empty answer immediately.
        If it's empty/too short, reprompt ONCE; then accept the second input.
        """
        self._log_turn("user", user_text)

        
        if self._too_short(user_text):
            reprompt = "I didn't quite catch that. Could you add a bit more detail?"
            print(f"Agent: {reprompt}")
            self._log_turn("agent", reprompt)
            user_text = input("You: ").strip()
            self._log_turn("user", user_text) 

            if len(user_text) == 0:
                user_text = "(no response provided)"

        # Save final
        self.answers[key] = user_text

    def acknowledge(self, user_text: str) -> str:
        """
        Light-touch acknowledgement so the conversation feels human.
        """
        if user_text and len(user_text) > 20:
            return "Thanks, that helps."
        return ""

    def build_session(self):
        return {
            "metadata": {"project": "AI Interview Agent", "version": "0.1.0"},
            "answers": self.answers,
            "turns": [{"role": r, "text": t, "time": ts} for (r, t, ts) in self.turns],
        }

    # ---------- Utilities ----------

    def _log_turn(self, role: str, text: str):
        self.turns.append((role, text, datetime.now().isoformat()))
