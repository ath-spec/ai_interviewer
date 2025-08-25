import argparse
import os
from datetime import datetime

from app.interview import InterviewAgent
from app.logger import SessionLogger
from app.summarize import summarize_session
from app.faq import faq_loop, render_faq_md
from app import config

# Voice modules (optional)
try:
    from app.voice_input import record_audio, transcribe_audio
    from app.voice_output import speak
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


def run_interview():
    agent = InterviewAgent()
    logger = SessionLogger()

    print("\n--- AI Interview Agent ---\n")
    print(
        f"Mode: User input = {'Voice' if config.USER_VOICE_INPUT else 'Text'} | "
        f"Agent output = {'Voice+Text' if config.AGENT_VOICE_OUTPUT else 'Text only'}\n"
    )

    if config.USER_VOICE_INPUT and not VOICE_AVAILABLE:
        print("[ERROR] Voice modules not available. Falling back to text input.\n")
        config.USER_VOICE_INPUT = False
    if config.AGENT_VOICE_OUTPUT and not VOICE_AVAILABLE:
        print("[ERROR] Voice modules not available. Agent will use text only.\n")
        config.AGENT_VOICE_OUTPUT = False

    faq_pairs = []

    try:
        # ---------- INTERVIEW LOOP ----------
        agent.start()
        while agent.has_next():
            q = agent.next_question()

            # Agent asks the question
            print(f"Agent: {q['text']}")
            if config.AGENT_VOICE_OUTPUT:
                speak(q["text"])

            # User answers
            if config.USER_VOICE_INPUT:
                path = record_audio()
                user = transcribe_audio(path)
                print(f"You (transcribed): {user}")
            else:
                user = input("You: ").strip()

            agent.record_answer(q["key"], user)
            logger.log_turn(role="user", text=user)

            # Agent acknowledgment
            ack = agent.acknowledge(user)
            if ack:
                print(f"Agent: {ack}")
                logger.log_turn(role="agent", text=ack)
                if config.AGENT_VOICE_OUTPUT:
                    speak(ack)

        # ---------- FAQ LOOP ----------
        print("\n--- Candidate Q&A ---")
        # let faq_loop handle the voice cue internally
        speak_fn = speak if (VOICE_AVAILABLE and config.AGENT_VOICE_OUTPUT) else None
        faq_pairs = faq_loop(speak_fn=speak_fn)  # returns list of {"question","answer"}

        # ---------- SUMMARY ----------
        session = agent.build_session()
        if faq_pairs:
            session["faq"] = faq_pairs

        summary_md, summary_json = summarize_session(session)
        if faq_pairs:
            faq_md = render_faq_md(faq_pairs)
            if faq_md:
                summary_md = summary_md + "\n---\n" + faq_md
                summary_json["faq_markdown"] = faq_md

        print("\n--- Summary ---")
        print(summary_md)
        if config.AGENT_VOICE_OUTPUT:
            speak("Here is a summary of your interview.")

        # ---------- SAVE ----------
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(config.SESSION_DIR, exist_ok=True)
        logger.save_transcript(os.path.join(config.SESSION_DIR, f"{now}_transcript.txt"))
        logger.save_session_json(
            os.path.join(config.SESSION_DIR, f"{now}_session.json"),
            session,
            summary_json,
        )

        print(f"\nSaved transcript and session JSON in '{config.SESSION_DIR}/'.")

    except KeyboardInterrupt:
        print("\n\nInterview interrupted. Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="AI Interview Agent")
    parser.add_argument("--faq", action="store_true", help="Run standalone FAQ mode (debug)")
    args = parser.parse_args()

    run_interview()

    if args.faq:
        print("\n--- FAQ Mode (standalone) ---")
        speak_fn = speak if (VOICE_AVAILABLE and config.AGENT_VOICE_OUTPUT) else None
        _ = faq_loop(speak_fn=speak_fn)


if __name__ == "__main__":
    main()
