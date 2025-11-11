import os
import time
import threading
from queue import Queue, Empty

import readchar
import pyttsx3

# -------- CONFIG --------
REPS_PER_SET = 10
REST_SECONDS = 10
MOTIVATION_INTERVAL = 15   # seconds between AI messages
ENABLE_TTS = True          # set False if you want silent mode
ENABLE_AI = True           # set False to disable AI motivation

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"

# --------- TTS WORKER (NON-BLOCKING) ---------
class TTSWorker(threading.Thread):
    def __init__(self, enable_tts=True):
        super().__init__(daemon=True)
        self.enable_tts = enable_tts
        self.q = Queue()
        self._stop = threading.Event()

    def run(self):
        engine = None
        if self.enable_tts:
            try:
                engine = pyttsx3.init()
            except Exception as e:
                print(f"[TTS disabled] {e}")
                self.enable_tts = False

        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.1)
            except Empty:
                continue

            # Always print immediately (instant feedback)
            print(f"🔊 {text}")

            if self.enable_tts and engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS error] {e}")
                    # Don’t crash; just keep printing text

    def speak(self, text: str):
        self.q.put(text)

    def stop(self):
        self._stop.set()

# --------- AI MOTIVATION WORKER (NON-BLOCKING) ---------
class AIMotivation(threading.Thread):
    def __init__(self, speaker: TTSWorker, interval: int):
        super().__init__(daemon=True)
        self.speaker = speaker
        self.interval = interval
        self.state_lock = threading.Lock()
        self.set_num = 0
        self.rep_num = 0
        self._stop = threading.Event()

        # Lazy import so it’s optional
        self.client = None
        if ENABLE_AI and GROQ_API_KEY:
            try:
                from groq import Groq
                self.client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                print(f"[AI disabled] {e}")

    def update_progress(self, set_num: int, rep_num: int):
        with self.state_lock:
            self.set_num = set_num
            self.rep_num = rep_num

    def run(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            if not self.client:
                continue
            with self.state_lock:
                s, r = self.set_num, self.rep_num
            if s == 0 and r == 0:
                continue  # don’t hype before starting

            prompt = f"Give one short energetic (not cringe) hype line for set {s}, rep {r}."
            try:
                resp = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.9,
                )
                msg = (resp.choices[0].message.content or "").strip()
                if msg:
                    self.speaker.speak(msg)
            except Exception:
                # Silent failure; never block the main loop
                pass

    def stop(self):
        self._stop.set()

# --------- MAIN PROGRAM ---------
def main():
    speaker = TTSWorker(enable_tts=ENABLE_TTS)
    speaker.start()

    ai = AIMotivation(speaker, MOTIVATION_INTERVAL)
    if ENABLE_AI and GROQ_API_KEY:
        ai.start()

    rep_count = 0
    set_count = 0

    print("\n--- BICEP COACH STARTED ---")
    print("Press ANY key to count a rep. Press Q to quit.\n")
    speaker.speak("Begin your first set!")

    while True:
        key = readchar.readkey()   # returns instantly when a key is pressed

        # Quit immediately on 'q' or 'Q'
        if key.lower() == 'q':
            speaker.speak("Workout finished. Strong work today.")
            break

        # Count rep instantly (no I/O blocking)
        rep_count += 1
        speaker.speak(str(rep_count))

        # Share progress with AI thread (non-blocking)
        if ENABLE_AI and GROQ_API_KEY:
            ai.update_progress(set_count, rep_count)

        # Completed a set?
        if rep_count >= REPS_PER_SET:
            set_count += 1
            speaker.speak(f"Set {set_count} complete. Rest for {REST_SECONDS} seconds.")
            print(f"\n✅ Set {set_count} done. Resting...\n")

            # Simple rest countdown without speech each second (keeps it snappy)
            for i in range(REST_SECONDS, 0, -1):
                print(f"Rest: {i:2d}s", end="\r")
                time.sleep(1)
            print(" " * 20, end="\r")  # clear line

            rep_count = 0
            speaker.speak("Begin your next set now.")

            # Update AI progress after reset
            if ENABLE_AI and GROQ_API_KEY:
                ai.update_progress(set_count, rep_count)

    # Cleanup
    if ENABLE_AI and GROQ_API_KEY:
        ai.stop()
    speaker.stop()

if __name__ == "__main__":
    main()