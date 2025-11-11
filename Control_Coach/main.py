import time
import threading
import pyttsx3
from groq import Groq
import readchar  # instant keypress in terminal

# ---------- CONFIG ----------
groq_client = Groq(api_key="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")
engine = pyttsx3.init()

REPS_PER_SET = 10
REST_SECONDS = 10
MOTIVATION_INTERVAL = 15  # seconds between motivational lines

rep_count = 0
set_count = 0
last_message_time = time.time()

# ---------- SPEECH (NON-BLOCKING) ----------
def speak(text):
    print(f"🎤 {text}")
    def _talk():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_talk, daemon=True).start()

# ---------- AI MOTIVATION ----------
def get_fast_coaching(set_num, rep_num):
    prompt = f"Give one short hype line for someone doing set {set_num}, rep {rep_num}. Make it energetic, not cringe."
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    except:
        return "Keep going."

# ---------- START ----------
print("\n--- BICEP COACH STARTED ---")
print("Press ANY key to count a rep. Press Q to quit.\n")
speak("Begin your first set!")

while True:
    key = readchar.readkey()  # fast, no buffering

    # Quit
    if key.lower() == "q":
        speak("Workout finished. Strong work today.")
        break

    # Count a rep instantly
    rep_count += 1
    speak(str(rep_count))

    # Completed a set?
    if rep_count >= REPS_PER_SET:
        set_count += 1
        speak(f"Set {set_count} complete. Rest for {REST_SECONDS} seconds.")
        print(f"\n✅ Set {set_count} done. Resting...\n")

        # Rest timer (no speech each second → no lag)
        for i in range(REST_SECONDS, 0, -1):
            print(f"Rest: {i}s", end="\r")
            time.sleep(1)

        rep_count = 0
        print("\n🔥 Begin next set!\n")
        speak("Begin your next set now.")

    # Periodic hype
    if rep_count > 0 and (time.time() - last_message_time > MOTIVATION_INTERVAL):
        speak(get_fast_coaching(set_count, rep_count))
        last_message_time = time.time()