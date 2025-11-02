from enum import Enum, auto
import torch
from transformers import pipeline
from gtts import gTTS
import os, time


class CoachState(Enum):
    ACTIVE = auto()
    RESTING = auto()
    STOPPED = auto()


class Coach:
    def __init__(self, participant_id):
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # or "mps" for Mac
        )

        self.language = "en"
        self.participant_id = participant_id
        self.state = CoachState.ACTIVE
        self.rest_start = None
        self.rest_duration = 0
        self.prev_fatigue_score = 0.0

        #logging setup
        self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
        os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
        if not os.path.exists(self.logging_file_path):
            with open(self.logging_file_path, "w") as f:
                f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm\n")

    #fatigue scale
    def scale_fatigue(self, fatigue_score):
        alpha = 0.3
        fatigue_score = alpha * fatigue_score + (1 - alpha) * self.prev_fatigue_score
        self.prev_fatigue_score = fatigue_score

        scaled = int(round(fatigue_score * 6 + 1))
        return max(1, min(scaled, 7)), fatigue_score

    #rule based message
    def rule_based_message(self, fatigue_level, fatigue_score):
        if fatigue_level <= 2:
            return "Continue your set and keep a steady pace."
        elif fatigue_level <= 4:
            return "Maintain your pace and focus on form and breathing."
        elif fatigue_level <= 6:
            rest = self.get_rest_duration(fatigue_score)
            return f"Stop this set and rest for about {rest} seconds."
        else:
            rest = self.get_rest_duration(fatigue_score)
            return f"Stop completely. You’ve reached exhaustion. Rest for at least {rest} seconds."

    #rest duration
    def get_rest_duration(self, fatigue_score):
        if fatigue_score < 0.4:
            return 0
        elif fatigue_score < 0.7:
            return 60
        else:
            return 120 + int((fatigue_score - 0.7) * 300)

    #rephrasing
    def llm_rephrase(self, rule_text, fatigue_level):
        """
        Rephrase a deterministic rule-based instruction into a motivational coaching tone.
        """
        prompt = (
            "You are a motivational fitness coach guiding a person doing bicep curls.\n"
            "Rephrase the following instruction naturally, keeping the same meaning.\n"
            "Do not add new actions or durations. Keep it concise and motivating.\n\n"
            f"Fatigue level: {fatigue_level}\n"
            f"Instruction: {rule_text}\n"
            "Coach response:"
        )
        outputs = self.pipe(prompt, max_new_tokens=50, do_sample=True, top_p=0.9)
        return outputs[0]["generated_text"].split("Coach response:")[-1].strip()

    #main command
    def give_command(self, fatigue_score):
        fatigue_level, fatigue_score = self.scale_fatigue(fatigue_score)

        #check recovery during rest
        if self.state == CoachState.RESTING:
            elapsed = time.time() - self.rest_start
            recovered = fatigue_score < 0.3

            if recovered:
                self.state = CoachState.ACTIVE
                msg = "You’ve recovered enough — great job. Start your next set when ready."
                self.speak(msg)
                self.log(fatigue_level, fatigue_score, msg, "Recovered early")
                return msg

            elif elapsed >= self.rest_duration:
                self.state = CoachState.ACTIVE
                msg = "Rest complete. Resume your next set."
                self.speak(msg)
                self.log(fatigue_level, fatigue_score, msg, "Rest complete")
                return msg

            remaining = int(self.rest_duration - elapsed)
            if remaining > 0 and remaining % 30 == 0:
                self.speak(f"{remaining} seconds left in your rest.")
            return

        #deciding what to say
        rule_text = self.rule_based_message(fatigue_level, fatigue_score)
        llm_text = self.llm_rephrase(rule_text, fatigue_level)

        self.speak(llm_text)
        self.log(fatigue_level, fatigue_score, llm_text, rule_text)

        #prompt rest at level 5
        if fatigue_level >= 5:
            self.state = CoachState.RESTING
            self.rest_duration = self.get_rest_duration(fatigue_score)
            self.rest_start = time.time()
        elif fatigue_level == 7:
            self.state = CoachState.STOPPED

        return llm_text

    #text to speech
    def speak(self, text):
        tts = gTTS(text=text, lang=self.language)
        filename = "coach_command.mp3"
        tts.save(filename)
        try:
            os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
        except Exception:
            print("Audio unable to be played.")
        print("AI Coach:", text)

    #logging
    def log(self, fatigue_level, fatigue_score, llm_text, rule_text):
        ts = int(time.time() * 1000)
        with open(self.logging_file_path, "a") as f:
            f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text}\n")

