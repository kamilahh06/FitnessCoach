# # from enum import Enum, auto
# # import torch
# # from transformers import pipeline
# # from gtts import gTTS
# # import os, time


# # class CoachState(Enum):
# #     ACTIVE = auto()
# #     RESTING = auto()
# #     STOPPED = auto()


# # class Coach:
# #     def __init__(self, participant_id):
# #         self.pipe = pipeline(
# #             "text-generation",
# #             model="google/gemma-2-2b-it",
# #             model_kwargs={"torch_dtype": torch.bfloat16},
# #             device="mps",  # or "mps" for Mac
# #         )

# #         self.language = "en"
# #         self.participant_id = participant_id
# #         self.state = CoachState.ACTIVE
# #         self.rest_start = None
# #         self.rest_duration = 0
# #         self.prev_fatigue_score = 0.0

# #         #logging setup
# #         self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
# #         # print(self.logging_file_path)
# #         os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
# #         if not os.path.exists(self.logging_file_path):
# #             with open(self.logging_file_path, "w") as f:
# #                 f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm\n")

# #     #fatigue scale
# #     def scale_fatigue(self, fatigue_score):
# #         alpha = 0.3
# #         fatigue_score = alpha * fatigue_score + (1 - alpha) * self.prev_fatigue_score
# #         self.prev_fatigue_score = fatigue_score

# #         scaled = int(round(fatigue_score * 6 + 1))
# #         return max(1, min(scaled, 7)), fatigue_score

# #     #rule based message
# #     def rule_based_message(self, fatigue_level, fatigue_score):
# #         if fatigue_level <= 2:
# #             return "Continue your set and keep a steady pace."
# #         elif fatigue_level <= 4:
# #             return "Maintain your pace and focus on form and breathing."
# #         elif fatigue_level <= 6:
# #             rest = self.get_rest_duration(fatigue_score)
# #             return f"Stop this set and rest for about {rest} seconds."
# #         else:
# #             rest = self.get_rest_duration(fatigue_score)
# #             return f"Stop completely. You’ve reached exhaustion. Rest for at least {rest} seconds."

# #     #rest duration
# #     def get_rest_duration(self, fatigue_score):
# #         if fatigue_score < 0.4:
# #             return 0
# #         elif fatigue_score < 0.7:
# #             return 60
# #         else:
# #             return 120 + int((fatigue_score - 0.7) * 300)

# #     #rephrasing
# #     def llm_rephrase(self, rule_text, fatigue_level):
# #         """
# #         Rephrase a deterministic rule-based instruction into a motivational coaching tone.
# #         """
# #         prompt = (
# #             "You are a motivational fitness coach guiding a person doing bicep curls.\n"
# #             "Rephrase the following instruction naturally, keeping the same meaning.\n"
# #             "Do not add new actions or durations. Keep it concise and motivating.\n\n"
# #             f"Fatigue level: {fatigue_level}\n"
# #             f"Instruction: {rule_text}\n"
# #             "Coach response:"
# #         )
# #         # outputs = self.pipe(prompt, max_new_tokens=50, do_sample=True, top_p=0.9)
# #         # return outputs[0]["generated_text"].split("Coach response:")[-1].strip()
# #         return ""

# #     #main command
# #     def give_command(self, fatigue_score):
# #         fatigue_level, fatigue_score = self.scale_fatigue(fatigue_score)
# #         print(f"Fatigue Score for Command: {fatigue_score}")

# #         #check recovery during rest
# #         if self.state == CoachState.RESTING:
# #             elapsed = time.time() - self.rest_start
# #             recovered = fatigue_score < 0.3

# #             if recovered:
# #                 self.state = CoachState.ACTIVE
# #                 msg = "You’ve recovered enough — great job. Start your next set when ready."
# #                 self.speak(msg)
# #                 self.log(fatigue_level, fatigue_score, msg, "Recovered early")
# #                 return msg

# #             elif elapsed >= self.rest_duration:
# #                 self.state = CoachState.ACTIVE
# #                 msg = "Rest complete. Resume your next set."
# #                 self.speak(msg)
# #                 self.log(fatigue_level, fatigue_score, msg, "Rest complete")
# #                 return msg

# #             remaining = int(self.rest_duration - elapsed)
# #             if remaining > 0 and remaining % 30 == 0:
# #                 self.speak(f"{remaining} seconds left in your rest.")
# #             return

# #         #deciding what to say
# #         rule_text = self.rule_based_message(fatigue_level, fatigue_score)
# #         llm_text = self.llm_rephrase(rule_text, fatigue_level)

# #         self.speak(llm_text)
# #         self.log(fatigue_level, fatigue_score, llm_text, rule_text)

# #         #prompt rest at level 5
# #         if fatigue_level >= 5:
# #             self.state = CoachState.RESTING
# #             self.rest_duration = self.get_rest_duration(fatigue_score)
# #             self.rest_start = time.time()
# #         elif fatigue_level == 7:
# #             self.state = CoachState.STOPPED

# #         return llm_text

# #     #text to speech
# #     def speak(self, text):
# #         # tts = gTTS(text=text, lang=self.language)
# #         # filename = "coach_command.mp3"
# #         # tts.save(filename)
# #         try:
# #             os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
# #         except Exception:
# #             print("Audio unable to be played.")
# #         # print("AI Coach:", text)

# #     #logging
# #     def log(self, fatigue_level, fatigue_score, llm_text, rule_text):
# #         ts = int(time.time() * 1000)
# #         with open(self.logging_file_path, "a") as f:
# #             # print(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text}\n")
# #             print(f"Fatigue Score: {fatigue_score}")
# #             f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text}\n")


from enum import Enum, auto
import torch
from transformers import pipeline
from gtts import gTTS
import os, time
import pandas as pd
import numpy as np

class CoachState(Enum):
    ACTIVE = auto()
    RESTING = auto()
    STOPPED = auto()

class Coach:
    def __init__(self, participant_id):
        # self.pipe = pipeline(... gated HF model ...)
        self.language = "en"
        self.participant_id = participant_id
        self.state = CoachState.ACTIVE
        # self.pipe = pipeline(
        #     "text-generation",
        #     model="google/gemma-2-2b-it",
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device="mps",  # or "mps" for Mac
        # )

        # recovery / rest tracking
        self.rest_start = None
        self.rest_duration = 0

        self.baseline_rms = None
        self.baseline_thesh = 2 # originally 1.2
        self.rms_history = []
        self.activity_threshold_std = 3.0


        # fatigue memory
        self.prev_fatigue_score = 0.0

        # activity gating (RMS-based)
        self.rms_baseline = None
        self.activity_mult = 3          # active if rms > baseline * this, originally 1.5
        self.min_active_rms = 2e-5        # absolute floor in volts (tune for your hardware)
        self.inactive_decay = 0.2        # per-window drop when inactive (0..1), originally 0.1

        # logging
        self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
        os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
        if not os.path.exists(self.logging_file_path):
            with open(self.logging_file_path, "w") as f:
                f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm\n")

    # -------- activity detection ----------
    def _update_activity_baseline(self, rms, active):
        if self.rms_baseline is None:
            self.rms_baseline = rms
            return
        # If inactive, let baseline track down slowly; if active, don’t chase it up too fast
        beta = 0.10 if not active else 0.02
        self.rms_baseline = (1 - beta) * self.rms_baseline + beta * rms

    # def _is_active(self, rms):
        # if rms is None:
        #     return True  # backward compatible: assume active if no signal provided
        # if self.rms_baseline is None:
        #     # Warm-up: treat as inactive unless clearly above a minimum floor
        #     active = rms > max(self.min_active_rms, 1e-5)
        #     self._update_activity_baseline(rms, active)
        #     return active
        # thresh = max(self.min_active_rms, self.rms_baseline * self.activity_mult)
        # active = rms > thresh
        # self._update_activity_baseline(rms, active)
        # return active

    # def _is_active(self, rms):
    #     # Establish baseline on the first complete window only
    #     if self.baseline_rms is None:
    #         self.baseline_rms = rms
    #         self.baseline_std = 0.0  # not really needed, but for compatibility
    #         print(f"📊 Baseline RMS set: {self.baseline_rms:.3f} (8s window)")
    #         return True  # Treat initial window as active just to start safely

    #     # Once baseline is set, compare current rms to threshold
    #     threshold = self.baseline_rms * self.activity_mult
    #     active = rms > threshold

    #     # Update baseline slowly during inactivity (optional)
    #     if not active:
    #         self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * rms  # gradual drift correction

    #     return active
    def _is_active(self, rms):
        """Return True if user is currently active based on RMS amplitude,
        using hysteresis to avoid flicker."""
        if self.baseline_rms is None:
            # First window — set baseline and assume active to bootstrap
            self.baseline_rms = rms
            print(f"📊 Baseline RMS set: {self.baseline_rms:.3f} (8s window)")
            self.active = True
            return True

        # Create persistent state flag
        if not hasattr(self, "active"):
            self.active = True

        # Two thresholds for hysteresis
        upper_thresh = self.baseline_rms * 2.0   # must exceed to count as active again
        lower_thresh = self.baseline_rms * 1.2   # must fall below to count as inactive

        # Decide state based on thresholds
        if rms < lower_thresh:
            self.active = False
        elif rms > upper_thresh:
            self.active = True

        # Update baseline slowly when inactive to track drift
        if not self.active:
            self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * rms

        return self.active



    # -------- scaling & smoothing ----------
    # def _scale_fatigue(self, fatigue_score, active=True):
    #     # Clamp first, then smooth (avoids bias)
    #     fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
    #     # Adaptive smoothing: slower (stable) when active, faster drop when inactive
    #     alpha = 0.30 if active else 0.70
    #     smoothed = alpha * fatigue_score + (1 - alpha) * self.prev_fatigue_score
    #     self.prev_fatigue_score = smoothed
    #     level = int(round(smoothed * 6 + 1))
    #     return max(1, min(level, 7)), smoothed
    def _scale_fatigue(self, fatigue_score, active=True):
        fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
        # Emphasize onset zone (0.4–0.7) by applying a mild curve
        fatigue_adj = fatigue_score ** 1.5
        alpha = 0.30 if active else 0.70
        smoothed = alpha * fatigue_adj + (1 - alpha) * self.prev_fatigue_score
        self.prev_fatigue_score = smoothed
        level = np.digitize(
            smoothed,
            bins=[0.15, 0.35, 0.55, 0.7, 0.85, 0.95],  # 7 bins for 1-7 levels
        ) + 1
        return level, smoothed


    # -------- rule text ----------
    # def rule_based_message(self, fatigue_level, fatigue_score):
    #     if fatigue_level <= 2:
    #         return "Continue your set and keep a steady pace."
    #     elif fatigue_level <= 4:
    #         return "Maintain your pace and focus on form and breathing."
    #     elif fatigue_level <= 6:
    #         rest = self.get_rest_duration(fatigue_score)
    #         return f"Stop this set and rest for about {rest} seconds."
    #     else:
    #         rest = self.get_rest_duration(fatigue_score)
    #         return f"Stop completely. You’ve reached exhaustion. Rest for at least {rest} seconds."
    def rule_based_message(self, fatigue_level, fatigue_score):
        """
        Updated thresholds based on observation:
        - 0.4 ≈ onset of fatigue / failure
        - 0.5+ ≈ must stop
        """
        if fatigue_score < 0.20:
            self.state = CoachState.ACTIVE
            return "You're fresh — keep your pace smooth and steady."

        elif fatigue_score < 0.30:
            self.state = CoachState.ACTIVE
            return "Good rhythm — you're working hard but still in control."

        elif fatigue_score < 0.40:
            self.state = CoachState.ACTIVE
            return "You're approaching fatigue — focus on breathing and get ready to stop soon."

        elif fatigue_score < 0.6:
            self.state = CoachState.RESTING
            self.get_rest_duration(fatigue_score)
            return f"That’s real fatigue setting in. Stop the set and rest for about {self.rest_duration} seconds."

        else:
            self.fatigue_state = CoachState.RESTING
            self.get_rest_duration(fatigue_score)
            return f"You’ve reached full fatigue. Stop completely and rest at least {self.rest_duration} seconds before continuing."

    # def get_rest_duration(self, fatigue_score):
    #     if fatigue_score < 0.4:   return 0
    #     if fatigue_score < 0.7:   return 60
    #     return 60 + int((fatigue_score - 0.7) * 300)
    def get_rest_duration(self, fatigue_score):
        if fatigue_score < 0.35:
            self.rest_duration = 0
        # Smoothly scale from 10 to 110 s between 0.35–0.9
        self.rest_duration = int(10 + (fatigue_score - 0.35) / 0.55 * 90)


    def llm_rephrase(self, rule_text, fatigue_level):
        # If you later re-enable your pipeline, return an LLM rewrite here.
        return ""  # keep empty to use fallback below

    # -------- main API (now with optional activity_rms) ----------
    def give_command(self, fatigue_score, activity_rms=None):
        # Determine activity from EMG RMS (pre-processed volts)
        active = self._is_active(activity_rms)

        # --- NEW: Use baseline RMS to scale fatigue ---
        if self.baseline_rms:
            # Ratio of current muscle activity to baseline
            rms_ratio = activity_rms / max(self.baseline_rms, 1e-6)
            # If user is close to baseline (resting), reduce effective fatigue
            fatigue_score *= np.clip(rms_ratio, 0.2, 1.0)
            # e.g., if rms is half baseline, fatigue is halved
        else:
            rms_ratio = 1.0

        # ---- After computing active ----
        if activity_rms is not None and activity_rms < self.baseline_rms * self.baseline_thesh:
            active = False

        # If inactive: decay toward recovery regardless of raw model output
        if not active and activity_rms < self.baseline_rms * 1.1:
            fatigue_score = 0.0  # force recovery if resting completely
            
        if not active:
            decayed = max(0.0, self.prev_fatigue_score - self.inactive_decay)
            fatigue_level, smoothed = self._scale_fatigue(decayed, active=False)
        else:
            fatigue_level, smoothed = self._scale_fatigue(fatigue_score, active=True)

        print(f"Fatigue Score for Command: {smoothed:.3f} (active={active}, rms={activity_rms})")

        # If resting, check recovery progress
        # if self.state == CoachState.RESTING:
        #     elapsed = time.time() - self.rest_start
        #     recovered = smoothed < 0.30 or not active  # recover faster when inactive
        #     if recovered or elapsed >= self.rest_duration:
        #         self.state = CoachState.ACTIVE
        #         msg = "Rest complete. Resume your next set." if elapsed >= self.rest_duration \
        #               else "You’ve recovered enough — start your next set when ready."
        #         self._speak(msg)
        #         self._log(fatigue_level, smoothed, msg, "Recovered/rest-complete")
        #         return msg
        #     remaining = int(self.rest_duration - elapsed)
        #     if remaining > 0 and remaining % 30 == 0:
        #         self._speak(f"{remaining} seconds left in your rest.")
        #     return

        if self.state == CoachState.RESTING:
            elapsed = time.time() - self.rest_start
            # actively decay fatigue each loop during rest
            recovery_rate = 0.15  # adjust: 0.1 = slow, 0.3 = faster recovery
            self.prev_fatigue_score = max(0.0, self.prev_fatigue_score - recovery_rate * (elapsed / self.rest_duration))
            smoothed = self.prev_fatigue_score

            recovered = smoothed < 0.30 or not active
            if recovered or elapsed >= self.rest_duration:
                self.state = CoachState.ACTIVE
                msg = (
                    "Rest complete. Resume your next set."
                    if elapsed >= self.rest_duration
                    else "You’ve recovered enough — start your next set when ready."
                )
                self._speak(msg)
                self._log(1, smoothed, msg, "Recovered/rest-complete")
                return msg

            remaining = int(self.rest_duration - elapsed)
            if remaining > 0 and remaining % 30 == 0:
                self._speak(f"{remaining} seconds left in your rest.")
            return
        
        # Decide what to say
        rule_text = self.rule_based_message(fatigue_level, smoothed)
        llm_text  = self.llm_rephrase(rule_text, fatigue_level) or rule_text  # fallback
        self._speak(llm_text)
        self._log(fatigue_level, smoothed, llm_text, rule_text)

        # # Enter rest if high fatigue
        # if fatigue_level >= 5:
        #     self.state = CoachState.RESTING
        #     self.rest_duration = self.get_rest_duration(smoothed)
        #     self.rest_start = time.time()
        # elif fatigue_level == 7:
        #     self.state = CoachState.STOPPED


        # --- Baseline recalibration after rest ---
        if self.state == CoachState.ACTIVE and not active:
            # When user just finished resting, baseline may have drifted down
            self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * (activity_rms or self.baseline_rms)

        return llm_text

    # -------- Audio & Logging --------
    def _speak(self, text):
        try:
            tts = gTTS(text=text, lang=self.language)
            filename = "coach_command.mp3"
            tts.save(filename)
            os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
        except Exception as e:
            print(f"Audio playback error: {e}")
            print("AI Coach:", text)

    def _log(self, fatigue_level, fatigue_score, llm_text, rule_text):
        ts = int(time.time() * 1000)
        with open(self.logging_file_path, "a") as f:
            print(f"Fatigue Score: {fatigue_score:.3f}")
            f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text}\n")




# from enum import Enum, auto
# import torch, os, time, numpy as np
# from transformers import pipeline
# from gtts import gTTS

# class CoachState(Enum):
#     ACTIVE = auto()
#     RESTING = auto()
#     STOPPED = auto()

# class Coach:
#     def __init__(self, participant_id):
#         self.language = "en"
#         self.participant_id = participant_id
#         self.state = CoachState.ACTIVE
#         self.rest_start = None
#         self.rest_duration = 0
#         self.prev_fatigue_score = 0.0

#         # --- LLM Setup (Gemma-2-2B-IT) ---
#         # self.pipe = pipeline(
#         #     "text-generation",
#         #     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         #     device="cpu"
#         # )
        
#         self.pipe = pipeline(
#             "text-generation",
#             model="microsoft/phi-2",
#             device="cpu",
#             torch_dtype=torch.float32
#         )
#         # --- Activity tracking ---
#         self.baseline_rms = None
#         self.activity_mult = 3.0
#         self.inactive_decay = 0.2

#         # --- Logging ---
#         self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
#         os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
#         if not os.path.exists(self.logging_file_path):
#             with open(self.logging_file_path, "w") as f:
#                 f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm\n")

#     # -------- Baseline & activity --------
#     def _is_active(self, rms):
#         if self.baseline_rms is None:
#             self.baseline_rms = rms
#             print(f"📊 Baseline RMS set: {self.baseline_rms:.3f} (8s window)")
#             return True
#         threshold = self.baseline_rms * self.activity_mult
#         active = rms > threshold
#         if not active:
#             self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * rms
#         return active

#     # -------- Fatigue scaling --------
#     def _scale_fatigue(self, fatigue_score, active=True):
#         fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
#         alpha = 0.3 if active else 0.7
#         smoothed = alpha * fatigue_score + (1 - alpha) * self.prev_fatigue_score
#         self.prev_fatigue_score = smoothed
#         level = int(round(smoothed * 6 + 1))
#         return max(1, min(level, 7)), smoothed

#     # -------- Rule-based fallback --------
#     def rule_based_message(self, fatigue_level, fatigue_score):
#         if fatigue_level <= 2:
#             return "Continue your set and keep a steady pace."
#         elif fatigue_level <= 4:
#             return "Maintain your pace and focus on form and breathing."
#         elif fatigue_level <= 6:
#             rest = self.get_rest_duration(fatigue_score)
#             return f"Stop this set and rest for about {rest} seconds."
#         else:
#             rest = self.get_rest_duration(fatigue_score)
#             return f"Stop completely. You’ve reached exhaustion. Rest for at least {rest} seconds."

#     def get_rest_duration(self, fatigue_score):
#         if fatigue_score < 0.4:
#             return 0
#         if fatigue_score < 0.7:
#             return 60
#         return 60 + int((fatigue_score - 0.7) * 300)

#     # -------- LLM rephrasing --------
#     def llm_rephrase(self, rule_text, fatigue_level):
#         print("Current rephrasing...")
#         prompt = (
#             "You are a motivational fitness coach guiding a person performing bicep curls.\n"
#             "Rephrase the following instruction in a natural, encouraging tone. Keep the same meaning.\n"
#             "Do not add new actions or durations. Keep it concise and motivating.\n\n"
#             f"Fatigue level: {fatigue_level}\n"
#             f"Instruction: {rule_text}\n"
#             "Coach response:"
#         )
#         try:
#             outputs = self.pipe(prompt, max_new_tokens=20, do_sample=True, top_p=0.9)
#             text = outputs[0]["generated_text"].split("Coach response:")[-1].strip()
#             return text
#         except Exception as e:
#             print(f"⚠️ LLM Error: {e}")
#             return rule_text

#     # -------- Main loop --------
#     def give_command(self, fatigue_score, activity_rms=None):
#         active = self._is_active(activity_rms)
#         if not active:
#             decayed = max(0.0, self.prev_fatigue_score - self.inactive_decay)
#             fatigue_level, smoothed = self._scale_fatigue(decayed, active=False)
#         else:
#             fatigue_level, smoothed = self._scale_fatigue(fatigue_score, active=True)

#         print(f"Fatigue Score for Command: {smoothed:.3f} (active={active}, rms={activity_rms})")

#         # Check rest progress
#         if self.state == CoachState.RESTING:
#             elapsed = time.time() - self.rest_start
#             recovered = smoothed < 0.3 or not active
#             if recovered or elapsed >= self.rest_duration:
#                 self.state = CoachState.ACTIVE
#                 msg = "Rest complete. Resume your next set." if elapsed >= self.rest_duration \
#                       else "You’ve recovered enough — start your next set when ready."
#                 self._speak(msg)
#                 self._log(fatigue_level, smoothed, msg, "Recovered/rest-complete")
#                 return msg
#             remaining = int(self.rest_duration - elapsed)
#             if remaining > 0 and remaining % 30 == 0:
#                 self._speak(f"{remaining} seconds left in your rest.")
#             return

#         # Generate message
#         rule_text = self.rule_based_message(fatigue_level, smoothed)
#         llm_text = self.llm_rephrase(rule_text, fatigue_level)
#         self._speak(llm_text)
#         self._log(fatigue_level, smoothed, llm_text, rule_text)

#         if fatigue_level >= 5:
#             self.state = CoachState.RESTING
#             self.rest_duration = self.get_rest_duration(smoothed)
#             self.rest_start = time.time()
#         elif fatigue_level == 7:
#             self.state = CoachState.STOPPED

#         return llm_text

#     # -------- Audio & Logging --------
#     def _speak(self, text):
#         print("Trying to speak")
#         try:
#             tts = gTTS(text=text, lang=self.language)
#             filename = "coach_command.mp3"
#             tts.save(filename)
#             os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
#         except Exception as e:
#             print(f"Audio playback error: {e}")
#             print("AI Coach:", text)

#     def _log(self, fatigue_level, fatigue_score, llm_text, rule_text):
#         ts = int(time.time() * 1000)
#         with open(self.logging_file_path, "a") as f:
#             f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text}\n")
#         print(f"📝 Logged: fatigue={fatigue_level}, state={self.state.name}")
