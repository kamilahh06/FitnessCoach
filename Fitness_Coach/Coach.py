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


# from enum import Enum, auto
# import torch
# from transformers import pipeline
# from gtts import gTTS
# import os, time
# import pandas as pd
# import numpy as np

# class CoachState(Enum):
#     ACTIVE = auto()
#     RESTING = auto()
#     STOPPED = auto()

# class Coach:
#     def __init__(self, participant_id):
#         # self.pipe = pipeline(... gated HF model ...)
#         self.language = "en"
#         self.participant_id = participant_id
#         self.state = CoachState.ACTIVE
#         # self.pipe = pipeline(
#         #     "text-generation",
#         #     model="google/gemma-2-2b-it",
#         #     model_kwargs={"torch_dtype": torch.bfloat16},
#         #     device="mps",  # or "mps" for Mac
#         # )

#         # recovery / rest tracking
#         self.rest_start = None
#         self.rest_duration = 0

#         self.baseline_rms = None
#         self.baseline_thesh = 2 # originally 1.2
#         self.rms_history = []
#         self.activity_threshold_std = 3.0
#         self.noise_floor = 0.05  # default

#         """
#         0.03 for very clean, low-amplitude setups (tight electrodes, short leads)
#         0.05 for most consumer EMG sensors
#         0.08–0.1 for noisy analog environments or dry electrodes
#         """


#         # fatigue memory
#         self.prev_fatigue_score = 0.0

#         # activity gating (RMS-based)
#         self.rms_baseline = None
#         self.activity_mult = 3          # active if rms > baseline * this, originally 1.5
#         self.min_active_rms = 2e-5        # absolute floor in volts (tune for your hardware)
#         self.inactive_decay = 0.2        # per-window drop when inactive (0..1), originally 0.1

#         # logging
#         self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
#         os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
#         if not os.path.exists(self.logging_file_path):
#             with open(self.logging_file_path, "w") as f:
#                 f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm\n")

#     # -------- activity detection ----------
#     def _update_activity_baseline(self, rms, active):
#         if self.rms_baseline is None:
#             self.rms_baseline = rms
#             return
#         # If inactive, let baseline track down slowly; if active, don’t chase it up too fast
#         beta = 0.10 if not active else 0.02
#         self.rms_baseline = (1 - beta) * self.rms_baseline + beta * rms

#     # def _is_active(self, rms):
#         # if rms is None:
#         #     return True  # backward compatible: assume active if no signal provided
#         # if self.rms_baseline is None:
#         #     # Warm-up: treat as inactive unless clearly above a minimum floor
#         #     active = rms > max(self.min_active_rms, 1e-5)
#         #     self._update_activity_baseline(rms, active)
#         #     return active
#         # thresh = max(self.min_active_rms, self.rms_baseline * self.activity_mult)
#         # active = rms > thresh
#         # self._update_activity_baseline(rms, active)
#         # return active

#     # def _is_active(self, rms):
#     #     # Establish baseline on the first complete window only
#     #     if self.baseline_rms is None:
#     #         self.baseline_rms = rms
#     #         self.baseline_std = 0.0  # not really needed, but for compatibility
#     #         print(f"📊 Baseline RMS set: {self.baseline_rms:.3f} (8s window)")
#     #         return True  # Treat initial window as active just to start safely

#     #     # Once baseline is set, compare current rms to threshold
#     #     threshold = self.baseline_rms * self.activity_mult
#     #     active = rms > threshold

#     #     # Update baseline slowly during inactivity (optional)
#     #     if not active:
#     #         self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * rms  # gradual drift correction

#     #     return active

#     # def _is_active(self, rms):
#     #     """Return True if user is currently active based on RMS amplitude,
#     #     using hysteresis to avoid flicker."""
#     #     if self.baseline_rms is None:
#     #         # First window — set baseline and assume active to bootstrap
#     #         self.baseline_rms = rms
#     #         print(f"📊 Baseline RMS set: {self.baseline_rms:.3f} (8s window)")
#     #         self.active = True
#     #         return True

#     #     # Create persistent state flag
#     #     if not hasattr(self, "active"):
#     #         self.active = True

#     #     # NEW THESHOLDS FOR NEW PROBLEM!!!!!
#     #     upper_thresh = 0.2   # must exceed 20% of max to count as active
#     #     lower_thresh = 0.08  # must drop below 8% to count as inactive

#     #     # Two thresholds for hysteresis
#     #     # upper_thresh = self.baseline_rms * 2.0   # must exceed to count as active again
#     #     # lower_thresh = self.baseline_rms * 1.2   # must fall below to count as inactive

#     #     # Decide state based on thresholds
#     #     if rms < lower_thresh:
#     #         self.active = False
#     #     elif rms > upper_thresh:
#     #         self.active = True

#     #     # Update baseline slowly when inactive to track drift
#     #     if not self.active:
#     #         self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * rms

#     #     return self.active

#     # nEW IS ACTIVE TO FIX FOR AAMI
#     def _is_active(self, rms):
#         if self.baseline_rms is None:
#             self.baseline_rms = rms
#             print(f"📊 Baseline RMS set: {self.baseline_rms:.3f} (8s window)")
#             self.active = True
#             return True

#         # Ensure persistent flag
#         if not hasattr(self, "active"):
#             self.active = True

#         # Two thresholds for hysteresis
#         # upper_thresh = max(0.05, self.baseline_rms * 1.8)
#         # lower_thresh = max(0.02, self.baseline_rms * 1.2)
#         upper_thresh = max(self.noise_floor, self.baseline_rms * 1.8)
#         lower_thresh = max(self.noise_floor * 0.6, self.baseline_rms * 1.2)


#         # --- ADAPTIVE RESET LOGIC ---
#         # If RMS has stayed low for a while (below 0.05), baseline slowly drifts down
#         if rms < 0.05:
#             self.baseline_rms = 0.98 * self.baseline_rms + 0.02 * rms

#         # Normal hysteresis switching
#         if rms < lower_thresh:
#             self.active = False
#         elif rms > upper_thresh:
#             self.active = True

#         self.baseline_rms = np.clip(self.baseline_rms, 0.02, 0.3)
#         return self.active




#     # -------- scaling & smoothing ----------
#     # def _scale_fatigue(self, fatigue_score, active=True):
#     #     # Clamp first, then smooth (avoids bias)
#     #     fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
#     #     # Adaptive smoothing: slower (stable) when active, faster drop when inactive
#     #     alpha = 0.30 if active else 0.70
#     #     smoothed = alpha * fatigue_score + (1 - alpha) * self.prev_fatigue_score
#     #     self.prev_fatigue_score = smoothed
#     #     level = int(round(smoothed * 6 + 1))
#     #     return max(1, min(level, 7)), smoothed
#     def _scale_fatigue(self, fatigue_score, active=True):
#         """Scale and bucket a raw fatigue score into a discrete fatigue level while applying smoothing.
#         Parameters
#         ----------
#         fatigue_score : float
#             Raw fatigue value expected in the range [0.0, 1.0]. Values outside this range are clipped.
#         active : bool, optional
#             If True, use a lower smoothing factor (alpha=0.30) to make the output respond more quickly
#             to changes. If False, use a higher smoothing factor (alpha=0.70) to make the output change
#             more slowly. Default: True.
#         Returns
#         -------
#         tuple[int, float]
#             A tuple (level, smoothed) where:
#             - level is an integer fatigue level in the range 1..7 (inclusive) obtained by binning the
#               smoothed score into predefined thresholds.
#             - smoothed is the floating-point internal fatigue estimate after clipping, non-linear
#               emphasis, and exponential-like smoothing. This value is also stored to self.prev_fatigue_score
#               as a side effect.
#         """
#         fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
#         # Emphasize onset zone (0.4–0.7) by applying a mild curve
#         fatigue_adj = fatigue_score ** 1.5
#         alpha = 0.30 if active else 0.70
#         smoothed = alpha * fatigue_adj + (1 - alpha) * self.prev_fatigue_score
#         self.prev_fatigue_score = smoothed
#         level = np.digitize(
#             smoothed,
#             bins=[0.15, 0.35, 0.55, 0.7, 0.85, 0.95],  # 7 bins for 1-7 levels
#         ) + 1
#         return level, smoothed


#     # -------- rule text ----------
#     # def rule_based_message(self, fatigue_level, fatigue_score):
#     #     if fatigue_level <= 2:
#     #         return "Continue your set and keep a steady pace."
#     #     elif fatigue_level <= 4:
#     #         return "Maintain your pace and focus on form and breathing."
#     #     elif fatigue_level <= 6:
#     #         rest = self.get_rest_duration(fatigue_score)
#     #         return f"Stop this set and rest for about {rest} seconds."
#     #     else:
#     #         rest = self.get_rest_duration(fatigue_score)
#     #         return f"Stop completely. You’ve reached exhaustion. Rest for at least {rest} seconds."
#     def rule_based_message(self, fatigue_level, fatigue_score):
#         """
#         Updated thresholds based on observation:
#         - 0.4 ≈ onset of fatigue / failure
#         - 0.5+ ≈ must stop
        
#         Args
#         fatigue_level: fatigue level
#         fatigue_score: smoothed fatigue level
#         """
#         if fatigue_score < 0.20:
#             self.state = CoachState.ACTIVE
#             return "You're fresh — keep your pace smooth and steady."

#         elif fatigue_score < 0.30:
#             self.state = CoachState.ACTIVE
#             return "Good rhythm — you're working hard but still in control."

#         elif fatigue_score < 0.40:
#             self.state = CoachState.ACTIVE
#             return "You're approaching fatigue — focus on breathing and get ready to stop soon."

#         elif fatigue_score < 0.6:
#             self.state = CoachState.RESTING
#             self.get_rest_duration(fatigue_score)
#             return f"That’s real fatigue setting in. Stop the set and rest for about {self.rest_duration} seconds."

#         else:
#             self.fatigue_state = CoachState.RESTING
#             self.get_rest_duration(fatigue_score)
#             return f"You’ve reached full fatigue. Stop completely and rest at least {self.rest_duration} seconds before continuing."

#     # def get_rest_duration(self, fatigue_score):
#     #     if fatigue_score < 0.4:   return 0
#     #     if fatigue_score < 0.7:   return 60
#     #     return 60 + int((fatigue_score - 0.7) * 300)
#     def get_rest_duration(self, fatigue_score):
#         if fatigue_score < 0.35:
#             self.rest_duration = 0
#         # Smoothly scale from 10 to 110 s between 0.35–0.9
#         self.rest_duration = int(10 + (fatigue_score - 0.35) / 0.55 * 90)


#     def llm_rephrase(self, rule_text, fatigue_level):
#         # If you later re-enable your pipeline, return an LLM rewrite here.
#         return ""  # keep empty to use fallback below

#     # -------- main API (now with optional activity_rms) ----------
#     def give_command(self, fatigue_score, activity_rms=None):
#         # Determine activity from EMG RMS (pre-processed volts)

#         # JUST NOW ADDED TO RESOLVE PROBLEM
#         V_MAX = 2.5
#         activity_rms = np.clip(activity_rms / V_MAX, 0.0, 1.0)

#         active = self._is_active(activity_rms)

#         # --- NEW: Use baseline RMS to scale fatigue ---
#         if self.baseline_rms:
#             # Ratio of current muscle activity to baseline
#             rms_ratio = activity_rms / max(self.baseline_rms, 1e-6)
#             # If user is close to baseline (resting), reduce effective fatigue
#             fatigue_score *= np.clip(rms_ratio, 0.2, 1.0)
#             # e.g., if rms is half baseline, fatigue is halved
#         else:
#             rms_ratio = 1.0

#         # ---- After computing active ----
#         if activity_rms is not None and activity_rms < self.baseline_rms * self.baseline_thesh:
#             active = False

#         # If inactive: decay toward recovery regardless of raw model output
#         if not active and activity_rms < self.baseline_rms * 1.1:
#             fatigue_score = 0.0  # force recovery if resting completely
            
#         if not active:
#             decayed = max(0.0, self.prev_fatigue_score - self.inactive_decay)
#             fatigue_level, smoothed = self._scale_fatigue(decayed, active=False)
#         else:
#             fatigue_level, smoothed = self._scale_fatigue(fatigue_score, active=True)

#         print(f"Fatigue Score for Command: {smoothed:.3f} (active={active}, rms={activity_rms})")

#         # If resting, check recovery progress
#         # if self.state == CoachState.RESTING:
#         #     elapsed = time.time() - self.rest_start
#         #     recovered = smoothed < 0.30 or not active  # recover faster when inactive
#         #     if recovered or elapsed >= self.rest_duration:
#         #         self.state = CoachState.ACTIVE
#         #         msg = "Rest complete. Resume your next set." if elapsed >= self.rest_duration \
#         #               else "You’ve recovered enough — start your next set when ready."
#         #         self._speak(msg)
#         #         self._log(fatigue_level, smoothed, msg, "Recovered/rest-complete")
#         #         return msg
#         #     remaining = int(self.rest_duration - elapsed)
#         #     if remaining > 0 and remaining % 30 == 0:
#         #         self._speak(f"{remaining} seconds left in your rest.")
#         #     return

#         if self.state == CoachState.RESTING:
#             elapsed = time.time() - self.rest_start
#             # actively decay fatigue each loop during rest
#             recovery_rate = 0.15  # adjust: 0.1 = slow, 0.3 = faster recovery
#             self.prev_fatigue_score = max(0.0, self.prev_fatigue_score - recovery_rate * (elapsed / self.rest_duration))
#             smoothed = self.prev_fatigue_score

#             recovered = smoothed < 0.30 or not active
#             # if recovered or elapsed >= self.rest_duration:
#             #     self.state = CoachState.ACTIVE
#             #     msg = (
#             #         "Rest complete. Resume your next set."
#             #         if elapsed >= self.rest_duration
#             #         else "You’ve recovered enough — start your next set when ready."
#             #     )
#             #     self._speak(msg)
#             #     self._log(1, smoothed, msg, "Recovered/rest-complete")
#             #     return msg
            
#             # ADDED TO SOLVE PROBLEMMMMM    
#             if recovered or elapsed >= self.rest_duration:
#                 self.state = CoachState.ACTIVE
#                 self.prev_fatigue_score = 0.0  # <<< reset fatigue
#                 msg = (
#                     "Rest complete. Resume your next set."
#                     if elapsed >= self.rest_duration
#                     else "You’ve recovered enough — start your next set when ready."
#                 )
#                 self._speak(msg)
#                 self._log(1, smoothed, msg, "Recovered/rest-complete")
#                 return msg


#             remaining = int(self.rest_duration - elapsed)
#             if remaining > 0 and remaining % 30 == 0:
#                 self._speak(f"{remaining} seconds left in your rest.")
#             return
        
#         # Decide what to say
#         rule_text = self.rule_based_message(fatigue_level, smoothed)
#         llm_text  = self.llm_rephrase(rule_text, fatigue_level) or rule_text  # fallback
#         self._speak(llm_text)
#         self._log(fatigue_level, smoothed, llm_text, rule_text)

#         # # Enter rest if high fatigue
#         # if fatigue_level >= 5:
#         #     self.state = CoachState.RESTING
#         #     self.rest_duration = self.get_rest_duration(smoothed)
#         #     self.rest_start = time.time()
#         # elif fatigue_level == 7:
#         #     self.state = CoachState.STOPPED


#         # --- Baseline recalibration after rest ---
#         if self.state == CoachState.ACTIVE and not active:
#             # When user just finished resting, baseline may have drifted down
#             self.baseline_rms = 0.9 * self.baseline_rms + 0.1 * (activity_rms or self.baseline_rms)

#         return llm_text

#     # -------- Audio & Logging --------
#     def _speak(self, text):
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
#             print(f"Fatigue Score: {fatigue_score:.3f}")
#             f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text}\n")


# from enum import Enum, auto
# import torch
# from transformers import pipeline
# from gtts import gTTS
# import os, time
# import pandas as pd
# import numpy as np

# class CoachState(Enum):
#     CALIBRATING = auto()
#     ACTIVE = auto()
#     RESTING = auto()
#     STOPPED = auto()

# class Coach:
#     def __init__(self, participant_id, require_calibration=True):
#         self.language = "en"
#         self.participant_id = participant_id
#         self.state = CoachState.CALIBRATING if require_calibration else CoachState.ACTIVE
        
#         # Calibration parameters - just collect a few windows like before
#         self.require_calibration = require_calibration
#         self.calibration_windows = []
#         self.calibration_target = 3  # 3 windows = ~24 seconds at 8s/window
        
#         self.is_calibrated = not require_calibration
        
#         # Adaptive normalization (set during calibration)
#         self.rms_percentiles = None  # will store (p10, p50, p90, p99)
#         self.baseline_rms = None
#         self.max_observed_rms = 0.0
        
#         # Activity detection thresholds (relative to calibrated baseline)
#         self.activity_multiplier = 1.5  # reduced from 1.8 for more sensitivity
#         self.rest_multiplier = 1.1      # slightly above baseline = resting
        
#         # Fatigue smoothing
#         self.prev_fatigue_score = 0.0
#         self.active = True
        
#         # Rest tracking
#         self.rest_start = None
#         self.rest_duration = 0
        
#         # Fatigue scaling - now subject-adaptive
#         self.fatigue_sensitivity = 1.0  # will be tuned during calibration
        
#         # Logging
#         self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
#         os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
#         if not os.path.exists(self.logging_file_path):
#             with open(self.logging_file_path, "w") as f:
#                 f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm,rms_raw,rms_normalized\n")

#     def add_calibration_sample(self, rms_volts):
#         """Collect baseline RMS samples during calibration phase"""
#         if self.state != CoachState.CALIBRATING:
#             return False
            
#         self.calibration_windows.append(rms_volts)
#         progress = len(self.calibration_windows) / self.calibration_target
        
#         if len(self.calibration_windows) % 5 == 0:
#             print(f"📊 Calibration progress: {int(progress * 100)}%")
        
#         if len(self.calibration_windows) >= self.calibration_target:
#             self._finalize_calibration()
#             return True
#         return False

#     def _finalize_calibration(self):
#         """Compute subject-specific normalization parameters"""
#         rms_samples = np.array(self.calibration_windows)
        
#         # Compute percentiles for robust normalization
#         self.rms_percentiles = {
#             'p10': np.percentile(rms_samples, 10),
#             'p50': np.percentile(rms_samples, 50),  # median baseline
#             'p90': np.percentile(rms_samples, 90),
#             'p99': np.percentile(rms_samples, 99)
#         }
        
#         self.baseline_rms = self.rms_percentiles['p50']
#         self.max_observed_rms = self.rms_percentiles['p99']
        
#         # Set dynamic thresholds based on signal quality
#         signal_range = self.rms_percentiles['p90'] - self.rms_percentiles['p10']
#         if signal_range < 0.01:  # very clean signal
#             self.activity_multiplier = 1.3
#         elif signal_range < 0.05:  # normal signal
#             self.activity_multiplier = 1.5
#         else:  # noisy signal
#             self.activity_multiplier = 1.8
        
#         self.state = CoachState.ACTIVE
#         self.is_calibrated = True
        
#         print("\n✅ Calibration complete!")
#         print(f"   Windows collected: {len(self.calibration_windows)}")
#         print(f"   Baseline RMS: {self.baseline_rms:.4f}V")
#         print(f"   Max RMS (p99): {self.max_observed_rms:.4f}V")
#         print(f"   Activity threshold: {self.baseline_rms * self.activity_multiplier:.4f}V")
#         print(f"   Signal quality: {'Clean' if signal_range < 0.01 else 'Normal' if signal_range < 0.05 else 'Noisy'}\n")
        
#         self._speak("Calibration complete. Begin your workout.")

#     def _normalize_rms(self, rms_volts):
#         """Normalize RMS to 0-1 range using subject-specific calibration"""
#         if not self.is_calibrated:
#             # Before calibration, just clip to reasonable range
#             return np.clip(rms_volts / 0.5, 0.0, 1.0)
        
#         # Update max if we see higher values during workout
#         self.max_observed_rms = max(self.max_observed_rms, rms_volts)
        
#         # Normalize using percentile-based scaling
#         # Maps p50 -> 0.3, p99 -> 1.0, with smooth interpolation
#         if rms_volts <= self.baseline_rms:
#             # Below baseline: scale 0-0.3
#             normalized = 0.3 * (rms_volts / max(self.baseline_rms, 1e-6))
#         else:
#             # Above baseline: scale 0.3-1.0
#             range_above = self.max_observed_rms - self.baseline_rms
#             normalized = 0.3 + 0.7 * ((rms_volts - self.baseline_rms) / max(range_above, 1e-6))
        
#         return np.clip(normalized, 0.0, 1.0)

#     def _is_active(self, rms_volts):
#         """Determine if subject is actively exercising"""
#         if not self.is_calibrated:
#             return True  # assume active during calibration
        
#         if not hasattr(self, 'active'):
#             self.active = True
        
#         # Hysteresis thresholds
#         upper_thresh = self.baseline_rms * self.activity_multiplier
#         lower_thresh = self.baseline_rms * self.rest_multiplier
        
#         # State machine with hysteresis
#         if rms_volts < lower_thresh:
#             self.active = False
#         elif rms_volts > upper_thresh:
#             self.active = True
#         # else: maintain previous state (hysteresis)
        
#         return self.active

#     def _scale_fatigue(self, fatigue_score, active=True):
#         """Scale and smooth fatigue score with subject-adaptive sensitivity"""
#         fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
        
#         # Apply subject-specific sensitivity (reduce for sensitive subjects)
#         fatigue_adjusted = fatigue_score ** (1.0 / self.fatigue_sensitivity)
        
#         # Adaptive smoothing
#         alpha = 0.35 if active else 0.65
#         smoothed = alpha * fatigue_adjusted + (1 - alpha) * self.prev_fatigue_score
#         self.prev_fatigue_score = smoothed
        
#         # More granular binning
#         level = np.digitize(
#             smoothed,
#             bins=[0.12, 0.25, 0.38, 0.50, 0.65, 0.80],  # 7 bins for levels 1-7
#         ) + 1
        
#         return level, smoothed

#     def rule_based_message(self, fatigue_level, fatigue_score):
#         """Generate coaching message based on fatigue state"""
#         if fatigue_score < 0.15:
#             self.state = CoachState.ACTIVE
#             return "You're fresh — maintain your steady pace."
        
#         elif fatigue_score < 0.30:
#             self.state = CoachState.ACTIVE
#             return "Good work — you're in your rhythm. Keep it controlled."
        
#         elif fatigue_score < 0.45:
#             self.state = CoachState.ACTIVE
#             return "Fatigue building — stay focused on form. Prepare to stop soon."
        
#         elif fatigue_score < 0.65:
#             if self.state != CoachState.RESTING:
#                 self.state = CoachState.RESTING
#                 self.rest_duration = self._get_rest_duration(fatigue_score)
#                 self.rest_start = time.time()
#             return f"Moderate fatigue detected. Stop and rest for {self.rest_duration} seconds."
        
#         else:
#             if self.state != CoachState.RESTING:
#                 self.state = CoachState.RESTING
#                 self.rest_duration = self._get_rest_duration(fatigue_score)
#                 self.rest_start = time.time()
#             return f"High fatigue. Stop immediately and rest at least {self.rest_duration} seconds."

#     def _get_rest_duration(self, fatigue_score):
#         """Calculate rest duration based on fatigue level"""
#         if fatigue_score < 0.40:
#             return 0
#         # Scale from 30s (at 0.4) to 120s (at 1.0)
#         return int(30 + (fatigue_score - 0.40) / 0.60 * 90)

#     def give_command(self, fatigue_score, activity_rms=None):
#         """Main coaching logic"""
        
#         # Handle calibration phase
#         if self.state == CoachState.CALIBRATING:
#             if activity_rms is not None:
#                 is_complete = self.add_calibration_sample(activity_rms)
#                 if not is_complete:
#                     return "Calibrating... Stay relaxed."
#             return
        
#         # Validate input
#         if activity_rms is None:
#             print("⚠️ Warning: No RMS provided, assuming active state")
#             active = True
#             rms_normalized = 0.5
#         else:
#             # Detect activity state
#             active = self._is_active(activity_rms)
            
#             # Normalize RMS for logging
#             rms_normalized = self._normalize_rms(activity_rms)
        
#         # Handle resting state
#         if self.state == CoachState.RESTING:
#             elapsed = time.time() - self.rest_start
            
#             # Active recovery: decay fatigue during rest
#             recovery_rate = 0.12 if active else 0.20  # recover faster when truly resting
#             time_fraction = elapsed / max(self.rest_duration, 1)
#             self.prev_fatigue_score = max(0.0, self.prev_fatigue_score - recovery_rate * time_fraction)
            
#             # Check if rest is complete
#             recovered = self.prev_fatigue_score < 0.25 or not active
#             if recovered or elapsed >= self.rest_duration:
#                 self.state = CoachState.ACTIVE
#                 self.prev_fatigue_score = 0.0  # reset for next set
#                 msg = "Rest complete. Resume your next set when ready."
#                 self._speak(msg)
#                 self._log(1, self.prev_fatigue_score, msg, "Recovered", activity_rms, rms_normalized)
#                 return msg
            
#             # Periodic rest reminders
#             remaining = int(self.rest_duration - elapsed)
#             if remaining > 0 and remaining % 30 == 0:
#                 self._speak(f"{remaining} seconds remaining.")
#             return
        
#         # Scale fatigue only when active
#         if not active and activity_rms < self.baseline_rms * 1.15:
#             # Force recovery when completely inactive
#             decayed = max(0.0, self.prev_fatigue_score - 0.15)
#             fatigue_level, smoothed = self._scale_fatigue(decayed, active=False)
#         else:
#             fatigue_level, smoothed = self._scale_fatigue(fatigue_score, active=True)
        
#         # Generate and deliver coaching message
#         rule_text = self.rule_based_message(fatigue_level, smoothed)
#         self._speak(rule_text)
#         self._log(fatigue_level, smoothed, rule_text, rule_text, activity_rms, rms_normalized)
        
#         return rule_text

#     def _speak(self, text):
#         """Convert text to speech"""
#         try:
#             tts = gTTS(text=text, lang=self.language)
#             filename = "coach_command.mp3"
#             tts.save(filename)
#             os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
#         except Exception as e:
#             print(f"Audio error: {e}")
#             print(f"🔊 Coach: {text}")

#     def _log(self, fatigue_level, fatigue_score, llm_text, rule_text, rms_raw=None, rms_norm=None):
#         """Log coaching session data"""
#         ts = int(time.time() * 1000)
#         with open(self.logging_file_path, "a") as f:
#             rms_r = f"{rms_raw:.4f}" if rms_raw is not None else "N/A"
#             rms_n = f"{rms_norm:.4f}" if rms_norm is not None else "N/A"
#             f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text},{rms_r},{rms_n}\n")
#         print(f"📊 Fatigue: {fatigue_score:.3f} | Level: {fatigue_level} | State: {self.state.name}")

#     def adjust_sensitivity(self, factor):
#         """Allow manual adjustment of fatigue sensitivity (1.0 = default, <1.0 = more sensitive, >1.0 = less sensitive)"""
#         self.fatigue_sensitivity = max(0.5, min(2.0, factor))
#         print(f"🎚️  Sensitivity adjusted to {self.fatigue_sensitivity:.2f}")

# from enum import Enum, auto
# import torch
# from transformers import pipeline
# from gtts import gTTS
# import os, time
# import pandas as pd
# import numpy as np

# class CoachState(Enum):
#     CALIBRATING = auto()
#     ACTIVE = auto()
#     RESTING = auto()
#     STOPPED = auto()

# class Coach:
#     def __init__(self, participant_id, require_calibration=True):
#         self.language = "en"
#         self.participant_id = participant_id
#         self.state = CoachState.CALIBRATING if require_calibration else CoachState.ACTIVE
        
#         # Calibration parameters - just collect a few windows like before
#         self.require_calibration = require_calibration
#         self.calibration_windows = []
#         self.calibration_target = 1  # 3 windows = ~24 seconds at 8s/window
        
#         self.is_calibrated = not require_calibration
        
#         # Adaptive normalization (set during calibration)
#         self.rms_percentiles = None  # will store (p10, p50, p90, p99)
#         self.baseline_rms = None
#         self.max_observed_rms = 0.0
        
#         # Activity detection thresholds (relative to calibrated baseline)
#         self.activity_multiplier = 1.5  # reduced from 1.8 for more sensitivity
#         self.rest_multiplier = 1.1      # slightly above baseline = resting
        
#         # Fatigue smoothing
#         self.prev_fatigue_score = 0.0
#         self.active = True
        
#         # Rest tracking
#         self.rest_start = None
#         self.rest_duration = 0
        
#         # Fatigue scaling - now subject-adaptive
#         self.fatigue_sensitivity = 1.0  # will be tuned during calibration
        
#         # Logging
#         self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
#         os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
#         if not os.path.exists(self.logging_file_path):
#             with open(self.logging_file_path, "w") as f:
#                 f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm,rms_raw,rms_normalized\n")

#     def add_calibration_sample(self, rms_volts):
#         """Collect baseline RMS samples during calibration phase"""
#         if self.state != CoachState.CALIBRATING:
#             return False
            
#         self.calibration_windows.append(rms_volts)
#         progress = len(self.calibration_windows) / self.calibration_target
        
#         if len(self.calibration_windows) % 5 == 0:
#             print(f"📊 Calibration progress: {int(progress * 100)}%")
        
#         if len(self.calibration_windows) >= self.calibration_target:
#             self._finalize_calibration()
#             return True
#         return False

#     def _finalize_calibration(self):
#         """Compute subject-specific normalization parameters"""
#         rms_samples = np.array(self.calibration_windows)
        
#         # Compute percentiles for robust normalization
#         self.rms_percentiles = {
#             'p10': np.percentile(rms_samples, 10),
#             'p50': np.percentile(rms_samples, 50),  # median baseline
#             'p90': np.percentile(rms_samples, 90),
#             'p99': np.percentile(rms_samples, 99)
#         }
        
#         self.baseline_rms = self.rms_percentiles['p50']
#         self.max_observed_rms = self.rms_percentiles['p99']
        
#         # Set dynamic thresholds based on signal quality
#         signal_range = self.rms_percentiles['p90'] - self.rms_percentiles['p10']
#         if signal_range < 0.01:  # very clean signal
#             self.activity_multiplier = 1.3
#         elif signal_range < 0.05:  # normal signal
#             self.activity_multiplier = 1.5
#         else:  # noisy signal
#             self.activity_multiplier = 1.8
        
#         self.state = CoachState.ACTIVE
#         self.is_calibrated = True
        
#         print("\n✅ Calibration complete!")
#         print(f"   Windows collected: {len(self.calibration_windows)}")
#         print(f"   Baseline RMS: {self.baseline_rms:.4f}V")
#         print(f"   Max RMS (p99): {self.max_observed_rms:.4f}V")
#         print(f"   Activity threshold: {self.baseline_rms * self.activity_multiplier:.4f}V")
#         print(f"   Signal quality: {'Clean' if signal_range < 0.01 else 'Normal' if signal_range < 0.05 else 'Noisy'}\n")
        
#         self._speak("Calibration complete. Begin your workout.")

#     def _normalize_rms(self, rms_volts):
#         """Normalize RMS to 0-1 range using subject-specific calibration"""
#         if not self.is_calibrated:
#             # Before calibration, just clip to reasonable range
#             return np.clip(rms_volts / 0.5, 0.0, 1.0)
        
#         # Update max if we see higher values during workout
#         self.max_observed_rms = max(self.max_observed_rms, rms_volts)
        
#         # Normalize using percentile-based scaling
#         # Maps p50 -> 0.3, p99 -> 1.0, with smooth interpolation
#         if rms_volts <= self.baseline_rms:
#             # Below baseline: scale 0-0.3
#             normalized = 0.3 * (rms_volts / max(self.baseline_rms, 1e-6))
#         else:
#             # Above baseline: scale 0.3-1.0
#             range_above = self.max_observed_rms - self.baseline_rms
#             normalized = 0.3 + 0.7 * ((rms_volts - self.baseline_rms) / max(range_above, 1e-6))
        
#         return np.clip(normalized, 0.0, 1.0)

#     def _is_active(self, rms_volts):
#         """Determine if subject is actively exercising"""
#         if not self.is_calibrated:
#             return True  # assume active during calibration
        
#         if not hasattr(self, 'active'):
#             self.active = True
        
#         # Hysteresis thresholds
#         upper_thresh = self.baseline_rms * self.activity_multiplier
#         lower_thresh = self.baseline_rms * self.rest_multiplier
        
#         # State machine with hysteresis
#         if rms_volts < lower_thresh:
#             self.active = False
#         elif rms_volts > upper_thresh:
#             self.active = True
#         # else: maintain previous state (hysteresis)
        
#         return self.active

#     def _scale_fatigue(self, fatigue_score, active=True):
#         """Scale and smooth fatigue score with subject-adaptive sensitivity"""
#         fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
        
#         # Apply subject-specific sensitivity (reduce for sensitive subjects)
#         fatigue_adjusted = fatigue_score ** (1.0 / self.fatigue_sensitivity)
        
#         # Adaptive smoothing
#         alpha = 0.35 if active else 0.65
#         smoothed = alpha * fatigue_adjusted + (1 - alpha) * self.prev_fatigue_score
#         self.prev_fatigue_score = smoothed
        
#         # More granular binning
#         level = np.digitize(
#             smoothed,
#             bins=[0.12, 0.25, 0.38, 0.50, 0.65, 0.80],  # 7 bins for levels 1-7
#         ) + 1
        
#         return level, smoothed

#     def rule_based_message(self, fatigue_level, fatigue_score):
#         """Generate coaching message based on fatigue state"""
#         if fatigue_score < 0.15:
#             self.state = CoachState.ACTIVE
#             return "You're fresh — maintain your steady pace."
        
#         elif fatigue_score < 0.30:
#             self.state = CoachState.ACTIVE
#             return "Good work — you're in your rhythm. Keep it controlled."
        
#         elif fatigue_score < 0.45:
#             self.state = CoachState.ACTIVE
#             return "Fatigue building — stay focused on form. Prepare to stop soon."
        
#         elif fatigue_score < 0.65:
#             if self.state != CoachState.RESTING:
#                 self.state = CoachState.RESTING
#                 self.rest_duration = self._get_rest_duration(fatigue_score)
#                 self.rest_start = time.time()
#                 return f"Moderate fatigue detected. Stop and rest for {self.rest_duration} seconds."
#             return None  # Return None to avoid repeating the message
        
#         else:
#             if self.state != CoachState.RESTING:
#                 self.state = CoachState.RESTING
#                 self.rest_duration = self._get_rest_duration(fatigue_score)
#                 self.rest_start = time.time()
#                 return f"High fatigue. Stop immediately and rest at least {self.rest_duration} seconds."
#             return None  # Return None to avoid repeating the message

#     def _get_rest_duration(self, fatigue_score):
#         """Calculate rest duration based on fatigue level.
#         Returns 0 for fatigue_score < 0.40, otherwise scales linearly from 10s (at 0.40)
#         to 60s (at 1.00)."""

#         if fatigue_score < 0.40:
#             return 0
#         # Scale from 10s (at 0.4) to 60s (at 1.0)
#         t = 10 + (fatigue_score - 0.40) / 0.60 * 50
#         return int(max(10, min(60, round(t))))

#     def give_command(self, fatigue_score, activity_rms=None):
#         """Main coaching logic - prioritizes RMS-based activity tracking"""
        
#         # Handle calibration phase
#         if self.state == CoachState.CALIBRATING:
#             if activity_rms is not None:
#                 is_complete = self.add_calibration_sample(activity_rms)
#                 if not is_complete:
#                     return "Calibrating... Stay relaxed."
#             return
        
#         # Validate input
#         if activity_rms is None:
#             print("⚠️ Warning: No RMS provided, assuming active state")
#             active = True
#             rms_normalized = 0.5
#         else:
#             # Detect activity state
#             active = self._is_active(activity_rms)
            
#             # Normalize RMS for logging
#             rms_normalized = self._normalize_rms(activity_rms)
        
#         # Handle resting state
#         if self.state == CoachState.RESTING:
#             elapsed = time.time() - self.rest_start
            
#             # Active recovery: decay fatigue during rest
#             recovery_rate = 0.12 if active else 0.20  # recover faster when truly resting
#             time_fraction = elapsed / max(self.rest_duration, 1)
#             self.prev_fatigue_score = max(0.0, self.prev_fatigue_score - recovery_rate * time_fraction)
            
#             # Check if rest is complete
#             recovered = self.prev_fatigue_score < 0.25 or not active
#             if recovered or elapsed >= self.rest_duration:
#                 self.state = CoachState.ACTIVE
#                 self.prev_fatigue_score = 0.0  # reset for next set
#                 msg = "Rest complete. Resume your next set when ready."
#                 self._speak(msg)
#                 self._log(1, self.prev_fatigue_score, msg, "Recovered", activity_rms, rms_normalized)
#                 return msg
            
#             # Periodic rest reminders
#             remaining = int(self.rest_duration - elapsed)
#             if remaining > 0 and remaining % 30 == 0:
#                 self._speak(f"{remaining} seconds remaining.")
#             return
        
#         # === NEW: RMS-DRIVEN FATIGUE LOGIC ===
#         # Build fatigue estimate primarily from RMS history
#         if not hasattr(self, 'rms_history'):
#             self.rms_history = []
#             self.rms_decay_counter = 0
        
#         self.rms_history.append(activity_rms)
#         if len(self.rms_history) > 10:  # keep last 10 windows
#             self.rms_history.pop(0)
        
#         # Calculate sustained high activity (fatigue proxy)
#         if len(self.rms_history) >= 3:
#             recent_avg = np.mean(self.rms_history[-5:])  # last 5 windows
#             sustained_ratio = recent_avg / max(self.baseline_rms, 1e-6)
            
#             # If sustaining high activity (>2x baseline), fatigue accumulates
#             if sustained_ratio > 2.0 and active:
#                 rms_fatigue = min(1.0, (sustained_ratio - 2.0) / 3.0)  # scale 2x->5x to 0->1
#                 # Blend with model prediction (70% RMS, 30% model)
#                 blended_fatigue = 0.70 * rms_fatigue + 0.30 * fatigue_score
#             else:
#                 # Light activity or resting: use model with heavy decay
#                 blended_fatigue = 0.30 * fatigue_score
#                 self.rms_decay_counter += 1
#         else:
#             # Not enough history yet
#             blended_fatigue = 0.30 * fatigue_score
        
#         # If inactive for multiple windows, force recovery
#         if not active:
#             self.rms_decay_counter += 1
#             if self.rms_decay_counter >= 2:  # 2+ windows of inactivity
#                 blended_fatigue = max(0.0, blended_fatigue - 0.2)
#         else:
#             self.rms_decay_counter = 0
        
#         # Scale and smooth
#         fatigue_level, smoothed = self._scale_fatigue(blended_fatigue, active=active)
        
#         # Generate and deliver coaching message
#         rule_text = self.rule_based_message(fatigue_level, smoothed)
#         self._speak(rule_text)
#         self._log(fatigue_level, smoothed, rule_text, rule_text, activity_rms, rms_normalized)
        
#         print(f"🔍 RMS: {activity_rms:.4f}V | Active: {active} | Model: {fatigue_score:.3f} | Blended: {blended_fatigue:.3f} | Final: {smoothed:.3f}")
        
#         return rule_text

#     def _speak(self, text):
#         """Convert text to speech"""
#         try:
#             tts = gTTS(text=text, lang=self.language)
#             filename = "coach_command.mp3"
#             tts.save(filename)
#             os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
#         except Exception as e:
#             print(f"Audio error: {e}")
#             print(f"🔊 Coach: {text}")

#     def _log(self, fatigue_level, fatigue_score, llm_text, rule_text, rms_raw=None, rms_norm=None):
#         """Log coaching session data"""
#         ts = int(time.time() * 1000)
#         with open(self.logging_file_path, "a") as f:
#             rms_r = f"{rms_raw:.4f}" if rms_raw is not None else "N/A"
#             rms_n = f"{rms_norm:.4f}" if rms_norm is not None else "N/A"
#             f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text},{rms_r},{rms_n}\n")
#         print(f"📊 Fatigue: {fatigue_score:.3f} | Level: {fatigue_level} | State: {self.state.name}")

#     def adjust_sensitivity(self, factor):
#         """Allow manual adjustment of fatigue sensitivity (1.0 = default, <1.0 = more sensitive, >1.0 = less sensitive)"""
#         self.fatigue_sensitivity = max(0.5, min(2.0, factor))
#         print(f"🎚️  Sensitivity adjusted to {self.fatigue_sensitivity:.2f}")


from enum import Enum, auto
import os, time
import numpy as np
from gtts import gTTS
from groq import Groq

class CoachState(Enum):
    CALIBRATING = auto()
    ACTIVE = auto()
    RESTING = auto()
    STOPPED = auto()

class Coach:
    def __init__(self, participant_id, require_calibration=True, groq_api_key=None):
        self.language = "en"
        self.participant_id = participant_id
        self.state = CoachState.CALIBRATING if require_calibration else CoachState.ACTIVE
        groq_api_key = ""
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
        self.use_llm = groq_api_key is not None or "GROQ_API_KEY" in os.environ
        
        # Track recent messages to avoid repetition
        self.recent_messages = []
        self.max_message_history = 3
        
        # Calibration parameters
        self.require_calibration = require_calibration
        self.calibration_windows = []
        self.calibration_target = 1
        self.is_calibrated = not require_calibration
        
        # Adaptive normalization
        self.rms_percentiles = None
        self.baseline_rms = None
        self.max_observed_rms = 0.0
        
        # Activity detection thresholds
        self.activity_multiplier = 1.5
        self.rest_multiplier = 1.1
        
        # Fatigue smoothing
        self.prev_fatigue_score = 0.0
        self.active = True
        
        # Rest tracking
        self.rest_start = None
        self.rest_duration = 0
        
        # Fatigue scaling
        self.fatigue_sensitivity = 1.0
        
        # Logging
        self.logging_file_path = f"Fitness_Coach/Logging/coach_log_{participant_id}.csv"
        os.makedirs(os.path.dirname(self.logging_file_path), exist_ok=True)
        if not os.path.exists(self.logging_file_path):
            with open(self.logging_file_path, "w") as f:
                f.write("timestamp_ms,fatigue_level,continuous_score,state,rule,llm,rms_raw,rms_normalized\n")

    def _llm_rephrase(self, rule_text, fatigue_level, context=""):
        """Use Groq to rephrase coaching message in a friendly, varied way"""
        if not self.use_llm:
            return rule_text
        
        try:
            # Build context from recent messages
            recent_context = ""
            if self.recent_messages:
                recent_context = f"\nRecent messages you've said: {', '.join(self.recent_messages[-2:])}"
            
            prompt = f"""You are an encouraging, supportive fitness coach guiding someone through bicep curls.

Your task: Rephrase the following instruction in a natural, motivational way.

Rules:
- Keep the SAME core meaning and any specific numbers (like rest duration)
- Make it sound friendly, personal, and varied (don't repeat phrases)
- Use 1-2 sentences maximum
- Be encouraging but not overly dramatic
- Vary your style: sometimes energetic, sometimes calm and supportive
{recent_context}

Fatigue level: {fatigue_level}/7
Instruction to rephrase: "{rule_text}"
{context}

Rephrased coaching message:"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fast and capable model
                messages=[
                    {"role": "system", "content": "You are a friendly, motivational fitness coach. Keep responses concise and natural."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher for more variety
                max_tokens=80,
                top_p=0.9
            )
            
            rephrased = response.choices[0].message.content.strip()
            
            # Remove quotes if LLM added them
            rephrased = rephrased.strip('"').strip("'")
            
            # Track message to avoid repetition
            self.recent_messages.append(rephrased[:50])  # Store first 50 chars
            if len(self.recent_messages) > self.max_message_history:
                self.recent_messages.pop(0)
            
            return rephrased
            
        except Exception as e:
            print(f"⚠️ LLM error: {e}. Falling back to rule-based message.")
            return rule_text

    def add_calibration_sample(self, rms_volts):
        """Collect baseline RMS samples during calibration phase"""
        if self.state != CoachState.CALIBRATING:
            return False
            
        self.calibration_windows.append(rms_volts)
        progress = len(self.calibration_windows) / self.calibration_target
        
        if len(self.calibration_windows) % 5 == 0:
            print(f"📊 Calibration progress: {int(progress * 100)}%")
        
        if len(self.calibration_windows) >= self.calibration_target:
            self._finalize_calibration()
            return True
        return False

    def _finalize_calibration(self):
        """Compute subject-specific normalization parameters"""
        rms_samples = np.array(self.calibration_windows)
        
        self.rms_percentiles = {
            'p10': np.percentile(rms_samples, 10),
            'p50': np.percentile(rms_samples, 50),
            'p90': np.percentile(rms_samples, 90),
            'p99': np.percentile(rms_samples, 99)
        }
        
        self.baseline_rms = self.rms_percentiles['p50']
        self.max_observed_rms = self.rms_percentiles['p99']
        
        signal_range = self.rms_percentiles['p90'] - self.rms_percentiles['p10']
        if signal_range < 0.01:
            self.activity_multiplier = 1.3
        elif signal_range < 0.05:
            self.activity_multiplier = 1.5
        else:
            self.activity_multiplier = 1.8
        
        self.state = CoachState.ACTIVE
        self.is_calibrated = True
        
        print("\n✅ Calibration complete!")
        print(f"   Baseline RMS: {self.baseline_rms:.4f}V")
        print(f"   Activity threshold: {self.baseline_rms * self.activity_multiplier:.4f}V\n")
        
        self._speak("Calibration complete. Begin your workout.")

    def _normalize_rms(self, rms_volts):
        """Normalize RMS to 0-1 range using subject-specific calibration"""
        if not self.is_calibrated:
            return np.clip(rms_volts / 0.5, 0.0, 1.0)
        
        self.max_observed_rms = max(self.max_observed_rms, rms_volts)
        
        if rms_volts <= self.baseline_rms:
            normalized = 0.3 * (rms_volts / max(self.baseline_rms, 1e-6))
        else:
            range_above = self.max_observed_rms - self.baseline_rms
            normalized = 0.3 + 0.7 * ((rms_volts - self.baseline_rms) / max(range_above, 1e-6))
        
        return np.clip(normalized, 0.0, 1.0)

    def _is_active(self, rms_volts):
        """Determine if subject is actively exercising"""
        if not self.is_calibrated:
            return True
        
        if not hasattr(self, 'active'):
            self.active = True
        
        upper_thresh = self.baseline_rms * self.activity_multiplier
        lower_thresh = self.baseline_rms * self.rest_multiplier
        
        if rms_volts < lower_thresh:
            self.active = False
        elif rms_volts > upper_thresh:
            self.active = True
        
        return self.active

    def _scale_fatigue(self, fatigue_score, active=True):
        """Scale and smooth fatigue score with subject-adaptive sensitivity"""
        fatigue_score = float(np.clip(fatigue_score, 0.0, 1.0))
        fatigue_adjusted = fatigue_score ** (1.0 / self.fatigue_sensitivity)
        
        alpha = 0.35 if active else 0.65
        smoothed = alpha * fatigue_adjusted + (1 - alpha) * self.prev_fatigue_score
        self.prev_fatigue_score = smoothed
        
        level = np.digitize(
            smoothed,
            bins=[0.12, 0.25, 0.38, 0.50, 0.65, 0.80],
            # bins=[0.10, 0.22, 0.35, 0.47, 0.58, 0.70],
        ) + 1
        
        return level, smoothed

    def rule_based_message(self, fatigue_level, fatigue_score):
        """Generate coaching message based on fatigue state"""
        if fatigue_score < 0.15:
            self.state = CoachState.ACTIVE
            return "You're fresh — maintain your steady pace."
        
        elif fatigue_score < 0.30:
            self.state = CoachState.ACTIVE
            return "Good work — you're in your rhythm. Keep it controlled."
        
        elif fatigue_score < 0.45:
            self.state = CoachState.ACTIVE
            return "Fatigue building — stay focused on form. Prepare to stop soon."
        
        elif fatigue_score < 0.65:
            if self.state != CoachState.RESTING:
                self.state = CoachState.RESTING
                self.rest_duration = self._get_rest_duration(fatigue_score)
                self.rest_start = time.time()
                return f"Moderate fatigue detected. Stop and rest for {self.rest_duration} seconds."
            return None
        
        else:
            if self.state != CoachState.RESTING:
                self.state = CoachState.RESTING
                self.rest_duration = self._get_rest_duration(fatigue_score)
                self.rest_start = time.time()
                return f"High fatigue. Stop immediately and rest at least {self.rest_duration} seconds."
            return None

    def _get_rest_duration(self, fatigue_score):
        """Calculate rest duration based on fatigue level"""
        if fatigue_score < 0.40:
            return 0
        t = 10 + (fatigue_score - 0.40) / 0.60 * 50
        return int(max(10, min(60, round(t))))

    def give_command(self, fatigue_score, activity_rms=None):
        """Main coaching logic - prioritizes RMS-based activity tracking"""
        
        if self.state == CoachState.CALIBRATING:
            if activity_rms is not None:
                is_complete = self.add_calibration_sample(activity_rms)
                if not is_complete:
                    return "Calibrating... Stay relaxed."
            return
        
        if activity_rms is None:
            print("⚠️ Warning: No RMS provided, assuming active state")
            active = True
            rms_normalized = 0.5
        else:
            active = self._is_active(activity_rms)
            rms_normalized = self._normalize_rms(activity_rms)
        
        # Handle resting state
        if self.state == CoachState.RESTING:
            elapsed = time.time() - self.rest_start
            
            recovery_rate = 0.12 if active else 0.20
            time_fraction = elapsed / max(self.rest_duration, 1)
            self.prev_fatigue_score = max(0.0, self.prev_fatigue_score - recovery_rate * time_fraction)
            
            recovered = self.prev_fatigue_score < 0.25 or not active
            if recovered or elapsed >= self.rest_duration:
                self.state = CoachState.ACTIVE
                self.prev_fatigue_score = 0.0
                base_msg = "Rest complete. Resume your next set when ready."
                msg = self._llm_rephrase(base_msg, 1, context="User just finished resting")
                self._speak(msg)
                self._log(1, self.prev_fatigue_score, msg, base_msg, activity_rms, rms_normalized)
                return msg
            
            remaining = int(self.rest_duration - elapsed)
            if remaining > 0 and remaining % 30 == 0:
                reminder = f"{remaining} seconds remaining."
                self._speak(self._llm_rephrase(reminder, 5, context="Rest reminder"))
            return
        
        # RMS-DRIVEN FATIGUE LOGIC
        if not hasattr(self, 'rms_history'):
            self.rms_history = []
            self.rms_decay_counter = 0
        
        self.rms_history.append(activity_rms)
        if len(self.rms_history) > 10:
            self.rms_history.pop(0)
        
        if len(self.rms_history) >= 3:
            recent_avg = np.mean(self.rms_history[-5:])
            sustained_ratio = recent_avg / max(self.baseline_rms, 1e-6)
            
            if sustained_ratio > 2.0 and active:
                rms_fatigue = min(1.0, (sustained_ratio - 2.0) / 3.0)
                blended_fatigue = 0.70 * rms_fatigue + 0.30 * fatigue_score
            else:
                blended_fatigue = 0.30 * fatigue_score
                self.rms_decay_counter += 1
        else:
            blended_fatigue = 0.30 * fatigue_score
        
        if not active:
            self.rms_decay_counter += 1
            if self.rms_decay_counter >= 2:
                blended_fatigue = max(0.0, blended_fatigue - 0.2)
        else:
            self.rms_decay_counter = 0
        
        fatigue_level, smoothed = self._scale_fatigue(blended_fatigue, active=active)
        
        # Generate and deliver coaching message
        rule_text = self.rule_based_message(fatigue_level, smoothed)
        
        if rule_text:  # Only speak if there's a new message
            llm_text = self._llm_rephrase(rule_text, fatigue_level)
            self._speak(llm_text)
            self._log(fatigue_level, smoothed, llm_text, rule_text, activity_rms, rms_normalized)
            print(f"🔍 Active: {active} | Model: {fatigue_score:.3f} | Final: {smoothed:.3f}")
            return llm_text
        
        return None

    def _speak(self, text):
        """Convert text to speech"""
        if not text:
            return
        try:
            tts = gTTS(text=text, lang=self.language)
            filename = "coach_command.mp3"
            tts.save(filename)
            os.system("afplay coach_command.mp3" if os.name == "posix" else "mpg123 coach_command.mp3")
        except Exception as e:
            print(f"Audio error: {e}")
            print(f"🔊 Coach: {text}")

    def _log(self, fatigue_level, fatigue_score, llm_text, rule_text, rms_raw=None, rms_norm=None):
        """Log coaching session data"""
        ts = int(time.time() * 1000)
        with open(self.logging_file_path, "a") as f:
            rms_r = f"{rms_raw:.4f}" if rms_raw is not None else "N/A"
            rms_n = f"{rms_norm:.4f}" if rms_norm is not None else "N/A"
            f.write(f"{ts},{fatigue_level},{fatigue_score:.3f},{self.state.name},{rule_text},{llm_text},{rms_r},{rms_n}\n")
        print(f"📊 Fatigue: {fatigue_score:.3f} | Level: {fatigue_level} | State: {self.state.name}")

    def adjust_sensitivity(self, factor):
        """Adjust fatigue sensitivity (1.0 = default, <1.0 = more sensitive, >1.0 = less sensitive)"""
        self.fatigue_sensitivity = max(0.5, min(2.0, factor))
        print(f"🎚️  Sensitivity adjusted to {self.fatigue_sensitivity:.2f}")