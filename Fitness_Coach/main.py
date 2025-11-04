# from .EMGReader import EMGReader
# from .Coach import Coach
# from .FatigueModel import ModelHandler
# from Fatigue_Model.Data_Processing.PreProcessor import PreProcessor
# import os

# def main():
#     print("Running Fitness Coach main()")

#     participant_id = 1
#     window_size = 4

#     model_file_path = "Models/best_model_4.pt"
#     model = ModelHandler(model_file_path)
#     processor = PreProcessor()
#     coach = Coach(participant_id)
    

#     reader = EMGReader(ip="172.20.10.2", port=8080, participant_id=participant_id)
#     reader.create_file(participant_id)

#     print("🎯 Fitness Coach started. Reading live EMG...")

#     while True:
#         # ---- Read a batch from ESP32 ----
#         window, start_time = reader.read_window(window_size)
#         if len(window) == 0:
#             print("Empty window, retrying...")
#             continue

#         # ---- Optional: log the window ----
#         with open(reader.filename, "a") as f:
#             for val in window:
#                 f.write(f"{int(start_time*1000)},{val}\n")

#         # ---- Process and predict ----
#         features = processor.full_process(window)
#         if features is not None:
#             prediction = model.predict(features)
#             coach.give_command(prediction)
#             # coach.log()

# if __name__ == "__main__":
#     main()

from .EMGReader import EMGReader
from .Coach import Coach, CoachState
from .FatigueModel import ModelHandler
from Fatigue_Model.Data_Processing.PreProcessor import PreProcessor
import numpy as np
import os
import time

def main():
    print("Running Fitness Coach main()")

    participant_id = -2
    window_size = 1600  # in samples, not seconds, og 1600
    model_file_path = "Models/best_model_4.pt"

    model = ModelHandler(model_file_path)
    processor = PreProcessor()
    coach = Coach(participant_id)
    reader = EMGReader(ip="172.20.10.2", port=8080, participant_id=participant_id)
    reader.create_file(participant_id)

    print("🎯 Fitness Coach started. Reading live EMG...")

    while True:
        try:
            # 💤 If the coach is resting, pause *before* new readings
            print(f"Coach State: {coach.state}")
            if coach.state == CoachState.RESTING:
                rest_time = coach.rest_duration
                print(f"⏸️  Pausing EMG reading for {rest_time} seconds...")
                coach._speak(f"Rest for {rest_time} seconds.")
                time.sleep(rest_time)

                # Reset fatigue slightly so next set starts fresh
                coach.prev_fatigue_score = max(0.0, coach.prev_fatigue_score * 0.3)
                coach.state = CoachState.ACTIVE
                coach._speak("Rest complete. Resume your next set.")
                print("▶️  Resuming EMG reading...")
                continue  # skip to next iteration (don’t read EMG during rest)

            # ---- Read a window from ESP32 ----
            window, start_time = reader.read_window(window_size)

            # Convert ADC → volts first
            window = (np.asarray(window, dtype=float) / 4095.0) * 3.3

            # 1) compute raw RMS in volts for activity detection
            activity_rms = PreProcessor.window_rms(window)

            # 2) then do your training-aligned preprocessing
            features = processor.full_process(window, normalize_mode="zscore")
            if features is None or np.isnan(features).any():
                continue

            prediction = float(model.predict(features))
            prediction = np.clip(prediction, 0.0, 1.0)

            # pass activity_rms to make the coach decay during rest
            coach.give_command(prediction, activity_rms=activity_rms)
            # coach.give_command(prediction)

        except (ConnectionResetError, OSError):
            print("⚠️ Connection lost — attempting to reconnect...")
            reader.reconnect()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Session ended by user.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
