from enum import IntEnum
import torch
from transformers import pipeline
from gtts import gTTS
import os


class Coach:
    def __init__(self, participant_id):
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # replace with "mps" for Mac
        )
        self.base_message = (
            "You are a fitness coach trying to improve the performance of an athlete "
            "who is cycling on a stationary bike while minimizing the risk of injury. "
            "Based on the athlete's fatigue level, provide a short, clear command "
            "to optimize their workout.\n\n"
            "Fatigue levels:\n"
            "1: Not Fatigued\n"
            "2: Slightly Fatigued\n"
            "3: Moderately Fatigued\n"
            "4: Very Fatigued\n"
            "5: Extremely Fatigued\n"
            "6: Near Exhaustion\n"
            "7: Exhausted\n\n"
        )
        self.language = "en"
        self.participant_id = participant_id
        self.logging_file_path = "Fitness_Coach/Logging/coach_log" + self.participant_id + ".csv"
        self.curr_fatigue_state = None
        self.curr_message = ""


    def give_command(self, prediction):
        self.curr_fatigue_state = prediction
        
        # Build prompt including fatigue state
        prompt = self.base_message + f"The athlete's fatigue level is: {prediction}.\nCoach response:"

        # Generate response
        outputs = self.pipe(prompt, max_new_tokens=50, do_sample=True, top_p=0.9)
        self.curr_message = outputs[0]["generated_text"].strip()

        print("AI Coach says:", self.curr_message)
        
        # Text-to-Speech
        tts = gTTS(text=self.curr_message, lang=self.language)
        filename = "coach_command.mp3"
        tts.save(filename)

        # Optionally play audio (Linux/WSL/macOS)
        try:
            os.system(f"mpg123 {filename}")  # install mpg123 or use 'afplay' on Mac
        except Exception:
            print(f"Audio unable to be played")

        return self.curr_message
    
    def log(self):
        time = int(time.time() * 1000)
        with open("coach_log.txt", "a") as f:
            f.write(time, self.curr_fatigue_state, message)
            
        if not os.path.exists(self.logging_file_path):
            with open(self.logging_file_path, "w") as f:
                f.write("time_ms, fatigue_state, command\n") #header