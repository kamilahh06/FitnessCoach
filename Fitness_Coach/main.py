from EMGReader import EMGReader
from Fitness_Coach.Coach import Coach
from FatigueModel import ModelHandler
from Fatigue_Model.Data_Processing.PreProcessor import PreProcessor
import os

def main():
    participant_id = -1 # Participant id for logging results    
    window_size = 2000  # Miliseconds for EMG window
        
    model_file_path = "/u/kamilah/FitnessCoach/Models/best_model_4.pt"
    model = ModelHandler(model_file_path)  # Load your trained model here
    processor = PreProcessor()
    coach = Coach(participant_id)
    
    reader = EMGReader(port="/dev/cu.SLAB_USBtoUART", participant_id=1)
    reader.create_file(participant_id)
    
    print("🎯 Fitness Coach started. Reading live EMG...")

    while True:
        emg_val = reader.read_window(window_size)
        features = processor.full_process(emg_val)

        if features is not None:
            prediction = model.predict(features)
            coach.give_command(prediction)
            coach.log()

if __name__ == "__main__":
    main()