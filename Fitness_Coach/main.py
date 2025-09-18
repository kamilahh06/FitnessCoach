from emg_reader import EMGReader
from processing import Processor
from model_handler import ModelHandler
from coach import Coach
from feedback import Feedback

def main():
    reader = EMGReader()
    processor = Processor()
    model = ModelHandler()
    coach = Coach()
    feedback = Feedback()

    print("🎯 Fitness Coach started. Reading live EMG...")

    while True:
        emg_val = reader.get_data()
        features = processor.process(emg_val)

        if features is not None:
            prediction = model.predict(features)
            command = coach.give_command(prediction)
            feedback.give_visual(command)
            feedback.give_audio(command)

if __name__ == "__main__":
    main()