import pyttsx3

class Feedback:
    def __init__(self):
        self.engine = pyttsx3.init()

    def give_audio(self, command):
        self.engine.say(command)
        self.engine.runAndWait()

    def give_visual(self, command):
        """ Placeholder for visual feedback logic."""