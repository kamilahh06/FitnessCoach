from enum import IntEnum  # Use IntEnum so values behave like integers

class Command(IntEnum):
    SPEED_UP = 1
    HOLD = 2
    SLOW_DOWN = 3
    RECOVER = 4

class Coach:
    def __init__(self):
        """Always start with SPEED_UP"""
        self.current_command = Command.SPEED_UP

    def give_command(self, prediction: int) -> Command:
        """
        Decide command based on model prediction.
        prediction: 0 (no fatigue), 1 (fatigue)
        Sample logic, unfinished
        """
        if prediction == 0:  # No fatigue
            # move "up" the command scale if possible
            if self.current_command < Command.SPEED_UP:
                self.current_command = Command.SPEED_UP
        else:  # Fatigue detected
            if self.current_command < Command.RECOVER: 
                self.current_command = Command(self.current_command + 1)
        return self.current_command