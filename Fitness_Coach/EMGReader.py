import random
import time
import serial

class EMGReader:
    def __init__(self, port, baud=115200):
        self.ser= serial.Serial(port, baud, timeout=1)
    
    #read one line and return (time_ms, emg_value) as ints
    def read_line(self):
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t = int(parts[0])
                    val = int(parts[1])
                    return t, val
                except ValueError:
                    return None
        return None

    #stream everything to a csv file until ctrl+c
    def log_to_file(self, filename):
        with open(filename, "w") as f:
            f.write("time_ms, emg_value\n") #header
            try:
                while True:
                    row= self.read_line()
                    if row:
                        f.write(f"{row[0]},{row[1]}\n")
                        print(row)
            except KeyboardInterrupt:
                print("Logging stopped.")
