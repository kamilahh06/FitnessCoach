import random
import time
import serial
import numpy as np

class EMGReader:
    def __init__(self, port, baud=115200):
        self.ser= serial.Serial(port, baud, timeout=1)
    
    # Read one line and return (time_ms, emg_value) as ints
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
    
    
    # Read a window of emg values of given size
    def read_window(self, window_size=30):
        window = np.array([])
        start_time = time.time()
        while len(window) < window_size:
            row = self.read_line()
            if row:
                window.append(row[1])
        return window, start_time
    
    
    # Create a new log file with header
    def create_file(self):
        filename = f"emg_log_{int(time.time())}.csv"
        print(f"Logging to {filename}...")
        with (open filename, "w") as f:
            f.write("time_ms, emg_value\n")
        return filename
    
    
    # Create a new log file with header
    def log_to_file(self, filename):
        if not filename:
            filename = self.create_file()
        with open(filename, "w") as f:
            row = self.read_line()
            f.write(f"{row[0]},{row[1]}\n")
         

    # Stream everything to a csv file until ctrl+c
    def log_all_to_file(self, filename):
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
