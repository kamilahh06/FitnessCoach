# import random
# import time
# import serial
# import numpy as np
# import pandas as pd
# import os
# import socket

# class EMGReader:
#     """
#     Reads EMG data from a serial port and logs it to a CSV file.
#     """
#     def __init__(self, ip="172.20.10.2", port=8080, participant_id=1):
#         self.participant_id = participant_id
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.connect((ip, port))
#         self.sock.settimeout(5.0)
#         self.filename = f"emg_log_{int(time.time())}_{participant_id}.csv"
#         print(f"Connected to ESP32 at {ip}:{port}")

#         with open(self.filename, "w") as f:
#             f.write("time_ms,emg_value\n")

#     # Read one line and return (time_ms, emg_value) as ints
#     def read_line(self):
#         try:
#             data = self.sock.recv(1024).decode("utf-8", errors="ignore")
#             for line in data.strip().splitlines():
#                 parts = line.split(",")
#                 if len(parts) == 2:
#                     try:
#                         t = int(parts[0])
#                         val = int(parts[1])
#                         return t, val
#                     except ValueError:
#                         continue
#         except socket.timeout:
#             return None
#         return None
    
#     # Read a window of emg values of given size
#     def read_window(self, window_size=4):
#         window = []                    # not np.array()
#         start_time = time.time()
#         while len(window) < window_size:
#             row = self.read_line()
#             if row:
#                 window.append(row[1])
#         return np.array(window), start_time
    
#     # Create a new log file with header
#     def create_file(self, participant_id=None):
#         print(f"Logging to {self.filename}...")
#         with open (self.filename, "w") as f:
#             f.write("time_ms, emg_value\n")
#         return self.filename
    
    
#     # Create a new log file with header
#     def log_to_file(self):
#         if not filename:
#             filename = self.create_file()
#         with open(filename, "w") as f:
#             row = self.read_line()
#             f.write(f"{row[0]},{row[1]}\n")
         

#     # Stream everything to a csv file until ctrl+c
#     def log_all_to_file(self):
#         with open(self.filename, "w") as f:
#             f.write("time_ms, emg_value\n") #header
#             try:
#                 while True:
#                     row= self.read_line()
#                     if row:
#                         f.write(f"{row[0]},{row[1]}\n")
#                         print(row)
#             except KeyboardInterrupt:
#                 print("Logging stopped.")

import time
import socket
import numpy as np
import os

class EMGReader:
    """
    Reads EMG data from an ESP32 over TCP and logs it to CSV.
    """

    def __init__(self, ip="172.20.10.2", port=8080, participant_id=1):
        self.ip = ip
        self.port = port
        self.participant_id = participant_id
        self.sock = None
        self.filename = f"emg_log_{int(time.time())}_{participant_id}.csv"
        self.connect()

    def connect(self):
        """Establish a TCP connection to the ESP32."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        print(f"🔌 Connecting to ESP32 at {self.ip}:{self.port}...")
        self.sock.connect((self.ip, self.port))
        print("✅ Connected.")
        with open(self.filename, "w") as f:
            f.write("time_ms,emg_value\n")

    def reconnect(self):
        """Reconnect if the socket dies."""
        try:
            self.connect()
        except Exception as e:
            print(f"Reconnect failed: {e}")
            time.sleep(2)
            self.reconnect()

    def read_line(self):
        """Read one line of 'timestamp,value' data from the socket."""
        try:
            data = self.sock.recv(1024).decode("utf-8", errors="ignore")
            for line in data.strip().splitlines():
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        t = int(parts[0])
                        val = float(parts[1])
                        return t, val
                    except ValueError:
                        continue
        except socket.timeout:
            return None
        return None

    def read_window(self, window_size=4):
        """Read a fixed-length EMG window."""
        window = []
        start_time = time.time()
        while len(window) < window_size:
            row = self.read_line()
            if row:
                window.append(row[1])
            else:
                # avoid tight loop on empty reads
                time.sleep(0.001)
        # print(f"Window: {window}")
        return np.array(window, dtype=float), start_time

    def create_file(self, participant_id=None):
        """Create a log file for EMG recording."""
        os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
        with open(self.filename, "w") as f:
            f.write("time_ms,emg_value\n")
        print(f"📝 Logging to {self.filename}...")
        return self.filename

