import serial
import time

PORT = "/dev/cu.SLAB_USBtoUART"
BAUD = 115200

#open serial connection
ser = serial.Serial(PORT, 115200)

#create file
filename = f"emg_log_{int(time.time())}.csv"
with open(filename, "w") as f:
    f.write("time_ms, emg_value\n") #headeer
    print(f"Logging to {filename}...")
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                f.write(line + "\n")
                print(line)
    except KeyboardInterrupt:
        print("\n Logging stopped manually.")

