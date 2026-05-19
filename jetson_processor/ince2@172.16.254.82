#!/usr/bin/env python3
import serial
import pandas as pd
import time
from datetime import datetime
from collections import deque
import argparse
import os

class CSIReader:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, output_csv=None):
        self.port = port
        self.baudrate = baudrate
        self.output_csv = output_csv or f"csi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.window = deque(maxlen=30)
        self.packet_count = 0
        self.last_save_time = time.time()

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        if not os.path.exists(self.output_csv):
            pd.DataFrame(columns=[
                "pc_time",
                "count",
                "rssi",
                "len",
                "mac",
                "esp_time"
            ]).to_csv(self.output_csv, index=False)

    def parse_line(self, line):
        parts = line.strip().split(",")
        if len(parts) != 6:
            return None
        if parts[0] != "CSI":
            return None

        try:
            return {
                "pc_time": datetime.now().isoformat(),
                "count": int(parts[1]),
                "rssi": int(parts[2]),
                "len": int(parts[3]),
                "mac": parts[4],
                "esp_time": float(parts[5]),
            }
        except ValueError:
            return None

    def append_csv(self, row):
        pd.DataFrame([row]).to_csv(
            self.output_csv,
            mode="a",
            header=False,
            index=False
        )

    def compute_motion_score(self):
        if len(self.window) < 10:
            return None
        series = pd.Series(self.window)
        return float(series.std())

    def run(self):
        self.connect()
        print(f"Listening on {self.port} at {self.baudrate} baud")
        print(f"Saving to {self.output_csv}")

        try:
            while True:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                row = self.parse_line(raw)
                if row is None:
                    continue

                self.packet_count += 1
                self.window.append(row["rssi"])
                self.append_csv(row)

                motion_score = self.compute_motion_score()

                if motion_score is None:
                    print(
                        f"count={row['count']} rssi={row['rssi']} len={row['len']} mac={row['mac']}"
                    )
                else:
                    state = "motion" if motion_score > 1.5 else "stable"
                    print(
                        f"count={row['count']} rssi={row['rssi']} len={row['len']} "
                        f"mac={row['mac']} std={motion_score:.2f} state={state}"
                    )

        except KeyboardInterrupt:
            print("Stopping reader")
        finally:
            self.ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    reader = CSIReader(port=args.port, baudrate=args.baudrate, output_csv=args.csv)
    reader.run()
