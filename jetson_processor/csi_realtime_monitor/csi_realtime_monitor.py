#!/usr/bin/env python3
import serial
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque
import argparse
import os

class CSIRealtimeMonitor:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, csv_path=None,
                 window_seconds=3.0, step_seconds=1.0):
        self.port = port
        self.baudrate = baudrate
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.csv_path = csv_path or f"csi_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.ser = None
        self.buffer = deque()
        self.last_inference_time = 0.0

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=["pc_time", "count", "rssi", "length", "mac", "esp_time"]).to_csv(
                self.csv_path, index=False
            )

    def parse_line(self, line):
        parts = line.strip().split(",")
        if len(parts) != 6:
            return None
        if parts[0] != "CSI":
            return None
        try:
            return {
                "pc_time": time.time(),
                "count": int(parts[1]),
                "rssi": int(parts[2]),
                "length": int(parts[3]),
                "mac": parts[4],
                "esp_time": float(parts[5]),
            }
        except ValueError:
            return None

    def append_csv(self, row):
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=False, index=False)

    def cleanup_buffer(self, now_ts):
        while self.buffer and (now_ts - self.buffer[0]["pc_time"] > self.window_seconds):
            self.buffer.popleft()

    def compute_features(self):
        if len(self.buffer) < 10:
            return None

        df = pd.DataFrame(list(self.buffer))
        rssi = df["rssi"].astype(float).values

        features = {
            "packet_count": len(rssi),
            "rssi_mean": float(np.mean(rssi)),
            "rssi_std": float(np.std(rssi)),
            "rssi_min": float(np.min(rssi)),
            "rssi_max": float(np.max(rssi)),
            "rssi_range": float(np.max(rssi) - np.min(rssi)),
            "rssi_mad": float(np.mean(np.abs(np.diff(rssi)))) if len(rssi) > 1 else 0.0,
        }
        return features

    def infer_state(self, features):
        if features is None:
            return "warming_up"

        if features["packet_count"] < 10:
            return "insufficient_data"

        if features["rssi_std"] < 0.8 and features["rssi_range"] < 3:
            return "stable"

        if features["rssi_std"] >= 0.8 or features["rssi_mad"] >= 0.8:
            return "motion"

        return "occupied_uncertain"

    def run(self):
        self.connect()
        print(f"Listening on {self.port} at {self.baudrate} baud")
        print(f"Saving raw data to {self.csv_path}")
        print(f"Window={self.window_seconds}s Step={self.step_seconds}s")

        try:
            while True:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
                now_ts = time.time()

                if raw:
                    row = self.parse_line(raw)
                    if row is not None:
                        self.buffer.append(row)
                        self.append_csv(row)

                self.cleanup_buffer(now_ts)

                if now_ts - self.last_inference_time >= self.step_seconds:
                    features = self.compute_features()
                    state = self.infer_state(features)

                    if features is None:
                        print("state=warming_up")
                    else:
                        print(
                            f"state={state} "
                            f"packets={features['packet_count']} "
                            f"mean={features['rssi_mean']:.2f} "
                            f"std={features['rssi_std']:.2f} "
                            f"range={features['rssi_range']:.2f} "
                            f"mad={features['rssi_mad']:.2f}"
                        )

                    self.last_inference_time = now_ts

        except KeyboardInterrupt:
            print("Stopping monitor")
        finally:
            if self.ser is not None:
                self.ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()

    monitor = CSIRealtimeMonitor(
        port=args.port,
        baudrate=args.baudrate,
        csv_path=args.csv,
        window_seconds=args.window,
        step_seconds=args.step,
    )
    monitor.run()
