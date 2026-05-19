#!/usr/bin/env python3
import serial
import numpy as np
import time
from datetime import datetime
from collections import deque
import argparse
import os
import csv

class CSIRealtimeMonitorV3:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        baudrate=115200,
        csv_path=None,
        window_seconds=3.0,
        step_seconds=1.0,
        calibration_seconds=8.0,
        min_packets=5,
        hold_seconds=3.0,
        min_calibration_windows=3,
    ):
        self.port = port
        self.baudrate = baudrate
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.calibration_seconds = calibration_seconds
        self.min_packets = min_packets
        self.hold_seconds = hold_seconds
        self.min_calibration_windows = min_calibration_windows
        self.csv_path = csv_path or f"csi_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.ser = None
        self.buffer = deque()
        self.last_inference_time = 0.0
        self.start_time = time.time()

        self.calibration_features = []
        self.calibrated = False
        self.baseline = None

        self.last_valid_state = "warming_up"
        self.last_valid_state_time = 0.0

        self.csv_file = None
        self.csv_writer = None

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.2)
        file_exists = os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["pc_time", "count", "rssi", "length", "mac", "esp_time"])
            self.csv_file.flush()

    def parse_line(self, line):
        parts = line.strip().split(",")
        if len(parts) != 6 or parts[0] != "CSI":
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
        self.csv_writer.writerow([
            row["pc_time"],
            row["count"],
            row["rssi"],
            row["length"],
            row["mac"],
            row["esp_time"]
        ])

    def cleanup_buffer(self, now_ts):
        while self.buffer and (now_ts - self.buffer[0]["pc_time"] > self.window_seconds):
            self.buffer.popleft()

    def compute_features(self):
        if len(self.buffer) < self.min_packets:
            return None

        rssi = np.array([x["rssi"] for x in self.buffer], dtype=float)
        if len(rssi) < 2:
            return None

        return {
            "packet_count": len(rssi),
            "rssi_mean": float(np.mean(rssi)),
            "rssi_std": float(np.std(rssi)),
            "rssi_min": float(np.min(rssi)),
            "rssi_max": float(np.max(rssi)),
            "rssi_range": float(np.max(rssi) - np.min(rssi)),
            "rssi_mad": float(np.mean(np.abs(np.diff(rssi))))
        }

    def update_calibration(self, features):
        if features is not None:
            self.calibration_features.append(features)

        elapsed = time.time() - self.start_time
        if elapsed >= self.calibration_seconds and len(self.calibration_features) >= self.min_calibration_windows:
            stds = [x["rssi_std"] for x in self.calibration_features]
            ranges = [x["rssi_range"] for x in self.calibration_features]
            mads = [x["rssi_mad"] for x in self.calibration_features]

            self.baseline = {
                "std_mean": float(np.mean(stds)),
                "std_p95": float(np.percentile(stds, 95)),
                "range_mean": float(np.mean(ranges)),
                "range_p95": float(np.percentile(ranges, 95)),
                "mad_mean": float(np.mean(mads)),
                "mad_p95": float(np.percentile(mads, 95)),
            }
            self.calibrated = True

    def infer_state(self, features):
        now_ts = time.time()

        if features is None:
            if now_ts - self.last_valid_state_time <= self.hold_seconds:
                return self.last_valid_state
            return "no_signal"

        if not self.calibrated or self.baseline is None:
            return "calibrating"

        std_th = max(self.baseline["std_p95"] * 1.25, self.baseline["std_mean"] + 0.4)
        range_th = max(self.baseline["range_p95"] * 1.20, self.baseline["range_mean"] + 1.0)
        mad_th = max(self.baseline["mad_p95"] * 1.20, self.baseline["mad_mean"] + 0.3)

        votes = 0
        if features["rssi_std"] > std_th:
            votes += 1
        if features["rssi_range"] > range_th:
            votes += 1
        if features["rssi_mad"] > mad_th:
            votes += 1

        state = "motion" if votes >= 2 else "stable"
        self.last_valid_state = state
        self.last_valid_state_time = now_ts
        return state

    def run(self):
        self.connect()
        print(f"Listening on {self.port} at {self.baudrate} baud")
        print(f"Saving raw data to {self.csv_path}")
        print(f"Window={self.window_seconds}s Step={self.step_seconds}s Calibration={self.calibration_seconds}s")
        print("Keep the room empty and still during calibration")

        flush_counter = 0

        try:
            while True:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
                now_ts = time.time()

                if raw:
                    row = self.parse_line(raw)
                    if row is not None:
                        self.buffer.append(row)
                        self.append_csv(row)
                        flush_counter += 1
                        if flush_counter >= 20:
                            self.csv_file.flush()
                            flush_counter = 0

                self.cleanup_buffer(now_ts)

                if now_ts - self.last_inference_time >= self.step_seconds:
                    features = self.compute_features()

                    if not self.calibrated:
                        self.update_calibration(features)
                        if self.calibrated:
                            print(
                                "Calibration done "
                                f"std_p95={self.baseline['std_p95']:.2f} "
                                f"range_p95={self.baseline['range_p95']:.2f} "
                                f"mad_p95={self.baseline['mad_p95']:.2f}"
                            )
                        else:
                            elapsed = time.time() - self.start_time
                            packet_info = 0 if features is None else features["packet_count"]
                            print(f"state=calibrating elapsed={elapsed:.1f}s packets={packet_info}")
                    else:
                        state = self.infer_state(features)
                        if features is None:
                            print(f"state={state}")
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
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
            if self.ser:
                self.ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--calibration", type=float, default=8.0)
    parser.add_argument("--min-packets", type=int, default=5)
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--min-calibration-windows", type=int, default=3)
    args = parser.parse_args()

    monitor = CSIRealtimeMonitorV3(
        port=args.port,
        baudrate=args.baudrate,
        csv_path=args.csv,
        window_seconds=args.window,
        step_seconds=args.step,
        calibration_seconds=args.calibration,
        min_packets=args.min_packets,
        hold_seconds=args.hold,
        min_calibration_windows=args.min_calibration_windows,
    )
    monitor.run()
