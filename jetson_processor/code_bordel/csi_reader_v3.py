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

        log_dir = "Log"
        os.makedirs(log_dir, exist_ok=True)

        default_csv = os.path.join(
            log_dir,
            f"csi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        self.output_csv = output_csv or default_csv
        self.window = deque(maxlen=30)
        self.packet_count = 0
        self.ignored_raw_count = 0
        self.ignored_other_count = 0
        self.ser = None

    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)

        output_dir = os.path.dirname(self.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

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

                if raw.startswith("CSI_RAW,"):
                    self.ignored_raw_count += 1
                    continue

                if not raw.startswith("CSI,"):
                    self.ignored_other_count += 1
                    continue

                row = self.parse_line(raw)
                if row is None:
                    self.ignored_other_count += 1
                    continue

                self.packet_count += 1
                self.window.append(row["rssi"])
                self.append_csv(row)

                motion_score = self.compute_motion_score()

                if motion_score is None:
                    print(
                        f"count={row['count']} "
                        f"rssi={row['rssi']} "
                        f"len={row['len']} "
                        f"mac={row['mac']}"
                    )
                else:
                    state = "motion" if motion_score > 1.5 else "stable"
                    print(
                        f"count={row['count']} "
                        f"rssi={row['rssi']} "
                        f"len={row['len']} "
                        f"mac={row['mac']} "
                        f"std={motion_score:.2f} "
                        f"state={state}"
                    )

        except KeyboardInterrupt:
            print("\nStopping reader")
            print(f"Saved CSI packets: {self.packet_count}")
            print(f"Ignored CSI_RAW lines: {self.ignored_raw_count}")
            print(f"Ignored other/invalid lines: {self.ignored_other_count}")

        finally:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                print("Serial port closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    reader = CSIReader(
        port=args.port,
        baudrate=args.baudrate,
        output_csv=args.csv
    )
    reader.run()
