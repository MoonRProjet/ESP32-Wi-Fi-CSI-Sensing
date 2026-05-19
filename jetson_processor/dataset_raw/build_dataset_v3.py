#!/usr/bin/env python3

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

LABELS = {
    "empty": 0,
    "1_static": 1,
    "1_moving": 2,
    "2_static": 3,
    "2_moving": 4,
    "2_mixed": 5,
}

DEFAULT_TARGET_MAC = "0EA06D7AA587"


def infer_label_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part in LABELS:
            return part

    filename = os.path.basename(path)
    for label in LABELS:
        if filename.startswith(label + "__") or filename.startswith(label + "_"):
            return label

    return None


def extract_take_id(path):
    filename = os.path.splitext(os.path.basename(path))[0]

    m = re.search(r"take[-_]?(\d+)", filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)$", filename)
    if m:
        return int(m.group(1))

    return -1


def looks_like_shifted_format(df):
    if len(df) == 0:
        return False

    row = df.iloc[0].astype(str)

    pc_time_val = row.get("pc_time", "")
    count_val = row.get("count", "")
    rssi_val = row.get("rssi", "")
    len_val = row.get("len", "")
    mac_val = row.get("mac", "")
    esp_time_val = row.get("esp_time", "")

    mac_pattern = r"^[0-9A-Fa-f]{12}$"
    timestamp_pattern = r"^\d{4}-\d{2}-\d{2}T"

    return (
        re.match(r"^\d+(\.0+)?$", str(pc_time_val)) is not None and
        re.match(r"^\d+(\.\d+)?$", str(count_val)) is not None and
        re.match(r"^\d+(\.0+)?$", str(rssi_val)) is not None and
        re.match(mac_pattern, str(len_val)) is not None and
        re.match(timestamp_pattern, str(mac_val)) is not None and
        re.match(r"^-?\d+(\.\d+)?$", str(esp_time_val)) is not None
    )


def normalize_csv(path):
    df = pd.read_csv(path)

    required_cols = {"pc_time", "count", "rssi", "len", "mac", "esp_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError("%s missing columns: %s" % (path, missing))

    df = df.copy()

    if looks_like_shifted_format(df):
        raw = df.copy()
        fixed = pd.DataFrame({
            "pc_time": raw["mac"],
            "count": raw["pc_time"],
            "rssi": raw["esp_time"],
            "len": raw["rssi"],
            "mac": raw["len"],
            "esp_time": raw["count"],
        })
        fixed["source_format"] = "shifted"
        return fixed

    df["source_format"] = "normal"
    return df


def load_csv(path, target_mac=None):
    df = normalize_csv(path)

    df["pc_time"] = df["pc_time"].astype(str).str.strip()
    df["mac"] = df["mac"].astype(str).str.strip().str.upper()

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df["len"] = pd.to_numeric(df["len"], errors="coerce")
    df["esp_time"] = pd.to_numeric(df["esp_time"], errors="coerce")

    if target_mac:
        df = df[df["mac"] == str(target_mac).upper()]

    if df.empty:
        return df

    df["pc_time_dt"] = pd.to_datetime(df["pc_time"], errors="coerce")
    df = df.dropna(subset=["pc_time_dt", "count", "rssi", "len", "esp_time"])

    if df.empty:
        return df

    df["pc_time_sec"] = df["pc_time_dt"].astype("int64") / 1e9

    df = df.sort_values("pc_time_sec").reset_index(drop=True)

    df = df[(df["rssi"] >= -100) & (df["rssi"] <= 0)]
    df = df[df["len"] > 0]

    return df.reset_index(drop=True)


def compute_window_features(win_df):
    rssi = np.array(win_df["rssi"], dtype=float)
    pkt_len = np.array(win_df["len"], dtype=float)
    pc_time = np.array(win_df["pc_time_sec"], dtype=float)
    count = np.array(win_df["count"], dtype=float)

    if len(rssi) < 2:
        return None

    rssi_diff = np.diff(rssi)
    time_diff = np.diff(pc_time)
    count_diff = np.diff(count)

    duration = pc_time[-1] - pc_time[0]
    packet_rate = float(len(rssi)) / float(duration) if duration > 0 else 0.0

    feats = {
        "packet_count": int(len(rssi)),
        "duration": float(duration),
        "packet_rate": float(packet_rate),

        "rssi_mean": float(np.mean(rssi)),
        "rssi_std": float(np.std(rssi)),
        "rssi_min": float(np.min(rssi)),
        "rssi_max": float(np.max(rssi)),
        "rssi_range": float(np.max(rssi) - np.min(rssi)),
        "rssi_median": float(np.median(rssi)),
        "rssi_q25": float(np.percentile(rssi, 25)),
        "rssi_q75": float(np.percentile(rssi, 75)),
        "rssi_iqr": float(np.percentile(rssi, 75) - np.percentile(rssi, 25)),

        "rssi_diff_mean": float(np.mean(rssi_diff)),
        "rssi_diff_abs_mean": float(np.mean(np.abs(rssi_diff))),
        "rssi_diff_std": float(np.std(rssi_diff)),
        "rssi_diff_max_abs": float(np.max(np.abs(rssi_diff))),

        "len_mean": float(np.mean(pkt_len)),
        "len_std": float(np.std(pkt_len)),
        "len_min": float(np.min(pkt_len)),
        "len_max": float(np.max(pkt_len)),

        "count_gap_mean": float(np.mean(count_diff)) if len(count_diff) > 0 else 0.0,
        "count_gap_std": float(np.std(count_diff)) if len(count_diff) > 0 else 0.0,
        "count_gap_max": float(np.max(count_diff)) if len(count_diff) > 0 else 0.0,

        "time_gap_mean": float(np.mean(time_diff)) if len(time_diff) > 0 else 0.0,
        "time_gap_std": float(np.std(time_diff)) if len(time_diff) > 0 else 0.0,
        "time_gap_max": float(np.max(time_diff)) if len(time_diff) > 0 else 0.0,
    }

    return feats


def build_windows(df, window_seconds=5.0, step_seconds=2.5, min_packets=10):
    rows = []

    if df.empty:
        return rows

    start_time = float(df["pc_time_sec"].iloc[0])
    end_time = float(df["pc_time_sec"].iloc[-1])

    t = start_time
    while t + window_seconds <= end_time:
        mask = (df["pc_time_sec"] >= t) & (df["pc_time_sec"] < t + window_seconds)
        win_df = df.loc[mask]

        if len(win_df) >= min_packets:
            feats = compute_window_features(win_df)
            if feats is not None:
                feats["window_start"] = float(t)
                feats["window_end"] = float(t + window_seconds)
                rows.append(feats)

        t += step_seconds

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build features dataset from raw CSI CSV files")
    parser.add_argument("--input-dir", required=True, help="Root folder containing raw CSV files")
    parser.add_argument("--output-csv", default="features_dataset_v3.csv", help="Output features CSV")
    parser.add_argument("--window", type=float, default=5.0, help="Window size in seconds")
    parser.add_argument("--step", type=float, default=2.5, help="Step size in seconds")
    parser.add_argument("--min-packets", type=int, default=10, help="Minimum packets per window")
    parser.add_argument("--target-mac", default=DEFAULT_TARGET_MAC, help="MAC address to keep")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.csv"), recursive=True))

    if not csv_files:
        print("No CSV files found.")
        return

    all_rows = []

    for path in csv_files:
        base = os.path.basename(path)

        if base.startswith("features_dataset"):
            continue

        label = infer_label_from_path(path)
        if label is None:
            print("Skipping unknown label file: %s" % path)
            continue

        try:
            df = load_csv(path, target_mac=args.target_mac)
        except Exception as e:
            print("Error loading %s: %s" % (path, e))
            continue

        if df.empty:
            print("%s -> label=%s, rows_after_mac_filter=0, windows=0" % (path, label))
            continue

        source_format = "unknown"
        if "source_format" in df.columns and len(df) > 0:
            source_format = str(df["source_format"].iloc[0])

        windows = build_windows(
            df,
            window_seconds=args.window,
            step_seconds=args.step,
            min_packets=args.min_packets
        )

        for w in windows:
            w["label"] = label
            w["label_id"] = LABELS[label]
            w["source_file"] = base
            w["source_path"] = path
            w["take_id"] = extract_take_id(path)
            w["target_mac"] = args.target_mac
            w["source_format"] = source_format
            all_rows.append(w)

        print("%s -> label=%s, format=%s, rows=%d, windows=%d" % (
            path, label, source_format, len(df), len(windows)
        ))

    if not all_rows:
        print("No windows generated.")
        return

    out_df = pd.DataFrame(all_rows)

    ordered_cols = [
        "label", "label_id", "source_file", "source_path", "take_id", "target_mac", "source_format",
        "window_start", "window_end",
        "packet_count", "duration", "packet_rate",
        "rssi_mean", "rssi_std", "rssi_min", "rssi_max", "rssi_range",
        "rssi_median", "rssi_q25", "rssi_q75", "rssi_iqr",
        "rssi_diff_mean", "rssi_diff_abs_mean", "rssi_diff_std", "rssi_diff_max_abs",
        "len_mean", "len_std", "len_min", "len_max",
        "count_gap_mean", "count_gap_std", "count_gap_max",
        "time_gap_mean", "time_gap_std", "time_gap_max"
    ]

    out_df = out_df[ordered_cols]
    out_df.to_csv(args.output_csv, index=False)

    print("")
    print("Saved dataset to: %s" % args.output_csv)
    print("Total windows: %d" % len(out_df))
    print("")
    print("Windows per label:")
    print(out_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
