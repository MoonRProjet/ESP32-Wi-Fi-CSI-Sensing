#!/usr/bin/env python3
import os
import shutil

PROJECT_ROOT = os.path.expanduser("~/Melik_Project")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")

LABELS = {
    "0": "empty",
    "1": "1_static",
    "2": "1_moving",
    "3": "2_static",
    "4": "2_moving",
    "5": "2_mixed",
    "s": "skip"
}

def main():
    csv_files = []
    for name in os.listdir(PROJECT_ROOT):
        full_path = os.path.join(PROJECT_ROOT, name)
        if os.path.isfile(full_path) and name.endswith(".csv"):
            csv_files.append(name)

    csv_files.sort()

    if not csv_files:
        print("No CSV files found in project root.")
        return

    print("CSV files found:")
    for i, name in enumerate(csv_files, 1):
        print(f"{i:02d}. {name}")

    print("\nChoose a label for each file:")
    print("0 = empty")
    print("1 = 1_static")
    print("2 = 1_moving")
    print("3 = 2_static")
    print("4 = 2_moving")
    print("5 = 2_mixed")
    print("s = skip")

    for idx, filename in enumerate(csv_files, 1):
        src = os.path.join(PROJECT_ROOT, filename)

        while True:
            choice = input(f"\n[{idx}/{len(csv_files)}] {filename} -> label: ").strip().lower()
            if choice in LABELS:
                break
            print("Invalid choice. Try again.")

        if choice == "s":
            print("Skipped.")
            continue

        label = LABELS[choice]
        dst_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)

        name, ext = os.path.splitext(filename)
        take_num = 1
        while True:
            new_name = f"{label}__take-{take_num:02d}{ext}"
            dst = os.path.join(dst_dir, new_name)
            if not os.path.exists(dst):
                break
            take_num += 1

        shutil.move(src, dst)
        print(f"Moved to: {dst}")

    print("\nDone.")

if __name__ == "__main__":
    main()
