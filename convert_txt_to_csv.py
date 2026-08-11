"""
convert_txt_to_csv.py
Converts the steering-dataset's data.txt (space-separated, no header,
lines like "99576.jpg -66.250000") into the CSV format dataset.py expects
(image_path,steering_angle).

Usage:
    python convert_txt_to_csv.py --txt data.txt --out driving_log.csv
"""

import argparse
import csv


def convert(txt_path, csv_path):
    n_written = 0
    n_skipped = 0
    with open(txt_path, "r") as f_in, open(csv_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["image_path", "steering_angle"])

        for line_num, raw_line in enumerate(f_in, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                print(f"  skipping malformed line {line_num}: {raw_line!r}")
                n_skipped += 1
                continue

            img_name, angle = parts
            try:
                float(angle)
            except ValueError:
                print(f"  skipping non-numeric angle on line {line_num}: {raw_line!r}")
                n_skipped += 1
                continue

            writer.writerow([img_name, angle])
            n_written += 1

    print(f"Wrote {n_written} rows to {csv_path} ({n_skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, required=True, help="Path to data.txt")
    parser.add_argument("--out", type=str, default="driving_log.csv", help="Output CSV path")
    args = parser.parse_args()

    convert(args.txt, args.out)
