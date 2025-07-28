import os
import numpy as np
import pandas as pd
import mne

# === PARAMETER ===
fs = 256
seg_dur = 5
default_sph = 120  # 2 menit
default_sop = 60   # 1 menit

# === ROOT FOLDER ===
input_root = r'D:\eeg\v2.0.3\edf'
output_root = r'E:\eeg'

def process_file(edf_file, csv_file, out_dir):
    try:
        print(f"🔍 Processing: {edf_file}")
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        data, times = raw.get_data(return_times=True)

        df = pd.read_csv(csv_file, sep=',', comment='#')
        df.columns = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
        df_seizure = df[df['label'] != 'bckg']
        if df_seizure.empty:
            print(f"[SKIP] No seizure found in: {edf_file}")
            return

        seizure_start = df_seizure['start_time'].min()
        total_window = default_sph + default_sop
        if seizure_start < total_window:
            total_window = seizure_start
            sph = int(total_window * 2 / 3)
            sop = int(total_window - sph)
            print(f"⚠️  Adjusted SPH/SOP → SPH: {sph}s, SOP: {sop}s")
        else:
            sph = default_sph
            sop = default_sop

        total_samples = int(total_window * fs)
        start_idx = int((seizure_start - total_window) * fs)
        end_idx = int(seizure_start * fs)
        data_window = data[:, start_idx:end_idx]

        samples_per_seg = seg_dur * fs
        num_segments = total_samples // samples_per_seg
        if num_segments < 3:
            print(f"[SKIP] Too few segments ({num_segments}) in: {edf_file}")
            return

        X, y = [], []
        for i in range(num_segments):
            s = i * samples_per_seg
            e = s + samples_per_seg
            X.append(data_window[:, s:e])
            y.append(1 if i >= sph // seg_dur else 0)

        X = np.stack(X)
        y = np.array(y)

        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(edf_file))[0]
        np.save(os.path.join(out_dir, f"{base_name}_X.npy"), X)
        np.save(os.path.join(out_dir, f"{base_name}_y.npy"), y)

        print(f"[OK] Saved {base_name}_X.npy | Segments: {X.shape[0]}")

    except Exception as e:
        print(f"[ERROR] Failed to process {edf_file} → {e}")

def walk_and_process_all():
    for root, _, files in os.walk(input_root):
        edf_files = [f for f in files if f.endswith('.edf')]
        for edf_name in edf_files:
            edf_path = os.path.join(root, edf_name)
            csv_path = edf_path.replace('.edf', '.csv')
            if not os.path.exists(csv_path):
                print(f"[MISS] CSV not found for: {edf_path}")
                continue

            rel_path = os.path.relpath(root, input_root)
            out_dir = os.path.join(output_root, rel_path)

            process_file(edf_path, csv_path, out_dir)

# === RUN ===
walk_and_process_all()
