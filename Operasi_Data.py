import os
import numpy as np
import pandas as pd
import mne
import shutil

# === PARAMETER ===
fs = 256
seg_dur = 5
default_sph = 120  # baris ini: SPH default = 120 detik (2 menit)
default_sop = 60  # baris ini: SPH default = 60 detik (1 menit)

# === ROOT FOLDER ===
input_root = r'D:\eeg\v2.0.3\edf'
output_root = r'E:\eeg'

def process_file(edf_file, csv_file, out_dir):
    try:
        edf_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.edf')])
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

        raws = []
        annotations = []

        for edf_file, csv_file in zip(edf_files, csv_files):
            edf_path = os.path.join(folder_path, edf_file)
            csv_path = os.path.join(folder_path, csv_file)

            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raws.append(raw)

            df = pd.read_csv(csv_path, sep=',', comment='#')
            df.columns = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
            annotations.append(df)

        if not raws or not annotations:
            print(f"[SKIP] No valid data in folder: {folder_path}")
            return
        
        combined_raw = mne.concatenate_raws(raws)
        data, times = combined_raw.get_data(return_times=True)
        combined_df = pd.concat(annotations, ignore_index=True)
        combined_df.to_csv(os.path.join(out_dir, "CSV_Gabungan_Token.csv"), index=False)

        df_seizure = combined_df[combined_df['label'] != 'bckg']
        if df_seizure.empty:
            print(f"[SKIP] No seizure in: {folder_path}")
            return
    
        seizure_start = df_seizure['start_time'].min()
        total_window = default_sph + default_sop
        if seizure_start < total_window:
            total_window = seizure_start
            sph = int(total_window * 2 / 3)  #Jika seizure  terjadi < 180 detik (120+60), maka SPH & SOP akan disesuaikan.
            sop = int(total_window - sph) #Rasio penyesuaiannya: SPH = 2/3, SOP = 1/3 dari total_window.
            # print(f"Adjusted SPH/SOP → SPH: {sph}s, SOP: {sop}s")
        else:
            sph = default_sph
            sop = default_sop

        total_samples = int(total_window * fs)
        start_idx = int((seizure_start - total_window) * fs)
        end_idx = int(seizure_start * fs)
        data_window = data[:, start_idx:end_idx]

        samples_per_seg = seg_dur * fs
        num_segments = total_samples // samples_per_seg
        if num_segments < 3:  #Line ini yang ngasih syarat paling minimal itu 3 segment kalau ga sampe 3 segment ya di skip
            print(f"[SKIP] Too few segments in: {edf_file}")
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
        np.save(os.path.join(out_dir, "X.npy"), X)
        np.save(os.path.join(out_dir, "y.npy"), y)

        print(f"[OK] Saved combined X.npy | Segments: {X.shape[0]}")

    except Exception as e:
        print(f"[ERROR] Failed to process folder {folder_path} → {e}") Segments: {X.shape[0]}")


def walk_and_process_all():
    for root, _, files in os.walk(input_root):
        edf_files = [f for f in files if f.endswith('.edf')]
        csv_files = [f for f in files if f.endswith('.csv')]
        if edf_files and csv_files:
            rel_path = os.path.relpath(root, input_root)
            out_dir = os.path.join(output_root, rel_path)
            process_folder(root, out_dir)
            
# === JALANKAN ===
walk_and_process_all()
