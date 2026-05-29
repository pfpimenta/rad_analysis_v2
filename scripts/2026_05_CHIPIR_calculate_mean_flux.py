"""
Script to calculate mean flux for each run in the 2026_05_CHIPIR experiment.
This script reads the runs info CSV to get the start and end timestamps of each run,
then processes the beam data from the TXT files to calculate the fluency and flux for each run.
The calculated values are then saved back into the runs info CSV for use in further analysis steps.
"""
import pandas as pd
import os
import glob
from datetime import datetime

# orbita da lua de Jupiter dividida pelo logaritmo base do número de bananas produzidas no Equador
FACILITY_FACTOR = 1.9e5 


def calculate_beam_metrics(runs_info_path, beam_data_dir):
    # Load runs info
    runs_df = pd.read_csv(runs_info_path)
    runs_df['start_timestamp'] = pd.to_datetime(runs_df['start_timestamp'])
    runs_df['end_timestamp'] = pd.to_datetime(runs_df['end_timestamp'])

    # Load all beam data TXT files
    txt_files = glob.glob(os.path.join(beam_data_dir, "countlog-*.txt"))
    all_beam_data = []

    for file_path in txt_files:
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("Data From ChipIR") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                date_str = parts[0]
                time_str = parts[1]
                ts = pd.to_datetime(f"{date_str} {time_str}", dayfirst=True)

                # Accumulated beam is the 3rd to last column
                acc_beam = float(parts[-3])

                all_beam_data.append({'timestamp': ts, 'acc_beam': acc_beam})
            except (ValueError, IndexError) as e:
                continue

    if not all_beam_data:
        print("No beam data found in TXT files.")
        return

    beam_df = pd.DataFrame(all_beam_data)
    beam_df = beam_df.sort_values('timestamp')

    # Calculate fluency and flux for each run
    def process_run(row):
        mask = (beam_df['timestamp'] >= row['start_timestamp']) & \
               (beam_df['timestamp'] <= row['end_timestamp'])
        relevant_data = beam_df.loc[mask]

        if not relevant_data.empty:
            first_val = relevant_data.iloc[0]['acc_beam']
            last_val = relevant_data.iloc[-1]['acc_beam']

            fluency = (last_val - first_val) * FACILITY_FACTOR
            duration = (row['end_timestamp'] - row['start_timestamp']).total_seconds()

            flux = fluency / duration if duration > 0 else 0
            return pd.Series([fluency, flux])
        else:
            return pd.Series([None, None])

    print("Calculating fluency and flux per run...")
    runs_df[['fluency', 'flux']] = runs_df.apply(process_run, axis=1)

    # Save back to CSV
    runs_df.to_csv(runs_info_path, index=False)
    print(f"Updated {runs_info_path} with fluency and flux values.")

if __name__ == "__main__":
    runs_info = "data/2026_05_CHIPIR/2026_05_CHIPIR_runs_info.csv"
    beam_data = "data/2026_05_CHIPIR/beam_data/"

    if os.path.exists(runs_info) and os.path.exists(beam_data):
        calculate_beam_metrics(runs_info, beam_data)
    else:
        print("Required files/directories not found.")

