"""
Script to create the runs info CSV for the 2026_05_CHIPIR experiment.
This script reads the raw beam data from the provided CSV, identifies the ON/OFF periods to determine run start and end times, and creates a structured CSV file that will be used in subsequent analysis steps. The output CSV will have columns for run_id, start_timestamp, end_timestamp, and placeholders
for other relevant information that will be filled in later (e.g., AccTime, model_ran, #SDCs, #DUEs, flux, fluency, SDC_cross_section, DUE_cross_section).
"""

import pandas as pd
import os

def format_chipir_beam_data(input_path, output_path):
    # Load the input CSV
    df = pd.read_csv(input_path)
    
    # Combine Date and Time into a single datetime column
    # The source format seems to be YYYY-MM-DD and HH:MM
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    runs = []
    current_run = {}
    run_id = 1
    
    for _, row in df.iterrows():
        status = str(row['Status']).strip().upper()
        
        if status == 'ON':
            current_run['start_timestamp'] = row['timestamp']
        elif status == 'OFF':
            if 'start_timestamp' in current_run:
                current_run['end_timestamp'] = row['timestamp']
                current_run['run_id'] = run_id
                runs.append(current_run)
                
                # Reset for next run
                current_run = {}
                run_id += 1
            else:
                print(f"Warning: Found OFF at {row['timestamp']} without preceding ON. Skipping.")
    
    # Create the output DataFrame with the required columns
    output_columns = [
        'run_id', 'start_timestamp', 'end_timestamp', 'AccTime', 
        'model_ran', '#SDCs', '#DUEs', 'flux', 'fluency', 
        'SDC_cross_section', 'DUE_cross_section'
    ]
    
    output_df = pd.DataFrame(runs)
    
    # Ensure all columns exist
    for col in output_columns:
        if col not in output_df.columns:
            output_df[col] = None
            
    # Reorder columns
    output_df = output_df[output_columns]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Formatted data saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/2026_05_CHIPIR/beam_data/ChipIR 05_2026 - Beam Register.csv"
    output_file = "data/2026_05_CHIPIR/2026_05_CHIPIR_runs_info.csv"
    
    if os.path.exists(input_file):
        format_chipir_beam_data(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
