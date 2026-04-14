"""
Utilities for parsing raw experiment log files and extracting metadata from log file paths.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.experiment_paths import ExperimentPaths


def filter_crazy_sdcs(sdc_details: List[List[dict]]) -> Tuple[int, List[List[dict]]]:
    """removes inconsistent SDCs from the parsed ones in sdc_details"""
    MAX_VALUE_ERRORS_PER_SDC = 500
    # 1. Filter out SDCs that exceed the error threshold
    # We use a list comprehension to avoid mutation issues during iteration
    filtered_details = [
        sdc for sdc in sdc_details if len(sdc) <= MAX_VALUE_ERRORS_PER_SDC
    ]
    # 2. Drop duplicates
    unique_sdcs = []
    seen_signatures = set()
    for sdc in filtered_details:
        # Create a unique "signature" for the SDC by converting the list of dicts
        # into a sorted tuple of items. Sorting ensures that if the dicts
        # are in a different order but have the same content, they are still flagged.
        signature = tuple(tuple(sorted(d.items())) for d in sdc)
        if signature not in seen_signatures:
            unique_sdcs.append(sdc)
            seen_signatures.add(signature)
    # Update count based on the final cleaned list
    new_sdc_count = len(unique_sdcs)
    return new_sdc_count, unique_sdcs


def get_stats_from_log(log_filepath: str) -> tuple:
    last_acc = None
    running_acc = None  # Track the most recent AccTime seen in the log
    sdc_count = 0
    sdc_details = []

    float_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    err_pattern = re.compile(
        rf"index:(\d+)\s+e:({float_pattern})\s+r:({float_pattern})"
    )
    acc_pattern = re.compile(r"AccTime:(\d+\.?\d*)")
    image_index_pattern = re.compile(r"image_index:(\d+)")

    try:
        with open(log_filepath, "r") as f:
            for line in f:
                # 1. Update the running_acc whenever we pass an #IT line
                if "AccTime:" in line:
                    match = acc_pattern.search(line)
                    if match:
                        running_acc = float(match.group(1))
                        last_acc = running_acc  # Also update the overall last_acc

                # 2. Check for the start of an SDC event
                if "SDC" in line:
                    img_match = image_index_pattern.search(line)
                    current_image_index = int(img_match.group(1)) if img_match else None

                    # Start a new list for this SDC
                    sdc_details.append([])

                # 3. Check for the error index lines
                if "index:" in line and sdc_details:
                    match = err_pattern.search(line)
                    if match:
                        detail = {
                            "index": int(match.group(1)),
                            "expected": float(match.group(2)),
                            "received": float(match.group(3)),
                            "image_index": current_image_index,
                            "acc_time_at_sdc": running_acc,
                        }
                        # Append to the latest SDC event's list
                        sdc_details[-1].append(detail)

    except Exception as e:
        print(f"Error reading {log_filepath}: {e}")

    sdc_count, sdc_details = filter_crazy_sdcs(sdc_details)

    return last_acc, sdc_count, sdc_details


def extract_info_from_log_path(path_string: str) -> Tuple[pd.Timestamp, str, str, str]:
    """
    Returns timestamp, model_name, experiment_name, and device from a log filepath.
    Example input path_string:
    /home/pfpimenta/rad_analysis/data/PARTREC_2026_01/logs/rasp4-coral/2026_01_21_23_44_01_run_ssd_mobilenetv2_coral_ECC_OFF_rasp4-coral.log
    """
    path_obj = Path(path_string)
    filename = path_obj.name

    # 1. Extract Timestamp (YYYY_MM_DD_HH_MM_SS)
    ts_match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", filename)
    timestamp = (
        pd.to_datetime(ts_match.group(1), format="%Y_%m_%d_%H_%M_%S")
        if ts_match
        else None
    )

    # 2. Extract Model Name
    model_match = re.search(r"(?<=run_)(.*?)(?=_ECC_OFF)", filename)
    model_name = model_match.group(1) if model_match else None

    # 3. Extract experiment_name and Device via Path Hierarchy
    # Structure: [Experiment]/logs/[Device]/[File]
    try:
        device = path_obj.parent.name  # 'rasp4-coral'
        experiment_name = path_obj.parents[2].name  # 'PARTREC_2026_01'
    except IndexError:
        device, experiment_name = None, None

    return timestamp, model_name, experiment_name, device


def create_logs_df(experiment_name: str, ignore_logs=[]):
    # TODO refactor: put ignore_logs on a experiment config json ?

    experiment_paths = ExperimentPaths(experiment_name)

    ### logs_df
    # get logs filepath list
    logs_filepaths = []
    for log_folderpath in experiment_paths.logs_folderpaths:
        log_paths = [str(f) for f in Path(log_folderpath).iterdir() if f.is_file()]
        logs_filepaths += log_paths
    # ignore crazy logs
    logs_filepaths = [
        path
        for path in logs_filepaths
        if not any(ignore in path for ignore in ignore_logs)
    ]
    # create logs DataFrame
    logs_df = pd.DataFrame(logs_filepaths, columns=["filepath"])
    # extract start timestamp, model_name, experiment_name, and device from filename
    logs_df[["log_start_timestamp", "model_name", "experiment_name", "device"]] = (
        logs_df["filepath"].apply(lambda x: pd.Series(extract_info_from_log_path(x)))
    )
    # extract last AccTime and amount of SDCs from log content
    logs_df[["last_acc_time", "sdc_count", "error_info"]] = logs_df["filepath"].apply(
        lambda x: pd.Series(get_stats_from_log(x))
    )
    # create results folder if it does not exist yet
    Path(experiment_paths.logs_info_csv).parent.mkdir(parents=True, exist_ok=True)
    # save logs_df into CSV
    logs_df.to_csv(
        experiment_paths.logs_info_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {experiment_paths.logs_info_csv}")

    return logs_df
