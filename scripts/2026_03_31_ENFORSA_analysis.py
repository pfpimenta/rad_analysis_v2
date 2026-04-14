import argparse
from typing import List
import numpy as np
from src.experiment_paths import ExperimentPaths
import pandas as pd
from src.goldens import load_golden_array
from src.sdc_processing import create_sdc_df
import hashlib
from src.geometric_distribution_analysis import geometric_distribution_analysis
from tqdm import tqdm
from src.plot import generate_all_plots

EXPERIMENT_NAME = "2026_03_31_ENFORSA"
CONV_20_NAME = "depthwise_conv_2d_1_1024_1024_3_20_20_3_1"
CONV_40_NAME = "simple_conv_2d_1_1024_1024_1_40_40_1_1"


def load_enforsa_goldens():
    goldens = {}
    goldens["conv20x20"] = load_golden_array(
        experiment_name=EXPERIMENT_NAME, model_name=CONV_20_NAME
    )[0]
    goldens["conv40x40"] = load_golden_array(
        experiment_name=EXPERIMENT_NAME, model_name=CONV_40_NAME
    )[0]
    return goldens


def load_enforsa_output(fault_tag: int) -> np.ndarray:
    """Output shape: (Batch, Width, Height, Depth)"""
    enforsa_paths = ExperimentPaths(experiment_name=EXPERIMENT_NAME)
    # TODO refactor so we have both before and after im2col
    outputs_folderpath = enforsa_paths.experiment_folderpath / "conv_out_samples_sw"
    # outputs_folderpath = enforsa_paths.experiment_folderpath / "conv_out_samples_sw_after_im2col"
    output_filepath = outputs_folderpath / f"fault_{fault_tag}.npz"
    corrupted_output_array = np.load(output_filepath)['data']
    return corrupted_output_array


def get_sdc_details(
    fault_row: pd.Series,
    corrupted_output_array: np.ndarray,
    golden_output_array: np.ndarray,
) -> List[dict]:
    row_hash = hashlib.md5(str(fault_row).encode()).hexdigest()
    sdc_id = f"{int(fault_row.fault_tag)}_conv40_{row_hash}"
    target_val = fault_row.get('target')
    # Convert to int only if it's not None
    fault_type = int(target_val) if target_val is not None else None
    # 1. Find the indices where they differ
    diff_indices = np.where(corrupted_output_array != golden_output_array)
    # 2. Extract the values using those indices
    expected_values = golden_output_array[diff_indices]
    received_values = corrupted_output_array[diff_indices]
    # get output shape
    assert corrupted_output_array.shape == golden_output_array.shape
    original_shape = golden_output_array.shape
    sdc_details_list = []
    for idx, exp, rec in zip(
        np.transpose(diff_indices), expected_values, received_values
    ):
        original_indexes = tuple(int(i) for i in idx)
        sdc_wrong_element_info = {
            "filepath": None,
            "log_start_timestamp": None,
            "model_name": CONV_40_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "device": "rasp4-coral",
            "sdc_id": sdc_id,
            "index": None,
            "expected": int(exp),
            "received": int(rec),
            "image_index": 0,
            "acc_time_at_sdc": None,
            "diff": int(rec) - int(exp),
            "original_indexes": original_indexes,
            "fault_type": fault_type,
            # "original_shape": original_shape,
        }
        sdc_details_list.append(sdc_wrong_element_info)
    return sdc_details_list

def create_enforsa_sdc_details_df(faults_df: pd.DataFrame, goldens: dict) -> pd.DataFrame: 
    ### create SDC details dataframe
    sdc_faults_df = faults_df[faults_df["sdc"] == 1]
    all_sdc_details_list = []
    for idx, row in tqdm(sdc_faults_df.iterrows(), total=len(sdc_faults_df), desc="Processing SDCs"):
        # check if non-SDC outputs match goldens
        corrupted_output_array = load_enforsa_output(row.fault_tag)
        # conv40x40 == simple_conv1k == 2d 40x40 == simple_conv_2d_1_1024_1024_1_40_40_1_1
        assert corrupted_output_array.shape == goldens["conv40x40"].shape
        sdc_details_list = get_sdc_details(
            row, corrupted_output_array, goldens["conv40x40"]
        )
        all_sdc_details_list += sdc_details_list
    sdc_details_df = pd.DataFrame(all_sdc_details_list)
    sdc_details_df_len = len(sdc_details_df)

    # get geometric classification of SDCs (single, row, square)
    sdc_geometric_dist_df = geometric_distribution_analysis(sdc_details_df)
    sdc_geometric_dist_df = sdc_geometric_dist_df[["sdc_id", "original_indexes", "sdc_class", "original_shape"]]
    sdc_details_df = sdc_details_df.merge(
        sdc_geometric_dist_df, on=["sdc_id", "original_indexes"], how="left"
    )
    # be sure that there are not extra rows
    assert len(sdc_details_df) == sdc_details_df_len

    return sdc_details_df

def analysis_2026_03_31_enforsa(plot_only: bool = False):
    
    if not plot_only:

        enforsa_paths = ExperimentPaths(experiment_name=EXPERIMENT_NAME)
        ### load data
        # load log CSV\
        # TODO refactor so we have both before and after im2col
        log_csv_path = enforsa_paths.experiment_folderpath / "trace.tsv"
        # log_csv_path = enforsa_paths.experiment_folderpath / "log_sw.csv"
        # log_csv_path = enforsa_paths.experiment_folderpath / "log_sw_after_im2col.csv"
        faults_df = pd.read_csv(log_csv_path, sep="\t", comment="#")
        # load golden
        goldens = load_enforsa_goldens()
        
        sdc_details_df = create_enforsa_sdc_details_df(faults_df, goldens)

        # create SDCs df
        sdc_df = create_sdc_df(sdc_details_df, EXPERIMENT_NAME)

        # save CSVs
        sdc_details_df.to_csv(enforsa_paths.sdc_details_csv, index=False)
        print(f"Saved {enforsa_paths.sdc_details_csv}")
        sdc_df.to_csv(enforsa_paths.sdcs_csv, index=False)
        print(f"Saved {enforsa_paths.sdcs_csv}")

    # generate all plots
    # TODO fix plots, ta meio bugado
    generate_all_plots(experiment_name=EXPERIMENT_NAME)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENFORSA Radiation Analysis")
    
    # Add the flag. 'action="store_true"' means it's False by default 
    # and becomes True if you include it in the command line.
    parser.add_argument(
        "--plot-only", 
        action="store_true", 
        help="Skip SDC computation and just generate plots from existing CSVs"
    )
    
    args = parser.parse_args()
    analysis_2026_03_31_enforsa(plot_only=args.plot_only)
