# 2026_02_04
# script to measure the geometric distribution of errors of SDCs in convolutions

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.experiment_paths import ExperimentPaths
from src.goldens import get_output_shape, load_golden_array


# TODO get this from 1 single golden file
def get_original_shapes(golden_paths):
    original_shapes_dict = {}
    for key, path in golden_paths.items():
        try:
            data = np.load(path)
            print(f"{key:<30} | {data.shape}")
            shape = data.shape
            # ignore first dimension, as it is only the num_samples, not output of the model itself
            original_shapes_dict[key] = shape[1:]
        except FileNotFoundError:
            print(f"{key:<30} | File not found at path.")
        except Exception as e:
            print(f"{key:<30} | Error: {e}")
    return original_shapes_dict


def get_nd_indices(row):
    # unravel_index returns a tuple of coordinates
    # We convert it to a list or tuple to store it in a single cell
    try:
        coords = np.unravel_index(row["index"], row["original_shape"])
    except:
        print(
            f"crazy index: {row['index']} ... original_shape: {row['original_shape']}"
        )
        breakpoint()
    return tuple(int(c) for c in coords)


def classify_sdc_pattern(one_sdc_df: pd.DataFrame) -> str:
    """Given the rows of sdc_details_df of a single SDC,
    returns the SDC pattern classification:
    "single": only one wrong element in the output
    "row": all wrong elements' positions are in a single column or row
    "semi-row": >90% of wrong elements' positions are in a single column or row
    "square": the wrong elements' positions can only be covered with a big rectangle or square
    """
    SEMI_ROW_THRESHOLD = 0.75
    sdc_pattern = "square"  # Default catch-all
    coords = np.array(one_sdc_df["original_indexes"].tolist())
    num_points = len(coords)

    # 1. "single" check
    if num_points == 1:
        sdc_pattern = "single"
    else:
        # Calculate unique values per dimension
        unique_counts = [len(np.unique(coords[:, i])) for i in range(coords.shape[1])]
        dimensions_that_varied = sum(1 for count in unique_counts if count > 1)

        # 2. "row" check: variation in exactly one dimension
        if dimensions_that_varied == 1:
            sdc_pattern = "row"

        # 3. "semi-row" check: > 90% of points share coordinates in all but one dimension
        else:
            for axis in range(coords.shape[1]):
                # Isolate all dimensions except the current one
                other_axes = [i for i in range(coords.shape[1]) if i != axis]
                other_coords = coords[:, other_axes]

                # Find the most frequent coordinate pattern in the other dimensions
                # 'return_counts=True' gives us the frequency of each unique row
                _, counts = np.unique(other_coords, axis=0, return_counts=True)
                max_alignment = np.max(counts)

                if (max_alignment / num_points) >= SEMI_ROW_THRESHOLD:
                    sdc_pattern = "semi-row"
                    break  # Stop looking if we found a semi-row alignment

    return sdc_pattern


def print_sdc_class_percentages(sdc_geometric_dist_df: pd.DataFrame):

    # 0. keep only 1 row per SDC
    df = sdc_geometric_dist_df.drop_duplicates(subset=["sdc_id"])

    # 1. Get absolute counts of unique SDC events
    counts = df.groupby("model_name")["sdc_class"].value_counts().unstack().fillna(0)

    # 2. Get percentages of unique SDC events
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # 3. Combine with multi-index columns
    combined = pd.concat(
        [counts, percentages], axis=1, keys=["Event_Count", "Percentage"]
    )

    print("SDC Event Classification Summary (Unique Events):")
    print(combined.round(2))


def check_original_indexes(one_sdc_df: pd.DataFrame) -> bool:
    """
    check if one_sdc_df is valid by checking if the expected values
    parsed from the logs match with the values in the golden data
    """
    experiment_name = one_sdc_df.iloc[0].experiment_name
    model_name = one_sdc_df.iloc[0].model_name

    # for each SDC element, check if the expected is there at the golden
    all_index_match = True
    for i in range(len(one_sdc_df)):
        row = one_sdc_df.iloc[i]
        golden_array = load_golden_array(experiment_name, model_name)
        golden_output = golden_array[row["image_index"]]
        original_indexes = row["original_indexes"]
        try:
            golden_value = golden_output[original_indexes]
        except:
            print(f"model {model_name}, original_indexes: {original_indexes}")
            print(f"DEBUG one_sdc_df: {one_sdc_df}")
            breakpoint()
        # if golden_value != row["expected"] and row["model_name"] != "ssd_mobilenetv2_coral":
        if golden_value != row["expected"]:
            print(
                f"Mismatch on {model_name} ...Real golden value: {golden_value}, but we have {row['expected']}"
            )
            all_index_match = False
            print(
                f"DEBUG one_sdc_df.columns: {one_sdc_df.columns}, one_sdc_df: {one_sdc_df}"
            )
            breakpoint()
    return all_index_match


def geometric_distribution_analysis(sdc_details_df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO description

    Returns:
    sdc_geometric_dist_df: pd.DataFrame,
    with 1 row per SDC wrong element, with column 'sdc_class' (single, row, square),
    as well as columns 'original_shape' and 'original_indexes
    """
    sdc_details_df_len = len(sdc_details_df)
    processed_SDCs = []  # 1. Initialize a list to hold processed groups
    # for each SDC:
    grouped_sdcs = sdc_details_df.groupby("sdc_id")
    for sdc_id, one_sdc_df in tqdm(
        grouped_sdcs, total=len(grouped_sdcs), desc="Geometric distribution analysis"
    ):
        # get model_name
        model_name = one_sdc_df.iloc[0].model_name
        experiment_name = one_sdc_df.iloc[0].experiment_name

        # skip analysis for some specific cases
        if (
            model_name == "conv_10x10_coral"
            or model_name == "base_vit_8"
            or model_name == "conv_3x3_on_1024x1024_int8_coral"
            # CNAO and PARTREC have different shape. CNAO is just bounding boxes, which is what Bruno's code actually uses
            or (
                model_name == "ssd_mobilenetv2_coral"
                and experiment_name == "CNAO_2026_01"
            )
        ):
            # if skip, then fill in None values and etc
            one_sdc_df.loc[:, "sdc_class"] = ["analysis_skipped"] * len(one_sdc_df)
            if not "original_shape" in one_sdc_df.columns:
                one_sdc_df.loc[:, "original_shape"] = [None] * len(one_sdc_df)
            if not "original_indexes" in one_sdc_df.columns:
                one_sdc_df["original_indexes"] = [None] * len(one_sdc_df)
        else:
            # get original_shape
            shape = get_output_shape(
                experiment_name=experiment_name, model_name=model_name
            )
            one_sdc_df.loc[:, "original_shape"] = [shape] * len(one_sdc_df)
            # get the original indexes,
            if not "original_indexes" in one_sdc_df.columns:
                one_sdc_df["original_indexes"] = one_sdc_df.apply(
                    get_nd_indices, axis=1
                )
            # check if the golden matches the expected values
            assert check_original_indexes(one_sdc_df)
            # then classify: single, etc
            sdc_class = classify_sdc_pattern(one_sdc_df)
            one_sdc_df.loc[:, "sdc_class"] = [sdc_class] * len(one_sdc_df)
        processed_SDCs.append(one_sdc_df)

    sdc_geometric_dist_df = pd.concat(processed_SDCs, ignore_index=True)
    assert len(sdc_geometric_dist_df) == sdc_details_df_len

    print_sdc_class_percentages(sdc_geometric_dist_df)

    return sdc_geometric_dist_df


if __name__ == "__main__":
    experiment_name = "NSREC26"
    experiment_paths = ExperimentPaths(experiment_name)
    # load SDC details df
    sdc_details_df = pd.read_csv(experiment_paths.sdc_details_csv)

    sdc_geometric_dist_df = geometric_distribution_analysis(sdc_details_df)

    # save in an CSV
    output_csv_filepath = experiment_paths.results_folderpath / "sdc_geometric_dist.csv"
    sdc_geometric_dist_df.to_csv(output_csv_filepath, index=False)
    print(f"Saved {output_csv_filepath}")
