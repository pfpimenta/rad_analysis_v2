"""
Methods for processing Single Event Upset (SDC) data into Pandas DataFrames.
"""

import hashlib
import json

import pandas as pd

from src.criticality_analysis import criticality_analysis
from src.geometric_distribution_analysis import geometric_distribution_analysis


def create_sdc_id(row):
    """uses hash and numbered sdc_id to create the final sdc_id"""
    # Create the unique fingerprint data
    data_to_hash = {
        "timestamp": str(row["log_start_timestamp"]),
        "errors": row["error_info"],
    }

    # Generate the MD5 hash
    json_string = json.dumps(data_to_hash, sort_keys=True).encode("utf-8")
    error_hash = hashlib.md5(json_string).hexdigest()

    # Format: EventNo_Model_Hash
    return f"{row['sdc_number']}_{row['model_name']}_{error_hash}"


def create_sdc_df(
    sdc_details_df: pd.DataFrame, experiment_name: str = None
) -> pd.DataFrame:
    """1 SDC per row. i.e. 1 row == information about 1 SDC"""
    # Base columns to group by
    group_cols = [
        "sdc_id",
        "model_name",
        "image_index",
        "acc_time_at_sdc",
        "filepath",
        "log_start_timestamp",
        "sdc_class",
    ]
    # Conditionally add fault_type if it exists in the input dataframe
    if "fault_type" in sdc_details_df.columns:
        group_cols.append("fault_type")

    sdc_df = (
        sdc_details_df.groupby(group_cols, dropna=False)
        .agg(
            # 1. Count rows per sdc_id
            count_wrong_elements=("diff", "count"),
            # 2. Stats for received values
            mean_received=("received", "mean"),
            min_received=("received", "min"),
            max_received=("received", "max"),
            # 3. Stats for expected values
            mean_expected=("expected", "mean"),
            min_expected=("expected", "min"),
            max_expected=("expected", "max"),
            # 4. Stats for difference values
            mean_diff=("diff", "mean"),
            min_diff=("diff", lambda x: x.abs().min()),
            max_diff=("diff", lambda x: x.abs().max()),
        )
        .reset_index()
    )

    # check if each row has only 1 sdc_id
    assert sdc_df["sdc_id"].is_unique, "Found duplicate sdc_id entries in sdc_df!"
    # check that we have 1 row for every sdc_id in sdc_details_df
    expected_count = sdc_details_df["sdc_id"].nunique()
    actual_count = len(sdc_df)
    assert actual_count == expected_count, (
        f"Data loss detected! Expected {expected_count} unique SDCs, "
        f"but result has {actual_count}."
    )

    # add criticality columns per SDC + save object detection criticality CSV
    sdc_criticality_df = criticality_analysis(sdc_details_df, experiment_name)
    sdc_df = pd.merge(sdc_df, sdc_criticality_df, on="sdc_id", how="left")

    print(f" --- Processed {len(sdc_df)} SDCs ---")

    return sdc_df


def create_sdc_details_df(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    creates dataframe where each row has information about a corrupted element (wrong value) in an SDC
    (one SDC -> one or more rows)
    """

    # 1. Explode the list of SDC events (outer list)
    sdc_details_df = logs_df.explode("error_info")

    # 2. Filter out empty SDCs before assigning IDs
    sdc_details_df = sdc_details_df[
        sdc_details_df["error_info"].map(
            lambda d: len(d) > 0 if isinstance(d, list) else False
        )
    ]

    # 3. Create a unique SDC event index
    # Since each row is now a distinct SDC event, we just number them 0 to n
    sdc_details_df["sdc_number"] = range(len(sdc_details_df))
    # add the hash of log_start_timestamp and error_info to be sure we do not have repeated sdc_ids
    sdc_details_df["sdc_id"] = sdc_details_df.apply(create_sdc_id, axis=1)

    # 4. Explode the error details (inner list)
    # Now, if an SDC event had 5 errors, it becomes 5 rows,
    # but all 5 rows will share the same 'sdc_event_id'
    sdc_details_df = sdc_details_df.explode("error_info")

    # 5. Reset index and normalize dictionaries
    sdc_details_df = sdc_details_df.reset_index(drop=True)
    details_df = pd.json_normalize(sdc_details_df["error_info"])

    # 6. Combine metadata, sdc_id, and details
    sdc_details_df = pd.concat(
        [
            sdc_details_df[
                [
                    "filepath",
                    "log_start_timestamp",
                    "model_name",
                    "experiment_name",
                    "device",
                    "sdc_id",
                ]
            ],
            details_df,
        ],
        axis=1,
    )

    # compute diff
    sdc_details_df["diff"] = sdc_details_df["received"] - sdc_details_df["expected"]

    # get geometric classification of SDCs (single, row, square)
    sdc_details_df_len = len(sdc_details_df)
    sdc_geometric_dist_df = geometric_distribution_analysis(sdc_details_df)

    sdc_geometric_dist_df = sdc_geometric_dist_df[
        ["sdc_id", "index", "original_indexes", "sdc_class", "original_shape"]
    ]
    sdc_details_df = sdc_details_df.merge(
        sdc_geometric_dist_df, on=["sdc_id", "index"], how="left"
    )
    # be sure that there are not extra rows
    assert len(sdc_details_df) == sdc_details_df_len

    print(f"--- Processed {len(sdc_details_df)} corrupted elements ---")
    return sdc_details_df
