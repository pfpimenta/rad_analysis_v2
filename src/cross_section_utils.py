"""
Utility functions related to calculating radiation cross-sections.

This module provides tools to merge experiment run metadata with execution logs,
aggregate event counts (Single Event Upsets - SDCs and Device Under Test Errors - DUEs),
and compute cross-sections with associated confidence intervals using Chi-squared statistics.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def merge_runs_df_and_logs_df(
    runs_df: pd.DataFrame, logs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges runs_df and logs_df, finding which log files correspond to each
    experiment run.
    If a run contains multiple models, the output will have one row per (run, model) pair.
    The output dataframe contains columns from runs_df plus:
    'sdc_count', 'model_name', 'num_logs', and 'total_acc_time_sum'.
    """
    # 1. Format timestamps
    date_format = "%Y-%m-%d %H:%M:%S"
    runs_df["start_timestamp"] = pd.to_datetime(
        runs_df["start_timestamp"], format=date_format
    )
    runs_df["end_timestamp"] = pd.to_datetime(
        runs_df["end_timestamp"], format=date_format
    )
    logs_df["log_start_timestamp"] = pd.to_datetime(
        logs_df["log_start_timestamp"], format=date_format
    )

    # 2. Prepare logs with end timestamps
    logs_df["log_end_timestamp"] = logs_df["log_start_timestamp"] + pd.to_timedelta(
        logs_df["last_acc_time"].fillna(0), unit="s"
    )

    # 3. Perform a cross merge to find all possible matches
    # We only need the columns necessary for matching and aggregation
    combined = pd.merge(
        runs_df[["run_id", "start_timestamp", "end_timestamp"]],
        logs_df[
            [
                "log_start_timestamp",
                "log_end_timestamp",
                "model_name",
                "last_acc_time",
                "sdc_count",
            ]
        ],
        how="cross",
    )

    # 4. Filter logs that fall within the run time window
    buffer = pd.Timedelta(seconds=10)
    mask = (
        (combined["log_start_timestamp"] <= (combined["end_timestamp"] + buffer))
        & (combined["log_end_timestamp"] >= (combined["start_timestamp"] - buffer))
    )
    matches = combined[mask]

    # 5. Aggregate metrics per (run_id, model_name)
    summary = (
        matches.groupby(["run_id", "model_name"])
        .agg(
            total_acc_time_sum=("last_acc_time", "sum"),
            sdc_count=("sdc_count", "sum"),
            num_logs=("model_name", "count"),
        )
        .reset_index()
    )

    # 6. Final Join with runs_df to regain metadata (flux, etc.)
    # We use 'left' to keep track of runs with no matches
    new_runs_df = pd.merge(runs_df, summary, on="run_id", how="left")

    # Cleaning up
    new_runs_df["num_logs"] = new_runs_df["num_logs"].fillna(0)
    new_runs_df["total_acc_time_sum"] = new_runs_df["total_acc_time_sum"].fillna(0)
    new_runs_df["sdc_count"] = new_runs_df["sdc_count"].fillna(0)

    # Check for missing logs
    missing_logs = new_runs_df[new_runs_df["num_logs"] == 0]
    if not missing_logs.empty:
        print(
            f"⚠️ Warning: No logs found for Run IDs: {missing_logs['run_id'].unique().tolist()}"
        )

    return new_runs_df


def add_cross_section_bounds(df, alpha=0.15):
    """
    Computes 1-alpha confidence intervals for SDC and DUE cross sections.
    Standard alpha = 0.15 for 85% confidence level.
    """

    for metric in ["sdc", "due"]:
        count_col = f"total_{metric}"
        fluency_col = "total_fluency"

        # 1. Compute the lower and upper bounds for the NUMBER of events (N)
        # Lower bound: 0.5 * Chi2_ppf(alpha/2, 2*N)
        # Upper bound: 0.5 * Chi2_ppf(1 - alpha/2, 2*N + 2)
        df[f"{metric}_n_low"] = df[count_col].apply(
            lambda n: chi2.ppf(alpha / 2, 2 * n) / 2 if n > 0 else 0
        )
        df[f"{metric}_n_high"] = df[count_col].apply(
            lambda n: chi2.ppf(1 - alpha / 2, 2 * n + 2) / 2
        )

        # 2. Divide by total fluency to get Cross Section bounds
        df[f"{metric}_cs_low"] = df[f"{metric}_n_low"] / df[fluency_col]
        df[f"{metric}_cs_high"] = df[f"{metric}_n_high"] / df[fluency_col]

        # Optional: Clean up intermediate count columns
        df.drop(columns=[f"{metric}_n_low", f"{metric}_n_high"], inplace=True)

    return df


def aggregate_per_model(cross_section_df: pd.DataFrame) -> pd.DataFrame:
    """
    input dataframe must have columns:
    'model_name', 'run_id', 'flux', 'total_acc_time_sum', 'sdc_count', 'due_count', 'run_fluency'
    """
    # 1. Calculate Fluency per run first to ensure we sum it correctly
    # Fluency [p/cm^2] = Flux [p/cm^2/s] * Time [s]
    cross_section_df["run_fluency"] = (
        cross_section_df["flux"] * cross_section_df["total_acc_time_sum"]
    )

    # 2. Group by the official model name
    model_summary = (
        cross_section_df.groupby("model_name")
        .agg(
            total_time_s=("total_acc_time_sum", "sum"),
            total_sdc=("sdc_count", "sum"),
            total_due=("due_count", "sum"),
            total_fluency=(
                "run_fluency",
                "sum",
            ),  # Summing fluencies is the correct way to aggregate
            num_runs=("run_id", "count"),
        )
        .reset_index()
    )

    # 3. Calculate the "Effective Flux" (Weighted Mean Flux)
    # This is useful for your final table to see the average conditions
    model_summary["weighted_avg_flux"] = (
        model_summary["total_fluency"] / model_summary["total_time_s"]
    )

    # 4. Calculate Final Cross Sections
    # Sigma = Events / Total Fluency
    model_summary["SDC_cross_section"] = (
        model_summary["total_sdc"] / model_summary["total_fluency"]
    )
    model_summary["DUE_cross_section"] = (
        model_summary["total_due"] / model_summary["total_fluency"]
    )  # TODO

    # Optional: Replace inf with NaN if total_fluency was 0
    model_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

    # compute lower and upper bound for cross sections
    model_summary = add_cross_section_bounds(model_summary)

    return model_summary


def create_cross_section_df(runs_df, logs_df) -> pd.DataFrame:
    """
    Merges experiment runs with system logs to calculate reliability cross-sections.
    First, computes the total fluency of each run.
    Then, computes SDC and DUE cross-section by dividing their count by the fluency.
    """
    cross_section_df = merge_runs_df_and_logs_df(runs_df, logs_df)

    # compute due_count
    cross_section_df["due_count"] = cross_section_df["num_logs"] - 1
    # compute fluency
    cross_section_df["fluency"] = (
        cross_section_df["total_acc_time_sum"] * cross_section_df["flux"]
    )
    # Handle cases where flux might be 0 to avoid Infinity values
    cross_section_df["fluency"] = cross_section_df["fluency"].replace(
        [np.inf, -np.inf], np.nan
    )
    # Compute SDC cross-section: SDCs divided by Fluency
    cross_section_df["SDC_cross_section"] = (
        cross_section_df["sdc_count"] / cross_section_df["fluency"]
    )
    # Compute DUE cross-section: DUEs divided by Fluency
    cross_section_df["DUE_cross_section"] = (
        cross_section_df["due_count"] / cross_section_df["fluency"]
    )
    # Optional: Clean up potential division by zero (if fluency was 0)
    cross_section_df["SDC_cross_section"] = cross_section_df[
        "SDC_cross_section"
    ].replace([np.inf, -np.inf], np.nan)
    # Optional: Clean up potential division by zero (if fluency was 0)
    cross_section_df["DUE_cross_section"] = cross_section_df[
        "DUE_cross_section"
    ].replace([np.inf, -np.inf], np.nan)

    return cross_section_df
