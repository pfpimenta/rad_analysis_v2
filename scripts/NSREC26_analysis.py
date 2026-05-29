# script that uses CNAO and PARTREC (and TIFPA?) analysis for other analysis

import numpy as np
import pandas as pd

from src.experiment_paths import ExperimentPaths
from src.plot import generate_all_plots
from src.sdc_processing import create_sdc_df

EXPERIMENT_NAME = "NSREC26"
nsrec_paths = ExperimentPaths(experiment_name=EXPERIMENT_NAME)
cnao_paths = ExperimentPaths(experiment_name="2026_01_CNAO")
partrec_paths = ExperimentPaths(experiment_name="2026_01_PARTREC")


def print_sdc_stats_per_model(sdc_details_df: pd.DataFrame):
    # Aggregating by model_name across all individual SDC events
    sdc_details_df = (
        sdc_details_df.groupby("model_name")
        .agg(
            # 1. Total number of SDC events found for this model
            total_sdc_events=("sdc_id", "nunique"),
            # 2. Total corrupted elements across all events
            total_corrupted_elements=("diff", "count"),
            # 3. Average corruption footprint (how many elements are usually wrong per SDC)
            avg_elements_per_sdc=("sdc_id", lambda x: len(x) / x.nunique()),
            # 4. Statistical range of the received values (the corrupted data)
            mean_received=("received", "mean"),
            max_received=("received", "max"),
            # 5. Magnitude of the corruption (how much the values shifted)
            mean_absolute_diff=("diff", lambda x: x.abs().mean()),
            max_diff=("diff", "max"),
            std_diff=("diff", "std"),
        )
        .reset_index()
    )

    # Display the result
    print("\n--- SDC Statistics Aggregated by Model ---")
    columns_to_print = ["model_name", "avg_elements_per_sdc", "mean_absolute_diff"]
    print(sdc_details_df[columns_to_print])


def nsrec_sdc_analysis():
    # error value analysis for both CNAO and PARTREC
    cnao_sdc_details_df = pd.read_csv(cnao_paths.sdc_details_csv)
    partrec_sdc_details_df = pd.read_csv(partrec_paths.sdc_details_csv)
    merged_sdc_details_df = pd.concat(
        [cnao_sdc_details_df, partrec_sdc_details_df], ignore_index=True
    )

    # save sdc_details_df into CSV
    merged_sdc_details_df.to_csv(
        nsrec_paths.sdc_details_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {nsrec_paths.sdc_details_csv}")

    # print SDC wrong values distribution
    print_sdc_stats_per_model(merged_sdc_details_df)

    # dataframe having one row per SDC:
    sdc_df = create_sdc_df(merged_sdc_details_df, EXPERIMENT_NAME)
    sdc_df.to_csv(nsrec_paths.sdcs_csv, index=False, sep=",", encoding="utf-8")
    print(f"Saved {nsrec_paths.sdcs_csv}")

    # generate all plots
    generate_all_plots(experiment_name=EXPERIMENT_NAME)


def nsrec_cross_section_analysis():
    cnao_cross_sections_per_model = pd.read_csv(cnao_paths.cross_sections_per_model_csv)
    partrec_cross_sections_per_model = pd.read_csv(
        partrec_paths.cross_sections_per_model_csv
    )
    ### SDC cross section per model
    # 1. Combine both DataFrames
    combined_df = pd.concat(
        [cnao_cross_sections_per_model, partrec_cross_sections_per_model],
        ignore_index=True,
    )
    # 2. Define the columns to keep and sum
    cols_to_sum = [
        "total_time_s",
        "total_sdc",
        "total_due",
        "total_fluency",
        "num_runs",
    ]
    # 3. Group by model_name and sum the metrics
    merged_cross_sections = (
        combined_df.groupby("model_name")[cols_to_sum].sum().reset_index()
    )

    # compute SDC cross section
    merged_cross_sections["SDC_cross_section"] = (
        merged_cross_sections["total_sdc"] / merged_cross_sections["total_fluency"]
    )
    # Optional: Clean up potential division by zero (if fluency was 0)
    merged_cross_sections["SDC_cross_section"] = merged_cross_sections[
        "SDC_cross_section"
    ].replace([np.inf, -np.inf], np.nan)

    # compute DUE cross section
    merged_cross_sections["DUE_cross_section"] = (
        merged_cross_sections["total_due"] / merged_cross_sections["total_fluency"]
    )
    # Optional: Clean up potential division by zero (if fluency was 0)
    merged_cross_sections["DUE_cross_section"] = merged_cross_sections[
        "DUE_cross_section"
    ].replace([np.inf, -np.inf], np.nan)

    print("Cross sections (CNAO and PARTREC merged):")
    print(merged_cross_sections)

    # save merged_cross_sections into CSV
    merged_cross_sections.to_csv(
        nsrec_paths.cross_sections_per_model_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {nsrec_paths.cross_sections_per_model_csv}")


def nsrec_analysis():
    nsrec_cross_section_analysis()
    nsrec_sdc_analysis()


if __name__ == "__main__":
    nsrec_analysis()
