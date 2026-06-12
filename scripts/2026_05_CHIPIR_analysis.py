import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cross_section_utils import aggregate_per_model, create_cross_section_df
from src.experiment_paths import ExperimentPaths
from src.log_parsing import create_logs_df
from src.plot import generate_all_plots
from src.sdc_processing import create_sdc_details_df, create_sdc_df

EXPERIMENT_NAME = "2026_05_CHIPIR"
chipir_paths = ExperimentPaths(EXPERIMENT_NAME)


def plot_sdc_cross_section_per_model(
    experiment_name: str, model_names: list, comparison_name: str
):
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    cross_section_per_model_df = pd.read_csv(
        experiment_paths.cross_sections_per_model_csv
    )

    df = cross_section_per_model_df[
        cross_section_per_model_df["model_name"].isin(model_names)
    ].copy()
    df["model_name"] = pd.Categorical(
        df["model_name"], categories=model_names, ordered=True
    )
    df = df.sort_values("model_name")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    y = df["SDC_cross_section"]
    # yerr must be a 2xN array for lower and upper error bars
    yerr = np.array([(y - df["sdc_cs_low"]).values, (df["sdc_cs_high"] - y).values])

    plt.bar(x, y, yerr=yerr, capsize=5, color="skyblue", edgecolor="black", alpha=0.8)

    plt.xticks(x, df["model_name"], rotation=45, ha="right")
    plt.ylabel("SDC Cross Section (cm²)")
    plt.title(f"SDC Cross Section: {comparison_name}\nExperiment: {experiment_name}")
    plt.yscale("log")
    plt.grid(axis="y", linestyle="--", alpha=0.7, which="both")

    plt.tight_layout()
    plot_path = (
        experiment_paths.plots_folderpath
        / f"{experiment_name}_{comparison_name}_sdc_cross_section.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")


def plot_avg_error_delta_per_model(
    experiment_name: str, model_names: list, comparison_name: str
):
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    df = sdc_df[sdc_df["model_name"].isin(model_names)].copy()
    df["model_name"] = pd.Categorical(
        df["model_name"], categories=model_names, ordered=True
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="model_name",
        y="mean_diff",
        hue="model_name",
        palette="viridis",
        errorbar=("ci", 95),
        legend=False,
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Error Delta (mean_diff)")
    plt.title(f"Average Error Delta: {comparison_name}\nExperiment: {experiment_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = (
        experiment_paths.plots_folderpath
        / f"{experiment_name}_{comparison_name}_avg_error_delta.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")


def plot_avg_wrong_element_count_per_model(
    experiment_name: str, model_names: list, comparison_name: str
):
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    df = sdc_df[sdc_df["model_name"].isin(model_names)].copy()
    df["model_name"] = pd.Categorical(
        df["model_name"], categories=model_names, ordered=True
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="model_name",
        y="count_wrong_elements",
        hue="model_name",
        palette="magma",
        errorbar=("ci", 95),
        legend=False,
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Corrupted Elements per SDC")
    plt.title(
        f"Average Corrupted Elements: {comparison_name}\nExperiment: {experiment_name}"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = (
        experiment_paths.plots_folderpath
        / f"{experiment_name}_{comparison_name}_avg_wrong_element_count.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")


def generate_conv_comparison_plots(experiment_name: str):
    """
    plots comparing metrics:
    [SDC cross section, avg_error_delta, avg_wrong_element_count]
    for each comparison:
    [kernel size, int8 x uint8, standard2d x depthwise]
    """
    kernel_size_comparison_models = [
        "conv_2d_int8_k3x3x64_in256x256x64",
        "conv_2d_int8_k5x5x64_in256x256x64",
        "conv_2d_int8_k8x8x64_in256x256x64",
        "conv_2d_int8_k16x16x64_in256x256x64",
    ]
    int8_uint8_comparison_models = [
        "conv_2d_int8_k3x3x64_in256x256x64",
        "conv_2d_uint8_k3x3x64_in256x256x64",
    ]
    standard_depthwise_comparison_models = [
        "conv_2d_int8_k3x3x64_in256x256x64",
        "depthwise_conv_2d_int8_k3x3x64_in256x256x64",
    ]
    comparisons = {
        "kernel_size": kernel_size_comparison_models,
        "int8_vs_uint8": int8_uint8_comparison_models,
        "standard_vs_depthwise": standard_depthwise_comparison_models,
    }
    for comparison_name, model_names in comparisons.items():
        print(f"Generating plots for comparison: {comparison_name}")
        plot_sdc_cross_section_per_model(experiment_name, model_names, comparison_name)
        plot_avg_error_delta_per_model(experiment_name, model_names, comparison_name)
        plot_avg_wrong_element_count_per_model(
            experiment_name, model_names, comparison_name
        )


def chipir_2026_05_analysis():
    """
    ### runs_df
    # load runs CSV
    runs_df = pd.read_csv(chipir_paths.runs_info_csv)

    ### logs_df
    logs_df = create_logs_df(experiment_name=EXPERIMENT_NAME)

    # create SDCs dataframe
    sdc_details_df = create_sdc_details_df(logs_df=logs_df)
    sdc_details_df.to_csv(
        chipir_paths.sdc_details_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {chipir_paths.sdc_details_csv}")

    # create sdc_df
    sdc_df = create_sdc_df(sdc_details_df, EXPERIMENT_NAME)
    sdc_df.to_csv(chipir_paths.sdcs_csv, index=False, sep=",", encoding="utf-8")
    print(f"Saved {chipir_paths.sdcs_csv}")

    ### now using both runs_df and logs_df:
    cross_section_df = create_cross_section_df(runs_df, logs_df)
    # save cross sections per run into CSV
    cross_section_df.to_csv(
        chipir_paths.cross_sections_per_run_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {chipir_paths.cross_sections_per_run_csv}")

    # compute cross sections per model
    cross_section_per_model_df = aggregate_per_model(cross_section_df)
    # save cross sections per model
    cross_section_per_model_df.to_csv(
        chipir_paths.cross_sections_per_model_csv,
        index=False,
        sep=",",
        encoding="utf-8",
    )
    print(f"Saved {chipir_paths.cross_sections_per_model_csv}")

    # generate and save all plots
    generate_all_plots(EXPERIMENT_NAME)
    """

    # conv comparison plots
    generate_conv_comparison_plots(EXPERIMENT_NAME)


if __name__ == "__main__":
    chipir_2026_05_analysis()
