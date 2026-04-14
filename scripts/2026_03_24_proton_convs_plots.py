# 2026_03_24
# script to create the following plots from convolution benchmarks from all PROTON radiation experiments:
# * distribution of SDC geometric class  (sigle, row, square/box/rectangle)
# * distribution of amount of corrupted output elements per SDC
# * distribution of the error delta per corrupted output element
# * distribution of corrupted VS expected output values

from src.experiment_paths import ExperimentPaths
import pandas as pd
from src.plot import generate_all_plots, print_general_SDC_stats

PROTON_EXPERIMENTS = [
    "2026_01_CNAO",
    # "2026_01_PARTREC", # strange data cause it is INT8 data
    "2024_08_CNAO",
    "2025_05_CNAO",
    "TIFPA_2025_01",
]
# TODO create another abstraction instead of experiment_name
EXPERIMENT_NAME = "2026_03_SC26_PROTON_CONVS"


def proton_convs_analysis():

    # gather data from all convs in proton experiments
    sdc_details_df_list = []
    sdc_df_list = []
    for experiment_name in PROTON_EXPERIMENTS:
        # load data
        experiment_paths = ExperimentPaths(experiment_name=experiment_name)
        sdc_details_df = pd.read_csv(experiment_paths.sdc_details_csv)
        sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
        # filter only convolution data
        sdc_details_df = sdc_details_df[
            sdc_details_df["model_name"].str.contains("conv")
        ]
        sdc_df = sdc_df[sdc_df["model_name"].str.contains("conv")]
        # add dataframes to proton_convs_sdc_details_df and proton_convs_sdc_df
        sdc_details_df_list.append(sdc_details_df)
        sdc_df_list.append(sdc_df)

    # concatenate data from all proton conv SDCs
    proton_convs_sdc_details_df = pd.concat(sdc_details_df_list, ignore_index=True)
    proton_convs_sdc_df = pd.concat(sdc_df_list, ignore_index=True)

    # replace all model_name with 'conv', to make all plots as aggregations?
    proton_convs_sdc_details_df.loc[:, "model_name"] = "conv"
    proton_convs_sdc_df.loc[:, "model_name"] = "conv"

    # save CSVs
    paths = ExperimentPaths(experiment_name=EXPERIMENT_NAME)
    proton_convs_sdc_details_df.to_csv(paths.sdc_details_csv, index=False)
    print(f"Saved {paths.sdc_details_csv}")
    proton_convs_sdc_df.to_csv(paths.sdcs_csv, index=False)
    print(f"Saved {paths.sdcs_csv}")

    # create and save plots
    generate_all_plots(experiment_name=EXPERIMENT_NAME)

    print_general_SDC_stats(proton_convs_sdc_details_df, proton_convs_sdc_df)


if __name__ == "__main__":
    proton_convs_analysis()
