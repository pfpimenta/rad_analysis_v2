# 2026_03_23
# comparison between experiments,
# considering only convolution benchmarks

from src.experiment_paths import ExperimentPaths
import pandas as pd
from src.plot import print_general_SDC_stats

experiments = [
    "2026_03_08_ENFORSA",
    "2026_03_16_ENFORSA",
    "2024_07_TRIUMF",
    "2026_01_CNAO",
    "2026_01_PARTREC",
    "2024_08_CNAO",
    "2025_05_CNAO",
    "TIFPA_2025_01"
]

for experiment_name in experiments:
    # load data
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_details_df = pd.read_csv(experiment_paths.sdc_details_csv)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    # filter only convolution data
    sdc_details_df = sdc_details_df[sdc_details_df["model_name"].str.contains("conv")]
    sdc_df = sdc_df[sdc_df["model_name"].str.contains("conv")]

    print(f"\n{'='*40}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"... considering only CONVOLUTION benchmarks")
    print(f"{'='*40}")
    print_general_SDC_stats(sdc_details_df, sdc_df)
    print(f"{'-'*40}")