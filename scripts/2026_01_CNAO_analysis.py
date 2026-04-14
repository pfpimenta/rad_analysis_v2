# script to compute and save:
# per model_run:
# * model per model run
# * num SDCs per model run
# * total Acctime per model run
# by using the server logs and the runs_info CSV

import pandas as pd

from src.experiment_paths import ExperimentPaths
from src.log_parsing import create_logs_df
from src.plot import generate_all_plots
from src.sdc_processing import create_sdc_details_df, create_sdc_df
from src.cross_section_utils import aggregate_per_model, create_cross_section_df

EXPERIMENT_NAME = "2026_01_CNAO"
cnao_paths = ExperimentPaths(experiment_name=EXPERIMENT_NAME)

# TODO fix! this script is failing because the golden values are not matching with the exp values in the logs! 

def analysis_2026_01_CNAO():
    ### runs_df
    # load runs CSV
    runs_df = pd.read_csv(cnao_paths.runs_info_csv)
    # transform flux values into INT
    runs_df["flux"] = (
        runs_df["flux"].str.replace("^", "**", regex=False).apply(pd.eval).astype(int)
    )

    ### logs_df
    ignore_logs = [
        # "2026_01_12_17_13_32_run_conv_3x3_coral_ECC_OFF_rasp4-coral.log",
        # "2026_01_12_17_24_41_run_conv_3x3_coral_ECC_OFF_rasp4-coral.log",
        # esse tem SDCs normais e TALVEZ um errado(?)
        # "2026_01_15_01_03_01_run_conv_3x3_coral_ECC_OFF_rasp4-coral.log",
    ]
    logs_df = create_logs_df(experiment_name=EXPERIMENT_NAME, ignore_logs=ignore_logs)

    # create sdc_details_df
    sdc_details_df = create_sdc_details_df(logs_df=logs_df)
    sdc_details_df.to_csv(
        cnao_paths.sdc_details_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {cnao_paths.sdc_details_csv}")

    # create sdc_df
    sdc_df = create_sdc_df(sdc_details_df, EXPERIMENT_NAME)
    sdc_df.to_csv(cnao_paths.sdcs_csv, index=False, sep=",", encoding="utf-8")
    print(f"Saved {cnao_paths.sdcs_csv}")

    ### now using both runs_df and logs_df:
    cross_section_df = create_cross_section_df(runs_df, logs_df)
    # save cross sections per model into CSV
    cross_section_df.to_csv(
        cnao_paths.cross_sections_per_run_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {cnao_paths.cross_sections_per_run_csv}")

    # compute cross sections per model
    cross_section_per_model_df = aggregate_per_model(cross_section_df)
    # save cross sections per model
    cross_section_per_model_df.to_csv(
        cnao_paths.cross_sections_per_model_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {cnao_paths.cross_sections_per_model_csv}")

    # generate and save plots
    generate_all_plots(EXPERIMENT_NAME)


if __name__ == "__main__":
    analysis_2026_01_CNAO()
