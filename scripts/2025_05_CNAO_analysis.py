from src.experiment_paths import ExperimentPaths
import pandas as pd
from src.log_parsing import create_logs_df
from src.sdc_processing import create_sdc_details_df, create_sdc_df
from src.plot import generate_all_plots

EXPERIMENT_NAME = "2025_05_CNAO"


def analysis_2025_05_CNAO():

    triumf_paths = ExperimentPaths(experiment_name=EXPERIMENT_NAME)

    ### runs_df
    # load runs CSV
    # runs_df = pd.read_csv(triumf_paths.runs_info_csv)
    # not formatted as expected. not needed now.

    ### logs_df
    ignore_logs = []
    logs_df = create_logs_df(experiment_name=EXPERIMENT_NAME, ignore_logs=ignore_logs)

    # for now, we are only interested in the convolutions
    conv_logs_df = logs_df[logs_df['model_name'].str.contains('conv')]

    # create sdc_details_df
    sdc_details_df = create_sdc_details_df(logs_df=conv_logs_df)
    sdc_details_df.to_csv(
        triumf_paths.sdc_details_csv, index=False, sep=",", encoding="utf-8"
    )
    print(f"Saved {triumf_paths.sdc_details_csv}")

    # create sdc_df
    sdc_df = create_sdc_df(sdc_details_df, EXPERIMENT_NAME)
    sdc_df.to_csv(triumf_paths.sdcs_csv, index=False, sep=",", encoding="utf-8")
    print(f"Saved {triumf_paths.sdcs_csv}")

    ## No need to compute cross section for now. Also, I dont have the flux data.

    # generate and save plots
    generate_all_plots(EXPERIMENT_NAME)


if __name__ == "__main__":
    analysis_2025_05_CNAO()
