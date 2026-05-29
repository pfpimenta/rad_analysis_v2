import pandas as pd

from src.experiment_paths import ExperimentPaths
from src.log_parsing import create_logs_df
from src.plot import generate_all_plots
from src.sdc_processing import create_sdc_details_df, create_sdc_df
from src.cross_section_utils import aggregate_per_model, create_cross_section_df

EXPERIMENT_NAME = "2026_05_CHIPIR"
chipir_paths = ExperimentPaths(EXPERIMENT_NAME)


def chipir_2026_05_analysis():
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


if __name__ == "__main__":
    chipir_2026_05_analysis()
