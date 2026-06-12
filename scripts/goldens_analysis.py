"""
Script to plot distributions of the golden data.
"""

from src.goldens import GOLDEN_PATHS
from src.plot import plot_golden_matrix_distribution


def golden_analysis():
    # collect all golden filepaths in a list (there are repeated ones)
    golden_filepaths_list = []
    for experiment_name in GOLDEN_PATHS:
        for model_name in GOLDEN_PATHS[experiment_name]:
            golden_filepath = GOLDEN_PATHS[experiment_name][model_name]
            golden_filepaths_list.append(golden_filepath)
    # remove repeated filepaths
    golden_filepaths_set = set(golden_filepaths_list)
    print(
        f"Collected {len(golden_filepaths_list)} golden filepaths, {len(golden_filepaths_set)} unique."
    )
    for golden_filepath in golden_filepaths_set:
        plot_golden_matrix_distribution(golden_filepath)


if __name__ == "__main__":
    golden_analysis()
