# 2026_03_17
# This script has methods to perform an statistical and geometric analysis
# of Silent Data Corruptions (SDCs) specifically classified as "row" errors.
# Usage:
# row_fault_model_analysis(experiment_name=EXPERIMENT_NAME)

import ast

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.experiment_paths import ExperimentPaths
from src.plot import plot_row_error_dimensions


def is_contiguous(group_df):
    """
    Determines if the indices in 'original_indexes' are contiguous
    along the dimension specified in 'spread_dimension'.
    """
    # 1. Get the spread dimension for this specific SDC group
    # We take the first value since it's the same for the whole group
    try:
        dim_to_check = int(group_df["spread_dimension"].iloc[0])
    except (ValueError, IndexError, TypeError):
        return False

    # 2. Parse the tuples
    indices = [
        ast.literal_eval(i) if isinstance(i, str) else i
        for i in group_df["original_indexes"]
    ]

    if len(indices) <= 1:
        # If only 1 element, it's technically contiguous
        return True

    # 3. Extract only the values for the spreading dimension
    # idx[dim_to_check] dynamically picks Height (1), Width (2), or Channel (3)
    changing_dim_values = [idx[dim_to_check] for idx in indices]

    # 4. Check for gaps
    # A range is contiguous if the span (max-min) equals (count - 1)
    # with no duplicates (which would happen if multiple bits in one pixel flipped)
    min_val = min(changing_dim_values)
    max_val = max(changing_dim_values)

    return (max_val - min_val) == (len(changing_dim_values) - 1)


def get_sdc_span(group_series):
    # Convert string tuples to actual tuples/lists
    indices = [ast.literal_eval(i) if isinstance(i, str) else i for i in group_series]
    if len(indices) < 2:
        return 0

    # Convert to numpy array for easy axis-wise calculation: shape (N, DIMS)
    arr = np.array(indices)

    # Calculate max distance in any single dimension (the row's axis)
    # For a row (0, 10, 5) to (0, 10, 50), this returns 45
    spans_per_dim = arr.max(axis=0) - arr.min(axis=0)
    return spans_per_dim.max()


def get_spread_dimension(group_series):
    """
    Returns the index of the dimension that is spreading.
    0: Batch, 1: Height, 2: Width, 3: Channel (typical 4D mapping)
    """
    indices = [ast.literal_eval(i) if isinstance(i, str) else i for i in group_series]
    if len(indices) < 2:
        return np.nan  # No spread in single-element SDCs

    arr = np.array(indices)

    # Calculate the span (max - min) for each dimension
    spans_per_dim = arr.max(axis=0) - arr.min(axis=0)

    # Return the index of the dimension with the largest spread
    spread_dimension = np.argmax(spans_per_dim)
    return spread_dimension


def get_dim_size(row):
    # 1. Convert the shape string to a tuple: (1, 256, 256, 64)
    shape_str = row["original_shape"]
    shape_tuple = (
        ast.literal_eval(shape_str) if isinstance(shape_str, str) else shape_str
    )

    # 2. Get the index of the spreading dimension from your previous series
    dim_idx = row["spread_dimension"]

    # 3. Return the size at that index (handle NaN if no spread)
    if pd.isna(dim_idx):
        return np.nan
    return shape_tuple[int(dim_idx)]


def is_bit_flip(value):
    """Checks if the absolute value is a power of 2 (1, 2, 4, 8, ...)."""
    if pd.isna(value) or value == 0:
        return False

    # Take absolute value and ensure it's an integer for bitwise check
    val = abs(int(value))

    # Power of two check: must be > 0 and (val & (val - 1) == 0)
    return (val > 0) and (val & (val - 1) == 0)


def print_1_diff_percentage(row_sdc_details_df: pd.DataFrame):
    # 1. Filter for exact +1 or -1 values
    is_plus_minus_one = row_sdc_details_df["diff"].abs() == 1

    # 2. Calculate Absolute Quantity
    abs_quantity_pm1 = is_plus_minus_one.sum()

    # 3. Calculate Percentage
    total_elements = len(row_sdc_details_df)
    percentage_pm1 = (
        (abs_quantity_pm1 / total_elements) * 100 if total_elements > 0 else 0
    )

    print(f"--- LSB Error Analysis (+/- 1) ---")
    print(f"Absolute Quantity: {abs_quantity_pm1}")
    print(f"Percentage: {percentage_pm1:.2f}%")


def plot_integer_histogram(
    df: pd.DataFrame,
    column_name: str,
    output_filepath: str,
    title=None,
    xlabel=None,
    ylabel="Frequency",
    color="skyblue",
    xticks_fontsize: int = 5,
    xticks_rotation: int = 0,
):
    """
    Creates a histogram where each bar represents exactly one integer value.
    """
    if df[column_name].empty:
        print(f"Column '{column_name}' is empty. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # discrete=True ensures bins are aligned with integers
    # shrink=.8 adds a small gap between bars for better readability
    ax = sns.histplot(
        data=df,
        x=column_name,
        discrete=True,
        shrink=0.8,
        color=color,
        edgecolor="black",
        alpha=0.7,
    )

    # Title and Labels
    if title:
        plt.title(title, fontsize=14)
    plt.xlabel(xlabel or column_name, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Generate a tick for every single integer in the range
    min_val = int(df[column_name].min())
    max_val = int(df[column_name].max())
    all_ticks = range(min_val, max_val + 1)
    plt.xticks(all_ticks, fontsize=xticks_fontsize, rotation=xticks_rotation)

    # Save plot to PNG
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filepath}")


def plot_continuous_histogram(
    df: pd.DataFrame,
    column_name: str,
    output_filepath: str,
    title=None,
    xlabel=None,
    ylabel="Density",
    color="skyblue",
):
    """
    Creates a continuous distribution plot (Histogram + KDE) for float ratios.
    """
    if df[column_name].dropna().empty:
        print(f"Column '{column_name}' is empty or all NaN. Skipping.")
        return

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Use stat="density" to make the KDE and Histogram scales match
    ax = sns.histplot(
        data=df,
        x=column_name,
        kde=True,
        element="step",  # Looks cleaner for continuous data
        color=color,
        edgecolor="black",
        alpha=0.4,
        stat="density",
    )

    if title:
        plt.title(title, fontsize=14)
    plt.xlabel(xlabel or column_name, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # For relative span, we know the range is 0 to 1
    if "relative" in column_name.lower():
        plt.xlim(-0.05, 1.05)  # Add a little padding
        # Set ticks at 10% intervals
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filepath}")


def plot_pie_chart(
    df: pd.DataFrame, column_name: str, output_filepath: str, title=None
):
    """
    Plots a pie chart for boolean or categorical columns showing both % and absolute counts.
    """
    if df[column_name].dropna().empty:
        print(f"Column '{column_name}' is empty. Skipping.")
        return

    # 1. Prepare data
    counts = df[column_name].value_counts()
    labels = counts.index
    values = counts.values  # Absolute quantities

    # 2. Setup colors
    colors = plt.get_cmap("Pastel1").colors

    plt.figure(figsize=(10, 8))

    # Define a helper function to format the labels
    def label_formatter(pct):
        absolute = int(np.round(pct / 100.0 * np.sum(values)))
        return f"{pct:.1f}%\n({absolute:d})"

    # 3. Create Pie
    wedges, texts, autotexts = plt.pie(
        values,
        labels=None,
        autopct=label_formatter,  # Use custom formatter here
        startangle=140,
        colors=colors,
        pctdistance=0.75,  # Moved slightly inward to accommodate two lines of text
        explode=[0.05] * len(counts),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    # 4. Make it a Donut chart
    centre_circle = plt.Circle(
        (0, 0), 0.60, fc="white"
    )  # Shrink circle slightly for more text room
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # 5. Formatting
    plt.title(title or f"Distribution of {column_name}", fontsize=16, pad=20)

    # Legend shows the Label and the Total Count for that category
    legend_labels = [f"{l} (Total: {v})" for l, v in zip(labels, values)]
    plt.legend(
        wedges,
        legend_labels,
        title=column_name,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    plt.setp(autotexts, size=10, weight="bold")
    plt.axis("equal")

    # 6. Save
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_filepath}")


def row_fault_model_analysis(experiment_name: str):
    ### load and prepare data
    # load SDC data
    paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(paths.sdcs_csv)
    sdc_details_df = pd.read_csv(paths.sdc_details_csv)
    # filter only convolution data
    sdc_df = sdc_df[sdc_df["model_name"].str.contains("conv")]
    # filter only ROW error SDCs
    sdc_df = sdc_df[sdc_df["sdc_class"] == "row"]
    print(f"Quantity of row-type SDCs: {len(sdc_df)}")
    # merge with sdc_details_df to get the indexes
    row_sdc_details_df = pd.merge(sdc_df, sdc_details_df, on="sdc_id", how="left")

    ### fault model analysis
    # Calculate span per SDC
    row_sdc_details_df["sdc_span"] = row_sdc_details_df.groupby("sdc_id")[
        "original_indexes"
    ].transform(get_sdc_span)
    # get dimension on which the row error spreads
    row_sdc_details_df["spread_dimension"] = row_sdc_details_df.groupby("sdc_id")[
        "original_indexes"
    ].transform(get_spread_dimension)
    # get the size of the specific dimension that is spreading
    row_sdc_details_df["spread_dim_max_size"] = row_sdc_details_df.apply(
        get_dim_size, axis=1
    )
    # get distribution of the relative size of the row (= row size / dimension size)
    # sdc_span_series / row_sdc_details_df['spread_dim_max_size']
    row_sdc_details_df["row_coverage_pct"] = (
        row_sdc_details_df["sdc_span"] / row_sdc_details_df["spread_dim_max_size"]
    ) * 100
    # get % of error_delta == power-of-two values (2, 4, 8, 16, 32, ...), indicating a single bit-flip
    row_sdc_details_df["is_single_bit_flip"] = row_sdc_details_df["diff"].apply(
        is_bit_flip
    )
    # check if it is contiguous (all original_indexes are neighbours, with no space in between)
    # Apply the logic per SDC ID
    contiguity_results = row_sdc_details_df.groupby("sdc_id").apply(is_contiguous)
    row_sdc_details_df["is_contiguous"] = row_sdc_details_df["sdc_id"].map(
        contiguity_results
    )
    # row_sdc_details_df["is_contiguous"] = row_sdc_details_df.groupby("sdc_id")[
    #     "original_indexes"
    # ].transform(is_contiguous)

    # print % and absolute quantity of diff that were +1 or -1
    print_1_diff_percentage(row_sdc_details_df)

    ### plots

    # plot distribution of the absolute size of the row
    row_sdc_df = row_sdc_details_df.drop_duplicates(subset=["sdc_id"])
    output_path = (
        paths.plots_folderpath / f"{experiment_name}_row_error_span_distribution.png"
    )
    plot_integer_histogram(
        df=row_sdc_df,
        column_name="sdc_span",
        output_filepath=str(output_path),
        title=f"Distribution of Absolute Row Error Spans\nExperiment: {experiment_name}",
        xlabel="Span Size (Max - Min Index)",
        color="darkorange",
    )

    # plot distribution of the relative size of the row (= row size / dimension size)
    output_path = (
        paths.plots_folderpath
        / f"{experiment_name}_row_relative_error_span_distribution.png"
    )
    plot_continuous_histogram(
        df=row_sdc_df,
        column_name="row_coverage_pct",
        output_filepath=str(output_path),
        title=f"Distribution of Relative Row Error Spans\nExperiment: {experiment_name}",
        xlabel="Relative Span Size = (Max - Min Index) / (total column size)",
        color="darkorange",
    )

    # plotar distribuição dos error_delta
    output_path = (
        paths.plots_folderpath
        / f"{experiment_name}_row_error_delta_distribution_discrete.png"
    )
    plot_integer_histogram(
        df=row_sdc_details_df,
        column_name="diff",
        output_filepath=str(output_path),
        title=f"Distribution of Error deltas\nExperiment: {experiment_name}",
        xlabel="Error delta = (Received - Expected)",
        xticks_fontsize=2,
        xticks_rotation=90,
        color="darkorange",
    )
    output_path = (
        paths.plots_folderpath
        / f"{experiment_name}_row_error_delta_distribution_continuous.png"
    )
    plot_continuous_histogram(
        df=row_sdc_details_df,
        column_name="diff",
        output_filepath=str(output_path),
        title=f"Distribution of Error deltas\nExperiment: {experiment_name}",
        xlabel="Error delta = (Received - Expected)",
        color="darkorange",
    )

    # plot % of error_delta == power-of-two values (2, 4, 8, 16, 32, ...), indicating a single bit-flip
    output_path = (
        paths.plots_folderpath / f"{experiment_name}_row_error_is_single_bit_flip.png"
    )
    plot_pie_chart(
        df=row_sdc_details_df,
        column_name="is_single_bit_flip",
        output_filepath=output_path,
        title=f"Row errors: is_single_bit_flip?\nExperiment: {experiment_name}",
    )

    # plot % of contiguous row errors
    output_path = (
        paths.plots_folderpath / f"{experiment_name}_row_error_is_contiguous.png"
    )
    plot_pie_chart(
        df=row_sdc_df,
        column_name="is_contiguous",
        output_filepath=output_path,
        title=f"Row errors: is_contiguous?\nExperiment: {experiment_name}",
    )

    # plot amount of wrong elements per SDC
    output_path = (
        paths.plots_folderpath / f"{experiment_name}_row_error_count_wrong_elements.png"
    )
    plot_integer_histogram(
        df=row_sdc_df,
        column_name="count_wrong_elements",
        output_filepath=str(output_path),
        title=f"Row errors: amount of corrupted output elements\nExperiment: {experiment_name}",
        xlabel="amount of wrong elements per SDC",
        color="darkorange",
    )

    # plot plot_row_error_dimensions
    plot_row_error_dimensions(experiment_name)

    # print mode, mean, and std  of count_wrong_elements
    print(f"count_wrong_elements mode: {row_sdc_df['count_wrong_elements'].mode()[0]}")
    print(f"count_wrong_elements mean: {row_sdc_df['count_wrong_elements'].mean()}")
    print(f"count_wrong_elements std: {row_sdc_df['count_wrong_elements'].std()}")


if __name__ == "__main__":
    row_fault_model_analysis(experiment_name="NSREC26")
