"""Methods that create and save plots
based on the radiation experiment data and analysis (SDCs, cross-sections, etc) data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.ENFORSA_FaultType import ENFORSA_FaultType
from src.experiment_paths import ExperimentPaths

# Define font sizes for easy adjustment
LABEL_SIZE = 20
TICK_SIZE = 20
TITLE_SIZE = 16

MODEL_COLORS = {
    "conv_3x3_on_1024x1024_int8_coral": "#344adb",
    "base_vit_8": "#344adb",
    "conv_10x10_coral": "#344adb",
    "conv_3x3_coral": "#344adb",
    "conv_5x5_coral": "#344adb",
    "conv_3x3_on_256x256x64_coral": "#344adb",
    "conv_5x5_on_256x256x64_coral": "#344adb",
    "mobilenet_v2_coral": "#8534db",
    "ssd_mobilenetv2_coral": "#ffdb65",
    # 2026_03_08_ENFORSA
    "depthwise_conv_2d_1_1024_1024_3_20_20_3_1": "#349bdb",
    "simple_conv_2d_1_1024_1024_1_40_40_1_1": "#9b34db",
    # 2024_07_TRIUMF
    "conv1k": "#349bdb",  # conv1k == Depthwise "3d" 20x20
    "simple_conv1k": "#9b34db",  # simple_conv1k == 2d 40x40
    # SC26 conv aggregations
    "conv": "#344adb",
}


def plot_histogram_comparisons_num_corrupted_elements(experiment_name: str):
    """
    Plots paired INT8 and UINT8 histograms to compare 3x3 vs 5x5 kernels.
    """
    # Font Size Constants
    LABEL_SIZE = 20
    TICK_SIZE = 20
    TITLE_SIZE = 16
    # load sdc_df
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    comparisons = [
        {
            "title": "Normalized SDC Error Spread (1024x1024x1, UINT8)",
            "filename": "convs_count_wrong_elements_UINT8",
            "labels": ["conv_3x3_coral", "conv_5x5_coral"],
            "label_name": ["conv_3x3_coral", "conv_5x5_coral"],
            "colors": ["#8534db", "#ffdb65"],
        },
        {
            "title": "Normalized SDC Error Spread (256x256x64, INT8)",
            "filename": "convs_count_wrong_elements_INT8",
            "labels": ["conv_3x3_on_256x256x64_coral", "conv_5x5_on_256x256x64_coral"],
            "label_name": [
                "conv_3x3_on_256x256x64_coral",
                "conv_5x5_on_256x256x64_coral",
            ],
            "colors": ["#8534db", "#ffdb65"],
        },
        {
            "title": "MobileNetV2 vs SSD-MobileNetV2 Error Distribution",
            "filename": "models_count_wrong_elements",
            "labels": ["mobilenet_v2_coral", "ssd_mobilenetv2_coral"],
            "label_name": ["Classification", "Detection"],
            "colors": ["#8534db", "#ffdb65"],
        },
    ]

    metric_column = "count_wrong_elements"

    for comp in comparisons:
        plt.figure(figsize=(10, 6))

        for model_name, color, label_name in zip(
            comp["labels"], comp["colors"], comp["label_name"]
        ):
            subset_df = sdc_df[sdc_df["model_name"] == model_name]

            if not subset_df.empty:
                # 'common_norm=False' + 'stat="percent"' ensures each distribution
                # is normalized to its own total independently.
                sns.histplot(
                    subset_df,
                    x=metric_column,
                    stat="percent",
                    common_norm=False,
                    discrete=True,
                    alpha=0.6,
                    label=label_name,
                    color=color,
                    edgecolor="black",
                )
            else:
                pass
                # print(f"⚠️ Warning: No data found for model '{model_name}'")

        # plt.title(comp["title"], fontsize=TITLE_SIZE)
        plt.xlabel("Number of corrupted elements in the output", fontsize=LABEL_SIZE)
        plt.ylabel("Frequency (%)", fontsize=LABEL_SIZE)
        # Adjust tick labels
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)

        plt.legend(loc="upper right", fontsize=LABEL_SIZE + 3)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        agg_path = f"{experiment_paths.plots_folderpath}/{experiment_name}_{comp['filename']}_normalized.png"
        plt.savefig(agg_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {agg_path}")


def plot_histogram_num_corrupted_elements_all_convs(experiment_name: str):
    """Saves PNG with plot of histogram of amount of corrupted elements per SDC,
    considering all convolution benchmarks at the same time.
    """
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    # --- 1. Aggregated plot for specific convolution models ---
    target_convs = [
        "conv_3x3_coral",
        "conv_5x5_coral",
        "conv_3x3_on_256x256x64_coral",
        "conv_5x5_on_256x256x64_coral",
    ]

    # Filter the DF for only these models
    agg_df = sdc_df[sdc_df["model_name"].isin(target_convs)]

    if not agg_df.empty:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        sns.histplot(
            data=agg_df,
            x="count_wrong_elements",
            discrete=True,
            color="#3498db",  # Different color to distinguish aggregation
            edgecolor="black",
            alpha=0.8,
        )

        # plt.title(
        #     f"Aggregated Convolution Models: Wrong elements per SDC\nTotal SDCs: {len(agg_df)}",
        #     fontsize=16,
        # )
        plt.xlabel("Number of Corrupted Elements", fontsize=14)
        plt.ylabel("Frequency (SDC Events)", fontsize=14)

        agg_path = f"{experiment_paths.plots_folderpath}/{experiment_name}_aggregated_convs_count_wrong_elements.png"
        plt.savefig(agg_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {agg_path}")


def plots_histogram_expected_vs_corrupted_output_values(experiment_name: str):
    """Saves PNGs with plots of, for each model,
    the distribution of the output expected values and received (corrupted) values.
    """
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    plots_folder = experiment_paths.plots_folderpath
    sdc_details_df = pd.read_csv(experiment_paths.sdc_details_csv)
    model_names = sdc_details_df["model_name"].unique()

    for model_name in model_names:
        model_sdcs_df = sdc_details_df[sdc_details_df["model_name"] == model_name]
        n_points = len(model_sdcs_df)

        if n_points == 0:
            continue

        # Weights for 0-100% normalization
        weights = np.ones(n_points) * 100.0 / n_points

        ### --- PLOT 1: ONLY DIFF ---
        plt.figure(figsize=(12, 6))
        # Use 300 bins for high granularity
        color = MODEL_COLORS[model_name]
        counts, bins, _ = plt.hist(
            model_sdcs_df["diff"],
            bins=300,
            weights=weights,
            alpha=0.7,
            edgecolor="black",
            color=color,
        )

        # Adjust Y-axis to 110% of the highest bin
        plt.ylim(0, counts.max() * 1.1)
        plt.xlabel("Difference (corrupted - expected)", fontsize=LABEL_SIZE)
        plt.ylabel("Frequency (%)", fontsize=LABEL_SIZE)
        # plt.title(
        #     f"Distribution of SDC difference \n model: {model_name}, experiment: {experiment_name}\n #points: {len(model_sdcs_df)}",
        #     fontsize=TITLE_SIZE,
        # )
        plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        plt.grid(True, alpha=0.3)

        output_path = (
            f"{plots_folder}/{experiment_name}_{model_name}_SDC_distribution.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

        ### --- PLOT 2: EXPECTED vs RECEIVED ---
        plt.figure(figsize=(12, 6))

        val_min = min(model_sdcs_df["expected"].min(), model_sdcs_df["received"].min())
        val_max = max(model_sdcs_df["expected"].max(), model_sdcs_df["received"].max())
        bin_edges = np.linspace(val_min, val_max, 300)

        # Plot and capture counts for normalization scaling
        counts_e, _, _ = plt.hist(
            model_sdcs_df["expected"],
            bins=bin_edges,
            weights=weights,
            alpha=0.5,
            label="Expected",
            color="#2ecc71",
        )
        counts_r, _, _ = plt.hist(
            model_sdcs_df["received"],
            bins=bin_edges,
            weights=weights,
            alpha=0.5,
            label="Received",
            color="#e74c3c",
        )

        # Adjust Y-axis to 110% of the peak of either distribution
        max_freq = max(counts_e.max(), counts_r.max())
        plt.ylim(0, max_freq * 1.1)

        plt.xlabel("Value", fontsize=LABEL_SIZE + 2)
        plt.ylabel("Frequency (%)", fontsize=LABEL_SIZE)
        # plt.title(
        #     f"Expected vs Received Values\nModel: {model_name}, experiment: {experiment_name}\n #points: {len(model_sdcs_df)}",
        #     fontsize=TITLE_SIZE,
        # )
        plt.legend(loc="upper left", fontsize=LABEL_SIZE + 3)
        plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        plt.grid(True, alpha=0.3)

        compare_path = (
            f"{plots_folder}/{experiment_name}_{model_name}_expected_vs_received.png"
        )
        plt.savefig(compare_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {compare_path}")


def plot_histogram_num_corrupted_elements_per_SDC(experiment_name: str):
    """Saves PNGs (Linear and Log Y-axis) with plots of histogram
    with the distribution of the amount of corrupted elements per SDC.
    """
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    plots_folder = experiment_paths.plots_folderpath
    sdc_details_df = pd.read_csv(experiment_paths.sdc_details_csv)

    # --- GLOBAL STYLE OVERRIDE ---
    # Matches the serif font used in LaTeX/IEEE templates
    plt.rcParams.update(
        {
            "font.size": 20,
        }
    )

    # 1. Reconstruct the count per SDC ID
    sdc_summary_df = (
        sdc_details_df.groupby("sdc_id")
        .agg(
            wrong_element_count=("original_indexes", "count"),
            model_name=("model_name", "first"),
        )
        .reset_index()
    )

    model_names = sdc_summary_df["model_name"].unique()

    for model_name in model_names:
        subset = sdc_summary_df[sdc_summary_df["model_name"] == model_name]
        counts = subset["wrong_element_count"]

        # Define the two scales we want to generate
        for scale_type in ["linear", "log"]:
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")

            max_val = int(counts.max())
            bins = np.arange(0.5, max_val + 1.5, 1)

            plt.hist(
                counts,
                bins=bins,
                color=(
                    "teal" if scale_type == "linear" else "indigo"
                ),  # Different color for distinction
                edgecolor="white",
                alpha=0.8,
                log=(scale_type == "log"),  # Matplotlib built-in log toggle
            )

            plt.xlabel("Number of Corrupted Elements", fontsize=22)
            plt.ylabel(f"Frequency ({scale_type} scale)", fontsize=22)

            if max_val <= 40:
                plt.xticks(range(1, max_val + 1))

            plt.grid(
                axis="y", linestyle=":", alpha=0.6, which="both"
            )  # 'both' shows minor log lines

            # Filename sanitization
            safe_model_name = str(model_name).replace(".", "_").replace("/", "_")
            suffix = "_logscale" if scale_type == "log" else ""
            plot_path = (
                plots_folder
                / f"{experiment_name}_{safe_model_name}_num_corrupted_elements_per_SDC{suffix}.png"
            )

            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved {plot_path}")


def plot_sdc_class_for_each_model(experiment_name: str):
    """
    Saves PNG of pie charts of the percentage and count of each SDC type.
    """
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    plots_folder = experiment_paths.plots_folderpath
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    model_names = sdc_df["model_name"].unique()

    # Custom function to return a formatted string with percentage and count
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val})"

        return my_autopct

    for model_name in model_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        model_data = sdc_df[sdc_df["model_name"] == model_name]
        # filter out skipped analysis
        model_data = model_data[model_data["sdc_class"] != "analysis_skipped"]
        # Replace semi-row with square
        model_data["sdc_class"] = model_data["sdc_class"].replace("semi-row", "square")
        # rename 'row' to 'line' and 'square' to 'box'
        model_data["sdc_class"] = model_data["sdc_class"].replace("row", "line")
        model_data["sdc_class"] = model_data["sdc_class"].replace("square", "box")

        class_counts = model_data["sdc_class"].value_counts()

        if class_counts.empty:
            print(f"Skipping {model_name}: No data found.")
            plt.close(fig)
            continue
        # create pie chart
        ax.pie(
            class_counts,
            labels=class_counts.index,
            autopct=make_autopct(class_counts),
            startangle=140,
            colors=plt.cm.Paired.colors,
        )
        ax.set_title(f"SDC Class Distribution\nModel: {model_name}")

        # save PNG
        safe_model_name = str(model_name).replace("/", "_").replace(" ", "_")
        plot_path = plots_folder / f"{experiment_name}_{safe_model_name}_sdc_class.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plot_path}")


def plot_sdc_criticality_comparisons(experiment_name: str):
    """Create a boxen plot showing the distribution of a few SDC metrics,
    for each model, for critical and non-critical errors.
    """
    # Font Settings
    LABEL_SIZE = 16
    TITLE_SIZE = 18

    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    # 1. Clean data
    sdc_df = sdc_df.dropna(subset=["is_critical"])
    sdc_df["is_critical"] = sdc_df["is_critical"].astype(bool)

    # TODO refactor dps: final_golden_score
    col1 = "golden_conf"
    col2 = "mean_golden_score_on_critical_errors"
    # Get the columns if they exist, otherwise use a Series of NaNs
    s1 = (
        sdc_df[col1]
        if col1 in sdc_df.columns
        else pd.Series(np.nan, index=sdc_df.index)
    )
    s2 = (
        sdc_df[col2]
        if col2 in sdc_df.columns
        else pd.Series(np.nan, index=sdc_df.index)
    )
    sdc_df["final_golden_score"] = s1.fillna(s2)

    metrics = [
        ("count_wrong_elements", "Number of Corrupted Elements"),
        ("mean_diff", "Mean Difference Magnitude"),
        ("max_diff", "Maximum Difference Magnitude"),
        ("golden_conf", "Model Confidence Image Classification"),
        ("final_golden_score", "Model Confidence"),
    ]
    for col, title in metrics:
        if col not in sdc_df.columns:
            continue
        plt.figure(figsize=(12, 8))
        # Boxenplot (Letter-Value Plot) is great for linear scales with outliers
        ax = sns.boxenplot(
            data=sdc_df, x="is_critical", y=col, hue="model_name", palette="Spectral"
        )
        # Aesthetics
        plt.title(
            f"Correlation: {title} vs SDC Criticality", fontsize=TITLE_SIZE, pad=20
        )
        plt.xlabel("Is Critical SDC?", fontsize=LABEL_SIZE)
        plt.ylabel(title, fontsize=LABEL_SIZE)
        plt.xticks([0, 1], ["False", "True"], fontsize=LABEL_SIZE)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        # Save plot
        plot_path = f"{experiment_paths.plots_folderpath}/{experiment_name}_SDC_criticality_corr_{col}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")


def plot_object_detection_mean_golden_score_X_criticality(experiment_name: str):
    """Plot 2 distributions for object detection, one of each column below, on the same plot image:
    * mean_golden_score_on_critical_errors
    * mean_golden_score_on_non_critical_errors
    """

    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    # if there is no Object Detection model in this experiment, skip
    if not "mean_golden_score_on_critical_errors" in sdc_df.columns:
        print(
            "Skipping golden_score_distribution_X_box_criticality plot because column mean_golden_score_on_critical_errors was not found in sdc_df."
        )
        return None

    # Set the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot Distribution 1: Critical Errors
    sns.kdeplot(
        data=sdc_df,
        x="mean_golden_score_on_critical_errors",
        fill=True,
        label="Critical Errors",
        color="crimson",
        common_norm=False,
    )

    # Plot Distribution 2: Non-Critical Errors
    sns.kdeplot(
        data=sdc_df,
        x="mean_golden_score_on_non_critical_errors",
        fill=True,
        label="Non-Critical Errors",
        color="royalblue",
        common_norm=False,
    )

    # Formatting the plot
    plt.title(f"Golden Score Distribution: {experiment_name}", fontsize=15)
    plt.xlabel("Mean Golden Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()

    # Save
    plt.tight_layout()
    plot_path = (
        experiment_paths.plots_folderpath
        / "golden_score_distribution_X_box_criticality.png"
    )
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")


def plot_criticality_percentage_per_model(experiment_name: str):
    """Saves a plot with the percentage of SDCs that were critical, per model.
    Also shows the total amount of SDCs and the amount of critical SDCs.
    """
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    # Compute the percentage of critical SDCs per model (ignoring NaNs)
    critical_stats = sdc_df.groupby("model_name")["is_critical"].mean() * 100
    # Absolute values
    critical_counts = sdc_df.groupby("model_name")["is_critical"].sum().astype(int)
    total_counts = sdc_df.groupby("model_name")["is_critical"].count()
    # generate bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    # Create vertical bar plot
    bars = ax.bar(
        critical_stats.index,
        critical_stats.values,
        color="#d9534f",
        edgecolor="black",
        alpha=0.8,
    )
    # Add data labels on top of each bar
    for bar, model_name in zip(bars, critical_stats.index):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%\n({critical_counts[model_name]}/{total_counts[model_name]})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    # Formatting
    ax.set_ylabel("Critical SDCs / Total SDCs (%)", fontsize=LABEL_SIZE)
    ax.set_xlabel("Model Name", fontsize=LABEL_SIZE)
    ax.set_title(
        f"Criticality Ratio per Model ({experiment_name})", fontsize=TITLE_SIZE, pad=25
    )
    # Ensure y-axis starts at 0 and leaves room for labels at the top
    max_val = critical_stats.max()
    ax.set_ylim(0, max_val * 3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    # save plot
    plots_folder = experiment_paths.plots_folderpath
    plot_path = (
        f"{plots_folder}/{experiment_name}_critical_sdc_percentage_per_model.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")


def save_describe_visual_report(experiment_name: str):
    """Saves a PNG with a summary table generated by Pandas' describe method,
    with SDC statistics grouped by model_name.
    """
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    # filter columns
    cols = [
        "model_name",
        "count_wrong_elements",
        "mean_diff",
        "max_diff",
        "critical_box_error_count",
        "real_critical_box_error_count",
        "class_id_error_count",
        "confidence_score_error_count",
        "mean_iou",
        "worst_iou",
        "mean_confidence_delta",
        "max_confidence_delta",
        "mean_golden_score_on_critical_errors",
        "golden_conf",
        "corrupted_conf",
        "golden_margin",
        "golden_score_ratio",
        "conf_loss",
    ]
    existing_cols = [c for c in cols if c in sdc_df.columns]
    sdc_df = sdc_df[existing_cols]
    # filter model_name
    model_names = ["ssd_mobilenetv2_coral", "mobilenet_v2_coral"]
    sdc_df = sdc_df[sdc_df["model_name"].isin(model_names)]
    # Generate the describe summary
    # We use numeric_only=True to ensure we only get the stats table
    summary = sdc_df.groupby("model_name").describe()
    # Flatten MultiIndex columns (e.g., ('mean_iou', 'mean') -> 'mean_iou_mean')
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    # Transpose so Metrics are on Y-axis and Models are on X-axis
    # This makes the table readable as a "Summary Sheet"
    summary_t = summary.T
    # Create the plot
    # We use a very large figure size because describe() output is typically massive
    plt.figure(figsize=(summary_t.shape[1] * 2 + 5, summary_t.shape[0] * 0.4 + 2))
    # Plotting raw values in a heatmap
    # cbar=False because with mixed scales (0.1 vs 1000), a single colorbar is misleading
    # we use 'annot=True' to show the exact numbers from describe()
    sns.heatmap(
        summary_t,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        annot_kws={"size": 10},
    )

    plt.title(f"Raw Describe() Statistics: {experiment_name}", fontsize=20, pad=30)
    plt.xlabel("Model Name", fontsize=16)
    plt.ylabel("Statistics (Metric_Stat)", fontsize=16)
    # Save the result
    plt.tight_layout()
    plots_folder = experiment_paths.plots_folderpath
    plot_path = f"{plots_folder}/{experiment_name}_sdc_stats_per_model.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_histogram_num_corrupted_elements_per_SDC_object_detection(
    experiment_name: str,
):
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    plots_folder = experiment_paths.plots_folderpath
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    # if SDC df has no Object Detection benchmark, just skip it
    if not "od_count_wrong_elements" in sdc_df.columns:
        print(
            f"Skipping OD wrong elements histogram for {experiment_name}, which has no od_count_wrong_elements column"
        )
        return None

    model_name_list = sdc_df.loc[
        sdc_df["od_count_wrong_elements"].notna(), "model_name"
    ].unique()
    for model_name in model_name_list:
        subset = sdc_df[sdc_df["model_name"] == model_name]

        plt.figure(figsize=(14, 7))  # Increased width slightly
        sns.set_style("whitegrid")

        # Determine bin range
        max_od = subset["od_count_wrong_elements"].max()
        max_gen = subset["count_wrong_elements"].fillna(0).max()
        max_val = int(max(max_od, max_gen))
        bins = np.arange(0.5, max_val + 1.5, 1)

        # Plot 1: OD Elements (%)
        sns.histplot(
            subset["od_count_wrong_elements"],
            bins=bins,
            label="Actual Quantity of Wrong Elements",
            color="skyblue",
            alpha=0.6,
            stat="percent",
            common_norm=False,
        )

        # Plot 2: General Elements (%)
        sns.histplot(
            subset["count_wrong_elements"],
            bins=bins,
            label="Naive Counting of Wrong Elements",
            color="salmon",
            alpha=0.6,
            stat="percent",
            common_norm=False,
        )

        plt.title(f"Comparison of Corrupted Elements (%): {model_name}", fontsize=14)
        plt.xlabel("Number of Wrong Elements per SDC", fontsize=11)
        plt.ylabel("Percentage of Samples (%)", fontsize=11)
        xticks_vals = range(1, max_val + 1)
        plt.xticks(xticks_vals, fontsize=5)
        plt.tick_params(axis="x", which="major")

        # Filename sanitization
        safe_model_name = str(model_name).replace(".", "_").replace("/", "_")
        plot_path = (
            plots_folder
            / f"{experiment_name}_{safe_model_name}_pct_corrupted_elements.png"
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved {plot_path}")


def plot_fault_type_percentage(experiment_name: str):
    """Plots the percentage and absolute count of each fault type in a pie chart."""
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    # Calculate counts
    counts = sdc_df["fault_type"].value_counts()
    # Map the integer indices to Enum names
    labels = [
        (
            ENFORSA_FaultType(val).name
            if val in [e.value for e in ENFORSA_FaultType]
            else str(val)
        )
        for val in counts.index
    ]

    # Custom function to display both percentage and count
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val})"

        return my_autopct

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("pastel", n_colors=len(counts))
    plt.pie(
        counts,
        labels=labels,
        autopct=make_autopct(counts),
        colors=colors,
        startangle=140,
    )
    plt.axis("equal")
    # save plot PNG
    plot_path = (
        experiment_paths.plots_folderpath
        / f"{experiment_name}_fault_type_pie_chart.png"
    )
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")


def plot_count_wrong_elements_distribution_per_fault_type(experiment_name: str):
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    unique_fault_types = sdc_df["fault_type"].unique()
    if np.isnan(unique_fault_types).all():
        print("⚠️ Warning: no fault type information to be used for plots.")
        return None
    for fault_type in unique_fault_types:
        subset = sdc_df[sdc_df["fault_type"] == fault_type]
        counts = subset["count_wrong_elements"]

        # Get Fault Name safely
        try:
            fault_type_name = ENFORSA_FaultType(fault_type).name
        except ValueError:
            fault_type_name = f"Type_{fault_type}"

        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        max_val = int(counts.max())
        threshold = 50

        # --- ADAPTIVE BINNING LOGIC ---
        if max_val <= threshold:
            # High granularity: 1 bin per integer
            bins = np.arange(-0.5, max_val + 1.5, 1)
            plt.hist(counts, bins=bins, color="teal", edgecolor="white", alpha=0.8)
            plt.xticks(range(0, max_val + 1))
        else:
            # Auto-binning for large ranges to avoid thin-bar clutter
            plt.hist(counts, bins="auto", color="teal", edgecolor="white", alpha=0.8)
            plt.locator_params(axis="x", nbins=10)

        # Formatting
        plt.title(
            f"Distribution of Corrupted Elements: {fault_type_name} (ID: {fault_type}) (#SDCs: {len(subset)})",
            fontsize=16,
        )
        plt.xlabel("Number of Corrupted Elements", fontsize=12)
        plt.ylabel("Frequency (Number of SDC Events)", fontsize=12)
        plt.grid(axis="y", linestyle=":", alpha=0.6)

        # save plot PNG
        plot_path = (
            experiment_paths.plots_folderpath
            / f"{experiment_name}_{fault_type_name}_num_corrupted_elements.png"
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved {plot_path}")


def plot_error_delta_distributions_per_fault_type(experiment_name: str):
    # plot the distribution of diff (SDC details df)
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_details_df = pd.read_csv(experiment_paths.sdc_details_csv)

    unique_fault_types = sdc_details_df["fault_type"].unique()
    for fault_type in unique_fault_types:
        subset = sdc_details_df[sdc_details_df["fault_type"] == fault_type]
        counts = subset["diff"]

        # Get Fault Name safely
        try:
            fault_type_name = ENFORSA_FaultType(fault_type).name
        except ValueError:
            fault_type_name = f"Type_{fault_type}"

        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        min_val, max_val = counts.min(), counts.max()
        range_val = max_val - min_val
        threshold = 50

        # --- ADAPTIVE BINNING LOGIC ---
        if range_val <= threshold:
            # Shift bins to center around integers based on the actual min/max
            bins = np.arange(np.floor(min_val) - 0.5, np.ceil(max_val) + 1.5, 1)
            plt.hist(counts, bins=bins, color="teal", edgecolor="white", alpha=0.8)
        else:
            plt.hist(counts, bins="auto", color="teal", edgecolor="white", alpha=0.8)

        # Formatting
        plt.title(
            f"Distribution of Error delta: {fault_type_name} (ID: {fault_type}) (# total wrong elements: {len(subset)})",
            fontsize=16,
        )
        plt.xlabel("Error delta", fontsize=12)
        plt.ylabel("Frequency (Number of SDC Events)", fontsize=12)
        plt.grid(axis="y", linestyle=":", alpha=0.6)

        # save plot PNG
        plot_path = (
            experiment_paths.plots_folderpath
            / f"{experiment_name}_{fault_type_name}_error_delta_distribution.png"
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved {plot_path}")


def print_general_SDC_stats(sdc_details_df: pd.DataFrame, sdc_df: pd.DataFrame):
    # 1. Row counts
    print(f"Total SDCs (sdc_df):         {len(sdc_df):>10}")
    print(f"Total Details (sdc_details): {len(sdc_details_df):>10}")

    # 2. count_wrong_elements stats (from sdc_df)
    if "count_wrong_elements" in sdc_df.columns:
        mean_cwe = sdc_df["count_wrong_elements"].mean()
        std_cwe = sdc_df["count_wrong_elements"].std()
        print(f"count_wrong_elements:        Mean: {mean_cwe:.2f} | Std: {std_cwe:.2f}")
    else:
        print("count_wrong_elements:        Column not found in sdc_df")

    # 3. diff stats (from sdc_details_df)
    if "diff" in sdc_details_df.columns:
        mean_diff = sdc_details_df["diff"].mean()
        std_diff = sdc_details_df["diff"].std()
        print(
            f"diff (numerical delta):      Mean: {mean_diff:.2f} | Std: {std_diff:.2f}"
        )
    else:
        print("diff (numerical delta):      Column not found in sdc_details_df")


def generate_fault_simulation_plots(experiment_name: str) -> None:
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)

    # if it is not a fault simulation experiment, do nothing
    if not "fault_type" in sdc_df.columns:
        return None

    plot_fault_type_percentage(experiment_name)
    plot_count_wrong_elements_distribution_per_fault_type(experiment_name)
    plot_error_delta_distributions_per_fault_type(experiment_name)


def generate_SDC_criticality_plots(experiment_name: str) -> None:
    # if the experiment does not have any model prediction benchmark, do nothing
    experiment_paths = ExperimentPaths(experiment_name=experiment_name)
    sdc_df = pd.read_csv(experiment_paths.sdcs_csv)
    if "is_critical" not in sdc_df.columns or sdc_df["is_critical"].isnull().all():
        return None

    # criticality % per model
    plot_criticality_percentage_per_model(experiment_name)

    # correlations between SDC criticality and other measurements
    plot_sdc_criticality_comparisons(experiment_name)

    # applies pandas describe() method on sdc_df and generates a PNG for that
    save_describe_visual_report(experiment_name)

    # check distribution of mean_golden_score on critical and non-critical errors
    plot_object_detection_mean_golden_score_X_criticality(experiment_name)


def generate_all_plots(experiment_name: str):
    # generate all plots of the experiment, if possible

    generate_fault_simulation_plots(experiment_name)

    plot_histogram_comparisons_num_corrupted_elements(experiment_name)
    plot_histogram_num_corrupted_elements_all_convs(experiment_name)
    plots_histogram_expected_vs_corrupted_output_values(experiment_name)
    plot_histogram_num_corrupted_elements_per_SDC(experiment_name)
    plot_sdc_class_for_each_model(experiment_name)
    generate_SDC_criticality_plots(experiment_name)
    plot_histogram_num_corrupted_elements_per_SDC_object_detection(experiment_name)
    # TODO plot row errors stuff: % of each dimension


if __name__ == "__main__":
    experiment_name = None  # TODO get from parameters
    generate_all_plots(experiment_name)
