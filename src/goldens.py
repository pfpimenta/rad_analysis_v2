from pathlib import Path

import numpy as np

from src.experiment_paths import DATA_FOLDER

GOLDEN_FOLDER = DATA_FOLDER / "goldens"

GOLDEN_PATHS = {
    "2026_01_CNAO": {
        "ssd_mobilenetv2_coral": GOLDEN_FOLDER
        / "golden_ILSVRC2012_selected_1024_300x300_SSD_mobilenetv2.npy",
        "mobilenet_v2_coral": GOLDEN_FOLDER
        / "golden_ILSVRC2012_selected_1024_mobilenetv2.npy",
        "conv_5x5_coral": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_5_5_1_1.npy",
        "conv_3x3_coral": GOLDEN_FOLDER
        / "golden_input_conv_3x3_on_128_1_1024_1024_3_random_values.npy",
    },
    "2026_01_PARTREC": {
        "ssd_mobilenetv2_coral": GOLDEN_FOLDER
        / "golden_input_COCO_for_SSD-MobileNetV2.npy",
        "mobilenet_v2_coral": GOLDEN_FOLDER
        / "golden_ILSVRC2012_selected_1024_mobilenetv2.npy",
        "conv_5x5_on_256x256x64_coral": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_256_256_64_5_5_64_1_int8_quant_edgetpu.npy",
        "conv_3x3_on_256x256x64_coral": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_256_256_64_3_3_64_1_int8_quant_edgetpu.npy",
    },
    "2026_03_08_ENFORSA": {
        "simple_conv_2d_1_1024_1024_1_40_40_1_1": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        "depthwise_conv_2d_1_1024_1024_3_20_20_3_1": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2024_07_TRIUMF": {
        # simple_conv1k == 2d 40x40
        "simple_conv1k": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        # conv1k == Depthwise 3d 20x20
        "conv1k": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2025_01_TIFPA": {
        # simple_conv1k == 2d 40x40
        "simple_conv1k": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        # conv1k == Depthwise 3d 20x20
        "conv1k": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2026_03_16_ENFORSA": {
        "simple_conv_2d_1_1024_1024_1_40_40_1_1": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        "depthwise_conv_2d_1_1024_1024_3_20_20_3_1": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2024_08_CNAO": {
        # simple_conv1k == 2d 40x40
        "simple_conv1k": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        # conv1k == Depthwise 3d 20x20
        "conv1k": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2025_05_CNAO": {
        # simple_conv1k == 2d 40x40
        "simple_conv1k": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        # conv1k == Depthwise 3d 20x20
        "conv1k": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "TIFPA_2025_01": {
        # simple_conv1k == 2d 40x40
        "simple_conv1k": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        # conv1k == Depthwise 3d 20x20
        "conv1k": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2026_03_31_ENFORSA": {
        "simple_conv_2d_1_1024_1024_1_40_40_1_1": GOLDEN_FOLDER
        / "golden_simple_conv_2d_1_1024_1024_1_40_40_1_1.npy",
        "depthwise_conv_2d_1_1024_1024_3_20_20_3_1": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_1_1024_1024_3_20_20_3_1.npy",
    },
    "2026_05_CHIPIR": {
        "conv_2d_int8_k3x3x64_in256x256x64": GOLDEN_FOLDER
        / "golden_conv_2d_int8_k3x3x64_in256x256x64_edgetpu.npy",
        "conv_2d_int8_k5x5x64_in256x256x64": GOLDEN_FOLDER
        / "golden_conv_2d_int8_k5x5x64_in256x256x64_edgetpu.npy",
        "conv_2d_int8_k16x16x64_in256x256x64": GOLDEN_FOLDER
        / "golden_conv_2d_int8_k16x16x64_in256x256x64_edgetpu.npy",
        "conv_2d_int8_k8x8x64_in256x256x64": GOLDEN_FOLDER
        / "golden_conv_2d_int8_k8x8x64_in256x256x64_edgetpu.npy",
        "conv_2d_uint8_k3x3x64_in256x256x64": GOLDEN_FOLDER
        / "golden_conv_2d_uint8_k3x3x64_in256x256x64_edgetpu.npy",
        "conv_2d_int8_k3x3x32_in256x256x32": GOLDEN_FOLDER
        / "golden_conv_2d_int8_k3x3x32_in256x256x32_edgetpu.npy",
        "depthwise_conv_2d_int8_k3x3x64_in256x256x64": GOLDEN_FOLDER
        / "golden_depthwise_conv_2d_int8_k3x3x64_in256x256x64_edgetpu.npy"
    },
}


def get_imagenet_labels(class_id: int) -> str:
    labels_filepath = GOLDEN_FOLDER / "imagenet_labels.txt"
    with open(labels_filepath, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    class_label = labels[class_id]
    return class_label


def get_coco_class_label(class_id: int) -> str:
    labels_filepath = GOLDEN_FOLDER / "coco_labels.txt"
    with open(labels_filepath, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    coco_class_label = labels[class_id]
    return coco_class_label


def get_class_label(class_id: int, model_name: str) -> str:
    if model_name == "ssd_mobilenetv2_coral":
        class_label = get_coco_class_label(class_id)
    elif model_name == "mobilenet_v2_coral":
        class_label = get_imagenet_labels(class_id)
    else:
        print(f"Warning: no labels file for model {model_name}")
        class_label = None
    return class_label


def get_golden_path(experiment_name: str, model_name: str) -> Path:
    try:
        golden_path = GOLDEN_PATHS[experiment_name][model_name]
    except:
        print(
            f"⚠️ Warning: Golden path not found for experiment='{experiment_name}', model='{model_name}'"
        )
        golden_path = None
    return golden_path


def load_golden_array(experiment_name: str, model_name: str) -> np.ndarray:
    filepath = get_golden_path(experiment_name, model_name)
    if filepath is None:
        return None
    golden_array = np.load(filepath)
    return golden_array


def get_output_shape(experiment_name: str, model_name: str) -> tuple:
    golden_array = load_golden_array(experiment_name, model_name)
    if golden_array is None:
        return None
    shape = golden_array.shape
    # ignore first dimension, as it is only the num_samples, not output of the model itself
    original_shape = shape[1:]
    return original_shape


def print_golden_shapes() -> None:
    """
    Iterates through GOLDEN_PATHS, retrieves golden file shapes and sizes, and prints them.
    Warns if a file is not found, cannot be loaded, or has an invalid path.
    """
    for exp_name, models in GOLDEN_PATHS.items():
        print(f"\nExperiment: {exp_name}")
        for model_name, path_obj in models.items():
            filepath_str = str(path_obj)
            if not filepath_str:
                print(f"  - Model: {model_name:<35} | Path is empty.")
                continue
            try:
                # Get file size in MB for better readability
                file_size_bytes = path_obj.stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)
                golden_array = np.load(filepath_str)
                print(
                    f"  - Model: {model_name:<35} | "
                    f"Size: {file_size_mb:.2f} MB | "
                    f"Shape: {str(golden_array.shape):<20} | "
                    f"Dtype: {golden_array.dtype}"
                )

            except FileNotFoundError:
                print(
                    f"  - Model: {model_name:<35} | Error: File not found at {filepath_str}"
                )
            except Exception as e:
                print(
                    f"  - Model: {model_name:<35} | Error loading file {filepath_str}: {e}"
                )
    print("\n------------------------")


if __name__ == "__main__":
    print_golden_shapes()
