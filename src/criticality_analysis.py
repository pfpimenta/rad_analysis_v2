"""
Utilities for assessing the criticality
(i.e. the effect on the model output's prediction)
of each SDC
"""

import ast
import statistics
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.experiment_paths import ExperimentPaths
from src.goldens import get_class_label, load_golden_array

models_tasks = {
    "ssd_mobilenetv2_coral": "object_detection",
    "mobilenet_v2_coral": "image_classification",
}


def get_corrupted_output(
    golden_output: np.ndarray, sdc_data: pd.DataFrame
) -> np.ndarray:
    """
    reconstructs the corrupted output based on the golden output and the SDC information
    """
    corrupted_output = golden_output.copy()
    for idx, received_val, expected_val in zip(
        sdc_data["original_indexes"], sdc_data["received"], sdc_data["expected"]
    ):
        # Using 'idx' directly as it is already a tuple (e.g., (0, 156))
        if isinstance(idx, str):
            idx = ast.literal_eval(idx)
        assert isinstance(idx, tuple)
        assert golden_output[idx] == expected_val
        corrupted_output[idx] = received_val
    return corrupted_output


def image_classification_criticality_analysis(sdc_data: pd.DataFrame) -> dict:
    """
    for each SDC,
    compare the corrupted output with the golden output
    check if the model prediction would change
    """

    model_name = sdc_data.model_name.iloc[0]
    experiment_name = sdc_data.experiment_name.iloc[0]
    image_index = sdc_data.image_index.iloc[0]

    # get golden prediction
    golden_output = load_golden_array(experiment_name, model_name)[image_index]
    golden_prediction = np.argmax(golden_output)
    # TODO what if there are 2 idx for argmax?

    # get corrupted prediction
    corrupted_output = get_corrupted_output(golden_output, sdc_data)
    corrupted_prediction = np.argmax(corrupted_output)

    # check if prediction has changed with SDC
    is_misprediction = bool(corrupted_prediction != golden_prediction)

    # get predicted_class and corrupted_predicted_class (label strings instead of IDs)
    golden_class_label = get_class_label(golden_prediction, model_name)
    corrupted_class_label = get_class_label(corrupted_prediction, model_name)

    ##### get prediction confidence
    # get value of confidence of selected class in golden_output
    golden_conf = float(golden_output[0][golden_prediction])
    # get value of confidence of selected class in corrupted_output
    corrupted_conf = float(corrupted_output[0][corrupted_prediction])
    # get Golden Margin (Difference between 1st and 2nd place)
    top_two_scores = np.partition(golden_output[0], -2)[-2:]
    top_score = float(top_two_scores[1])
    second_best_score = float(top_two_scores[0])
    golden_margin = top_score - second_best_score
    # get Score Ratio (How many times larger 1st is than 2nd)
    golden_score_ratio = top_score / (second_best_score + 1e-9)
    # How much the original class dropped
    conf_loss = golden_conf - corrupted_output[0][golden_prediction]

    criticality_dict = {
        "is_misprediction": is_misprediction,
        "golden_class_label": golden_class_label,
        "corrupted_class_label": corrupted_class_label,
        "golden_conf": golden_conf,
        "corrupted_conf": corrupted_conf,
        "golden_margin": golden_margin,
        "golden_score_ratio": golden_score_ratio,
        "conf_loss": conf_loss,
    }

    return criticality_dict


def pair_bounding_boxes(
    golden_boxes: np.ndarray, corrupted_boxes: np.ndarray
) -> List[int]:
    """
    create a 1-to-1 pairing between golden_boxes and corrupted_boxes,
    matching according to the closest euclidian distances between boxes
    """

    # 1. Remove the extra batch dimension so we have (N, 4)
    g_boxes = golden_boxes[0]
    c_boxes = corrupted_boxes[0]

    # 2. Compute Euclidean distance between every pair
    # cost_matrix[i, j] is the distance between golden[i] and corrupted[j]
    cost_matrix = cdist(g_boxes, c_boxes, metric="euclidean")

    # 3. Solve for the minimum total distance (1-to-1 pairing)
    gold_indices, corr_indices = linear_sum_assignment(cost_matrix)

    # corr_indices is now your List[int] where corr_indices[i] is the
    # index in corrupted_boxes that pairs with golden_boxes[i]
    box_pairings = corr_indices.tolist()

    return box_pairings


def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area
    union_area = float(boxA_area + boxB_area - inter_area)

    # Avoid division by zero
    if union_area == 0:
        return 0.0
    iou = float(inter_area / union_area)
    return iou


def object_detection_criticality_analysis(
    sdc_data: pd.DataFrame,
) -> Tuple[dict, List[dict]]:
    IOU_CRITICALITY_THRESHOLD = 0.5  # if IoU < threshold, then it is critical

    model_name = sdc_data.model_name.iloc[0]
    experiment_name = sdc_data.experiment_name.iloc[0]
    image_index = sdc_data.image_index.iloc[0]

    # get golden prediction
    golden_output = load_golden_array(experiment_name, model_name)[image_index]
    # split the different outputs from object detection.
    # in the TPU, they were concatenated as such:
    # output_data = np.concatenate([boxes, classes_expanded, scores_expanded], axis=2)
    # The indices [4, 5] tell NumPy where to "cut" along axis 2
    golden_boxes, golden_class_ids, golden_scores = np.split(
        golden_output, [4, 5], axis=2
    )

    # get corrupted prediction
    corrupted_output = get_corrupted_output(golden_output, sdc_data)
    corrupted_boxes, corrupted_class_ids, corrupted_scores = np.split(
        corrupted_output, [4, 5], axis=2
    )

    # find out with corrupted box correspond to each golden box
    box_pairings = pair_bounding_boxes(golden_boxes, corrupted_boxes)

    # check if prediction has changed with SDC
    criticality_per_box = []
    for golden_idx, corrupted_idx in enumerate(box_pairings):
        # compare bounding boxes
        g_box = golden_boxes[0][golden_idx]
        c_box = corrupted_boxes[0][corrupted_idx]
        # number of coordinates that are not equal between golden and corrupted
        count_different_coords = np.sum(g_box != c_box)
        iou = calculate_iou(g_box, c_box)
        has_critical_box_error = iou < IOU_CRITICALITY_THRESHOLD
        # compare class_id
        golden_class = int(golden_class_ids[0][golden_idx][0])
        corrupted_class = int(corrupted_class_ids[0][corrupted_idx][0])
        has_class_id_error = golden_class != corrupted_class
        golden_class_label = get_class_label(golden_class, model_name)
        corrupted_class_label = get_class_label(corrupted_class, model_name)
        # compare confidence scores
        golden_score = golden_scores[0][golden_idx][0]
        corrupted_score = corrupted_scores[0][corrupted_idx][0]
        confidence_score_diff = corrupted_score - golden_score
        box_criticality_dict = {
            "golden_box_index": golden_idx,
            "count_different_coords": count_different_coords,
            "iou": iou,
            "has_critical_box_error": has_critical_box_error,
            "golden_class": golden_class,
            "corrupted_class": corrupted_class,
            "has_class_id_error": has_class_id_error,
            "golden_class_label": golden_class_label,
            "corrupted_class_label": corrupted_class_label,
            "golden_score": golden_score,
            "corrupted_score": corrupted_score,
            "confidence_score_diff": confidence_score_diff,
        }
        criticality_per_box.append(box_criticality_dict)

    ### metrics per SDC
    # amount of BBs where IoU < 0.5
    critical_box_error_count = sum(
        1 for box in criticality_per_box if box["has_critical_box_error"]
    )
    # amount of BBs where has_class_id_error
    class_id_error_count = sum(
        1 for box in criticality_per_box if box["has_class_id_error"]
    )
    # amount of BBs where corrupted_score != golden_score
    confidence_score_error_count = sum(
        1 for box in criticality_per_box if abs(box["confidence_score_diff"]) > 0
    )
    # count total number of coordinates that are not equal between golden and corrupted
    total_count_different_coords = sum(
        box["count_different_coords"] for box in criticality_per_box
    )
    # average of IoU (golden X corrupted) in all BBs
    mean_iou = statistics.mean(box["iou"] for box in criticality_per_box)
    worst_iou = min(box["iou"] for box in criticality_per_box)
    # mean_confidence_delta shows if the SDC generally suppresses confidence (negative value) or creates "false confidence" (positive value)
    # = average of delta between golden and corrupted confidence scores of each BB
    mean_confidence_delta = statistics.mean(
        box["confidence_score_diff"] for box in criticality_per_box
    )
    # maximum delta value between golden and corrupted confidence scores of each BB
    max_confidence_delta = max(
        box["confidence_score_diff"] for box in criticality_per_box
    )
    # Counts boxes where confidence > 0.5 AND (IoU is poor OR class is wrong)
    critical_error_count = sum(
        1
        for box in criticality_per_box
        if box["golden_score"] > 0.5
        and (box["has_critical_box_error"] or box["has_class_id_error"])
    )
    critical_boxes_scores = [
        box["golden_score"]
        for box in criticality_per_box
        if box["has_critical_box_error"] or box["has_class_id_error"]
    ]
    non_critical_boxes_scores = [
        box["golden_score"]
        for box in criticality_per_box
        if not box["has_critical_box_error"] and not box["has_class_id_error"]
    ]
    # mean confidence of the golden (golden_score) in the cases where (IoU is poor OR class is wrong)
    # = average golden confidence score, considering only BBs with critical errors (IoU is poor OR class is wrong)
    mean_golden_score_on_critical_errors = (
        statistics.mean(critical_boxes_scores) if critical_boxes_scores else 0.0
    )
    # = average golden confidence score, considering only BBs without critical errors (IoU is poor OR class is wrong)
    mean_golden_score_on_non_critical_errors = (
        statistics.mean(non_critical_boxes_scores) if non_critical_boxes_scores else 0.0
    )
    # amount of wrong elements corrected by the bounding boxes order change
    od_count_wrong_elements = (
        total_count_different_coords
        + class_id_error_count
        + confidence_score_error_count
    )
    criticality_dict = {
        "critical_box_error_count": critical_box_error_count,
        "critical_error_count": critical_error_count,
        "class_id_error_count": class_id_error_count,
        "confidence_score_error_count": confidence_score_error_count,
        "mean_iou": mean_iou,
        "worst_iou": worst_iou,
        "mean_confidence_delta": mean_confidence_delta,
        "max_confidence_delta": max_confidence_delta,
        "mean_golden_score_on_critical_errors": mean_golden_score_on_critical_errors,
        "mean_golden_score_on_non_critical_errors": mean_golden_score_on_non_critical_errors,
        "od_count_wrong_elements": od_count_wrong_elements,
    }

    # return both criticality_dict (info aggregated per SDC)
    # and criticality_per_box (detailed information per bounding box)
    return criticality_dict, criticality_per_box


def build_sdc_criticality_df(sdc_criticality_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(sdc_criticality_dict, orient="index")
    # Move the SDC IDs from the index into an actual column named 'sdc_id'
    df = df.reset_index().rename(columns={"index": "sdc_id"})
    return df


def criticality_analysis(
    sdc_details_df: pd.DataFrame, experiment_name: str = None
) -> pd.DataFrame:
    """
    For each SDC, we compared the corrupted outputs with the golden outputs, and then generate
    some metrics that indicate whether it was a Critical SDC or Non-Critical SDC (is_critical column).
    Also, a Object Detection criticality per bounding box DataFrame is created and saved on a CSV.
    At the end, the sdc_criticality_df DataFrame with all the computed Image Classification
    and Object Detection criticality metrics per SDC is returned (1 row per SDC).
    """
    # if there is no model supported in sdc_details_df
    model_names = sdc_details_df["model_name"].unique()
    if not any(name in models_tasks for name in model_names):
        # return empty dataframe
        sdc_criticality_df = pd.DataFrame(columns=["sdc_id"])
        print(f"No model supported for Criticality Analysis found in sdc_details_df")
        return sdc_criticality_df

    # Group by sdc_id
    grouped_sdcs = sdc_details_df.groupby("sdc_id")

    # object detection and image classification criticality data
    od_sdc_criticality_dict = {}
    od_criticality_per_box_dict = {}
    ic_sdc_criticality_dict = {}
    for sdc_id, sdc_data in grouped_sdcs:
        # Get the model name (taking the first entry since it's 1:1)
        model_name = sdc_data["model_name"].iloc[0]
        sdc_id = sdc_data["sdc_id"].iloc[0]
        sdc_experiment_name = sdc_data["experiment_name"].iloc[0]

        # skip if it is this specific case
        if (
            model_name == "ssd_mobilenetv2_coral"
            and sdc_experiment_name == "CNAO_2026_01"
        ):
            # CNAO and PARTREC have different shape. CNAO is just bounding boxes, which is what Bruno's code actually uses
            continue

        # skip if the model is not supported in sdc_details_df
        if not model_name in models_tasks.keys():
            continue

        # get task
        task = models_tasks.get(model_name)

        # DEBUG TODO remove
        if sdc_data["original_indexes"].nunique() == 1 and len(sdc_data) > 1:
            print("repeated values")
            breakpoint()

        if task is None:
            # it is a benchmark, not a model.
            # so there is not any criticality analysis to be done
            continue

        if task == "object_detection":
            od_sdc_criticality, criticality_per_box = (
                object_detection_criticality_analysis(sdc_data)
            )
            od_sdc_criticality_dict[sdc_id] = od_sdc_criticality
            od_criticality_per_box_dict[sdc_id] = criticality_per_box
        elif task == "image_classification":
            ic_sdc_criticality = image_classification_criticality_analysis(sdc_data)
            ic_sdc_criticality_dict[sdc_id] = ic_sdc_criticality

    ### save od_criticality_per_box in an CSV
    if experiment_name is None:
        # get the experiment_name of the first line
        experiment_name = sdc_details_df.experiment_name.iloc[0]
    # put od_criticality_per_box_dict into a pandas dataframe
    rows = []
    for sdc_id, boxes in od_criticality_per_box_dict.items():
        for box in boxes:
            # We create a new dictionary to avoid modifying the original data
            # and add the sdc_id to each row
            row_data = {"sdc_id": sdc_id, **box}
            rows.append(row_data)
    # Create the Object Detection criticality per bounding box DataFrame
    od_criticality_per_box_df = pd.DataFrame(rows)
    # save CSV
    experiment_paths = ExperimentPaths(experiment_name)
    output_csv_filepath = (
        experiment_paths.results_folderpath / "sdc_criticality_per_box.csv"
    )
    od_criticality_per_box_df.to_csv(output_csv_filepath, index=False)
    print(f"Saved {output_csv_filepath}")

    od_sdc_criticality_df = build_sdc_criticality_df(od_sdc_criticality_dict)
    ic_sdc_criticality_df = build_sdc_criticality_df(ic_sdc_criticality_dict)
    sdc_criticality_df = pd.merge(
        od_sdc_criticality_df, ic_sdc_criticality_df, on="sdc_id", how="outer"
    )

    ### classify each SDC between critical and non-critical
    # 1. Ensure is_misprediction is treated as a boolean (handles 'True'/'False' strings or NaNs)
    is_mis_bool = (
        sdc_criticality_df["is_misprediction"].astype(str).str.lower() == "true"
    )
    # 2. Check for box errors (filling NaNs with 0 just in case)
    has_box_error = (
        sdc_criticality_df["critical_error_count"].fillna(0) > 0
        if "critical_error_count" in sdc_criticality_df.columns
        else False
    )  # 3. Create the 'is_critical' column
    sdc_criticality_df["is_critical"] = is_mis_bool | has_box_error

    return sdc_criticality_df
