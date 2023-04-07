from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path, PosixPath, WindowsPath

import pandas as pd
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import yaml


def get_3D_df_keys(key: str)-> Tuple[str, str, str]:
        return key + "_x", key + "_y", key + "_z"

def get_3D_array(df: pd.DataFrame, key: str, index: Optional[int]=None)->np.array:
    x, y, z = get_3D_df_keys(key)
    if index is None:
        return np.array([df[x], df[y], df[z]])
    else:
        return np.array([df[x][index], df[y][index], df[z][index]])

def check_keys(dictionary: Dict, list_of_keys: List[str]) -> List:
    missing_keys = []
    for key in list_of_keys:
        try:
            dictionary[key]
        except KeyError:
            missing_keys.append(key)
    return missing_keys


def read_config(path: Path) -> Dict:
    """
    Reads structured config file defining a project.
    """
    path = convert_to_path(path)
    if path.exists() and path.suffix == ".yaml":
        with open(path, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    else:
        raise FileNotFoundError(
            f"Could not open the yaml file at {path}\n Please make sure the path is correct and the file exists!"
        )
    return cfg


class Coordinates:
    def __init__(
        self, y_or_row: int, x_or_column: int, z: Optional[int] = None
    ) -> None:
        self.y = y_or_row
        self.row = y_or_row
        self.x = x_or_column
        self.column = x_or_column
        self.z = z


def load_image(filepath: Path, idx: int = 0) -> np.ndarray:
    iio_reader = iio.get_reader(filepath)
    return np.asarray(iio_reader.get_data(idx))


def load_single_frame_of_video(filepath: Path, frame_idx: int = 0) -> np.ndarray:
    return load_image(filepath=filepath, idx=frame_idx)


def plot_image(
    filepath: Path, idx: int = 0, plot_size: Tuple[int, int] = (9, 6)
) -> None:
    fig = plt.figure(figsize=plot_size, facecolor="white")
    image = load_image(filepath=filepath, idx=idx)
    plt.imshow(image)


def plot_single_frame_of_video(
    filepath: Path, frame_idx: int = 0, plot_size: Tuple[int, int] = (9, 6)
) -> None:
    plot_image(filepath=filepath, idx=frame_idx, plot_size=plot_size)


def convert_to_path(attribute: Union[str, Path]) -> Path:
    if type(attribute) == PosixPath or type(attribute) == WindowsPath:
        return attribute
    elif type(attribute) == str:
        return Path(attribute)


def construct_dlc_output_style_df_from_manual_marker_coords(
    manual_annotated_marker_coords_pred: Dict,
) -> pd.DataFrame:
    multi_index = get_multi_index(
        markers=manual_annotated_marker_coords_pred.keys()
    )
    df = pd.DataFrame(data={}, columns=multi_index)
    for scorer, marker_id, key in df.columns:
        df[(scorer, marker_id, key)] = manual_annotated_marker_coords_pred[
            marker_id
        ][key]
    return df


def get_multi_index(markers: List) -> pd.MultiIndex:
    multi_index_column_names = [[], [], []]
    for marker_id in markers:
        for column_name in ("x", "y", "likelihood"):
            multi_index_column_names[0].append("annotated_markers")
            multi_index_column_names[1].append(marker_id)
            multi_index_column_names[2].append(column_name)
    return pd.MultiIndex.from_arrays(
        multi_index_column_names, names=("scorer", "bodyparts", "coords")
    )


def create_calibration_key(
    videos: List[str], recording_date: str, calibration_index: int, iteration: Optional[int]=None,
) -> str:
    key = ""
    videos.sort()
    for elem in videos:
        key = key + "_" + elem
    if iteration is None:
        calibration_key = recording_date + "_" + str(calibration_index) + key
    else:
        calibration_key = recording_date + "_" + str(calibration_index) + key + "_" + str(iteration)
    return calibration_key


KEYS_TO_CHECK_PROJECT = [
        "valid_cam_ids",
        "paradigms",
        "animal_lines",
        "led_extraction_type",
        "led_extraction_filepath",
        "max_calibration_frames",
        "max_cpu_cores_to_pool",
        "max_ram_digestible_frames",
        "rapid_aligner_path",
        "use_gpu",
        "load_calibration",
        "calibration_tag",
        "calibration_validation_tag",
        "allowed_num_diverging_frames",
        "handle_synchro_fails",
        "default_offset_ms",
        "start_pattern_match_ms",
        "end_pattern_match_ms",
        "synchro_error_threshold",
        "synchro_marker",
        "led_box_size",
        "use_2D_filter",
        "score_threshold",
        'num_frames_to_pick',
        'triangulation_type'
    ]

KEYS_TO_CHECK_RECORDING = [
        "led_pattern",
        "target_fps",
        "calibration_index",
        "recording_date",
    ]


KEYS_TO_CHECK_CAMERA = [
        "processing_type",
        "calibration_evaluation_type",
        "processing_filepath",
        "calibration_evaluation_filepath",
        "led_extraction_type",
        "led_extraction_filepath",
    ]

STANDARD_ATTRIBUTES_TRIANGULATION = ["all_cameras",
                                     "markers_excluded_manually",
                                     "calibration_toml_filepath",
                                     "csv_output_filepath",
                                     "markers",
                                     "triangulation_dlc_cams_filepaths",
                                     "project_config_filepath",
                                     "output_directory",
                                     "normalised_dataframe",
                                     "anipose_io",
                                     "video_plotting_config",
                                     "rotated_filepath",
                                     "rotation_error",
                                     "synchronization_individuals",
                                     "led_detection_individuals",
                                     "ground_truth_config",
                                     "synchronized_videos"]

STANDARD_ATTRIBUTES_CALIBRATION = ["camera_group",
                                   "report_filepath",
                                   "reprojerr",
                                   "valid_videos",
                                   "synchronized_charuco_videofiles",
                                   "camera_objects",
                                   "synchronization_individuals",
                                   "led_detection_individuals",
                                   "project_config_filepath",
                                   "output_directory"]

SYNCHRO_METADATA_KEYS = ["handle_synchro_fails",
                         "default_offset_ms",
                         "start_pattern_match_ms",
                          "end_pattern_match_ms",
                         "synchro_error_threshold",
                         "synchro_marker",
                         "use_2D_filter",
                         'num_frames_to_pick',
                         "rapid_aligner_path",
                         "use_gpu",
                         "led_box_size"]
