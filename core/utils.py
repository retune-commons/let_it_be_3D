from pathlib import Path, PosixPath, WindowsPath
from typing import List, Tuple, Dict, Optional, Union

import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def convert_to_path(attribute: Union[str, Path]) -> Path:
    """Convert strings to Path and returns them."""
    if type(attribute) == PosixPath or type(attribute) == WindowsPath:
        return attribute
    elif type(attribute) == str:
        return Path(attribute)


def check_keys(dictionary: Dict, list_of_keys: List[str]) -> List:
    """
    Check, whether list_of_keys are in a dictionary and returns a list of
    keys missing in the dictionary.
    """
    missing_keys = []
    for key in list_of_keys:
        if key not in dictionary:
            missing_keys.append(key)
    return missing_keys


def read_config(path: Path) -> Dict:
    """Read structured config file defining a project."""
    path = convert_to_path(path)
    if path.exists() and path.suffix == ".yaml":
        with open(path, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    else:
        raise FileNotFoundError(
            f"Could not open the yaml file at {path}\n"
            f"Please make sure the path is correct and the file exists!"
        )
    return cfg


class Coordinates:
    """
    Class to store xyz coordinates.

    Parameters
    ----------
    y_or_row: int
    x_or_column: int
    z: int

    Attributes
    __________
    x: int, column: int
        Represent x coordinates in a system and columns in an image.
    y, row: int
        Represent y coordinates in a system and rows in an image.
    z: int
        Represent z coordinates in a system.
    """
    def __init__(
            self, y_or_row: int, x_or_column: int, z: Optional[int] = None
    ) -> None:
        """
        Construct attributes for Class Coordinates.

        Parameters
        ----------
        y_or_row: int
        x_or_column: int
        z: int
        """
        self.y = y_or_row
        self.row = y_or_row
        self.x = x_or_column
        self.column = x_or_column
        self.z = z


def get_3D_df_keys(key: str) -> Tuple[str, str, str]:
    """
    Construct a tuple of strings representing keys in a aniposelib created 3
    dimensional DataFrame.
    """
    return key + "_x", key + "_y", key + "_z"


def create_calibration_key(
        videos: List[str], recording_date: str, calibration_index: int, iteration: Optional[int] = None,
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


def get_3D_array(df: pd.DataFrame, key: str, index: Optional[int] = None) -> np.array:
    """
    Construct a ndarray of shape (3, N) representing one marker in a 3
    dimensional DataFrame with N number of indices.
    """
    x, y, z = get_3D_df_keys(key)
    if index is None:
        return np.array([df[x], df[y], df[z]])
    else:
        return np.array([df[x][index], df[y][index], df[z][index]])


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


def construct_dlc_output_style_df_from_dictionary(marker_predictions: Dict) -> pd.DataFrame:
    """
    Create a DataFrame from dictionary with DLC-like multi-index.

    Parameters
    ----------
    marker_predictions: {str: {str: list of int}}
        Dictionary containing markers as keys and dictionaries as values with
        x, y and z as keys and lists of int as values. The length of the lists
        is equivalent to the number of frames annotated.

    Returns
    -------
    df: pd.DataFrame
        The dataframe with DLC-like multiindex and data from input dictionary.
    """
    multi_index = get_multi_index(
        markers=marker_predictions.keys()
    )
    df = pd.DataFrame(data={}, columns=multi_index)
    for scorer, marker_id, key in df.columns:
        df[(scorer, marker_id, key)] = marker_predictions[
            marker_id
        ][key]
    return df


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
"""
Keys
____
calibration_evaluation_filepath: {str: str} 
    Dictionary with keys for all valid cam_ids, defining the filepath to the 
    file to use for marker detection for calibration_validation per cam.
calibration_evaluation_type: {str: str}
    Dictionary with keys for all valid cam_ids, defining the type of marker 
    detection to use per cam for calibration_evaluation files. Values: DLC, manual
processing_filepath: {str: str} 
    Dictionary with keys for all valid cam_ids, defining the filepath to the 
    file to use for marker detection for recordings per cam.
processing_type: {str: str}
    Dictionary with keys for all valid cam_ids, defining the type of marker 
    detection to use per cam for recordings. Values: DLC, manual
led_extraction_filepath: {str: str}
    Dictionary with keys for all valid cam_ids, defining the filepath to the 
    file to use for marker detection for synchronization per cam.
led_extraction_type: {str: str}
    Dictionary with keys for all valid cam_ids, defining the type of marker 
    detection to use per cam for synchronization. Values: DLC, manual
animal_lines: list of str
    List of all animal_lines to search for in recording filenames.
    If the lines are numbers, you need to set the lines as str.
paradigms: list of str
    List of all paradigms to search for in filenames and directories.
valid_cam_ids: list of str
    List of all cam_ids to search for in filenames.
calibration_tag: str
    Filename tag to search for in the calibration files.
    "Calvin" is an invalid value.
calibration_validation_tag: str 
    Filename tag to search for in the calibration_validation files.
    "Calvin" is an invalid value.
use_gpu: str, default ""
    Whether to restrict the usage of GPU for DLC analyses. Values: "", "prevent",
    "low", "full"
    "prevent" disables GPU usage
    "low" restricts GPU memory for synchronization 
    "full" uses entire GPU capacity, equivalent to ""
intrinsic_calibration_directory: str
    The directory, in which the intrinsic calibration pickle .p files or the 
    intrinsic calibration checkerboard videos are stored. Intrinsic calibration 
    videos have to have "checkerboard" and a valid cam_id in their filename. 
    They have to be recorded in same resolution as the recording/calibration 
    videos without cropping, using a 6x6 checkerboard.
load_calibration: bool
    If True, then use this package to set the intrinsics of the cameras instead
    of using the aniposelib function to calibrate intrinsics.
    Requires checkerboard videos or .p pickle files for all cameras in 
    intrinsic_calibration_directory. 
triangulation_type: str, default "triangulate"
    Specify the method of aniposelib triangulation to use.
    Values: "triangulate", "triangulate_optim_ransac_True", 
    "triangulate_optim_ransac_False"
allowed_num_diverging_frames: int
    Specify how many frames a synchronized file can differ from the median of
    all synchronized files before it will be excluded from analysis.
handle_synchro_fails: str, default error
    How to proceed if the first synchronisation try exceeds 
    synchro_error_threshold. Values: "repeat", "default", "manual", "error"
    "repeat" run the same method of synchronisation again.
    "default" using default value default_offset_ms as synchro offset.
    "manual" using manual marker detection for synchronisation
    "error" raises an error and breaks the analysis
default_offset_ms: int
    Synchro offset to use if first synchro try fails and handle_synchro_fails is
    "default". In milliseconds.
start_pattern_match_ms: int, default 0
    Start of time range in which to search for matching synchro pattern. In milliseconds.
end_pattern_match_ms: int
    End of time range in which to search for matching synchro pattern. In milliseconds.
synchro_error_threshold: int, default 100
    Below this threshold, a synchro patter alignment error will be considered as
    good synchro, above, as failed synchro. 
synchro_marker: str
    The marker to use for synchronisation. Has to be detectable by the 
    led_extraction method.
led_box_size: int
    Pixel range around predicted synchro marker position to calculate mean pixel
    intensity for blinking pattern from.
use_2D_filter: bool, default True
    Whether to use filtering on 2D marker detection predictions in recordings. 
    At the moment only available for DLC.
score_threshold: float, default 0.9
    Only predictions with likelihood above this threshold will be taken into 
    account for triangulation.
num_frames_to_pick: int, default 5
    The number of frames to use to find optimal prediction for the synchro_marker.
max_ram_digestible_frames: int, default 3000
    Maximum number of frames to keep in RAM during writing of synchronised videos. 
    Increase to speed up analysis, reduce to adapt to available RAM.
max_cpu_cores_to_pool: int, default 0
    If 0, then no multiprocessing will be used for writing of synchronised videos.
    Set to as many CPU cores, as you would like to use for multiprocessing.
max_calibration_frames: int
    Number of frames to take into account for intrinsic calibration. 300 works 
    well, depending on CPU speed, it can be necessary to reduce.
rapid_aligner_path: str, default ""
    If "", then no rapid_aligner (GPU based pattern alignment) will be used. 
    Insert path to locally installed clone of the rapid_aligner package to use 
    GPU for pattern synchronisation.
"""

KEYS_TO_CHECK_RECORDING = [
    "led_pattern",
    "target_fps",
    "calibration_index",
    "recording_date",
]
"""
led_pattern: dict
    Blinking pattern to use for temporal synchronisation.
target_fps: int
    Fps rate, to which the videos should be synchronized.
calibration_index: int, default 0
    Index of a calibration.
    Together with recording_date, it creates a unique calibration key.
recording_date: str
    Date at which the calibration was done.
"""

# ToDo: default values
# fps not necessary if all cams have the same fps
# offsets not necessary if no cropping was performed or use_intrinsic_calibration False
# fisheye not necessary if no camera is fisheye
KEYS_TO_CHECK_CAMERA_RECORDING = ["fps", "offset_row_idx", "offset_col_idx",
                                  "flip_h", "flip_v", "fisheye"]
"""
fps: int
    The framerate of the camera.
offset_row_idx: int, default 0
    If cropping was performed, specify the row or y index initial here.
offset_col_idx: int, default 0
    If cropping was performed, specify the col or x index initial here.
flip_h: bool, default False
    Based on the size from uncropped intrinsic calibration videos, the row or y
    index end is calculated and used instead of offset_row_idx.
flip_v: bool, default False
    Based on the size from uncropped intrinsic calibration videos, the col or x
    index end is calculated and used instead of offset_col_idx.
fisheye: bool, default False
    If a fisheye lens was used, set True.
"""

KEYS_TO_CHECK_CAMERA_PROJECT = [
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
                                     "ground_truth_config"]

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

KEYS_PER_CAM_PROJECT = ["processing_type", "calibration_evaluation_type", "processing_filepath",
                    "calibration_evaluation_filepath", "led_extraction_type", "led_extraction_filepath"]

KEYS_VIDEOMETADATA_PROJECT = ["valid_cam_ids", "paradigms", "load_calibration", "max_calibration_frames",
                          "max_ram_digestible_frames", "max_cpu_cores_to_pool", "animal_lines"]

# ToDo
# led pattern not necessary if no synchronisation necessary
KEYS_VIDEOMETADATA_RECORDING = ["led_pattern", "target_fps", "calibration_index"]