from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path, PosixPath

import pandas as pd
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt


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
    if type(attribute) == PosixPath:
        return attribute
    elif type(attribute) == str:
        return Path(attribute)


def construct_dlc_output_style_df_from_manual_marker_coords(
    manual_test_position_marker_coords_pred: Dict,
) -> pd.DataFrame:
    multi_index = get_multi_index(
        markers=manual_test_position_marker_coords_pred.keys()
    )
    df = pd.DataFrame(data={}, columns=multi_index)
    for scorer, marker_id, key in df.columns:
        df[(scorer, marker_id, key)] = manual_test_position_marker_coords_pred[
            marker_id
        ][key]
    return df


def get_multi_index(markers: List) -> pd.MultiIndex:
    multi_index_column_names = [[], [], []]
    for marker_id in markers:
        for column_name in ("x", "y", "likelihood"):
            multi_index_column_names[0].append("manually_annotated_marker_positions")
            multi_index_column_names[1].append(marker_id)
            multi_index_column_names[2].append(column_name)
    return pd.MultiIndex.from_arrays(
        multi_index_column_names, names=("scorer", "bodyparts", "coords")
    )


def create_calibration_key(
    videos: List[str], recording_date: str, calibration_index: int
) -> str:
    key = ""
    videos.sort()
    for elem in videos:
        key = key + "_" + elem
    return recording_date + "_" + str(calibration_index) + key
