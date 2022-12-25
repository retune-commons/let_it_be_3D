from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

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

    
def convert_to_path(
    attribute: Union[str, Path]
) -> Path:
    if type(attribute) == Path:
        return attribute
    elif type(attribute) == str:
        return Path(attribute)