import math
from pathlib import Path
from typing import List, Optional, Union, Dict

import cv2
import imageio as iio
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from moviepy.video.io.bindings import mplfig_to_npimage

from .utils import Coordinates, load_single_frame_of_video, convert_to_path, get_3D_df_keys


def _zscore(array: np.ndarray) -> np.ndarray:
    return (array - np.mean(array)) / np.std(array, ddof=0)


def _save_figure(filepath: Union[str, Path]):
    if convert_to_path(filepath).exists():
        convert_to_path(filepath).unlink()
    plt.savefig(filepath, dpi=400)


def _connect_one_set_of_markers(
        ax: plt.Figure,
        all_markers: Dict,
        group: Dict
) -> None:
    if all(x in list(all_markers.keys()) for x in group['markers']):
        x = [all_markers[marker]['x'] for marker in group['markers']]
        y = [all_markers[marker]['y'] for marker in group['markers']]
        z = [all_markers[marker]['z'] for marker in group['markers']]
        ax.plot(x, y, z, alpha=group['alpha'], c=group['color'])


def _fill_one_set_of_markers(
        ax: plt.Figure,
        all_markers: Dict,
        group: Dict
) -> None:
    if all(x in list(all_markers.keys()) for x in group['markers']):
        points_2d = [[all_markers[marker]['x'], all_markers[marker]['y']] for marker in group['markers']]
        z = [all_markers[marker]['z'] for marker in group['markers']]
        artist = Polygon(np.array(points_2d), closed=False, color=group['color'], alpha=group['alpha'])
        ax.add_patch(artist)
        art3d.pathpatch_2d_to_3d(artist, z=z, zdir='z')


def _connect_one_set_of_marker_ids(
        ax: plt.Figure,
        points: Dict,
        bps: str,
        bp_dict: Dict,
        color: np.ndarray,
) -> None:
    ixs = [bp_dict[bp] for bp in bps]
    ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2], color=color)


def _connect_all_marker_ids(
        ax: plt.Figure,
        points: Dict,
        scheme: List[str],
        bodyparts: List[str],
) -> None:
    cmap = plt.get_cmap("tab10")
    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
    for i, bps in enumerate(scheme):
        _connect_one_set_of_marker_ids(
            ax=ax, points=points, bps=bps, bp_dict=bp_dict, color=cmap(i)[:3]
        )


class RotationVisualization:
    def __init__(
            self,
            rotated_markers: List,
            config: Dict,
            rotation_error: float,
            output_filepath: Optional[Path] = None,
    ) -> None:
        self.rotated_markers = rotated_markers
        self.config = config
        self.rotation_error = rotation_error
        self.output_filepath = self._create_filepath(
            filepath=convert_to_path(output_filepath)) if output_filepath is not None else ""

    def create_plot(self, plot: bool, save: bool) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for elem in self.rotated_markers:
            ax.scatter(elem[0], elem[1], elem[2], color='orange', alpha=0.5)
        for elem in self.config["REFERENCE_ROTATION_COORDS"]:
            ax.scatter(elem[0], elem[1], elem[2], color='blue', alpha=0.5)
        x = [point[0] for point in self.rotated_markers]
        y = [point[1] for point in self.rotated_markers]
        z = [point[2] for point in self.rotated_markers]
        ax.plot(x, y, z, c="blue")
        ax.scatter(self.config["INVISIBLE_MARKERS"]["x"], self.config["INVISIBLE_MARKERS"]["y"],
                   self.config["INVISIBLE_MARKERS"]["z"], alpha=0)
        fig.suptitle(f"Rotation Error: {self.rotation_error}")

        if save:
            _save_figure(filepath=self.output_filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self, filepath):
        return filepath.parent.joinpath(filepath.stem + ".png")


# ToDo: rework, idx as argument of create_plot, not __init!
class TriangulationVisualization:
    def __init__(
            self,
            df_3D_filepath: Path,
            config: Dict,
            idx: int = 0,
            filename_tag: str = "",
            output_directory: Optional[Path] = None,
    ) -> None:
        self.df_3D = pd.read_csv(df_3D_filepath)
        self.idx = idx
        self.config = config
        self.filename_tag = filename_tag
        self.output_directory = output_directory if output_directory is not None else Path.cwd()
        self.bodyparts = list(set(key.split('_')[0] for key in self.df_3D.keys() if
                                  not any([label in key for label in self.config["markers_to_exclude"]])))
        self.filepath = self._create_filepath()

    def create_plot(self, plot: bool, save: bool, return_fig: bool = False) -> Optional[np.ndarray]:
        fig = plt.figure(figsize=(15, 15))
        fig.clf()
        ax_3d = fig.add_subplot(111, projection='3d')
        all_markers = {marker['name']: marker for marker in self.config["additional_markers_to_plot"]}
        for bodypart in self.bodyparts:
            x, y, z = get_3D_df_keys(bodypart)
            if not math.isnan(self.df_3D.loc[self.idx, x]):
                all_markers[bodypart] = {'name': bodypart,
                                         'x': self.df_3D.loc[self.idx, x],
                                         'y': self.df_3D.loc[self.idx, y],
                                         'z': self.df_3D.loc[self.idx, z],
                                         'alpha': self.config["body_marker_alpha"],
                                         'color': self.config["body_marker_color"],
                                         'size': self.config["body_marker_size"]}
        for marker in all_markers.values():
            ax_3d.text(marker['x'], marker['y'], marker['z'], marker['name'], size=self.config["body_label_size"],
                       alpha=self.config["body_label_alpha"], c=self.config["body_label_color"])
            ax_3d.scatter(marker['x'], marker['y'], marker['z'], s=marker['size'], alpha=marker['alpha'],
                          c=marker['color'])
        for group in self.config['markers_to_connect']:
            _connect_one_set_of_markers(ax=ax_3d, all_markers=all_markers, group=group)
        for group in self.config["markers_to_fill"]:
            _fill_one_set_of_markers(ax=ax_3d, all_markers=all_markers, group=group)
        if return_fig:
            npimage = mplfig_to_npimage(fig)
            plt.close()
            return npimage
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def return_fig(self) -> np.ndarray:
        return self.create_plot(plot=False, save=False, return_fig=True)

    def _create_filepath(self) -> str:
        filename = f"3D_plot_{self.filename_tag}"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)


# ToDo: rewrite function to take df_filepath instead of p3d and function to get bodyparts from df
class CalibrationValidationPlot:
    def __init__(
            self,
            p3d: Dict,
            bodyparts: List[str],
            output_directory: Optional[Union[str, Path]] = None,
            marker_ids_to_connect: List[str] = [],
            filename_tag: str = "",
    ) -> None:
        self.p3d = p3d
        self.bodyparts = bodyparts
        self.filename_tag = filename_tag
        self.output_directory = convert_to_path(output_directory) if output_directory is not None else Path.cwd()
        self.filepath = self._create_filepath()
        self.marker_ids_to_connect = marker_ids_to_connect

    def create_plot(self, plot: bool, save: bool) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.p3d[:, 0], self.p3d[:, 1], self.p3d[:, 2], c="black", s=15)
        _connect_all_marker_ids(
            ax=ax,
            points=self.p3d,
            scheme=self.marker_ids_to_connect,
            bodyparts=self.bodyparts,
        )
        for i in range(len(self.bodyparts)):
            ax.text(
                self.p3d[i, 0],
                self.p3d[i, 1] + 0.01,
                self.p3d[i, 2],
                self.bodyparts[i],
                size=5,
                alpha=0.5,
            )
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self) -> str:
        filename = f"3D_plot_{self.filename_tag}"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)


class PredictionsPlot:
    def __init__(
            self,
            image: Path,
            predictions: Path,
            cam_id: str = "",
            output_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        self.predictions = predictions
        self.image = image
        self.cam_id = cam_id
        if output_directory is None:
            output_directory = predictions.parent
        self.output_directory = convert_to_path(output_directory)
        self.filepath = self._create_filepath()

    def create_plot(self, plot: bool, save: bool) -> None:
        df = pd.read_hdf(self.predictions)
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        image = iio.v3.imread(self.image, index=0)
        plt.imshow(image)
        for scorer, marker, _ in df.columns:
            if df.loc[0, (scorer, marker, "likelihood")] > 0.6:
                x, y = (
                    df.loc[0, (scorer, marker, "x")],
                    df.loc[0, (scorer, marker, "y")],
                )
                plt.scatter(x, y)
                plt.text(x, y, marker)
        plt.title(f"Predictions_{self.cam_id}")
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self) -> str:
        filename = f"predictions_{self.cam_id}"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)


class AlignmentPlotIndividual:
    def __init__(
            self,
            template: np.ndarray,
            led_timeseries: np.ndarray,
            filename: str = "",
            cam_id: str = "",
            output_directory: Optional[Union[str, Path]] = None,
            led_box_size: Optional[int] = None,
            alignment_error: Optional[int] = None,
    ) -> None:
        self.template = template
        self.led_timeseries = led_timeseries
        self.output_directory = convert_to_path(output_directory) if output_directory is not None else Path.cwd()
        self.led_box_size = led_box_size
        self.alignment_error = alignment_error
        self.cam_id = cam_id
        self.filepath = self._create_filepath(filename=filename)

    def create_plot(self, plot: bool, save: bool) -> None:
        end_idx = self.template.shape[0]
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        plt.plot(_zscore(array=self.led_timeseries[:end_idx]))
        plt.plot(_zscore(array=self.template))
        plt.title(f"{self.cam_id}")
        plt.suptitle(
            f"LED box size: {self.led_box_size}\nAlignment error: {self.alignment_error}"
        )
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self, filename: str) -> str:
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)


class AlignmentPlotCrossvalidation:
    def __init__(
            self,
            template: np.ndarray,
            led_timeseries: Dict,
            filename: str = "",
            output_directory: Optional[Union[str, Path]] = None,
    ):
        self.template = template
        self.led_timeseries = led_timeseries
        self.output_directory = convert_to_path(output_directory) if output_directory is not None else Path.cwd()
        self.filepath = self._create_filepath(filename=filename)

    def create_plot(self, plot: bool, save: bool):
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        end_idx = self.template.shape[0]
        for label in self.led_timeseries.keys():
            led_timeseries = self.led_timeseries[label]
            plt.plot(_zscore(array=led_timeseries[:end_idx]), label=label)
        plt.plot(_zscore(array=self.template), c="black", label="Template")
        plt.legend()
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self, filename: str) -> str:
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)


class LEDMarkerPlot:
    def __init__(
            self,
            image: np.ndarray,
            led_center_coordinates: Coordinates,
            box_size: Optional[int] = None,
            cam_id: str = "",
            filename: str = "",
            output_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        self.image = image
        self.led_center_coordinates = led_center_coordinates
        self.box_size = box_size
        self.output_directory = convert_to_path(output_directory) if output_directory is not None else Path.cwd()
        self.filepath = self._create_filepath(filename=filename)
        self.cam_id = cam_id

    def create_plot(self, plot: bool, save: bool):
        fig = plt.figure()
        plt.imshow(self.image)
        plt.scatter(self.led_center_coordinates.x, self.led_center_coordinates.y)

        x_start_index = self.led_center_coordinates.x - (self.box_size // 2)
        x_end_index = self.led_center_coordinates.x + (
                self.box_size - (self.box_size // 2)
        )
        y_start_index = self.led_center_coordinates.y - (self.box_size // 2)
        y_end_index = self.led_center_coordinates.y + (
                self.box_size - (self.box_size // 2)
        )
        plt.plot(
            [x_start_index, x_start_index, x_end_index, x_end_index, x_start_index],
            [y_start_index, y_end_index, y_end_index, y_start_index, y_start_index],
        )
        plt.title(f"{self.cam_id}")
        plt.suptitle(
            f"LED box size: {self.box_size}"
        )
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self, filename: str) -> Path:
        filepath = self.output_directory.joinpath(filename)
        return filepath


class Intrinsics:
    def __init__(self, video_filepath: Union[Path, str],
                 intrinsic_calibration: Dict,
                 filename: str = "",
                 fisheye: bool = False,
                 output_directory: Optional[Union[str, Path]] = None) -> None:
        """
        Construct all necessary attributes for the Intrinsics Class.

        Parameters
        ----------
        video_filepath: Path or str
            The path to the video, that should be used to visualize undistortion.
        intrinsic_calibration: dict
            Intrinsic calibration results containing camera matrix and
            distorsion coefficient at keys 'K' and 'D'.
        filename: str, default ""
            Filename how the plot will be saved to disk.
        fisheye: bool, default False
            If True, the fisheye undistorsion method will be used.
        output_directory: Path or str, optional
            Directory, where the plot will be saved.
        """
        self.video_filepath = convert_to_path(video_filepath)
        self.output_directory = convert_to_path(output_directory) if output_directory is not None else Path.cwd()
        self.filepath = self._create_filepath(filename=filename)
        self.fisheye = fisheye
        self.intrinsic_calibration = intrinsic_calibration
        self._create_all_images(frame_idx=0)

    def create_plot(self, plot: bool, save: bool) -> None:
        fig = plt.figure(figsize=(12, 5), facecolor="white")
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.imshow(self.distorted_input_image)
        plt.title("raw image")
        ax2 = fig.add_subplot(gs[0, 1])
        plt.imshow(self.undistorted_output_image)
        plt.title("undistorted image based on intrinsic calibration")
        if save:
            _save_figure(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self, filename: str) -> Path:
        filepath = self.output_directory.joinpath(filename)
        return filepath

    def _create_all_images(self, frame_idx: int = 0) -> None:
        self.distorted_input_image = load_single_frame_of_video(
            filepath=self.video_filepath, frame_idx=frame_idx
        )
        if self.fisheye:
            self.undistorted_output_image = (
                self._undistort_fisheye_image_for_inspection(
                    image=self.distorted_input_image
                )
            )
        else:
            self.undistorted_output_image = cv2.undistort(
                self.distorted_input_image,
                self.intrinsic_calibration["K"],
                self.intrinsic_calibration["D"],
            )

    def _undistort_fisheye_image_for_inspection(self, image: np.ndarray) -> np.ndarray:
        k_for_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.intrinsic_calibration["K"],
            self.intrinsic_calibration["D"],
            self.intrinsic_calibration["size"],
            np.eye(3),
            balance=0,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.intrinsic_calibration["K"],
            self.intrinsic_calibration["D"],
            np.eye(3),
            k_for_fisheye,
            (
                self.intrinsic_calibration["size"][0],
                self.intrinsic_calibration["size"][1],
            ),
            cv2.CV_16SC2,
        )
        return cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
