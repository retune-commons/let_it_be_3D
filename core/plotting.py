from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import math

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
import pandas as pd
import imageio as iio

from .video_metadata import VideoMetadata
from .utils import Coordinates, load_single_frame_of_video, convert_to_path


class Plotting(ABC):
    def _save(self, filepath: str):
        if convert_to_path(filepath).exists():
            convert_to_path(filepath).unlink()
        plt.savefig(filepath, dpi=400)

    @abstractmethod
    def _create_plot(self):
        pass

    @abstractmethod
    def _create_filepath(self):
        pass

    def _zscore(self, array: np.ndarray) -> np.ndarray:
        return (array - np.mean(array)) / np.std(array, ddof=0)


class Plot3D(Plotting):
    def _create_filepath(self) -> str:
        filename = f"3D_plot_{self.filename_tag}"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)


class Triangulation_Visualization(Plot3D):
    def __init__(
        self,
        triangulation_object: "Triangulation",
        plot: bool = False,
        save: bool = True,
        idx: int = 0,
        likelihood_threshold: float = 0.6,
    ) -> None:
        self.p3d = triangulation_object.anipose_io["p3ds"][idx]
        self.idx = idx
        self.likelihood_threshold = likelihood_threshold
        self.bodyparts = triangulation_object.anipose_io["bodyparts"]
        self.filename_tag = triangulation_object.csv_output_filepath.stem
        self.output_directory = triangulation_object.output_directory
        self.triangulation_dlc_cams_filepaths = (
            triangulation_object.triangulation_dlc_cams_filepaths
        )
        self.videos_cams_filepaths = {
            video_metadata.cam_id: video_metadata.filepath
            for video_metadata in triangulation_object.metadata_from_videos.values()
        }
        self.filepath = self._create_filepath()

        self._create_plot(plot=plot, save=save)

    def _create_plot(self, plot: bool, save: bool, return_fig: bool = False) -> None:
        ncols = len(self.triangulation_dlc_cams_filepaths.keys())
        nrows = 2
        grid = GridSpec(
            nrows,
            ncols,
            left=0.1,
            bottom=0.15,
            right=0.94,
            top=0.94,
            wspace=0.3,
            hspace=0.3,
        )

        fig = plt.figure(figsize=(15, 15))
        fig.clf()

        ax_3d = fig.add_subplot(grid[0, 0 : ncols - 1], projection="3d")
        ax_3d.scatter(self.p3d[:, 0], self.p3d[:, 1], self.p3d[:, 2], s=15, c="blue")
        for i in range(len(self.bodyparts)):
            if not math.isnan(self.p3d[i, 0]):
                ax_3d.text(
                    self.p3d[i, 0],
                    self.p3d[i, 1] + 0.01,
                    self.p3d[i, 2],
                    self.bodyparts[i],
                    size=5,
                    alpha=0.5,
                )
        for n, cam in enumerate(self.triangulation_dlc_cams_filepaths.keys()):
            df = pd.read_hdf(self.triangulation_dlc_cams_filepaths[cam])
            ax_2d = fig.add_subplot(grid[1, n])
            image = iio.v3.imread(self.videos_cams_filepaths[cam], index=self.idx)
            ax_2d.imshow(image)
            for scorer, marker, _ in df.columns:
                idx = df.index[self.idx]
                try:
                    if (
                        df.loc[idx, (scorer, marker, "likelihood")]
                        > self.likelihood_threshold
                    ):
                        x, y = (
                            df.loc[idx, (scorer, marker, "x")],
                            df.loc[idx, (scorer, marker, "y")],
                        )
                        ax_2d.scatter(x, y, s=10, c="blue")
                except ValueError:
                    if (
                        df.loc[idx, (scorer, marker, "likelihood")].values[0]
                        > self.likelihood_threshold
                    ):
                        x, y = (
                            df.loc[idx, (scorer, marker, "x")].values[0],
                            df.loc[idx, (scorer, marker, "y")].values[0],
                        )
                        ax_2d.scatter(x, y, s=10, c="blue")
            ax_2d.set_title(f"{cam}")
        fig.suptitle("3D_Plot_with_2D_Plots")

        if return_fig:
            from moviepy.video.io.bindings import mplfig_to_npimage

            npimage = mplfig_to_npimage(fig)
            plt.close()
            return npimage
        if save:
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)

    def return_fig(self) -> None:
        return self._create_plot(plot=False, save=False, return_fig=True)


class Calibration_Validation_Plot(Plot3D):
    def __init__(
        self,
        p3d: Dict,
        bodyparts: List[str],
        output_directory: Path,
        marker_ids_to_connect: List[str] = [],
        plot: bool = False,
        save: bool = True,
    ) -> None:
        self.p3d = p3d
        self.bodyparts = bodyparts
        self.filename_tag = "calvin"
        self.output_directory = convert_to_path(output_directory)
        self.filepath = self._create_filepath()
        self._create_plot(
            plot=plot, save=save, marker_ids_to_connect=marker_ids_to_connect
        )

    def _create_plot(self, plot: bool, save: bool, marker_ids_to_connect: List) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.p3d[:, 0], self.p3d[:, 1], self.p3d[:, 2], c="black", s=15)
        self._connect_all_marker_ids(
            ax=ax,
            points=self.p3d,
            scheme=marker_ids_to_connect,
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
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)

    def _connect_all_marker_ids(
        self,
        ax: plt.Figure,
        points: np.ndarray,
        scheme: List[Tuple[str]],
        bodyparts: List[str],
    ) -> List[plt.Figure]:
        # ToDo: correct type hints
        cmap = plt.get_cmap("tab10")
        bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
        lines = []
        for i, bps in enumerate(scheme):
            line = self._connect_one_set_of_marker_ids(
                ax=ax, points=points, bps=bps, bp_dict=bp_dict, color=cmap(i)[:3]
            )
            lines.append(line)
        return lines  # return neccessary?

    def _connect_one_set_of_marker_ids(
        self,
        ax: plt.Figure,
        points: np.ndarray,
        bps: List[str],
        bp_dict: Dict,
        color: np.ndarray,
    ) -> plt.Figure:
        # ToDo: correct type hints
        ixs = [bp_dict[bp] for bp in bps]
        return ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2], color=color)


class Predictions_Plot(Plotting):
    def __init__(
        self,
        image: Path,
        predictions: Path,
        cam_id: str,
        plot: bool = False,
        save: bool = True,
        output_directory: Optional[Path] = None,
    ) -> None:
        self.predictions = predictions
        self.image = image
        self.cam_id = cam_id
        if output_directory == None:
            output_directory = predictions.parent
        self.output_directory = convert_to_path(output_directory)
        self.filepath = self._create_filepath()
        self._create_plot(plot=plot, save=save)

    def _create_filepath(self) -> str:
        filename = f"predictions_{self.cam_id}"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)

    def _create_plot(self, plot: bool, save: bool) -> None:
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
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)


class Alignment_Plot_Individual(Plotting):
    def __init__(
        self,
        template: np.ndarray,
        led_timeseries: np.ndarray,
        video_metadata: VideoMetadata,
        output_directory: Path,
        led_box_size: int,
        alignment_error: int,
        plot: bool = False,
        save: bool = True,
    ) -> None:
        self.template = template
        self.led_timeseries = led_timeseries
        self.video_metadata = video_metadata
        self.output_directory = convert_to_path(output_directory)
        self.led_box_size = led_box_size
        self.alignment_error = alignment_error
        self.filepath = self._create_filepath()
        self._create_plot(plot=plot, save=save)

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)

    def _create_filepath(self) -> str:
        if self.video_metadata.fps > self.video_metadata.target_fps:
            tag = f"_downsampled{self.video_metadata.target_fps}"
        else:
            tag = f"_upsampled{self.video_metadata.target_fps}"

        if self.video_metadata.charuco_video:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_synchronization_individual{tag}"
        else:
            filename = f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_synchronization_individual{tag}"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)

    def _create_plot(self, plot: bool, save: bool) -> None:
        end_idx = self.template.shape[0]
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        plt.plot(self._zscore(array=self.led_timeseries[:end_idx]))
        plt.plot(self._zscore(array=self.template))
        plt.title(f"{self.video_metadata.cam_id}")
        plt.suptitle(
            f"LED box size: {self.led_box_size}\nAlignment error: {self.alignment_error}"
        )
        if save:
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()


class Alignment_Plot_Crossvalidation(Plotting):
    def __init__(
        self,
        template: np.ndarray,
        led_timeseries: Dict,
        metadata: Dict,
        output_directory: Path,
        plot: bool = False,
        save: bool = True,
    ):
        self.template = template
        self.led_timeseries = led_timeseries
        self.metadata = metadata
        self.output_directory = convert_to_path(output_directory)
        self.filepath = self._create_filepath()
        self._create_plot(plot=plot, save=save)

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)

    def _create_filepath(self) -> str:
        if self.metadata["charuco_video"]:
            filename = f'{self.metadata["recording_date"]}_charuco_synchronization_crossvalidation_{self.metadata["fps"]}'
        else:
            filename = f'{self.metadata["mouse_id"]}_{self.metadata["recording_date"]}_{self.metadata["paradigm"]}_synchronization_crossvalidation'
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)

    def _create_plot(self, plot: bool, save: bool):
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        end_idx = self.template.shape[0]
        for label in self.led_timeseries.keys():
            led_timeseries = self.led_timeseries[label]
            plt.plot(self._zscore(array=led_timeseries[:end_idx]), label=label)
        plt.plot(self._zscore(array=self.template), c="black", label="Template")
        plt.legend()
        if save:
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()


class LED_Marker_Plot(Plotting):
    def __init__(
        self,
        image: np.ndarray,
        led_center_coordinates: Coordinates,
        box_size: int,
        video_metadata: VideoMetadata,
        output_directory: Path,
        plot: bool = False,
        save: bool = True,
    ) -> None:
        self.image = image
        self.led_center_coordinates = led_center_coordinates
        self.box_size = box_size
        self.video_metadata = video_metadata
        self.output_directory = convert_to_path(output_directory)
        self.filepath = self._create_filepath()
        self._create_plot(plot=plot, save=save)

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)

    def _create_filepath(self) -> Path:
        if self.video_metadata.charuco_video:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_LED_marker"
        else:
            filename = f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_LED_marker"
        filepath = self.output_directory.joinpath(filename)
        return filepath

    def _create_plot(self, plot: bool, save: bool):
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
        plt.title(f"{self.video_metadata.cam_id}")
        plt.suptitle(
            f"LED box size: {self.box_size}"
        )
        if save:
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()


class Intrinsics(Plotting):
    def __init__(
        self,
        video_metadata: VideoMetadata,
        output_dir: Path = "",
        plot: bool = False,
        save: bool = True,
    ) -> None:
        self.video_metadata = video_metadata
        self._create_all_images(frame_idx=0)
        if save:
            self.filepath = self._create_filepath(output_dir=output_dir)
        self._create_plot(plot=plot, save=save)

    def _create_all_images(self, frame_idx: int = 0) -> None:
        self.distorted_input_image = load_single_frame_of_video(
            filepath=self.video_metadata.filepath, frame_idx=frame_idx
        )
        if self.video_metadata.fisheye:
            self.undistorted_output_image = (
                self._undistort_fisheye_image_for_inspection(
                    image=self.distorted_input_image
                )
            )
        else:
            self.undistorted_output_image = cv2.undistort(
                self.distorted_input_image,
                self.video_metadata.intrinsic_calibration["K"],
                self.video_metadata.intrinsic_calibration["D"],
            )

    def plot(self) -> None:
        self._create_plot(plot=True, save=False)

    def _create_plot(self, plot: bool, save: bool) -> None:
        fig = plt.figure(figsize=(12, 5), facecolor="white")
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.imshow(self.distorted_input_image)
        plt.title("raw image")
        ax2 = fig.add_subplot(gs[0, 1])
        plt.imshow(self.undistorted_output_image)
        plt.title("undistorted image based on intrinsic calibration")
        if save:
            self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

    def _create_filepath(self, output_dir: Path) -> Path:
        if self.video_metadata.charuco_video:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_undistorted_image"
        elif self.video_metadata.recording:
            filename = f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_undistorted_image"
        elif self.video_metadata.calvin:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_calvin_undistorted_image"
        filepath = output_dir.joinpath(filename)
        return filepath

    def _undistort_fisheye_image_for_inspection(self, image: np.ndarray) -> np.ndarray:
        k_for_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.video_metadata.intrinsic_calibration["K"],
            self.video_metadata.intrinsic_calibration["D"],
            self.video_metadata.intrinsic_calibration["size"],
            np.eye(3),
            balance=0,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.video_metadata.intrinsic_calibration["K"],
            self.video_metadata.intrinsic_calibration["D"],
            np.eye(3),
            k_for_fisheye,
            (
                self.video_metadata.intrinsic_calibration["size"][0],
                self.video_metadata.intrinsic_calibration["size"][1],
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
