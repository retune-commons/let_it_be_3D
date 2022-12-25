from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2

from .video_metadata import VideoMetadata
from .utils import Coordinates, load_single_frame_of_video


class Plotting(ABC):
    def _save(self, filepath: str):
        plt.savefig(filepath, dpi=400)

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def _create_filename(self):
        pass

    def _zscore(self, array: np.ndarray) -> np.ndarray:
        return (array - np.mean(array)) / np.std(array, ddof=0)


class Alignment_Plot_Individual(Plotting):
    def __init__(
        self,
        template: np.ndarray,
        led_timeseries: np.ndarray,
        video_metadata: VideoMetadata,
        output_directory: Path,
        led_box_size: int
    ) -> None:
        self.template = template
        self.led_timeseries = led_timeseries
        self.video_metadata = video_metadata
        self.output_directory = output_directory
        self.led_box_size = led_box_size
        self.filepath = self._create_filename()
        self._create_plot(plot = False)

    def plot(self) -> None:
        self._create_plot(plot = True)

    def _create_filename(self) -> str:
        if self.video_metadata.charuco_video:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_synchronization_individual"
        else:
            filename = f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_synchronization_individual"
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)

    def _create_plot(self, plot: bool) -> None:
        end_idx = self.template.shape[0]
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        plt.plot(self._zscore(array=self.led_timeseries[:end_idx]))
        plt.plot(self._zscore(array=self.template))
        plt.title(f"{self.video_metadata.cam_id}")
        plt.suptitle(f"LED box size: {self.led_box_size}")
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
    ):
        self.template = template
        self.led_timeseries = led_timeseries
        self.metadata = metadata
        self.output_directory = output_directory
        self.filepath = self._create_filename()
        self._create_plot(plot = False)

    def plot(self) -> None:
        self._create_plot(plot = True)

    def _create_filename(self) -> str:
        if self.metadata["charuco_video"]:
            filename = f'{self.metadata["recording_date"]}_charuco_synchronization_crossvalidation'
        else:
            filename = f'{self.metadata["mouse_id"]}_{self.metadata["recording_date"]}_{self.metadata["paradigm"]}_synchronization_crossvalidation'
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)

    def _create_plot(self, plot: bool):
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        end_idx = self.template.shape[0]
        for label in self.led_timeseries.keys():
            led_timeseries = self.led_timeseries[label]
            plt.plot(self._zscore(array=led_timeseries[:end_idx]), label=label)
        plt.plot(self._zscore(array=self.template), c="black", label="Template")
        plt.legend()
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
    ) -> None:
        self.image = image
        self.led_center_coordinates = led_center_coordinates
        self.box_size = box_size
        self.video_metadata = video_metadata
        self.output_directory = output_directory
        self.filepath = self._create_filename()
        self._create_plot(plot=False)

    def plot(self) -> None:
        self._create_plot(plot = True)

    def _create_filename(self) -> Path:
        if self.video_metadata.charuco_video:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_LED_marker"
        else:
            filename = f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_LED_marker"
        filepath = self.output_directory.joinpath(filename)
        return filepath

    def _create_plot(self, plot: bool):
        fig = plt.figure()
        plt.imshow(self.image)
        plt.scatter(self.led_center_coordinates.x, self.led_center_coordinates.y)
        
        x_start_index = self.led_center_coordinates.x - (self.box_size // 2)
        x_end_index = self.led_center_coordinates.x + (self.box_size - (self.box_size // 2))
        y_start_index = self.led_center_coordinates.y - (self.box_size // 2)
        y_end_index = self.led_center_coordinates.y + (self.box_size - (self.box_size // 2))
        plt.plot([x_start_index, x_start_index, x_end_index, x_end_index, x_start_index], [y_start_index, y_end_index, y_end_index, y_start_index, y_start_index])
        
        self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()

class Intrinsics(Plotting):
    def __init__(self, metadata: VideoMetadata, output_dir: Path)->None:
        self.metadata = metadata
        self._create_all_images(frame_idx = 0)
        self.filepath = self._create_filename(output_dir = output_dir)
        self.plot(plot=False)
        
    def _create_all_images(self, frame_idx: int = 0) -> None:
        self.distorted_input_image = load_single_frame_of_video(
            filepath=self.metadata.filepath, frame_idx=frame_idx
        )
        if self.metadata.fisheye:
            self.undistorted_output_image = self._undistort_fisheye_image_for_inspection(
                image=self.distorted_input_image
            )
        else:
            self.undistorted_output_image = cv2.undistort(
                self.distorted_input_image,
                self.metadata.intrinsic_calibration["K"],
                self.metadata.intrinsic_calibration["D"],
            )
        
    def plot(
        self, plot: bool) -> None:
        fig = plt.figure(figsize=(12, 5), facecolor="white")
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.imshow(self.distorted_input_image)
        plt.title("raw image")
        ax2 = fig.add_subplot(gs[0, 1])
        plt.imshow(self.undistorted_output_image)
        plt.title("undistorted image based on intrinsic calibration")
        self._save(filepath=self.filepath)
        if plot:
            plt.show()
        plt.close()
        
    def _create_filename(self, output_dir: Path)->Path:
        if self.metadata.charuco_video:
            filename = f"{self.metadata.recording_date}_{self.metadata.cam_id}_charuco_undistorted_image"
        else:
            filename = f"{self.metadata.mouse_id}_{self.metadata.recording_date}_{self.metadata.paradigm}_{self.metadata.cam_id}_undistorted_image"
        filepath = output_dir.joinpath(filename)
        return filepath
        

    def _undistort_fisheye_image_for_inspection(self, image: np.ndarray) -> np.ndarray:
        k_for_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.metadata.intrinsic_calibration["K"],
            self.metadata.intrinsic_calibration["D"],
            self.metadata.intrinsic_calibration["size"],
            np.eye(3),
            balance=0,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.metadata.intrinsic_calibration["K"],
            self.metadata.intrinsic_calibration["D"],
            np.eye(3),
            k_for_fisheye,
            (
                self.metadata.intrinsic_calibration["size"][1],
                self.metadata.intrinsic_calibration["size"][0],
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
    