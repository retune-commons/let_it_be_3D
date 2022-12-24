from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod

from pathlib import Path
import datetime
import json
import pickle
import itertools as it
import imageio.v3 as iio
import numpy as np
import aniposelib as ap_lib
import cv2
import matplotlib.pyplot as plt

from .utils import load_single_frame_of_video
from .video_synchronization import Synchronizer
from .video_metadata import VideoMetadata


class VideoInterface:
    def __init__(self, metadata: VideoMetadata) -> None:
        self.metadata = metadata

    def inspect_intrinsic_calibration(self, frame_idx: int = 0) -> None:
        distorted_input_image = load_single_frame_of_video(
            filepath=self.metadata.filepath, frame_idx=frame_idx
        )
        if self.metadata.fisheye:
            undistorted_output_image = self._undistort_fisheye_image_for_inspection(
                image=distorted_input_image
            )
        else:
            undistorted_output_image = cv2.undistort(
                distorted_input_image,
                self.metadata.intrinsic_calibration["K"],
                self.metadata.intrinsic_calibration["D"],
            )
        self._plot_distorted_and_undistorted_image(
            distorted_image=distorted_input_image,
            undistorted_image=undistorted_output_image,
        )

    def run_synchronizer(
        self,
        synchronizer: Synchronizer,
        use_gpu: bool,
        output_directory: Path,
        overwrite: bool,
        synchronize_only: bool
    ) -> None:
        self.synchronizer_object = synchronizer(
            video_metadata=self.metadata,
            use_gpu=use_gpu,
            output_directory=output_directory,
        )
        (
            self.synchronized_object_filepath,
            self.already_synchronized,
        ) = self.synchronizer_object.run_synchronization(overwrite=overwrite, synchronize_only = synchronize_only)

    def export_for_aniposelib(self) -> Union[ap_lib.cameras.Camera, Path]:
        if self.synchronized_object_filepath.name.endswith(".h5"):
            return self.synchronized_object_filepath
        elif self.synchronized_object_filepath.name.endswith(".mp4"):
            return self._export_as_aniposelib_Camera_object()

    def _export_as_aniposelib_Camera_object(self) -> ap_lib.cameras.Camera:
        if self.metadata.fisheye:
            camera = ap_lib.cameras.FisheyeCamera(
                name=self.metadata.cam_id,
                matrix=self.metadata.intrinsic_calibration["K"],
                dist=self.metadata.intrinsic_calibration["D"],
                extra_dist=False,
            )
        else:
            camera = ap_lib.cameras.Camera(
                name=self.metadata.cam_id,
                matrix=self.metadata.intrinsic_calibration["K"],
                dist=self.metadata.intrinsic_calibration["D"],
                extra_dist=False,
            )
        return camera

    def _plot_distorted_and_undistorted_image(
        self, distorted_image: np.ndarray, undistorted_image: np.ndarray
    ) -> None:
        fig = plt.figure(figsize=(12, 5), facecolor="white")
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.imshow(distorted_image)
        plt.title("raw image")
        ax2 = fig.add_subplot(gs[0, 1])
        plt.imshow(undistorted_image)
        plt.title("undistorted image based on intrinsic calibration")
        plt.show()

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
