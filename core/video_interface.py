from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
import datetime
from pathlib import Path

import pickle
import imageio.v3 as iio
import numpy as np
import aniposelib as ap_lib
import cv2
import matplotlib.pyplot as plt

from .video_synchronization import Synchronizer
from .video_metadata import VideoMetadata
from .plotting import Intrinsics


class VideoInterface:
    def __init__(self, video_metadata: VideoMetadata, output_dir: Path) -> None:
        self.video_metadata = video_metadata
        self.plot_camera_intrinsics = Intrinsics(
            video_metadata=video_metadata, output_dir=output_dir
        )

    def run_synchronizer(
        self,
        synchronizer: Synchronizer,
        use_gpu: bool,
        output_directory: Path,
        synchronize_only: bool,
        test_mode: bool,
    ) -> None:
        self.synchronizer_object = synchronizer(
            video_metadata=self.video_metadata,
            use_gpu=use_gpu,
            output_directory=output_directory,
        )
        (
            self.marker_detection_filepath,
            self.synchronized_video_filepath,
        ) = self.synchronizer_object.run_synchronization(
            synchronize_only=synchronize_only, test_mode=test_mode
        )

    def export_for_aniposelib(self) -> Union[ap_lib.cameras.Camera, Path]:
        if not self.video_metadata.charuco_video:
            return self.marker_detection_filepath
        else:
            return self._export_as_aniposelib_Camera_object()

    def inspect_intrinsic_calibration(self) -> None:
        self.plot_camera_intrinsics.plot(plot=True)

    def _export_as_aniposelib_Camera_object(self) -> ap_lib.cameras.Camera:
        if self.video_metadata.fisheye:
            camera = ap_lib.cameras.FisheyeCamera(
                name=self.video_metadata.cam_id,
                matrix=self.video_metadata.intrinsic_calibration["K"],
                dist=self.video_metadata.intrinsic_calibration["D"],
                extra_dist=False,
            )
        else:
            camera = ap_lib.cameras.Camera(
                name=self.video_metadata.cam_id,
                matrix=self.video_metadata.intrinsic_calibration["K"],
                dist=self.video_metadata.intrinsic_calibration["D"],
                extra_dist=False,
            )
        return camera
