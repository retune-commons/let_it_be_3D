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

from .video_synchronization import Synchronizer
from .video_metadata import VideoMetadata
from .plotting import Intrinsics


class VideoInterface:
    def __init__(self, metadata: VideoMetadata, output_dir: Path) -> None:
        self.metadata = metadata
        self.plot_camera_intrinsics = Intrinsics(metadata = metadata, output_dir = output_dir)
        
    def inspect_intrinsic_calibration(self) -> None:
        self.plot_camera_intrinsics.plot(plot = True)

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
            self.marker_detection_path,
            self.synchronized_video_filepath,
            self.already_synchronized,
        ) = self.synchronizer_object.run_synchronization(overwrite=overwrite, synchronize_only = synchronize_only)

    def export_for_aniposelib(self) -> Union[ap_lib.cameras.Camera, Path]:
        if not self.metadata.charuco_video:
            return self.marker_detection_path
        else:
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
