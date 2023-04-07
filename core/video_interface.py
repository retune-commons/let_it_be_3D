from pathlib import Path
from typing import Union, Dict, Type

import aniposelib as ap_lib

from .plotting import Intrinsics
from .video_metadata import VideoMetadata


class VideoInterface:
    def __init__(
            self, video_metadata: VideoMetadata, output_dir: Path, test_mode: bool = False
    ) -> None:
        self.video_metadata = video_metadata
        if self.video_metadata.calibration:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_undistorted_image"
        elif self.video_metadata.recording:
            filename = f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_undistorted_image"
        elif self.video_metadata.calvin:
            filename = f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_calvin_undistorted_image"
        self.plot_camera_intrinsics = Intrinsics(video_filepath=self.video_metadata.filepath,
                                                 intrinsic_calibration=self.video_metadata.intrinsic_calibration,
                                                 filename=filename, fisheye=self.video_metadata.fisheye,
                                                 output_directory=output_dir)
        self.plot_camera_intrinsics.create_plot(save=not test_mode, plot=False)

    def run_synchronizer(
            self,
            synchronizer: Type,
            output_directory: Path,
            synchronize_only: bool,
            test_mode: bool,
            synchro_metadata: Dict,
    ) -> None:
        self.synchronizer_object = synchronizer(
            video_metadata=self.video_metadata,
            output_directory=output_directory,
            synchro_metadata=synchro_metadata,
        )
        (
            self.marker_detection_filepath,
            self.synchronized_video_filepath,
        ) = self.synchronizer_object.run_synchronization(
            synchronize_only=synchronize_only, test_mode=test_mode
        )

    def export_for_aniposelib(self) -> Union:
        if not self.video_metadata.calibration:
            return self.marker_detection_filepath
        else:
            return self._export_as_aniposelib_camera_object()

    def inspect_intrinsic_calibration(self) -> None:
        self.plot_camera_intrinsics.plot()

    def _export_as_aniposelib_camera_object(self):
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
