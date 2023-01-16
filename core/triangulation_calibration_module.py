from typing import List, Tuple, Dict, Union, Optional
import datetime
from pathlib import Path
from abc import ABC, abstractmethod

import aniposelib as ap_lib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .video_metadata import VideoMetadata
from .video_interface import VideoInterface
from .video_synchronization import (
    RecordingVideoDownSynchronizer,
    RecordingVideoUpSynchronizer,
    CharucoVideoSynchronizer,
)
from .plotting import Alignment_Plot_Crossvalidation
from .marker_detection import ManualAnnotation
from .utils import convert_to_path, create_calibration_key


def exclude_by_framenum(metadata_from_videos: Dict, target_fps: int) -> None:
    # makes no sense since only the beginning, not the end of the videos is synchronized
    synch_framenum_median = np.median(
        [
            video_metadata.framenum_synchronized
            for video_metadata in metadata_from_videos.values()
        ]
    )
    synch_duration_median = synch_framenum_median / target_fps
    for video_metadata in metadata_from_videos.values():
        if video_metadata.duration_synchronized < synch_duration_median - 1:
            video_metadata.exclusion_state = "exclude"


class TestPositionsGroundTruth:

    # ToDo:
    # Add method that allows to remove marker id?

    # ToDo:
    # include "add_marker_ids_to_be_connected_in_3d_plots" and "reference_distance_ids_with_corresponding_marker_ids" in save & load functions?

    def __init__(self) -> None:
        self.marker_ids_with_distances = {}
        self.unique_marker_ids = []
        self.reference_distance_ids_with_corresponding_marker_ids = []
        self.marker_ids_to_connect_in_3D_plot = []
        self._add_maze_corners()
        self.add_marker_ids_to_be_connected_in_3d_plots(
            marker_ids=(
                "maze_corner_open_left",
                "maze_corner_open_right",
                "maze_corner_closed_right",
                "maze_corner_closed_left",
            )
        )
        self.add_marker_ids_and_distance_id_as_reference_distance(
            marker_ids=("maze_corner_open_left", "maze_corner_closed_left"),
            distance_id="maze_length_left",
        )
        self.add_marker_ids_and_distance_id_as_reference_distance(
            marker_ids=("maze_corner_open_right", "maze_corner_closed_right"),
            distance_id="maze_length_right",
        )

    def add_new_marker_id(
        self,
        marker_id: str,
        other_marker_ids_with_distances: List[Tuple[str, Union[int, float]]],
    ) -> None:
        for other_marker_id, distance in other_marker_ids_with_distances:
            self._add_ground_truth_information(
                marker_id_a=marker_id, marker_id_b=other_marker_id, distance=distance
            )
            self._add_ground_truth_information(
                marker_id_a=other_marker_id, marker_id_b=marker_id, distance=distance
            )

    def add_marker_ids_and_distance_id_as_reference_distance(
        self, marker_ids: Tuple[str, str], distance_id: str
    ) -> None:
        self.reference_distance_ids_with_corresponding_marker_ids.append(
            (distance_id, marker_ids)
        )

    def add_marker_ids_to_be_connected_in_3d_plots(
        self, marker_ids: Tuple[str]
    ) -> None:
        if marker_ids not in self.marker_ids_to_connect_in_3D_plot:
            self.marker_ids_to_connect_in_3D_plot.append(marker_ids)

    def load_from_disk(self, filepath: Path) -> None:
        with open(filepath, "rb") as io:
            marker_ids_with_distances = pickle.load(io)
        unique_marker_ids = list(marker_ids_with_distances.keys())
        setattr(self, "marker_ids_with_distances", marker_ids_with_distances)
        setattr(self, "unique_marker_ids", unique_marker_ids)

    def save_to_disk(self, filepath: Path) -> None:
        # ToDo: validate filepath, -name, -extension & provide default alternative
        with open(filepath, "wb") as io:
            pickle.dump(self.marker_ids_with_distances, io)

    def _add_ground_truth_information(
        self, marker_id_a: str, marker_id_b: str, distance: Union[int, float]
    ) -> None:
        if marker_id_a not in self.marker_ids_with_distances.keys():
            self.marker_ids_with_distances[marker_id_a] = {}
            self.unique_marker_ids.append(marker_id_a)
        self.marker_ids_with_distances[marker_id_a][marker_id_b] = distance

    def _add_maze_corners(self) -> None:
        maze_width, maze_length = 4, 50
        maze_diagonal = (maze_width**2 + maze_length**2) ** 0.5
        maze_corner_distances = {
            "maze_corner_open_left": [
                ("maze_corner_open_right", maze_width),
                ("maze_corner_closed_right", maze_diagonal),
                ("maze_corner_closed_left", maze_length),
            ],
            "maze_corner_open_right": [
                ("maze_corner_closed_right", maze_length),
                ("maze_corner_closed_left", maze_diagonal),
            ],
            "maze_corner_closed_left": [("maze_corner_closed_right", maze_width)],
        }
        for marker_id, distances in maze_corner_distances.items():
            self.add_new_marker_id(
                marker_id=marker_id, other_marker_ids_with_distances=distances
            )


class Calibration:
    def __init__(
        self,
        calibration_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        output_directory: Optional[Path] = None,
        subgroup: bool = False,
    ) -> None:

        self.calibration_directory = convert_to_path(calibration_directory)
        self.project_config_filepath = convert_to_path(project_config_filepath)
        self.recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        if not subgroup:
            self._create_video_objects(
                calibration_directory=self.calibration_directory,
                project_config_filepath=self.project_config_filepath,
                recording_config_filepath=self.recording_config_filepath,
            )

    def run_synchronization(self, test_mode: bool = False) -> None:
        self.synchronized_charuco_videofiles = {}
        self.camera_objects = []
        self.synchronization_individuals = []
        self.led_detection_individuals = []

        for video_interface in self.charuco_interfaces.values():
            video_interface.run_synchronizer(
                synchronizer=CharucoVideoSynchronizer,
                use_gpu=video_interface.video_metadata.use_gpu,
                output_directory=self.output_directory,
                synchronize_only=True,
                test_mode=test_mode,
            )
            self.synchronized_charuco_videofiles[
                video_interface.video_metadata.cam_id
            ] = str(video_interface.synchronized_video_filepath)
            self.camera_objects.append(video_interface.export_for_aniposelib())

        template = list(self.charuco_interfaces.values())[
            0
        ].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(
            fps=self.target_fps
        )[
            0
        ][
            0
        ]
        self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
            template=template,
            led_timeseries={
                video_interface.video_metadata.cam_id: video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
                for video_interface in self.charuco_interfaces.values()
            },
            metadata={"recording_date": self.recording_date, "charuco_video": True},
            output_directory=self.output_directory,
        )
        self._validate_unique_cam_ids()
        self.initialize_camera_group()
        #exclude_by_framenum(metadata_from_videos=self.metadata_from_videos, target_fps=self.target_fps)

    def run_calibration(
        self,
        use_own_intrinsic_calibration: bool = True,
        verbose: bool = False,
        charuco_calibration_board: Optional = None,
    ) -> None:

        cams = list(self.metadata_from_videos.keys())
        filename = f"{create_calibration_key(videos = cams, recording_date = self.recording_date, calibration_index = self.calibration_index)}.toml"

        self.calibration_output_filepath = self.output_directory.joinpath(filename)
        if charuco_calibration_board == None:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            charuco_calibration_board = ap_lib.boards.CharucoBoard(
                7,
                5,
                square_length=1,
                marker_length=0.8,
                marker_bits=6,
                aruco_dict=aruco_dict,
            )
        videos = [[video] for video in self.synchronized_charuco_videofiles.values()]
        self.camera_group.calibrate_videos(
            videos=videos,
            board=charuco_calibration_board,
            init_intrinsics=not use_own_intrinsic_calibration,
            init_extrinsics=True,
            verbose=verbose,
        )
        self._save_calibration()

    def create_subgroup(self, cam_ids: List[str]) -> None:
        subgroup = Calibration(
            calibration_directory=self.calibration_directory,
            project_config_filepath=self.project_config_filepath,
            recording_config_filepath=self.recording_config_filepath,
            output_directory=self.output_directory,
            subgroup=True,
        )

        subgroup.recording_date = self.recording_date
        subgroup.target_fps = self.target_fps
        subgroup.calibration_index = self.calibration_index

        subgroup.metadata_from_videos = {}
        subgroup.camera_objects = {}
        subgroup.synchronized_charuco_videofiles = {}
        for cam in cam_ids:
            if cam in self.metadata_from_videos.keys():
                if self.metadata_from_videos[cam].exclusion_state == "valid":
                    subgroup.metadata_from_videos[cam] = self.metadata_from_videos[cam]
                    subgroup.camera_objects[cam] = self.camera_objects[cam]
                    subgroup.synchronized_charuco_videofiles[
                        cam
                    ] = self.synchronized_charuco_videofiles[cam]

        subgroup.initialize_camera_group()
        return subgroup

    def initialize_camera_group(self) -> None:
        self.camera_group = ap_lib.cameras.CameraGroup(self.camera_objects)

    def _check_output_directory(self, output_directory: Path) -> None:
        if output_directory != None:
            try:
                if output_directory.exists():
                    self.output_directory = output_directory
                else:
                    self._make_output_directory(
                        project_config_filepath=self.project_config_filepath
                    )
            except AttributeError:
                self._make_output_directory(
                    project_config_filepath=self.project_config_filepath
                )
        else:
            self._make_output_directory(
                project_config_filepath=self.project_config_filepath
            )

    def _make_output_directory(self, project_config_filepath: Path) -> None:
        unnamed_idx = 0
        for file in project_config_filepath.parent.iterdir():
            if str(file.name).startswith("unnamed_calibration_"):
                idx = int(file.stem[20:])
                if idx > unnamed_idx:
                    unnamed_idx = idx
        self.output_directory = project_config_filepath.parent.joinpath(
            f"unnamed_calibration_{str(unnamed_idx+1)}"
        )
        if not self.output_directory.exists():
            Path.mkdir(self.output_directory)

    def _create_video_objects(
        self,
        calibration_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
    ) -> None:
        avi_files = [
            file
            for file in calibration_directory.iterdir()
            if ("Charuco" in file.name or "charuco" in file.name)
            and file.name.endswith(".AVI")
        ]
        avi_files.sort()
        top_cam_file = avi_files[-1]  # hard coded!
        charuco_videofiles = [
            file
            for file in calibration_directory.iterdir()
            if ("Charuco" in file.name or "charuco" in file.name)
            and file.name.endswith(".mp4")
            and "synchronized" not in file.name
        ]
        charuco_videofiles.append(top_cam_file)

        self.charuco_interfaces = {}
        self.metadata_from_videos = {}
        for filepath in charuco_videofiles:
            video_metadata = VideoMetadata(
                video_filepath=filepath,
                recording_config_filepath=recording_config_filepath,
                project_config_filepath=project_config_filepath,
                calibration_dir=calibration_directory,
            )

            self.charuco_interfaces[video_metadata.cam_id] = VideoInterface(
                video_metadata=video_metadata, output_dir=self.output_directory
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata

        self._validate_and_save_metadata_for_recording()

    def _validate_and_save_metadata_for_recording(self) -> None:
        if all(
            [
                video_metadata.recording_date
                for video_metadata in self.metadata_from_videos.values()
            ]
        ):
            self.recording_date = list(self.metadata_from_videos.values())[
                0
            ].recording_date
            self.target_fps = list(self.metadata_from_videos.values())[0].target_fps
            self.led_pattern = list(self.metadata_from_videos.values())[0].led_pattern
            self.calibration_index = list(self.metadata_from_videos.values())[
                0
            ].calibration_index
        else:
            raise ValueError(
                f"The metadata, that was read from the videos in {self.recording_directory}, is not identical.\n"
                "Please check the files and rename them properly!"
            )

    def _validate_unique_cam_ids(self) -> None:
        cam_ids = []
        for ap_cam in self.camera_objects:
            if ap_cam.name not in cam_ids:
                cam_ids.append(ap_cam.name)
            else:
                raise ValueError(
                    f"You added multiple cameras with the cam_id {ap_cam.name}, "
                    "however, all cam_ids must be unique! Please check for duplicates "
                    "in the calibration directory and rename them!"
                )

    def _save_calibration(self) -> None:
        self.camera_group.dump(self.calibration_output_filepath)


class Triangulation(ABC):
    @property
    def score_threshold(self) -> float:
        return 0.5

    @abstractmethod
    def _validate_unique_cam_ids(self) -> None:
        pass

    @abstractmethod
    def _validate_and_save_metadata_for_recording(self) -> None:
        pass

    @abstractmethod
    def _create_video_objects(self) -> None:
        pass

    def run_triangulation(
        self, calibration_toml_filepath: Path, adapt_to_calibration: bool = False
    ):
        self.calibration_toml_filepath = convert_to_path(calibration_toml_filepath)
        self._load_calibration(filepath=self.calibration_toml_filepath)
        self._validate_unique_cam_ids(adapt_to_calibration=adapt_to_calibration)
        self._preprocess_dlc_predictions_for_anipose()
        p3ds_flat = self.camera_group.triangulate(
            self.anipose_io["points_flat"], progress=True
        )
        self._postprocess_triangulations_and_calculate_reprojection_error(
            p3ds_flat=p3ds_flat
        )
        self._get_dataframe_of_triangulated_points()
        self._save_dataframe_as_csv()

    def _check_output_directory(self, output_directory: Path) -> None:
        if output_directory != None:
            try:
                if output_directory.exists():
                    self.output_directory = output_directory
                else:
                    self._make_output_directory(
                        project_config_filepath=self.project_config_filepath
                    )
            except AttributeError:
                self._make_output_directory(
                    project_config_filepath=self.project_config_filepath
                )
        else:
            self._make_output_directory(
                project_config_filepath=self.project_config_filepath
            )

    def _make_output_directory(self, project_config_filepath: Path) -> None:
        unnamed_idx = 0
        for file in project_config_filepath.parent.iterdir():
            if str(file.name).startswith("unnamed_calibration_"):
                idx = int(file.stem[20:])
                if idx > unnamed_idx:
                    unnamed_idx = idx
        self.output_directory = project_config_filepath.parent.joinpath(
            f"unnamed_calibration_{str(unnamed_idx+1)}"
        )
        if not self.output_directory.exists():
            Path.mkdir(self.output_directory)

    def _load_calibration(self, filepath: Path) -> None:
        if filepath.name.endswith(".toml") and filepath.exists():
            self.camera_group = ap_lib.cameras.CameraGroup.load(filepath)
        else:
            raise FileNotFoundError(
                f"The path, given as calibration_toml_filepath\n"
                "does not end with .toml or does not exist!\n"
                "Make sure, that you enter the correct path!"
            )

    def _preprocess_dlc_predictions_for_anipose(self) -> None:
        anipose_io = ap_lib.utils.load_pose2d_fnames(
            fname_dict=self.triangulation_dlc_cams_filepaths
        )
        self.anipose_io = self._add_additional_information_and_continue_preprocessing(
            anipose_io=anipose_io
        )

    def _add_additional_information_and_continue_preprocessing(
        self, anipose_io: Dict
    ) -> Dict:
        n_cams, anipose_io["n_points"], anipose_io["n_joints"], _ = anipose_io[
            "points"
        ].shape
        anipose_io["points"][anipose_io["scores"] < self.score_threshold] = np.nan
        # ??? possibility to weigh cameras differently
        # remove the line above to use built-in methods of aniposelib for error handling?

        anipose_io["points_flat"] = anipose_io["points"].reshape(n_cams, -1, 2)
        anipose_io["scores_flat"] = anipose_io["scores"].reshape(n_cams, -1)
        return anipose_io

    def _postprocess_triangulations_and_calculate_reprojection_error(
        self, p3ds_flat: np.array
    ) -> None:
        self.reprojerr_flat = self.camera_group.reprojection_error(
            p3ds_flat, self.anipose_io["points_flat"], mean=True
        )
        self.p3ds = p3ds_flat.reshape(
            self.anipose_io["n_points"], self.anipose_io["n_joints"], 3
        )
        self.reprojerr = self.reprojerr_flat.reshape(
            self.anipose_io["n_points"], self.anipose_io["n_joints"]
        )
        self.reprojerr_nonan = self.reprojerr[np.logical_not(np.isnan(self.reprojerr))]

    def _get_dataframe_of_triangulated_points(self) -> None:
        all_points_raw = self.anipose_io["points"]
        all_scores = self.anipose_io["scores"]
        _cams, n_frames, n_joints, _ = all_points_raw.shape
        good_points = ~np.isnan(all_points_raw[:, :, :, 0])
        num_cams = np.sum(good_points, axis=0).astype("float")
        all_points_3d = self.p3ds.reshape(n_frames, n_joints, 3)
        all_errors = self.reprojerr_flat.reshape(n_frames, n_joints)
        all_scores[~good_points] = 2
        scores_3d = np.min(all_scores, axis=0)

        # try how df looks like without those 3 lines
        scores_3d[num_cams < 2] = np.nan
        all_errors[num_cams < 2] = np.nan
        num_cams[num_cams < 2] = np.nan

        all_points_3d_adj = all_points_3d
        M = np.identity(3)
        center = np.zeros(3)
        df = pd.DataFrame()
        for bp_num, bp in enumerate(self.anipose_io["bodyparts"]):
            for ax_num, axis in enumerate(["x", "y", "z"]):
                df[bp + "_" + axis] = all_points_3d_adj[:, bp_num, ax_num]
            df[bp + "_error"] = self.reprojerr[:, bp_num]
            df[bp + "_score"] = scores_3d[:, bp_num]
        for i in range(3):
            for j in range(3):
                df["M_{}{}".format(i, j)] = M[i, j]
        for i in range(3):
            df["center_{}".format(i)] = center[i]
        df["fnum"] = np.arange(n_frames)
        self.df = df

    def _save_dataframe_as_csv(self) -> None:
        if self.csv_output_filepath.exists():
            self.csv_output_filepath.unlink()
        self.df.to_csv(self.csv_output_filepath, index=False)


class Triangulation_Recordings(Triangulation):
    def __init__(
        self,
        recording_directory: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        calibration_directory: Path,
        output_directory: Optional[Path] = None,
    ) -> None:

        self.recording_directory = convert_to_path(recording_directory)
        self.project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        self._create_video_objects(
            recording_directory=recording_directory,
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=self.project_config_filepath,
            calibration_directory=calibration_directory,
        )

    def run_synchronization(
        self, synchronize_only: bool = False, test_mode: bool = False
    ) -> None:
        self.synchronization_individuals = []
        self.led_detection_individuals = []
        self.synchronized_videos = {}

        for video_interface in self.recording_interfaces.values():
            if (
                video_interface.video_metadata.fps
                >= video_interface.video_metadata.target_fps
            ):
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoDownSynchronizer,
                    use_gpu=video_interface.video_metadata.use_gpu,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
                    test_mode=test_mode,
                )
            else:
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoUpSynchronizer,
                    use_gpu=video_interface.video_metadata.use_gpu,
                    output_directory=self.output_directory,
                    synchronize_only=self.synchronize_only,
                    test_mode=test_mode,
                )
            self.synchronized_videos[
                video_interface.video_metadata.cam_id
            ] = video_interface.synchronized_video_filepath

        template = list(self.recording_interfaces.values())[
            0
        ].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(
            fps=self.target_fps
        )[
            0
        ][
            0
        ]
        self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
            template=template,
            led_timeseries={
                video_interface.video_metadata.cam_id: video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
                for video_interface in self.recording_interfaces.values()
            },
            metadata={
                "mouse_id": self.mouse_id,
                "recording_date": self.recording_date,
                "paradigm": self.paradigm,
                "charuco_video": False,
            },
            output_directory=self.output_directory,
        )

        if not synchronize_only:
            self.csv_output_filepath = self.create_csv_filepath()
            self.triangulation_dlc_cams_filepaths = {
                video_interface: self.recording_interfaces[
                    video_interface
                ].export_for_aniposelib()
                for video_interface in self.recording_interfaces
            }

        #exclude_by_framenum(metadata_from_videos=self.metadata_from_videos, target_fps=self.target_fps)

    def create_csv_filepath(self) -> None:
        filepath_out = self.output_directory.joinpath(
            f"{self.mouse_id}_{self.recording_date}_{self.paradigm}.csv"
        )
        return filepath_out

    def _create_video_objects(
        self,
        recording_directory: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        calibration_directory: Path,
    ) -> None:
        recording_videofiles = [
            file
            for file in recording_directory.iterdir()
            if file.name.endswith(".mp4") and "synchronized" not in file.name and "Front" not in file.name
        ]
        avi_files = [
            file for file in recording_directory.iterdir() if file.name.endswith(".AVI")
        ]
        avi_files.sort()
        try:
            top_cam_file = avi_files[-1]
            recording_videofiles.append(top_cam_file)
        except:
            pass
            
        self.recording_interfaces = {}
        self.metadata_from_videos = {}
        for filepath in recording_videofiles:
            video_metadata = VideoMetadata(
                video_filepath=filepath,
                recording_config_filepath=recording_config_filepath,
                project_config_filepath=project_config_filepath,
                calibration_dir=calibration_directory,
            )

            self.recording_interfaces[video_metadata.cam_id] = VideoInterface(
                video_metadata=video_metadata, output_dir=self.output_directory
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata
        self._validate_and_save_metadata_for_recording()

    def _validate_and_save_metadata_for_recording(self) -> None:
        if (
            all(
                [
                    video_metadata.recording_date
                    for video_metadata in self.metadata_from_videos.values()
                ]
            )
            and all(
                [
                    video_metadata.paradigm
                    for video_metadata in self.metadata_from_videos.values()
                ]
            )
            and all(
                [
                    video_metadata.mouse_id
                    for video_metadata in self.metadata_from_videos.values()
                ]
            )
            and all(
                [
                    video_metadata.led_pattern
                    for video_metadata in self.metadata_from_videos.values()
                ]
            )
        ):
            self.recording_date = list(self.metadata_from_videos.values())[
                0
            ].recording_date
            self.paradigm = list(self.metadata_from_videos.values())[0].paradigm
            self.mouse_id = list(self.metadata_from_videos.values())[0].mouse_id
            self.target_fps = list(self.metadata_from_videos.values())[0].target_fps
            self.led_pattern = list(self.metadata_from_videos.values())[0].led_pattern
        else:
            raise ValueError(
                f"The metadata, that was read from the videos in {self.recording_directory}, is not identical.\n"
                "Please check the files and rename them properly!"
            )

    def _validate_unique_cam_ids(self, adapt_to_calibration: bool) -> None:
        self.cameras = [camera.name for camera in self.camera_group.cameras]
        # possibility to create empty .h5 for missing recordings?
        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.cameras.sort()
        if not adapt_to_calibration:
            if filepath_keys != self.cameras:
                raise ValueError(
                    f"The cam_ids of the recordings in {self.recording_directory} do not match the cam_ids of the camera_group at {self.calibration_toml_filepath}.\n"
                    "Are there missing or additional files in the calibration or the recording folder?"
                )
        else:
            for camera in filepath_keys:
                if camera not in self.cameras:
                    self.triangulation_dlc_cams_filepaths.pop(camera)

            for camera in self.cameras:
                if camera not in filepath_keys:
                    raise ValueError(f"Missing filepath for {camera}!")


class Triangulation_Positions(Triangulation):
    def __init__(
        self,
        positions_directory: Path,
        calibration_directory: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        output_directory: Optional[Path] = None,
    ) -> None:

        self.positions_directory = convert_to_path(positions_directory)
        self.project_config_filepath = convert_to_path(project_config_filepath)
        self.recording_config_filepath = convert_to_path(recording_config_filepath)
        self.calibration_directory = convert_to_path(calibration_directory)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        self._create_video_objects(
            positions_directory=self.positions_directory,
            project_config_filepath=self.project_config_filepath,
            recording_config_filepath=self.recording_config_filepath,
            calibration_directory=self.calibration_directory,
        )
        self.csv_output_filepath = self.output_directory.joinpath(
            f"Positions_{self.recording_date}.csv"
        )

    def get_marker_predictions(self) -> None:
        self.triangulation_dlc_cams_filepaths = {}
        for cam in self.metadata_from_videos.values():
            h5_output_filepath = self.output_directory.joinpath(
                f"Positions_{cam.recording_date}_{cam.cam_id}.h5"
            )
            self.triangulation_dlc_cams_filepaths[cam.cam_id] = h5_output_filepath
            if cam.calibration_evaluation_type == "manual":
                config = cam.calibration_evaluation_filepath
                if not h5_output_filepath.exists():  # and test_mode
                    manual_interface = ManualAnnotation(
                        object_to_analyse=cam.filepath,
                        output_directory=self.output_directory,
                        marker_detection_directory=config,
                    )
                    manual_interface.analyze_objects(
                        filepath=h5_output_filepath, only_first_frame=True
                    )
            else:
                print(
                    "Template Matching and DLC are not yet implemented for Marker Detection in Positions!"
                )

    def evaluate_calibration(
        self, test_positions_gt: TestPositionsGroundTruth, verbose: bool = True
    ):
        self._add_reprojection_errors_of_all_test_position_markers()
        self._add_all_real_distances_errors(test_positions_gt=test_positions_gt)
        if verbose:
            print(f'Mean reprojection error: {self.anipose_io["reproj_nonan"].mean()}')
            for reference_distance_id, distance_errors in self.anipose_io[
                "distance_errors_in_cm"
            ].items():
                print(
                    f'Using {reference_distance_id} as reference distance, the mean distance error is: {distance_errors["mean_error"]} cm.'
                )

    def _create_video_objects(
        self,
        positions_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        calibration_directory: Path,
    ) -> None:
        avi_files = [
            file
            for file in positions_directory.iterdir()
            if file.name.endswith(".AVI") and ("Positions" in file.name or "positions" in file.name)
        ]
        avi_files.sort()
        top_cam_file = avi_files[-1]  # hardcoded
        position_files = [
            file
            for file in positions_directory.iterdir()
            if (
                file.name.endswith(".tiff")
                or file.name.endswith(".bmp")
                or file.name.endswith(".jpg")
                or file.name.endswith(".png")
            )
            and "Positions" in file.name
        ]
        position_files.append(top_cam_file)

        self.metadata_from_videos = {}
        for filepath in position_files:
            video_metadata = VideoMetadata(
                video_filepath=filepath,
                recording_config_filepath=recording_config_filepath,
                project_config_filepath=project_config_filepath,
                calibration_dir=calibration_directory,
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata
        self.cameras = list(self.metadata_from_videos.keys())
        self._validate_and_save_metadata_for_recording()

    def _validate_and_save_metadata_for_recording(self):
        if all(
            [
                video_metadata.recording_date
                for video_metadata in self.metadata_from_videos.values()
            ]
        ):
            self.recording_date = list(self.metadata_from_videos.values())[
                0
            ].recording_date
        else:
            raise ValueError(
                f"The metadata, that was read from the positions.jpgs in {self.positions_directory}, is not identical.\n"
                "Please check the files and rename them properly!"
            )

    def _validate_unique_cam_ids(self):
        cameras = [camera.name for camera in self.camera_group.cameras]
        # possibility to create empty .h5 for missing recordings?
        if self.cameras.sort() != cameras.sort():
            raise ValueError(
                f"The cam_ids of the recordings in {self.positions_directory} do not match the cam_ids of the camera_group at {self.calibration_toml_filepath}.\n"
                "Are there missing or additional files in the calibration or the recording folder?"
            )

    """
    # not yet implemented/tested
       
    def _add_reprojection_errors_of_all_test_position_markers(self) -> None:
        self.reprojection_errors_test_position_markers = {}
        all_reprojection_errors = []
        for key in self.df.iloc[0].keys():
            if "error" in key:
                reprojection_error = self.df[key].iloc[0]
                marker_id = key[: key.find("_error")]
                self.reprojection_errors_test_position_markers[
                    marker_id
                ] = reprojection_error  # since we only have a single image
                if type(reprojection_error) != np.nan:
                    # ToDo:
                    # confirm that it would actually be a numpy nan
                    # or as alternative, use something like this after blindly appending all errors to drop the nan´s:
                    # anipose_io['reprojerr'][np.logical_not(np.isnan(anipose_io['reprojerr']))]
                    all_reprojection_errors.append(reprojection_error)
        self.reprojection_errors_test_position_markers["mean"] = np.asarray(
            all_reprojection_errors
        ).mean()

    def _add_all_real_distances_errors(
        self, test_positions_gt: TestPositionsGroundTruth
    ) -> None:
        all_distance_to_cm_conversion_factors = self._get_conversion_factors_from_different_references(
            test_positions_gt=test_positions_gt
        )
        self._add_distances_in_cm_for_each_conversion_factor(
            anipose_io=anipose_io,
            conversion_factors=all_distance_to_cm_conversion_factors,
        )
        self._add_distance_errors(
            gt_distances=test_positions_gt.marker_ids_with_distances
        )

    def _get_conversion_factors_from_different_references(
        self, test_positions_gt: TestPositionsGroundTruth
    ) -> Dict:  # Tuple? List?
        all_conversion_factors = {}
        for (
            reference_distance_id,
            reference_marker_ids,
        ) in test_positions_gt.reference_distance_ids_with_corresponding_marker_ids:
            distance_in_cm = test_positions_gt.marker_ids_with_distances[
                reference_marker_ids[0]
            ][reference_marker_ids[1]]
            distance_to_cm_conversion_factor = self._get_xyz_to_cm_conversion_factor(
                reference_marker_ids=reference_marker_ids, distance_in_cm=distance_in_cm
            )
            all_conversion_factors[
                reference_distance_id
            ] = distance_to_cm_conversion_factor
        return all_conversion_factors

    def _get_xyz_to_cm_conversion_factor(
        self, reference_marker_ids: Tuple[str, str], distance_in_cm: Union[int, float]
    ) -> float:
        distance_in_triangulation_space = self._get_xyz_distance_in_triangulation_space(
            marker_ids=reference_marker_ids
        )
        return distance_in_triangulation_space / distance_in_cm

    def _get_xyz_distance_in_triangulation_space(
        self, marker_ids: Tuple[str, str]
    ) -> float:
        squared_differences = [
            (self.df[f"{marker_ids[0]}_{axis}"] - self.df[f"{marker_ids[1]}_{axis}"])
            ** 2
            for axis in ["x", "y", "z"]
        ]
        return sum(squared_differences) ** 0.5

    def _add_distances_in_cm_for_each_conversion_factor(self, conversion_factors: Dict):
        self.distances_in_cm = {}
        for reference_distance_id, conversion_factor in conversion_factors.items():
            self.distances_in_cm[
                reference_distance_id
            ] = self._convert_all_xyz_distances(conversion_factor=conversion_factor)

    def _convert_all_xyz_distances(self, conversion_factor: float) -> Dict:
        marker_id_combinations = it.combinations(self.anipose_io["bodyparts"], 2)
        all_distances_in_cm = {}
        for marker_id_a, marker_id_b in marker_id_combinations:
            if marker_id_a not in all_distances_in_cm.keys():
                all_distances_in_cm[marker_id_a] = {}
            xyz_distance = self._get_xyz_distance_in_triangulation_space(
                marker_ids=(marker_id_a, marker_id_b)
            )
            all_distances_in_cm[marker_id_a][marker_id_b] = (
                xyz_distance / conversion_factor
            )
        return all_distances_in_cm

    def _add_distance_errors(self, gt_distances: Dict) -> None:
        self.distance_errors_in_cm = {}
        for (
            reference_distance_id,
            triangulated_distances,
        ) in self.distances_in_cm.items():
            self.distance_errors_in_cm[reference_distance_id] = {}
            marker_ids_with_distance_error = self._compute_differences_between_triangulated_and_gt_distances(
                triangulated_distances=triangulated_distances, gt_distances=gt_distances
            )
            all_distance_errors = [
                distance_error
                for marker_id_a, marker_id_b, distance_error in marker_ids_with_distance_error
            ]
            mean_distance_error = np.asarray(all_distance_errors).mean()
            self.distance_errors_in_cm[reference_distance_id] = {
                "individual_errors": marker_ids_with_distance_error,
                "mean_error": mean_distance_error,
            }

    def _compute_differences_between_triangulated_and_gt_distances(
        self, triangulated_distances: Dict, gt_distances: Dict
    ) -> List[Tuple[str, str, float]]:
        marker_ids_with_distance_error = []
        for marker_id_a in triangulated_distances.keys():
            for marker_id_b in triangulated_distances[marker_id_a].keys():
                if (marker_id_a in gt_distances.keys()) & (
                    marker_id_b in gt_distances[marker_id_a].keys()
                ):
                    gt_distance = gt_distances[marker_id_a][marker_id_b]
                    triangulated_distance = triangulated_distances[marker_id_a][
                        marker_id_b
                    ]
                    distance_error = gt_distance - triangulated_distance
                    marker_ids_with_distance_error.append(
                        (marker_id_a, marker_id_b, distance_error)
                    )
        return marker_ids_with_distance_error
    """
