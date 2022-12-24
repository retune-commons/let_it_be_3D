from typing import List, Tuple, Dict, Union, Optional
import datetime
from abc import ABC, abstractmethod

from tqdm.auto import tqdm as TQDM
from pathlib import Path
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
        maze_diagonal = (maze_width ** 2 + maze_length ** 2) ** 0.5
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
        overwrite: bool = False,
        load_calibration: bool = False,
        max_frame_count: int = 300,
        use_gpu: bool = True,
    ) -> None:
        self.output_directory = output_directory
        self._create_video_objects(
            calibration_directory=calibration_directory,
            load_calibration=load_calibration,
            project_config_filepath=project_config_filepath,
            recording_config_filepath=recording_config_filepath,
            overwrite=overwrite,
            max_frame_count=max_frame_count,
            use_gpu=use_gpu,
            synchronize_only = True,
        )
        self._validate_unique_cam_ids()
        # user input to choose the correct file, if there are multiple for one cam_id?
        self._initialize_camera_group()

    def run_calibration(
        self,
        use_own_intrinsic_calibration: bool = True,
        verbose: bool = False,
        charuco_calibration_board: Optional[ap_lib.boards.CharucoBoard] = None,
    ) -> None:
        # ToDo
        # confirm type hints
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
        videos = [[video] for video in self.synchronized_charuco_videofiles]
        self.camera_group.calibrate_videos(
            videos=videos,
            board=charuco_calibration_board,
            init_intrinsics=not use_own_intrinsic_calibration,
            init_extrinsics=True,
            verbose=verbose,
        )
        self._save_calibration()

    def _create_video_objects(
        self,
        calibration_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        max_frame_count: int,
        overwrite: bool,
        load_calibration: bool,
        synchronize_only: bool,
        use_gpu: bool = True,
    ) -> None:
        avi_files = [
            file
            for file in calibration_directory.iterdir()
            if ("Charuco" in file.name or "charuco" in file.name)
            and file.name.endswith(".AVI")
            and "synchronized" not in file.name
        ]
        avi_files.sort()
        top_cam_file = avi_files[
            -1
        ]  # if there are multiple .AVI files, since CinePlex doesnt overwrite files if the recording was started more than once, the last file is used as topcam file (alternative: file with highest filesize based on pathlib)
        charuco_videofiles = [
            file
            for file in calibration_directory.iterdir()
            if ("Charuco" in file.name or "charuco" in file.name)
            and file.name.endswith(".mp4")
            and "synchronized" not in file.name
        ]
        charuco_videofiles.append(top_cam_file)
        charuco_metadata = [
            VideoMetadata(
                video_filepath=filepath,
                recording_config_filepath=recording_config_filepath,
                project_config_filepath=project_config_filepath,
                load_calibration=load_calibration,
                calibration_dir = calibration_directory,
            )
            for filepath in charuco_videofiles
        ]
        charuco_interfaces = [
            VideoInterface(metadata=video_metadata)
            for video_metadata in charuco_metadata
        ]
        self.metadata_from_videos = [
            video_interface.metadata for video_interface in charuco_interfaces
        ]
        self._validate_and_save_metadata_for_recording()

        if self.output_directory != None:
            try:
                if self.output_directory.exists():
                    pass
                else:
                    self._make_output_dir(
                        project_config_filepath=project_config_filepath
                    )
            except AttributeError:
                self._make_output_dir(project_config_filepath=project_config_filepath)
        else:
            self._make_output_dir(project_config_filepath=project_config_filepath)
        print(f"Started analysis. Saving files at {self.output_directory}.")

        bar = TQDM(
            total=len(charuco_interfaces),
            desc=f"Synchronizing calibration videos from {self.recording_date}",
        )
        for video_interface in charuco_interfaces:
            video_interface.run_synchronizer(
                synchronizer=CharucoVideoSynchronizer,
                use_gpu=use_gpu,
                output_directory=self.output_directory,
                overwrite=overwrite,
                synchronize_only = synchronize_only
            )
            bar.update(1)
        bar.close()
        self.synchronized_charuco_videofiles = [
            str(video_interface.synchronized_object_filepath)
            for video_interface in charuco_interfaces
        ]
        self.camera_objects = [
            video_interface.export_for_aniposelib()
            for video_interface in charuco_interfaces
        ]
        # better solution for the 3 lines above?

        self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
            template=charuco_interfaces[
                0
            ].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(
                fps=self.target_fps
            )[
                0
            ][
                0
            ],
            led_timeseries={
                video_interface.metadata.cam_id: video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
                for video_interface in charuco_interfaces
                if not video_interface.already_synchronized
            },
            metadata={"recording_date": self.recording_date, "charuco_video": True},
            output_directory=self.output_directory,
        )
        self.synchronization_individuals = [
            video_interface.synchronizer_object.synchronization_individual
            for video_interface in charuco_interfaces
            if not video_interface.already_synchronized
        ]
        self.led_detection_individuals = [
            video_interface.synchronizer_object.led_detection
            for video_interface in charuco_interfaces
            if not video_interface.already_synchronized
        ]

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

    def _validate_and_save_metadata_for_recording(self) -> None:
        if all([metadata.recording_date for metadata in self.metadata_from_videos]):
            self.recording_date = self.metadata_from_videos[0].recording_date
            self.target_fps = self.metadata_from_videos[0].target_fps
        else:
            raise ValueError(
                f"The metadata, that was read from the videos in {self.recording_directory}, is not identical.\n"
                "Please check the files and rename them properly!"
            )

    def _make_output_dir(self, project_config_filepath: Path) -> None:
        self.output_directory = project_config_filepath.parent.joinpath(
            f"{self.recording_date}_Analysis"
        )
        if not self.output_directory.exists():
            Path.mkdir(self.output_directory)

    def _initialize_camera_group(self) -> None:
        self.camera_group = ap_lib.cameras.CameraGroup(self.camera_objects)

    def _save_calibration(self) -> None:
        calibration_output_filepath = self.output_directory.joinpath(
            f"calibration_{self.recording_date}.toml"
        )
        self.camera_group.dump(calibration_output_filepath)


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

    def _make_output_dir(self, project_config_filepath: Path) -> None:
        self.output_directory = project_config_filepath.parent.joinpath(
            f"{self.recording_date}_Analysis"
        )
        if not self.output_directory.exists():
            Path.mkdir(self.output_directory)

    def run_triangulation(self):
        self._preprocess_dlc_predictions_for_anipose()
        p3ds_flat = self.camera_group.triangulate(
            self.anipose_io["points_flat"], progress=True
        )
        self._postprocess_triangulations_and_calculate_reprojection_error(
            p3ds_flat=p3ds_flat
        )
        self._get_dataframe_of_triangulated_points()
        self._save_dataframe_as_h5()

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
        # proove type hints
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

    def _save_dataframe_as_h5(self) -> None:
        self.df.to_csv(self.csv_output_filepath)


class Triangulation_Recordings(Triangulation):
    def __init__(
        self,
        recording_directory: Path,
        calibration_toml_filepath: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        overwrite: bool = False,
        max_frame_count: int = 300,
        load_calibration: bool = False,
        output_directory: Optional[Path] = None,
        use_gpu: bool = True,
        synchronize_only: bool = False
    ) -> None:
        self.recording_directory = recording_directory
        self.output_directory = output_directory
        self.synchronize_only = synchronize_only
        self._load_calibration(filepath=calibration_toml_filepath)
        self._create_video_objects(
            recording_directory=recording_directory,
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
            load_calibration=load_calibration,
            max_frame_count=max_frame_count,
            calibration_directory=calibration_toml_filepath.parent,
            overwrite=overwrite,
            use_gpu=use_gpu,
        )
        self._validate_unique_cam_ids()
        # user input to choose the correct file, if there are multiple files for one cam_id?

    def _create_video_objects(
        self,
        recording_directory: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        max_frame_count: int,
        overwrite: bool,
        calibration_directory: Path,
        load_calibration: bool = False,
        use_gpu: bool = True,
    ) -> None:
        avi_files = [
            file
            for file in recording_directory.iterdir()
            if file.name.endswith(".AVI") and "synchronized" not in file.name
        ]
        avi_files.sort()
        top_cam_file = avi_files[
            -1
        ]  # if there are multiple .AVI files, since CinePlex doesnt overwrite files if the recording was started more than once, the last file is used as topcam file (alternative: file with highest filesize based on pathlib)
        recording_videofiles = [
            file
            for file in recording_directory.iterdir()
            if file.name.endswith(".mp4") and "synchronized" not in file.name
        ]
        recording_videofiles.append(top_cam_file)
        recording_metadata = [
            VideoMetadata(
                video_filepath=filepath,
                recording_config_filepath=recording_config_filepath,
                project_config_filepath=project_config_filepath,
                load_calibration=load_calibration,
                calibration_dir = calibration_directory
            )
            for filepath in recording_videofiles
        ]
        recording_interfaces = [
            VideoInterface(metadata=video_metadata)
            for video_metadata in recording_metadata
        ]
        self.metadata_from_videos = [
            video_interface.metadata for video_interface in recording_interfaces
        ]
        self._validate_and_save_metadata_for_recording()

        if self.output_directory != None:
            try:
                if self.output_directory.exists():
                    pass
                else:
                    self._make_output_dir(
                        project_config_filepath=project_config_filepath
                    )
            except AttributeError:
                self._make_output_dir(project_config_filepath=project_config_filepath)
        else:
            self._make_output_dir(project_config_filepath=project_config_filepath)
        print(f"Started analysis. Saving files at {self.output_directory}.")

        bar = TQDM(
            total=len(recording_interfaces),
            desc=f"Synchronizing recording videos from {self.mouse_id}_{self.paradigm}_{self.recording_date}",
        )
        for video_interface in recording_interfaces:
            if video_interface.metadata.fps >= video_interface.metadata.target_fps:
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoDownSynchronizer,
                    use_gpu=use_gpu,
                    overwrite=overwrite,
                    output_directory=self.output_directory,
                    synchronize_only = self.synchronize_only
                )
            else:
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoUpSynchronizer,
                    use_gpu=use_gpu,
                    overwrite=overwrite,
                    output_directory=self.output_directory,
                    synchronize_only = self.synchronize_only
                )
            bar.update(1)
        bar.close()
        
        self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
            template=recording_interfaces[
                0
            ].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(
                fps=self.target_fps
            )[
                0
            ][
                0
            ],
            led_timeseries={
                video_interface.metadata.cam_id: video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
                for video_interface in recording_interfaces
                if not video_interface.already_synchronized
            },
            metadata={
                "mouse_id": self.mouse_id,
                "recording_date": self.recording_date,
                "paradigm": self.paradigm,
            },
            output_directory=self.output_directory,
        )
        self.synchronization_individuals = [
            video_interface.synchronizer_object.synchronization_individual
            for video_interface in recording_interfaces
            if not video_interface.already_synchronized
        ]
        self.led_detection_individuals = [
            video_interface.synchronizer_object.led_detection
            for video_interface in recording_interfaces
            if not video_interface.already_synchronized
        ]
        
        if not self.synchronize_only:
            self.csv_output_filepath = self.output_directory.joinpath(
                f"{self.mouse_id}_{self.recording_date}_{self.paradigm}.csv"
            )
            self.triangulation_dlc_cams_filepaths = {
                video_interface.metadata.cam_id: video_interface.export_for_aniposelib()
                for video_interface in recording_interfaces
            }

        

    def _validate_unique_cam_ids(self) -> None:
        self.cameras = [camera.name for camera in self.camera_group.cameras]
        # possibility to create empty .h5 for missing recordings?
        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.cameras.sort()
        if filepath_keys != self.cameras:
            raise ValueError(
                f"The cam_ids of the recordings in {self.recording_directory} do not match the cam_ids of the camera_group at {self.calibration_toml_filepath}.\n"
                "Are there missing or additional files in the calibration or the recording folder?"
            )

    def _validate_and_save_metadata_for_recording(self) -> None:
        # check for length of videos!!
        if (
            all([metadata.recording_date for metadata in self.metadata_from_videos])
            and all([metadata.paradigm for metadata in self.metadata_from_videos])
            and all([metadata.mouse_id for metadata in self.metadata_from_videos])
        ):
            self.recording_date = self.metadata_from_videos[0].recording_date
            self.paradigm = self.metadata_from_videos[0].paradigm
            self.mouse_id = self.metadata_from_videos[0].mouse_id
            self.target_fps = self.metadata_from_videos[0].target_fps
        else:
            raise ValueError(
                f"The metadata, that was read from the videos in {self.recording_directory}, is not identical.\n"
                "Please check the files and rename them properly!"
            )


class Triangulation_Positions(Triangulation):
    def __init__(
        self,
        positions_directory: Path,
        calibration_toml_filepath: Path,
        config_filepath: Path,
        output_directory: Optional[Path] = None,
        overwrite: bool = False,
        load_calibration: bool = False,
    ) -> None:
        self.positions_directory = positions_directory
        self._load_calibration(filepath=calibration_toml_filepath)
        self.output_directory = output_directory
        self._create_video_objects(
            positions_directory=positions_directory,
            config_filepath=config_filepath,
            load_calibration=load_calibration,
            calibration_directory=calibration_toml_filepath.parent,
            overwrite=overwrite,
        )
        self._validate_unique_cam_ids()
        # user input to choose the correct file, if there are multiple files for one cam_id?

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
        positions_directory: Path,
        config_filepath: Path,
        load_calibration: bool,
        calibration_directory: Path,
        overwrite: bool,
    ) -> None:
        avi_files = [
            file
            for file in recording_directory.iterdir()
            if file.name.endswith(".AVI") and "Positions" in file.name
        ]
        avi_files.sort()
        top_cam_file = avi_files[
            -1
        ]  # if there are multiple .AVI files, since CinePlex doesnt overwrite files if the recording was started more than once, the last file is used as topcam file (alternative: file with highest filesize based on pathlib)
        position_files = [
            file for file in positions_directory.iterdir() if file.name.endswith(".mp4")
        ]
        position_files.append(top_cam_file)
        positions_metadata = [
            VideoMetadata(
                filepath=filepath,
                config_filepath=config_filepath,
                load_calibration=load_calibration,
                calibration_dir = calibration_directory,
            )
            for filepath in position_files
        ]
        self.metadata_from_videos = positions_metadata
        self._validate_and_save_metadata_for_recording()

        if self.output_directory != None:
            try:
                if self.output_directory.exists():
                    pass
                else:
                    self._make_output_dir(
                        project_config_filepath=project_config_filepath
                    )
            except AttributeError:
                self._make_output_dir(project_config_filepath=project_config_filepath)
        else:
            self._make_output_dir(project_config_filepath=project_config_filepath)
        print(f"Started analysis. Saving files at {self.output_directory}.")

        self.csv_output_filepath = self.positions_directory.joinpath(
            f"Positions_{self.recording_date}.csv"
        )
        self.triangulation_dlc_cams_filepaths = {
            metadata.cam_id: metadata.filepath for metadata in positions_metadata
        }

    def _validate_unique_cam_ids():
        self.cameras = [camera.name for camera in self.camera_group.cameras]
        # possibility to create empty .h5 for missing recordings?
        if self.triangulation_dlc_cams_filepaths.keys().sort() != self.cameras.sort():
            raise ValueError(
                f"The cam_ids of the recordings in {self.positions_directory} do not match the cam_ids of the camera_group at {self.calibration_toml_filepath}.\n"
                "Are there missing or additional files in the calibration or the recording folder?"
            )

    def _validate_and_save_metadata_for_recording():
        if all([metadata.recording_date for metadata in self.metadata_from_videos]):
            self.recording_date = self.metadata_from_videos[0].recording_date
        else:
            raise ValueError(
                f"The metadata, that was read from the positions.jpgs in {self.positions_directory}, is not identical.\n"
                "Please check the files and rename them properly!"
            )

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
