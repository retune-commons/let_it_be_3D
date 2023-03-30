from typing import List, Tuple, Dict, Union, Optional, OrderedDict
import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import itertools as it
import math

from scipy.spatial.transform import Rotation
import aniposelib as ap_lib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip

from .video_metadata import VideoMetadata
from .video_interface import VideoInterface
from .video_synchronization import (
    RecordingVideoDownSynchronizer,
    RecordingVideoUpSynchronizer,
    CharucoVideoSynchronizer,
)
from .plotting import (
    Alignment_Plot_Crossvalidation,
    Predictions_Plot,
    Calibration_Validation_Plot,
    Triangulation_Visualization,
    Rotation_Visualization,
)
from .marker_detection import ManualAnnotation, DeeplabcutInterface
from .utils import (
    convert_to_path,
    create_calibration_key,
    read_config,
    check_keys,
    get_multi_index,
    get_3D_df_keys,
    get_3D_array,
)
from .angles_and_distances import (
    add_reprojection_errors_of_all_calibration_validation_markers,
    set_distances_and_angles_for_evaluation,
    fill_in_distances,
    add_all_real_distances_errors,
    set_angles_error_between_screws_and_plane,
)



class Triangulation_Calibration(ABC):
    @abstractmethod
    def _validate_unique_cam_ids(self) -> None:
        pass

    @abstractmethod
    def _validate_and_save_metadata_for_recording(self) -> None:
        pass

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
        directory: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict,
        videometadata_tag: str,
        filetypes: [str],
        filename_tag: str = "",
        test_mode: bool = False,
    ) -> None:
        videofiles = [
            file
            for file in directory.iterdir()
            if filename_tag.lower() in file.name.lower()
            and file.suffix in filetypes
            and "synchronized" not in file.name]

        self.video_interfaces = {}
        self.metadata_from_videos = {}
        for filepath in videofiles:
            video_metadata = VideoMetadata(
                video_filepath=filepath,
                recording_config_dict=recording_config_dict,
                project_config_dict=project_config_dict,
                tag=videometadata_tag,
            )

            self.video_interfaces[video_metadata.cam_id] = VideoInterface(
                video_metadata=video_metadata,
                output_dir=self.output_directory,
                test_mode=test_mode,
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata

        self._validate_and_save_metadata_for_recording()


class Calibration(Triangulation_Calibration):
    def __init__(
        self,
        calibration_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        output_directory: Optional[Path] = None,
        test_mode: bool = False,
    ) -> None:
        self.calibration_directory = convert_to_path(calibration_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        recording_config_dict, project_config_dict = self._get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self._create_video_objects(
            directory=self.calibration_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="calibration",
            filetypes=[".AVI", ".avi", ".mov", ".mp4"],
            filename_tag=self.calibration_tag,
            test_mode=test_mode,
        )
        self.target_fps = min([video_metadata.fps for video_metadata in self.metadata_from_videos.values()])
        for video_metadata in self.metadata_from_videos.values():
            video_metadata.target_fps = self.target_fps

    def run_synchronization(self, test_mode: bool = False) -> None:
        self.synchronized_charuco_videofiles = {}
        self.camera_objects = []
        self.synchronization_individuals = []
        self.led_detection_individuals = []

        for video_interface in self.video_interfaces.values():
            video_interface.run_synchronizer(
                synchronizer=CharucoVideoSynchronizer,
                rapid_aligner_path=self.rapid_aligner_path,
                use_gpu=self.use_gpu,
                output_directory=self.output_directory,
                synchronize_only=True,
                test_mode=test_mode,
                synchro_metadata=self.synchro_metadata,
            )
            self.synchronized_charuco_videofiles[
                video_interface.video_metadata.cam_id
            ] = str(video_interface.synchronized_video_filepath) 
            self.camera_objects.append(video_interface.export_for_aniposelib())

        template = list(self.video_interfaces.values())[0].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(
            fps=self.target_fps)[0][0]
        led_timeseries_crossvalidation = {}
        for video_interface in self.video_interfaces.values():
            try:
                led_timeseries_crossvalidation[
                    video_interface.video_metadata.cam_id
                ] = (video_interface.synchronizer_object.led_timeseries_for_cross_video_validation)
            except:
                pass
        if len(led_timeseries_crossvalidation.keys()) > 0:
            self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                metadata={
                    "recording_date": self.recording_date,
                    "charuco_video": True,
                    "fps": self.target_fps,
                },
                output_directory=self.output_directory,
            )
        self._validate_unique_cam_ids()
        self.initialize_camera_group()

    def run_calibration(
        self,
        use_own_intrinsic_calibration: bool = True,
        verbose: int = 0,
        charuco_calibration_board: Optional = None,
        test_mode: bool = False,
        iteration: Optional[int] = None,
    ) -> None:
        cams = list(self.metadata_from_videos.keys())
        filename = f"{create_calibration_key(videos = cams, recording_date = self.recording_date, calibration_index = self.calibration_index, iteration = iteration)}.toml"

        calibration_filepath = self.output_directory.joinpath(filename)
        if (not test_mode) or (
            test_mode and not calibration_filepath.exists()
        ):
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
            sorted_videos = OrderedDict(
                sorted(self.synchronized_charuco_videofiles.items())
            )
            videos = [[video] for video in sorted_videos.values()]
            self.reprojerr , _ = self.camera_group.calibrate_videos(
                videos=videos,
                board=charuco_calibration_board,
                init_intrinsics=not use_own_intrinsic_calibration,
                init_extrinsics=True,
                verbose=verbose > 1,
            )
            self._save_calibration(calibration_filepath=calibration_filepath)
        else:
            self.reprojerr = 0
        return calibration_filepath

    def initialize_camera_group(self) -> None:
        self.camera_group = ap_lib.cameras.CameraGroup(self.camera_objects)

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(
            video_metadata.recording_date
            for video_metadata in self.metadata_from_videos.values()
        )
        for attribute in [recording_dates]:
            if len(attribute) > 1:
                raise ValueError(
                    f"The filenames of the videos give different metadata! Reasons could be:\n"
                    f"  - video belongs to another calibration\n"
                    f"  - video filename is valid, but wrong\n"
                    f"Go the folder {self.calibration_directory} and check the filenames manually!"
                )
        self.recording_date = list(recording_dates)[0]

    def _get_metadata_from_configs(
        self, recording_config_filepath: Path, project_config_filepath: Path
    ) -> Tuple[Dict]:
        project_config_dict = read_config(path=project_config_filepath)
        recording_config_dict = read_config(path=recording_config_filepath)
        keys_to_check_project = [
            "valid_cam_IDs",
            "paradigms",
            "animal_lines",
            "led_extraction_type",
            "led_extraction_filepath",
            "max_calibration_frames",
            "max_cpu_cores_to_pool",
            "max_ram_digestible_frames",
            "rapid_aligner_path",
            "use_gpu",
            "load_calibration",
            "calibration_tag",
            "calibration_validation_tag",
            "handle_synchro_fails",
            "default_offset_ms",
            "start_pattern_match_ms",
            "end_pattern_match_ms",
            "synchro_error_threshold",
            "synchro_marker",
            "use_2D_filter",
            "score_threshold",
            'num_frames_to_pick'
        ]
        missing_keys = check_keys(
            dictionary=project_config_dict, list_of_keys=keys_to_check_project
        )
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing metadata information in the project_config_file {project_config_filepath} for {missing_keys}."
            )
            
        self.synchro_metadata={key: project_config_dict[key] for key in ["handle_synchro_fails", "default_offset_ms", "start_pattern_match_ms", "end_pattern_match_ms", "synchro_error_threshold", "synchro_marker", 'num_frames_to_pick']}

        keys_to_check_recording = [
            "led_pattern",
            "target_fps",
            "calibration_index",
            "recording_date",
        ]
        missing_keys = check_keys(
            dictionary=recording_config_dict, list_of_keys=keys_to_check_recording
        )
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing information for {missing_keys} in the config_file {recording_config_filepath}!"
            )

        self.use_gpu = project_config_dict["use_gpu"]
        self.rapid_aligner_path = convert_to_path(
            project_config_dict["rapid_aligner_path"]
        )
        self.valid_cam_ids = project_config_dict["valid_cam_IDs"]
        self.recording_date = recording_config_dict["recording_date"]
        self.led_pattern = recording_config_dict["led_pattern"]
        self.calibration_index = recording_config_dict["calibration_index"]
        self.calibration_tag = project_config_dict["calibration_tag"]
        self.calibration_validation_tag = project_config_dict[
            "calibration_validation_tag"
        ]

        self.cameras_missing_in_recording_config = check_keys(
            dictionary=recording_config_dict, list_of_keys=self.valid_cam_ids
        )

        for dictionary_key in [
            "processing_type",
            "calibration_evaluation_type",
            "processing_filepath",
            "calibration_evaluation_filepath",
            "led_extraction_type",
            "led_extraction_filepath",
        ]:
            missing_keys = check_keys(
                dictionary=project_config_dict[dictionary_key],
                list_of_keys=self.valid_cam_ids,
            )
            if len(missing_keys) > 0:
                raise KeyError(
                    f"Missing information {dictionary_key} for cam {missing_keys} in the config_file {project_config_filepath}!"
                )
        return recording_config_dict, project_config_dict

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
        self.camera_objects.sort(key=lambda x: x.name, reverse=False)

    def _save_calibration(self, calibration_filepath: Path) -> None:
        if calibration_filepath.exists():
            calibration_filepath.unlink()
        self.camera_group.dump(calibration_filepath)

    def calibrate_optimal(
        self,
        calibration_validation: "Calibration_Validation",
        max_iters: int = 5,
        p_threshold: float = 0.1,
        angle_threshold: int = 5,
        verbose: int = 1,
        test_mode: bool = False,
    ):
        """finds optimal calibration through repeated optimisations of anipose"""
        # ToDo make max_iters and p_threshold adaptable?
        
        report = pd.DataFrame()
        calibration_found = False
        cams = list(self.metadata_from_videos.keys())
        good_calibration_filepath = self.output_directory.joinpath(f"{create_calibration_key(videos = cams, recording_date = self.recording_date, calibration_index = self.calibration_index)}.toml")

        for cal in range(max_iters):
            if good_calibration_filepath.exists() and test_mode:
                calibration_filepath = good_calibration_filepath
                self.reprojerr=0
            else:
                calibration_filepath = self.run_calibration(verbose=verbose, test_mode=test_mode, iteration=cal)
            
            calibration_validation.run_triangulation(
                calibration_toml_filepath=calibration_filepath
            )

            calibration_validation.evaluate_triangulation_of_calibration_validation_markers()
            calibration_errors = calibration_validation.anipose_io[
                "distance_errors_in_cm"
            ]
            calibration_angles_errors = calibration_validation.anipose_io[
                "angles_error_screws_plan"
            ]
            reprojerr_nonan = calibration_validation.anipose_io["reproj_nonan"].mean()

            for reference in calibration_errors.keys():
                all_percentage_errors = [
                    percentage_error
                    for marker_id_a, marker_id_b, distance_error, percentage_error in calibration_errors[
                        reference
                    ][
                        "individual_errors"
                    ]
                ]
                
            for reference in calibration_angles_errors.keys():
                all_angle_errors = list(calibration_angles_errors.values())

            mean_dist_err_percentage = np.nanmean(np.asarray(all_percentage_errors))
            mean_angle_err = np.nanmean(np.asarray(all_angle_errors))

            if verbose > 0:
                print(
                    f"Calibration {cal}"
                    + "\n mean percentage error: "
                    + str(mean_dist_err_percentage)
                    + "\n mean angle error: "
                    + str(mean_angle_err)
                )

            report.loc[cal, "mean_distance_error_percentage"] = mean_dist_err_percentage
            report.loc[cal, "mean_angle_error"] = mean_angle_err
            report.loc[cal, "reprojerror"] = reprojerr_nonan

            if (
                mean_dist_err_percentage < p_threshold
                and mean_angle_err < angle_threshold
            ):
                calibration_found = True
                calibration_filepath.rename(good_calibration_filepath)
                calibration_filepath=good_calibration_filepath
                print(f"Good Calibration reached at iteration {cal}! Named it {good_calibration_filepath}.")
                break

        self.report_filepath = self.output_directory.joinpath(
            f"{self.recording_date}_calibration_report.csv"
        )
        report.to_csv(self.report_filepath, index=False)
        
        if not calibration_found:
            print("No optimal calibration found with given thresholds! Returned last executed calibration!")
        return calibration_filepath


class Triangulation(Triangulation_Calibration):
    @abstractmethod
    def _create_csv_filepath(self) -> Path:
        pass

    @abstractmethod
    def _validate_and_save_metadata_for_recording(self) -> None:
        pass

    def run_triangulation(
        self,
        calibration_toml_filepath: Path,
        save_first_frame: bool = False,
        test_mode: bool = False,
    ):
        self.calibration_toml_filepath = convert_to_path(calibration_toml_filepath)
        self._load_calibration(filepath=self.calibration_toml_filepath)
        self._validate_unique_cam_ids()

        framenums = []
        for path in self.triangulation_dlc_cams_filepaths.values():
            framenums.append(pd.read_hdf(path).shape[0])
        framenum = min(framenums)
        cams_in_calibration = self.camera_group.get_names()
        markers = self.markers
        self._fake_missing_files(cams_in_calibration=cams_in_calibration, framenum=framenum, markers=markers)

        self._preprocess_dlc_predictions_for_anipose(test_mode=test_mode)
        
        if self.triangulation_type == "triangulate":
            p3ds_flat = self.camera_group.triangulate(self.anipose_io["points_flat"], progress=True)
        elif self.triangulation_type == "triangulate_optim_ransac_False":
            p3ds_flat = self.camera_group.triangulate_optim(self.anipose_io["points"], init_ransac = False, init_progress=True).reshape(self.anipose_io["n_points"] * self.anipose_io["n_joints"], 3)
        elif self.triangulation_type == "triangulate_optim_ransac_True":
            p3ds_flat = self.camera_group.triangulate_optim(self.anipose_io["points"], init_ransac = True, init_progress=True).reshape(self.anipose_io["n_points"] * self.anipose_io["n_joints"], 3)
        else:
            raise ValueError("Supported methods for triangulation are triangulate, triangulate_optim_ransac_True, triangulate_optim_ransac_False!")
        
        self._postprocess_triangulations_and_calculate_reprojection_error(p3ds_flat=p3ds_flat)
        
        self._get_dataframe_of_triangulated_points()
        if (not test_mode) or (test_mode and not self.csv_output_filepath.exists()):
            self._save_dataframe_as_csv(filepath = self.csv_output_filepath, df = self.df)
        for path in self.triangulation_dlc_cams_filepaths.values():
            if "temp" in path.name:
                path.unlink()
            
    def exclude_markers(self, all_markers_to_exclude_config_path: Path, verbose: bool=True):
        all_markers_to_exclude = read_config(all_markers_to_exclude_config_path)
        
        missing_cams = check_keys(all_markers_to_exclude, list(self.triangulation_dlc_cams_filepaths))
        if len(missing_cams)>0:
            if verbose:
                print(f"Found no markers to exclude for {missing_cams} in {str(all_markers_to_exclude_config_path)}!")
        
        for cam_id in self.triangulation_dlc_cams_filepaths:
            h5_file = self.triangulation_dlc_cams_filepaths[cam_id]
            df = pd.read_hdf(h5_file)
            markers = set(b for a, b, c in df.keys())
            markers_to_exclude_per_cam = all_markers_to_exclude[cam_id]
            existing_markers_to_exclude = list(set(markers) & set(markers_to_exclude_per_cam))
            not_existing_markers = [marker for marker in markers_to_exclude_per_cam if marker not in markers]
            if len(not_existing_markers) > 0:
                if verbose:
                    print(f"The following markers were not found in the dataframe, but were given as markers to exclude for {cam_id}: {not_existing_markers}!")
            if len(existing_markers_to_exclude)>0:
                for i, keys in enumerate(df.columns):
                    a, b, c = keys
                    if b in existing_markers_to_exclude and c == "likelihood":
                        df.isetitem(i, 0)
                df.to_hdf(h5_file, key="key", mode="w")
        
        self.markers_excluded_manually = True

    def _get_metadata_from_configs(
        self, recording_config_filepath: Path, project_config_filepath: Path
    ) -> Tuple[Dict]:
        project_config_dict = read_config(path=project_config_filepath)
        recording_config_dict = read_config(path=recording_config_filepath)
        keys_to_check_project = [
            "valid_cam_IDs",
            "paradigms",
            "animal_lines",
            "led_extraction_type",
            "led_extraction_filepath",
            "max_calibration_frames",
            "max_cpu_cores_to_pool",
            "max_ram_digestible_frames",
            "rapid_aligner_path",
            "load_calibration",
            "use_gpu",
            "calibration_tag",
            "calibration_validation_tag",
            "handle_synchro_fails",
            "default_offset_ms",
            "start_pattern_match_ms",
            "end_pattern_match_ms",
            "synchro_error_threshold",
            "synchro_marker",
            "use_2D_filter",
            "score_threshold",
            'num_frames_to_pick',
            "triangulation_type",
        ]
        missing_keys = check_keys(
            dictionary=project_config_dict, list_of_keys=keys_to_check_project
        )
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing metadata information in the project_config_file {project_config_filepath} for {missing_keys}."
            )
            
        self.synchro_metadata={key: project_config_dict[key] for key in ["handle_synchro_fails", "default_offset_ms", "start_pattern_match_ms", "end_pattern_match_ms", "synchro_error_threshold", "synchro_marker", "use_2D_filter", 'num_frames_to_pick']}

        keys_to_check_recording = [
            "led_pattern",
            "target_fps",
            "calibration_index",
            "recording_date",
        ]
        missing_keys = check_keys(
            dictionary=recording_config_dict, list_of_keys=keys_to_check_recording
        )
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing information for {missing_keys} in the config_file {recording_config_filepath}!"
            )

        self.use_gpu = project_config_dict["use_gpu"]
        self.rapid_aligner_path = convert_to_path(
            project_config_dict["rapid_aligner_path"]
        )
        self.valid_cam_ids = project_config_dict["valid_cam_IDs"]
        self.recording_date = recording_config_dict["recording_date"]
        self.led_pattern = recording_config_dict["led_pattern"]
        self.calibration_index = recording_config_dict["calibration_index"]
        self.target_fps = recording_config_dict["target_fps"]
        self.calibration_tag = project_config_dict["calibration_tag"]
        self.calibration_validation_tag = project_config_dict["calibration_validation_tag"]
        self.score_threshold = project_config_dict["score_threshold"]
        self.triangulation_type = project_config_dict["triangulation_type"]

        for dictionary_key in [
            "processing_type",
            "calibration_evaluation_type",
            "processing_filepath",
            "calibration_evaluation_filepath",
            "led_extraction_type",
            "led_extraction_filepath",
        ]:
            missing_keys = check_keys(
                dictionary=project_config_dict[dictionary_key],
                list_of_keys=self.valid_cam_ids,
            )
            if len(missing_keys) > 0:
                raise KeyError(
                    f"Missing information {dictionary_key} for cam {missing_keys} in the config_file {project_config_filepath}!"
                )
        return recording_config_dict, project_config_dict

    def _fake_missing_files(
        self, cams_in_calibration: List[str], framenum: int, markers: List[str]
    ) -> None:
        for cam in cams_in_calibration:
            try:
                self.triangulation_dlc_cams_filepaths[cam]
            except KeyError:
                h5_output_filepath = self.output_directory.joinpath(
                    self.csv_output_filepath.stem + f"empty_{cam}.h5"
                )
                cols = get_multi_index(markers)
                df = pd.DataFrame(data=np.zeros((framenum, len(cols))), columns=cols, dtype=int)
                df.to_hdf(h5_output_filepath, "empty")
                self._validate_calibration_validation_marker_ids(
                    calibration_validation_markers_df_filepath=h5_output_filepath,
                    framenum=framenum,
                )
                self.triangulation_dlc_cams_filepaths[cam] = h5_output_filepath

    def _validate_calibration_validation_marker_ids(
        self,
        calibration_validation_markers_df_filepath: Path,
        framenum: int,
        add_missing_marker_ids_with_0_likelihood: bool = True,
    ) -> None:
        defined_marker_ids = self.markers
        calibration_validation_markers_df = pd.read_hdf(calibration_validation_markers_df_filepath)

        prediction_marker_ids = list(set([marker_id for scorer, marker_id, key in calibration_validation_markers_df.columns]))

        marker_ids_not_in_ground_truth = self._find_non_matching_marker_ids(
            prediction_marker_ids, defined_marker_ids
        )
        marker_ids_not_in_prediction = self._find_non_matching_marker_ids(
            defined_marker_ids, prediction_marker_ids
        )
        if add_missing_marker_ids_with_0_likelihood & (
            len(marker_ids_not_in_prediction) > 0
        ):
            self._add_missing_marker_ids_to_prediction(
                missing_marker_ids=marker_ids_not_in_prediction,
                df=calibration_validation_markers_df,
                framenum=framenum,
            )
            print(
                "The following marker_ids were missing and added to the dataframe with a "
                f"likelihood of 0: {marker_ids_not_in_prediction}."
            )
        if len(marker_ids_not_in_ground_truth) > 0:
            self._remove_marker_ids_not_in_ground_truth(
                marker_ids_to_remove=marker_ids_not_in_ground_truth,
                df=calibration_validation_markers_df,
            )
            print(
                "The following marker_ids were deleted from the dataframe, since they were "
                f"not present in the ground truth: {marker_ids_not_in_ground_truth}."
            )
        calibration_validation_markers_df.to_hdf(
            calibration_validation_markers_df_filepath, "empty", mode="w"
        )

    def _add_missing_marker_ids_to_prediction(
        self, missing_marker_ids: List[str], df: pd.DataFrame, framenum: int = 1
    ) -> None:
        try:
            scorer = list(df.columns)[0][0]
        except IndexError:
            scorer = "zero_likelihood_fake_markers"
        for marker_id in missing_marker_ids:
            for key in ["x", "y", "likelihood"]:
                for i in range(framenum):
                    df.loc[i, (scorer, marker_id, key)] = 0

    def _find_non_matching_marker_ids(
        self, marker_ids_to_match: List[str], template_marker_ids: List[str]
    ) -> List:
        return [
            marker_id
            for marker_id in marker_ids_to_match
            if marker_id not in template_marker_ids
        ]

    def _remove_marker_ids_not_in_ground_truth(
        self, marker_ids_to_remove: List[str], df: pd.DataFrame
    ) -> None:
        columns_to_remove = [
            column_name
            for column_name in df.columns
            if column_name[1] in marker_ids_to_remove
        ]
        df.drop(columns=columns_to_remove, inplace=True)

    def _load_calibration(self, filepath: Path) -> None:
        if filepath.name.endswith(".toml") and filepath.exists():
            self.camera_group = ap_lib.cameras.CameraGroup.load(filepath)
        else:
            raise FileNotFoundError(
                f"The path, given as calibration_toml_filepath\n"
                "does not end with .toml or does not exist!\n"
                "Make sure, that you enter the correct path!"
            )

    def _preprocess_dlc_predictions_for_anipose(self, test_mode: bool = False) -> None:
        anipose_io = ap_lib.utils.load_pose2d_fnames(
            fname_dict=self.triangulation_dlc_cams_filepaths
        )
        self.anipose_io = self._add_additional_information_and_continue_preprocessing(
            anipose_io=anipose_io, test_mode=test_mode
        )

    def _add_additional_information_and_continue_preprocessing(
        self, anipose_io: Dict, test_mode: bool = False
    ) -> Dict:
        n_cams, anipose_io["n_points"], anipose_io["n_joints"], _ = anipose_io["points"].shape
        if test_mode:
            a, b = 0, 2
            anipose_io["points"] = anipose_io["points"][:, a:b, :, :]
            anipose_io["n_points"] = b-a
            anipose_io["scores"] = anipose_io["scores"][:, a:b, :]
        anipose_io["points"][anipose_io["scores"] < self.score_threshold] = np.nan

        anipose_io["points_flat"] = anipose_io["points"].reshape(n_cams, -1, 2)
        anipose_io["scores_flat"] = anipose_io["scores"].reshape(n_cams, -1)
        return anipose_io

    def _postprocess_triangulations_and_calculate_reprojection_error(
        self, p3ds_flat: np.array
    ) -> None:
        self.reprojerr_flat = self.camera_group.reprojection_error(p3ds_flat, self.anipose_io["points_flat"], mean=True)
        self.p3ds = p3ds_flat.reshape(self.anipose_io["n_points"], self.anipose_io["n_joints"], 3)
        
        self.anipose_io["p3ds"] = self.p3ds
        self.reprojerr = self.reprojerr_flat.reshape(self.anipose_io["n_points"], self.anipose_io["n_joints"])
        
        self.reprojerr_nonan = self.reprojerr[np.logical_not(np.isnan(self.reprojerr))]
        self.anipose_io["reproj_nonan"] = self.reprojerr_nonan

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
        self.anipose_io["df_xyz"] = df

    def _save_dataframe_as_csv(self, filepath: str, df: pd.DataFrame) -> None:
        filepath = convert_to_path(filepath)
        if filepath.exists():
            filepath.unlink()
        df.to_csv(filepath, index=False)


class Triangulation_Recordings(Triangulation):
    def __init__(
        self,
        recording_directory: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        output_directory: Optional[Path] = None,
        test_mode: bool = False,
    ) -> None:
        self.recording_directory = convert_to_path(recording_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)
        self.normalised_dataframe = False

        recording_config_dict, project_config_dict = self._get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self._create_video_objects(
            directory=self.recording_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="recording",
            filetypes=[".AVI", ".avi", ".mov", ".mp4"],
            test_mode=test_mode,
        )

    def run_synchronization(
        self, synchronize_only: bool = False, test_mode: bool = False
    ) -> None:
        self.synchronization_individuals = []
        self.led_detection_individuals = []
        self.synchronized_videos = {}
        self.markers_excluded_manually = False

        for video_interface in self.video_interfaces.values():
            if (
                video_interface.video_metadata.fps
                >= video_interface.video_metadata.target_fps
            ):
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoDownSynchronizer,
                    rapid_aligner_path=self.rapid_aligner_path,
                    use_gpu=self.use_gpu,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
                    test_mode=test_mode,
                    synchro_metadata=self.synchro_metadata,
                )
            else:
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoUpSynchronizer,
                    rapid_aligner_path=self.rapid_aligner_path,
                    use_gpu=self.use_gpu,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
                    test_mode=test_mode,
                    synchro_metadata=self.synchro_metadata,
                )
            self.synchronized_videos[
                video_interface.video_metadata.cam_id
            ] = video_interface.synchronized_video_filepath

        template = list(self.video_interfaces.values())[0].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(fps=self.target_fps)[0][0]

        led_timeseries_crossvalidation = {}
        for video_interface in self.video_interfaces.values():
            try:
                led_timeseries_crossvalidation[
                    video_interface.video_metadata.cam_id
                ] = (
                    video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
                )
            except:
                pass
        if len(led_timeseries_crossvalidation.keys()) > 0:
            self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                metadata={
                    "mouse_id": self.mouse_id,
                    "recording_date": self.recording_date,
                    "paradigm": self.paradigm,
                    "charuco_video": False,
                    "fps": self.target_fps,
                },
                output_directory=self.output_directory,
            )

        if not synchronize_only:
            self.csv_output_filepath = self._create_csv_filepath()
            self.triangulation_dlc_cams_filepaths = {
                video_interface: self.video_interfaces[
                    video_interface
                ].export_for_aniposelib()
                for video_interface in self.video_interfaces
            }

        try:
            self.markers = list(pd.read_hdf(list(self.triangulation_dlc_cams_filepaths.values())[0]).columns.levels[1])
        except IndexError:
            pass

    def _create_csv_filepath(self) -> None:
        filepath_out = self.output_directory.joinpath(
                f"{self.mouse_id}_{self.recording_date}_{self.paradigm}_{self.target_fps}fps_{self.score_threshold}p_excludedmarkers{self.markers_excluded_manually}_filtered{self.synchro_metadata['use_2D_filter']}_normalised{self.normalised_dataframe}_{self.triangulation_type}.csv"
            )
        return filepath_out

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(
            video_metadata.recording_date
            for video_metadata in self.metadata_from_videos.values()
        )
        paradigms = set(
            video_metadata.paradigm
            for video_metadata in self.metadata_from_videos.values()
        )
        mouse_ids = set(
            video_metadata.mouse_id
            for video_metadata in self.metadata_from_videos.values()
        )
        for attribute in [recording_dates, paradigms, mouse_ids]:
            if len(attribute) > 1:
                raise ValueError(
                    f"The filenames of the videos give different metadata! Reasons could be:\n"
                    f"  - video belongs to another recording\n"
                    f"  - video filename is valid, but wrong\n"
                    f"Go the folder {self.recording_directory} and check the filenames manually!"
                )
        self.recording_date = list(recording_dates)[0]
        self.paradigm = list(paradigms)[0]
        self.mouse_id = list(mouse_ids)[0]

    def _validate_unique_cam_ids(self) -> None:
        self.cameras = [camera.name for camera in self.camera_group.cameras]
        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.cameras.sort()
        for camera in filepath_keys:
            if camera not in self.cameras:
                self.triangulation_dlc_cams_filepaths.pop(camera)

        for camera in self.cameras:
            if camera not in filepath_keys:
                print(f"Creating empty .h5 file for {camera}!")

    def normalize(self, normalization_config_path: Path, test_mode: bool=False)->None:
        normalization_config_path = convert_to_path(normalization_config_path)
        config = read_config(normalization_config_path)
        
        best_frame = self._get_best_frame_for_normalisation(config=config, df=self.df)
        
        x, y, z = get_3D_array(self.df, config['center'], best_frame)
        for key in self.df.keys():
            if '_x' in key:
                self.df[key] = self.df[key]-x
            if '_y' in key:
                self.df[key] = self.df[key]-y
            if '_z' in key:
                self.df[key] = self.df[key]-z

        marker0, marker1 = get_3D_array(self.df, config['ReferenceLengthMarkers'][0], best_frame), get_3D_array(self.df, config['ReferenceLengthMarkers'][1], best_frame)
        lengthleftside = np.sqrt((marker0[0]-marker1[0])**2 + (marker0[1]-marker1[1])**2 + (marker0[2]-marker1[2])**2)
        conversionfactor = config['ReferenceLengthCm']/lengthleftside
        
        bp_keys_unflat=set(get_3D_df_keys(key[:-2]) for key in self.df.keys() if not 'error' in key and not 'score' in key and not "M_" in key and not 'center' in key and not 'fn' in key)
        bp_keys = list(it.chain(*bp_keys_unflat))
        
        normalised = self.df
        normalised[bp_keys]*=conversionfactor
        reference_rotation_markers = []
        for marker in config['ReferenceRotationMarkers']:
            reference_rotation_markers.append(get_3D_array(normalised, marker, best_frame))
        # the rotation matrix between the referencespace and the reconstructedspace is calculated. 
        r = Rotation.align_vectors(config["ReferenceRotationCoords"], reference_rotation_markers)
        self.rotation_error = r[1]
        
        rotated = normalised
        for key in bp_keys_unflat:
            rot_points = r[0].apply(normalised.loc[:, [key[0], key[1], key[2]]])
            rotated.loc[:, key[0]]= rot_points[:, 0]
            rotated.loc[:, key[1]]= rot_points[:, 1]
            rotated.loc[:, key[2]]= rot_points[:, 2]
        
        rotated_markers = []
        for marker in config['ReferenceRotationMarkers']:
            rotated_markers.append(get_3D_array(rotated, marker, best_frame))
        
        self.normalised_dataframe = True
        self.rotated_filepath = self._create_csv_filepath()
        if (not test_mode) or (test_mode and not self.rotated_filepath.exists()):
            self._save_dataframe_as_csv(filepath = self.rotated_filepath, df = rotated)
        Rotation_Visualization(rotated_markers = rotated_markers, config = config, filepath = self.rotated_filepath, rotation_error = self.rotation_error)
        
    def _get_best_frame_for_normalisation(self, config: Dict, df: pd.DataFrame)->int:
        all_normalization_markers = [config['center']]
        for marker in config["ReferenceLengthMarkers"]:
            all_normalization_markers.append(marker)
        for marker in config["ReferenceRotationMarkers"]:
            all_normalization_markers.append(marker)
        all_normalization_markers = set(all_normalization_markers)
        
        normalization_keys_nested = [get_3D_df_keys(marker) for marker in all_normalization_markers]
        normalization_keys = list(set(it.chain(*normalization_keys_nested)))
        df_normalization_keys = df.loc[:, normalization_keys]
        valid_frames_for_normalization = list(df_normalization_keys.dropna(axis=0).index)
        
        if len(valid_frames_for_normalization) > 0:
            return valid_frames_for_normalization[0]
        else:
            raise ValueError ("Could not normalize the dataframe!")
            
    def create_triangulated_video(
        self,
        filename: str,
        config_path: Path
    ) -> None:
        config_path = convert_to_path(config_path)
        self.video_plotting_config = read_config(config_path)
        triangulated_video = VideoClip(
            self._get_triangulated_plots,
            duration=(self.video_plotting_config["end_s"] - self.video_plotting_config["start_s"]),
        )
        triangulated_video.write_videofile(
            f"{filename}.mp4", fps=self.target_fps, logger=None
        )
        
    def _get_triangulated_plots(self, idx: int) -> np.ndarray:
        idx = int(
            (self.video_plotting_config["start_s"] + idx) * self.target_fps)
        
        t = Triangulation_Visualization(df_filepath = self.rotated_filepath, output_directory=self.output_directory, idx=idx, config=self.video_plotting_config, plot=False, save=False)
        return t.return_fig()


class Calibration_Validation(Triangulation):
    def __init__(
        self,
        calibration_validation_directory: Path,
        recording_config_filepath: Path,
        ground_truth_config_filepath: Path,
        project_config_filepath: Path,
        output_directory: Optional[Path] = None,
        test_mode: bool = False,
    ) -> None:
        ground_truth_config_filepath = convert_to_path(ground_truth_config_filepath)
        self.ground_truth_config = read_config(ground_truth_config_filepath)
        self.calibration_validation_directory = convert_to_path(calibration_validation_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        recording_config_dict, project_config_dict = self._get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self._create_video_objects(
            directory=self.calibration_validation_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="calvin",
            filetypes=[".bmp", ".tiff", ".png", ".jpg", ".AVI", ".avi"],
            filename_tag=self.calibration_validation_tag,
            test_mode=test_mode,
        )

        self.csv_output_filepath = self.output_directory.joinpath(
            f"Calvin_{self.recording_date}.csv"
        )
        self.markers = self.ground_truth_config["unique_ids"]

    def _validate_and_save_metadata_for_recording(self):
        recording_dates = set(
            video_metadata.recording_date
            for video_metadata in self.metadata_from_videos.values()
        )
        for attribute in [recording_dates]:
            if len(attribute) > 1:
                raise ValueError(
                    f"The filenames of the calibration_validation images give different metadata! Reasons could be:\n"
                    f"  - image belongs to another calibration\n"
                    f"  - image filename is valid, but wrong\n"
                    f"Go the folder {self.calibration_validation_directory} and check the filenames manually!"
                )
        self.recording_date = list(recording_dates)[0]

    def _validate_unique_cam_ids(self):
        self.cameras = [camera.name for camera in self.camera_group.cameras]
        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.cameras.sort()
        for camera in filepath_keys:
            if camera not in self.cameras:
                self.triangulation_dlc_cams_filepaths.pop(camera)

        for camera in self.cameras:
            if camera not in filepath_keys:
                print(f"Creating empty .h5 file for {camera}!")

    def get_marker_predictions(self) -> None:
        self.markers_excluded_manually = False
        self.csv_output_filepath = self._create_csv_filepath()
        self.triangulation_dlc_cams_filepaths = {}
        for cam in self.metadata_from_videos.values():
            h5_output_filepath = self.output_directory.joinpath(
                f"Calvin_{cam.recording_date}_{cam.cam_id}.h5"
            )
            self.triangulation_dlc_cams_filepaths[cam.cam_id] = h5_output_filepath
            if cam.calibration_evaluation_type == "manual":
                config = cam.calibration_evaluation_filepath
                if not h5_output_filepath.exists():  # implement test_mode
                    manual_interface = ManualAnnotation(
                        object_to_analyse=cam.filepath,
                        output_directory=self.output_directory,
                        marker_detection_directory=config,
                    )
                    manual_interface.analyze_objects(
                        filepath=h5_output_filepath, only_first_frame=True
                    )
            elif cam.calibration_evaluation_type == "DLC":
                config = cam.calibration_evaluation_filepath
                if not h5_output_filepath.exists():  # implement test_mode
                    dlc_interface = DeeplabcutInterface(
                        object_to_analyse=cam.filepath,
                        output_directory=self.output_directory,
                        marker_detection_directory=config,
                    )
                    h5_output_filepath = dlc_interface.analyze_objects(
                        filtering=False, filepath=h5_output_filepath
                    )
            else:
                print(
                    "Template Matching is not yet implemented for Marker Detection in Calvin!"
                )
            self._validate_calibration_validation_marker_ids(
                calibration_validation_markers_df_filepath=h5_output_filepath, framenum=1
            )
            Predictions_Plot(
                image=cam.filepath,
                predictions=h5_output_filepath,
                output_directory=self.output_directory,
                cam_id=cam.cam_id,
            )

    def evaluate_triangulation_of_calibration_validation_markers(
        self, show_3D_plot: bool = True, verbose: int = 1
    ) -> None:
        self.anipose_io = add_reprojection_errors_of_all_calibration_validation_markers(
            anipose_io=self.anipose_io
        )
        self.anipose_io = set_distances_and_angles_for_evaluation(
            self.ground_truth_config, self.anipose_io
        )
        gt_distances = fill_in_distances(self.ground_truth_config["distances"])
        self.anipose_io = add_all_real_distances_errors(
            anipose_io=self.anipose_io, ground_truth_distances=gt_distances
        )
        self.anipose_io = set_angles_error_between_screws_and_plane(
            self.ground_truth_config["angles"], self.anipose_io
        )

        if verbose > 0:
            print(f'Mean reprojection error: {self.anipose_io["reproj_nonan"].mean()}')
            for reference_distance_id, distance_errors in self.anipose_io[
                "distance_errors_in_cm"
            ].items():
                print(
                    f'Using {reference_distance_id} as reference distance, the mean distance error is: {distance_errors["mean_error"]} cm.'
                )
            for angle, angle_error in self.anipose_io[
                "angles_error_screws_plan"
            ].items():
                print(f"Considering {angle}, the angle error is: {angle_error}")
        if show_3D_plot:
            Calibration_Validation_Plot(
                p3d=self.anipose_io["p3ds"][0],
                bodyparts=self.anipose_io["bodyparts"],
                output_directory=self.output_directory,
                marker_ids_to_connect=self.ground_truth_config[
                    "marker_ids_to_connect_in_3D_plot"
                ],
                plot=True,
                save=True,
            )

    def _create_csv_filepath(self) -> None:
        filepath_out = self.output_directory.joinpath(f"{self.recording_date}_{self.target_fps}fps_{self.score_threshold}p_excludedmarkers{self.markers_excluded_manually}_filtered{self.synchro_metadata['use_2D_filter']}_{self.triangulation_type}.csv")
        return filepath_out
