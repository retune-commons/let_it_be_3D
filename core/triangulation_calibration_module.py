from typing import List, Tuple, Dict, Union, Optional
import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import itertools as it
import math

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
from .utils import convert_to_path, create_calibration_key, read_config, check_keys, get_multi_index


def exclude_by_framenum(metadata_from_videos: Dict, target_fps: int) -> None:
    # makes little sense since only the beginning, not the end of the videos is synchronized
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

class Calibration:
    def __init__(
        self,
        calibration_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        output_directory: Optional[Path] = None
    ) -> None:

        self.calibration_directory = convert_to_path(calibration_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        recording_config_dict, project_config_dict = self._get_metadata_from_configs(recording_config_filepath = recording_config_filepath, project_config_filepath = project_config_filepath)
        self._create_video_objects(
            calibration_directory=self.calibration_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
        )

    def run_synchronization(self, test_mode: bool = False) -> None:
        self.synchronized_charuco_videofiles = {}
        self.camera_objects = []
        self.synchronization_individuals = []
        self.led_detection_individuals = []

        for video_interface in self.charuco_interfaces.values():
            video_interface.run_synchronizer(
                synchronizer=CharucoVideoSynchronizer,
                rapid_aligner_path=self.rapid_aligner_path,
                use_gpu = self.use_gpu,
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
        led_timeseries_crossvalidation = {}
        for video_interface in self.charuco_interfaces.values():
            try:
                led_timeseries_crossvalidation[video_interface.video_metadata.cam_id] = video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
            except:
                pass
        if len(led_timeseries_crossvalidation.keys()) > 0:
            self.synchronization_crossvalidation = Alignment_Plot_Crossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
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
        recording_config_dict: Dict,
        project_config_dict: Dict
    ) -> None:
        
        charuco_videofiles = [
            file
            for file in calibration_directory.iterdir()
            if self.calibration_tag.lower() in file.name.lower()
            and (file.name.endswith(".mp4") or file.name.endswith(".mov"))
            and ("synchronized" not in file.name and "Front" not in file.name)
        ]
        # add possibility to exclude filenames by flags in meta (and remove "Front")
        
        avi_files = [
            file
            for file in calibration_directory.iterdir()
            if self.calibration_tag.lower() in file.name.lower()
            and file.suffix == ".AVI"
        ]
        avi_files.sort()
        top_cam_file = avi_files[-1]  # hard coded!
        
        try:
            top_cam_file = avi_files[-1]
            charuco_videofiles.append(top_cam_file)
        except:
            pass
        

        self.charuco_interfaces = {}
        self.metadata_from_videos = {}
        for filepath in charuco_videofiles:
            video_metadata = VideoMetadata(
                video_filepath=filepath,
                recording_config_dict=recording_config_dict,
                project_config_dict=project_config_dict,
                tag = "calibration"
            )

            self.charuco_interfaces[video_metadata.cam_id] = VideoInterface(
                video_metadata=video_metadata, output_dir=self.output_directory
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata

        self._validate_and_save_metadata_for_recording()

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(video_metadata.recording_date for video_metadata in self.metadata_from_videos.values())
        for attribute in [recording_dates]:
            if len(attribute) > 1: 
                raise ValueError(
                    f"The filenames of the videos give different metadata! Reasons could be:\n"
                    f"  - video belongs to another calibration\n"
                    f"  - video filename is valid, but wrong\n"
                    f"Go the folder {self.calibration_directory} and check the filenames manually!"
                )
        self.recording_date = list(recording_dates)[0]
        
    def _get_metadata_from_configs(self, recording_config_filepath: Path, project_config_filepath: Path)->Tuple[Dict]:
        project_config_dict = read_config(path = project_config_filepath)
        recording_config_dict = read_config(path = recording_config_filepath)
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
            "calibration_validation_tag"
        ]
        missing_keys = check_keys(dictionary = project_config_dict, list_of_keys = keys_to_check_project)
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing metadata information in the project_config_file {project_config_filepath} for {missing_keys}."
            )
            
        keys_to_check_recording = ["led_pattern", "target_fps", "calibration_index", "recording_date"]
        missing_keys = check_keys(dictionary = recording_config_dict, list_of_keys = keys_to_check_recording)
        if len(missing_keys) > 0:    
            raise KeyError(
                f"Missing information for {missing_keys} in the config_file {recording_config_filepath}!"
            )
            
        self.use_gpu = project_config_dict["use_gpu"]
        self.rapid_aligner_path = convert_to_path(project_config_dict['rapid_aligner_path'])
        self.valid_cam_ids = project_config_dict['valid_cam_IDs']
        self.recording_date = recording_config_dict['recording_date']
        self.led_pattern = recording_config_dict['led_pattern']
        self.calibration_index = recording_config_dict['calibration_index']
        self.target_fps = recording_config_dict["target_fps"]
        self.calibration_tag = project_config_dict["calibration_tag"]
        self.calibration_validation_tag = project_config_dict["calibration_validation_tag"]
        
        self.cameras_missing_in_recording_config = check_keys(dictionary = recording_config_dict, list_of_keys = self.valid_cam_ids) 
        
        for dictionary_key in ["processing_type", "calibration_evaluation_type", "processing_filepath", "calibration_evaluation_filepath", "led_extraction_type", "led_extraction_filepath"]:
            missing_keys = check_keys(dictionary = project_config_dict[dictionary_key], list_of_keys = self.valid_cam_ids)
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

    def _save_calibration(self) -> None:
        if self.calibration_output_filepath.exists():
            self.calibration_output_filepath.unlink()
        self.camera_group.dump(self.calibration_output_filepath)
        
    def calibrate_optimal(self,
                         triangulation_positions: 'Triangulation_Positions',
                         max_iters: int=10,
                         p_threshold: float=0.1,
                         angle_threshold: int=5,
                         verbose: bool=True
                         ):
        """ finds optimal calibration through repeated optimisations of anipose """
        report = pd.DataFrame()
        calibration_found = False 

        for cal in range(max_iters):
            self.run_calibration(verbose=verbose)
            # change/return toml filename? or delete toml if threshold not reached? currently overwritten

            triangulation_positions.run_triangulation(calibration_toml_filepath = self.calibration_output_filepath)
            
            triangulation_positions.evaluate_triangulation_of_test_position_markers()
            calibration_errors = triangulation_positions.anipose_io['distance_errors_in_cm']
            calibration_angles_errors = triangulation_positions.anipose_io['angles_error_screws_plan']
            reprojerr_nonan = triangulation_positions.anipose_io["reproj_nonan"].mean()

            for reference in calibration_errors.keys():
                all_percentage_errors = [percentage_error for marker_id_a, marker_id_b, distance_error, percentage_error in calibration_errors[reference]['individual_errors']]

            for reference in calibration_angles_errors.keys():
                all_angle_errors = list(calibration_angles_errors.values())

            mean_dist_err_percentage = np.asarray(all_percentage_errors).mean()
            mean_angle_err = np.asarray(all_angle_errors).mean()

            # pritn output necessary if we have the report log?
            print(f"Calibration {cal}" +
                  "\n mean percentage error: "+ str(mean_dist_err_percentage) + 
                  "\n mean angle error: "+ str(mean_angle_err) )        

            report.loc[cal, 'mean_distance_error_percentage'] = mean_dist_err_percentage
            report.loc[cal, 'mean_angle_error'] = mean_angle_err
            report.loc[cal, 'reprojerror'] = reprojerr_nonan

            if mean_dist_err_percentage < p_threshold and mean_angle_err < angle_threshold:
                calibration_found = True
                print("Good Calibration reached!")
                break

        if not calibration_found:
            print('No optimal calibration found with given thresholds')
        
        report_filepath = self.output_directory.joinpath(f"{self.recording_date}_calibration_report.csv")
        report.to_csv(report_filepath, index = False)


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
        
        framenums = []
        for path in self.triangulation_dlc_cams_filepaths.values():
            framenums.append(pd.read_hdf(path).shape[0])
        framenum = min(framenums)
        cams_in_calibration = self.camera_group.get_names()
        markers = self.markers
        self._fake_missing_files(cams_in_calibration=cams_in_calibration, framenum = framenum, markers = markers)
        
        self._preprocess_dlc_predictions_for_anipose()
        p3ds_flat = self.camera_group.triangulate(
            self.anipose_io["points_flat"], progress=True
        )
        self._postprocess_triangulations_and_calculate_reprojection_error(
            p3ds_flat=p3ds_flat
        )
        self._get_dataframe_of_triangulated_points()
        self._save_dataframe_as_csv()
        
    def _get_metadata_from_configs(self, recording_config_filepath: Path, project_config_filepath: Path)->Tuple[Dict]:
        project_config_dict = read_config(path = project_config_filepath)
        recording_config_dict = read_config(path = recording_config_filepath)
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
            "calibration_validation_tag"
        ]
        missing_keys = check_keys(dictionary = project_config_dict, list_of_keys = keys_to_check_project)
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing metadata information in the project_config_file {project_config_filepath} for {missing_keys}."
            )
            
        keys_to_check_recording = ["led_pattern", "target_fps", "calibration_index", "recording_date"]
        missing_keys = check_keys(dictionary = recording_config_dict, list_of_keys = keys_to_check_recording)
        if len(missing_keys) > 0:    
            raise KeyError(
                f"Missing information for {missing_keys} in the config_file {recording_config_filepath}!"
            )
        
        self.use_gpu = project_config_dict["use_gpu"]
        self.rapid_aligner_path = convert_to_path(project_config_dict['rapid_aligner_path'])
        self.valid_cam_ids = project_config_dict['valid_cam_IDs']
        self.recording_date = recording_config_dict['recording_date']
        self.led_pattern = recording_config_dict['led_pattern']
        self.calibration_index = recording_config_dict['calibration_index']
        self.target_fps = recording_config_dict["target_fps"]
        self.calibration_tag = project_config_dict["calibration_tag"]
        self.calibration_validation_tag = project_config_dict["calibration_validation_tag"]
        
        for dictionary_key in ["processing_type", "calibration_evaluation_type", "processing_filepath", "calibration_evaluation_filepath", "led_extraction_type", "led_extraction_filepath"]:
            missing_keys = check_keys(dictionary = project_config_dict[dictionary_key], list_of_keys = self.valid_cam_ids)
            if len(missing_keys) > 0:    
                raise KeyError(
                    f"Missing information {dictionary_key} for cam {missing_keys} in the config_file {project_config_filepath}!"
                )
        return recording_config_dict, project_config_dict
        
    def _fake_missing_files(self, cams_in_calibration: List[str], framenum: int, markers: List[str])->None:
        for cam in cams_in_calibration:
            try:
                self.triangulation_dlc_cams_filepaths[cam]
            except KeyError:
                h5_output_filepath = self.output_directory.joinpath(f"Positions_{self.recording_date}_{cam}.h5")
                df = pd.DataFrame(data = {}, columns = get_multi_index(markers), dtype = int)
                for i in range(framenum):
                    df.loc[i, :] = 0
                df.to_hdf(h5_output_filepath, 'Positions')
                self._validate_test_position_marker_ids(test_position_markers_df_filepath = h5_output_filepath, framenum=framenum)
                self.triangulation_dlc_cams_filepaths[cam] = h5_output_filepath
                
    def _validate_test_position_marker_ids(self, test_position_markers_df_filepath: Path, framenum: int, add_missing_marker_ids_with_0_likelihood: bool = True) -> None:
        defined_marker_ids = self.markers
        test_position_markers_df = pd.read_hdf(test_position_markers_df_filepath)
        
        prediction_marker_ids = list(
            set([marker_id for scorer, marker_id, key in test_position_markers_df.columns]))
        
        marker_ids_not_in_ground_truth = self._find_non_matching_marker_ids(prediction_marker_ids,
                                                                            defined_marker_ids)
        marker_ids_not_in_prediction = self._find_non_matching_marker_ids(defined_marker_ids,
                                                                          prediction_marker_ids)
        if add_missing_marker_ids_with_0_likelihood & (len(marker_ids_not_in_prediction) > 0):
            self._add_missing_marker_ids_to_prediction(missing_marker_ids=marker_ids_not_in_prediction, df = test_position_markers_df, framenum = framenum)
            print('The following marker_ids were missing and added to the dataframe with a '
                  f'likelihood of 0: {marker_ids_not_in_prediction}.')
        if len(marker_ids_not_in_ground_truth) > 0:
            self._remove_marker_ids_not_in_ground_truth(marker_ids_to_remove=marker_ids_not_in_ground_truth, df = test_position_markers_df)
            print('The following marker_ids were deleted from the dataframe, since they were '
                  f'not present in the ground truth: {marker_ids_not_in_ground_truth}.')
        test_position_markers_df.to_hdf(test_position_markers_df_filepath, 'Positions', mode = 'w')

    def _add_missing_marker_ids_to_prediction(self, missing_marker_ids: List[str], df: pd.DataFrame, framenum: int=1) -> None:
        try:
            scorer = list(df.columns)[0][0]
        except IndexError:
            scorer = 'zero_likelihood_fake_markers'
        for marker_id in missing_marker_ids:
            for key in ['x', 'y', 'likelihood']:
                for i in range(framenum):
                    df.loc[i, (scorer, marker_id, key)] = 0

    def _find_non_matching_marker_ids(self, marker_ids_to_match: List[str], template_marker_ids: List[str]) -> List:
        return [marker_id for marker_id in marker_ids_to_match if marker_id not in template_marker_ids]

    def _remove_marker_ids_not_in_ground_truth(self, marker_ids_to_remove: List[str], df: pd.DataFrame) -> None:
        columns_to_remove = [column_name for column_name in df.columns if column_name[1] in marker_ids_to_remove]
        df.drop(columns=columns_to_remove, inplace=True) 

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
            self.anipose_io["n_points"], self.anipose_io["n_joints"], 3)
        self.anipose_io['p3ds'] = self.p3ds
        self.reprojerr = self.reprojerr_flat.reshape(
            self.anipose_io["n_points"], self.anipose_io["n_joints"]
        )
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
        self.anipose_io['df_xyz'] = df

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
        output_directory: Optional[Path] = None,
    ) -> None:

        self.recording_directory = convert_to_path(recording_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)
        
        recording_config_dict, project_config_dict = self._get_metadata_from_configs(recording_config_filepath = recording_config_filepath, project_config_filepath = project_config_filepath)
        self._create_video_objects(
            recording_directory=self.recording_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
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
                    rapid_aligner_path=self.rapid_aligner_path,
                    use_gpu=self.use_gpu,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
                    test_mode=test_mode,
                )
            else:
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoUpSynchronizer,
                    rapid_aligner_path=self.rapid_aligner_path,
                    use_gpu = self.use_gpu,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
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

        led_timeseries_crossvalidation = {}
        for video_interface in self.recording_interfaces.values():
            try:
                led_timeseries_crossvalidation[video_interface.video_metadata.cam_id] = video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
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
        
        try:
            self.markers = list(pd.read_hdf(list(self.triangulation_dlc_cams_filepaths.values())[0]).columns.levels[1])
        except IndexError: 
            pass
        #exclude_by_framenum(metadata_from_videos=self.metadata_from_videos, target_fps=self.target_fps)

    def create_csv_filepath(self) -> None:
        filepath_out = self.output_directory.joinpath(
            f"{self.mouse_id}_{self.recording_date}_{self.paradigm}.csv"
        )
        return filepath_out

    def _create_video_objects(
        self,
        recording_directory: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict
    ) -> None:
        recording_videofiles = [
            file
            for file in recording_directory.iterdir()
            if (file.name.endswith(".mp4") or file.name.endswith(".mov"))
            and ("synchronized" not in file.name and "Front" not in file.name)
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
                    recording_config_dict=recording_config_dict,
                    project_config_dict=project_config_dict,
                    tag = "recording"
                )

            self.recording_interfaces[video_metadata.cam_id] = VideoInterface(
                video_metadata=video_metadata, output_dir=self.output_directory
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata
        self._validate_and_save_metadata_for_recording()

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(video_metadata.recording_date for video_metadata in self.metadata_from_videos.values())
        paradigms = set(video_metadata.paradigm for video_metadata in self.metadata_from_videos.values())
        mouse_ids = set(video_metadata.mouse_id for video_metadata in self.metadata_from_videos.values())
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
        recording_config_filepath: Path,
        ground_truth_config_filepath: Path,
        project_config_filepath: Path,
        output_directory: Optional[Path] = None,
    ) -> None:
        ground_truth_config_filepath = convert_to_path(ground_truth_config_filepath)
        self.test_positions_gt = read_config(ground_truth_config_filepath)
        self.positions_directory = convert_to_path(positions_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self._check_output_directory(output_directory=output_directory)

        recording_config_dict, project_config_dict = self._get_metadata_from_configs(recording_config_filepath = recording_config_filepath, project_config_filepath = project_config_filepath)
        self._create_video_objects(
            positions_directory=self.positions_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict)
        
        self.csv_output_filepath = self.output_directory.joinpath(
            f"Positions_{self.recording_date}.csv"
        )
        self.markers = self.test_positions_gt['unique_ids']

    def _create_video_objects(
        self,
        positions_directory: Path,
        project_config_dict: Dict,
        recording_config_dict: Dict
    ) -> None:
        position_files = [
            file
            for file in positions_directory.iterdir()
            if (
                file.name.endswith(".tiff")
                or file.name.endswith(".bmp")
                or file.name.endswith(".jpg")
                or file.name.endswith(".png")
            )
            and self.calibration_validation_tag.lower() in file.name.lower()
        ]
        
        avi_files = [
                file
                for file in positions_directory.iterdir()
                if file.name.endswith(".AVI") and self.calibration_validation_tag.lower() in file.name.lower()
            ]
        try:
            avi_files.sort()
            top_cam_file = avi_files[-1]  # hardcoded
            position_files.append(top_cam_file)
        except:
            pass

        self.metadata_from_videos = {}
        for filepath in position_files:
            video_metadata = VideoMetadata(
                video_filepath=filepath,
                recording_config_dict=recording_config_dict,
                project_config_dict=project_config_dict,
                tag = "positions"
            )
            self.metadata_from_videos[video_metadata.cam_id] = video_metadata
        self.cameras = list(self.metadata_from_videos.keys())
        self._validate_and_save_metadata_for_recording()

    def _validate_and_save_metadata_for_recording(self):
        recording_dates = set(video_metadata.recording_date for video_metadata in self.metadata_from_videos.values())
        for attribute in [recording_dates]:
            if len(attribute) > 1: 
                raise ValueError(
                    f"The filenames of the position images give different metadata! Reasons could be:\n"
                    f"  - image belongs to another calibration\n"
                    f"  - image filename is valid, but wrong\n"
                    f"Go the folder {self.positions_directory} and check the filenames manually!"
                )
        self.recording_date = list(recording_dates)[0]

    def _validate_unique_cam_ids(self, adapt_to_calibration: bool=False):
        cameras = [camera.name for camera in self.camera_group.cameras]
        # possibility to create empty .h5 for missing recordings?
        if self.cameras.sort() != cameras.sort():
            raise ValueError(
                f"The cam_ids of the recordings in {self.positions_directory} do not match the cam_ids of the camera_group at {self.calibration_toml_filepath}.\n"
                "Are there missing or additional files in the calibration or the recording folder?"
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
                    marker_detection_directory=config)
                    dlc_ending = dlc_interface.analyze_objects(filtering=False)
                    h5_output_filepath = self.output_directory.joinpath(cam.filepath.stem + dlc_ending + ".h5")
            else:
                print(
                    "Template Matching is not yet implemented for Marker Detection in Positions!"
                )
            self._validate_test_position_marker_ids(test_position_markers_df_filepath = h5_output_filepath, framenum = 1)

    def evaluate_triangulation_of_test_position_markers(self, show_3D_plot: bool = True, verbose: bool = True) -> None:        
        self.anipose_io = self._add_reprojection_errors_of_all_test_position_markers(anipose_io=self.anipose_io)
        self._set_distances_and_angles_for_evaluation(self.test_positions_gt)
        gt_distances = self._fill_in_distances(self.test_positions_gt["distances"])
        self.anipose_io = self._add_all_real_distances_errors(anipose_io=self.anipose_io,
                                                              test_positions_distances=gt_distances)

        self._set_angles_error_between_screws_and_plane(self.test_positions_gt["angles"])
        if verbose:
            print(f'Mean reprojection error: {self.anipose_io["reproj_nonan"].mean()}')
            for reference_distance_id, distance_errors in self.anipose_io['distance_errors_in_cm'].items():
                print(
                    f'Using {reference_distance_id} as reference distance, the mean distance error is: {distance_errors["mean_error"]} cm.')
            for angle, angle_error in self.anipose_io["angles_error_screws_plan"].items():
                print(
                    f'Considering {angle}, the angle error is: {angle_error}'
                )
        if show_3D_plot:
            self._show_3D_plot(frame_idx=0, anipose_io=self.anipose_io,
                               marker_ids_to_connect=self.test_positions_gt["marker_ids_to_connect_in_3D_plot"])

    def _fill_in_distances(self, distances_dict):
        filled_d = {}
        for key, value in distances_dict.items():
            filled_d[key] = value
            for k, v in value.items():
                if k in filled_d.keys():
                    filled_d[k][key] = v
                else:
                    filled_d[k] = {}
                    filled_d[k][key] = v
        return filled_d

    def _set_distances_and_angles_for_evaluation(self, parameters_dict):
        if "distances" in parameters_dict:
            self._set_distances_from_configuration(parameters_dict["distances"])
        else:
            print(
                "WARNING: No distances were computed. If this is unexpected please edit the ground truth file accordingly")

        if "angles" in parameters_dict:
            self._set_angles_to_plane(parameters_dict["angles"])
        else:
            print(
                "WARNING: No angles were computed. If this is unexpected please edit the ground truth file accordingly")

        return

    def _set_distances_from_configuration(self, distances_to_compute):
        conversion_factors = self._get_conversion_factors_from_different_references(self.anipose_io,
                                                                                    distances_to_compute)
        self._add_distances_in_cm_for_each_conversion_factor(self.anipose_io, conversion_factors)
        return

    def _add_additional_information_and_continue_preprocessing(self, anipose_io: Dict) -> Dict:
        n_cams, anipose_io['n_points'], anipose_io['n_joints'], _ = anipose_io['points'].shape
        anipose_io['points'][anipose_io['scores'] < self.score_threshold] = np.nan
        anipose_io['points_flat'] = anipose_io['points'].reshape(n_cams, -1, 2)
        anipose_io['scores_flat'] = anipose_io['scores'].reshape(n_cams, -1)
        return anipose_io

    def _add_all_real_distances_errors(self, anipose_io: Dict, test_positions_distances: Dict) -> Dict:
        all_distance_to_cm_conversion_factors = self._get_conversion_factors_from_different_references(
            anipose_io=anipose_io, test_positions_distances=test_positions_distances)
        anipose_io = self._add_distances_in_cm_for_each_conversion_factor(anipose_io=anipose_io,
                                                                          conversion_factors=all_distance_to_cm_conversion_factors)
        anipose_io = self._add_distance_errors(anipose_io=anipose_io,
                                               gt_distances=test_positions_distances)
        return anipose_io

    def _add_distance_errors(self, anipose_io: Dict, gt_distances: Dict) -> Dict:
        anipose_io['distance_errors_in_cm'] = {}
        for reference_distance_id, triangulated_distances in anipose_io['distances_in_cm'].items():
            anipose_io['distance_errors_in_cm'][reference_distance_id] = {}
            marker_ids_with_distance_error = self._compute_differences_between_triangulated_and_gt_distances(
                triangulated_distances=triangulated_distances,
                gt_distances=gt_distances)
            all_distance_errors = [distance_error for marker_id_a, marker_id_b, distance_error, percentage_error in
                                   marker_ids_with_distance_error]
            mean_distance_error = np.asarray(all_distance_errors).mean()
            all_percentage_errors = [percentage_error for marker_id_a, marker_id_b, distance_error, percentage_error in
                                   marker_ids_with_distance_error]
            mean_percentage_error = np.asarray(all_percentage_errors).mean()
            anipose_io['distance_errors_in_cm'][reference_distance_id] = {
                'individual_errors': marker_ids_with_distance_error,
                'mean_error': mean_distance_error,
                'mean_percentage_error': mean_percentage_error}

        return anipose_io

    def _add_distances_in_cm_for_each_conversion_factor(self, anipose_io: Dict, conversion_factors: Dict) -> Dict:
        anipose_io['distances_in_cm'] = {}
        for reference_distance_id, conversion_factor in conversion_factors.items():
            anipose_io['distances_in_cm'][reference_distance_id] = self._convert_all_xyz_distances(
                anipose_io=anipose_io, conversion_factor=conversion_factor)
        return anipose_io

    def _add_reprojection_errors_of_all_test_position_markers(self, anipose_io: Dict) -> Dict:
        anipose_io['reprojection_errors_test_position_markers'] = {}
        all_reprojection_errors = []
        for key in anipose_io['df_xyz'].iloc[0].keys():
            if "error" in key:
                reprojection_error = anipose_io['df_xyz'][key].iloc[0]
                marker_id = key[:key.find('_error')]
                anipose_io['reprojection_errors_test_position_markers'][
                    marker_id] = reprojection_error  # since we only have a single image
                if type(reprojection_error) != np.nan:
                    # ToDo:
                    # confirm that it would actually be a numpy nan
                    # or as alternative, use something like this after blindly appending all errors to drop the nan´s:
                    # anipose_io['reprojerr'][np.logical_not(np.isnan(anipose_io['reprojerr']))]
                    all_reprojection_errors.append(reprojection_error)
        anipose_io['reprojection_errors_test_position_markers']['mean'] = np.asarray(all_reprojection_errors).mean()
        return anipose_io

    def _compute_differences_between_triangulated_and_gt_distances(self, triangulated_distances: Dict,
                                                                   gt_distances: Dict) -> List[Tuple[str, str, float]]:
        marker_ids_with_distance_error = []
        for marker_id_a in triangulated_distances.keys():
            for marker_id_b in triangulated_distances[marker_id_a].keys():
                if (marker_id_a in gt_distances.keys()) & (marker_id_b in gt_distances[marker_id_a].keys()):
                    ground_truth = gt_distances[marker_id_a][marker_id_b]
                    triangulated_distance = triangulated_distances[marker_id_a][marker_id_b]
                    distance_error = abs(ground_truth - abs(triangulated_distance))
                    percentage_error = distance_error / ground_truth
                    marker_ids_with_distance_error.append((marker_id_a, marker_id_b, distance_error, percentage_error))

        return marker_ids_with_distance_error

    def _wrap_angles_360(self, angle):
        """
        Wraps negative angle on 360 space
        :param angle: input angle
        :return: returns angle if positive or 360+angle if negative
        """
        return angle if angle > 0 else 360 + angle

    def _compute_differences_between_triangulated_and_gt_angles(self, gt_angles: Dict) -> Dict[str, float]:
        """
        Computes the difference between the triangulated screw angles and the provided ground truth ones.
        :param gt_angles: ground truth angles
        :return: list with angle errors
        """
        triangulates_angles: Dict = self.anipose_io["angles_to_plane"]
        marker_ids_with_angles_error = {}
        if triangulates_angles.keys() == gt_angles.keys():
            for key in triangulates_angles:
                wrapped_tri_angle = self._wrap_angles_360(triangulates_angles[key])
                angle_error = abs(gt_angles[key]["value"] - wrapped_tri_angle)
                half_pi_corrected_angle = angle_error if angle_error < 180 else angle_error - 180
                marker_ids_with_angles_error[key] = half_pi_corrected_angle
        else:
            raise ValueError("Please check the ground truth angles passed. The screws angles needed are:",
                             ', '.join(str(key) for key in triangulates_angles),
                             "\n But the angles in the passed ground truth are:",
                             ', '.join(str(key) for key in gt_angles))
        return marker_ids_with_angles_error

    def _computes_angles(self, angles_to_compute) -> Dict[str, float]:
        """
        Computes the triangulated angles
        :return: dictionary of the angles computed
        """
        triangulated_angles = {}
        for angle, markers_dictionary in angles_to_compute.items():
            if len(markers_dictionary["marker"]) == 3:
                pt_a = self._get_vector_from_label(label=markers_dictionary["marker"][0])
                pt_b = self._get_vector_from_label(label=markers_dictionary["marker"][1])
                pt_c = self._get_vector_from_label(label=markers_dictionary["marker"][2])
                triangulated_angles[angle] = self._get_angle_between_three_points_at_PointA(PointA=pt_a,
                                                                                            PointB=pt_b,
                                                                                            PointC=pt_c)
            elif len(markers_dictionary["marker"]) == 5:
                pt_a = self._get_vector_from_label(label=markers_dictionary["marker"][2])
                pt_b = self._get_vector_from_label(label=markers_dictionary["marker"][3])
                pt_c = self._get_vector_from_label(label=markers_dictionary["marker"][4])
                plane_coord = self._get_coordinates_plane_equation_from_three_points(PointA=pt_a,
                                                                                     PointB=pt_b,
                                                                                     PointC=pt_c)
                N = self._get_vector_product(A=plane_coord[0], B=plane_coord[2])

                pt_d = self._get_vector_from_label(label=markers_dictionary["marker"][0])
                pt_e = self._get_vector_from_label(label=markers_dictionary["marker"][1])
                triangulated_angles[angle] = self._get_angle_between_two_points_and_plane(PointA=pt_d, PointB=pt_e, N=N)
            else:
                raise ValueError("Invalid number (%d) of markers to compute the angle " + angle,
                                 (len(markers_dictionary["marker"])))
        return triangulated_angles

    def _set_angles_to_plane(self, angles_to_compute):
        """
        Sets the angles between the screws and the plane
        :param self:
        :param angles_dict:
        :return:
        """
        self.anipose_io["angles_to_plane"] = self._computes_angles(angles_to_compute)

    def _set_angles_error_between_screws_and_plane(self, gt_angles):
        """
        Sets the angles between the screws and the plane
        :param self:
        :param angles_dict:
        :return:
        """
        self.anipose_io["angles_error_screws_plan"] = self._compute_differences_between_triangulated_and_gt_angles(
            gt_angles)

    def _connect_all_marker_ids(self, ax: plt.Figure, points: np.ndarray, scheme: List[Tuple[str]],
                                bodyparts: List[str]) -> List[plt.Figure]:
        # ToDo: correct type hints
        cmap = plt.get_cmap('tab10')
        bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
        lines = []
        for i, bps in enumerate(scheme):
            line = self._connect_one_set_of_marker_ids(ax=ax, points=points, bps=bps, bp_dict=bp_dict,
                                                       color=cmap(i)[:3])
            lines.append(line)
        return lines  # return neccessary?

    def _connect_one_set_of_marker_ids(self, ax: plt.Figure, points: np.ndarray, bps: List[str], bp_dict: Dict,
                                       color: np.ndarray) -> plt.Figure:
        # ToDo: correct type hints
        ixs = [bp_dict[bp] for bp in bps]
        return ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2], color=color)

    def _convert_all_xyz_distances(self, anipose_io: Dict, conversion_factor: float) -> Dict:
        marker_id_combinations = it.combinations(anipose_io['bodyparts'], 2)
        all_distances_in_cm = {}
        for marker_id_a, marker_id_b in marker_id_combinations:
            if marker_id_a not in all_distances_in_cm.keys():
                all_distances_in_cm[marker_id_a] = {}
            xyz_distance = self._get_xyz_distance_in_triangulation_space(marker_ids=(marker_id_a, marker_id_b),
                                                                         df_xyz=anipose_io['df_xyz'])
            all_distances_in_cm[marker_id_a][marker_id_b] = xyz_distance / conversion_factor
        return all_distances_in_cm

    def _get_conversion_factors_from_different_references(self, anipose_io: Dict,
                                                          test_positions_distances: Dict) -> Dict:  # Tuple? List?
        all_conversion_factors = {}
        for reference, markers in test_positions_distances.items():
            for m in markers:
                reference_marker_ids = (reference, m)
                distance_in_cm = test_positions_distances[reference][m]
                reference_distance_id = reference + "_" + m
                distance_to_cm_conversion_factor = self._get_xyz_to_cm_conversion_factor(
                    reference_marker_ids=reference_marker_ids,
                    distance_in_cm=distance_in_cm,
                    df_xyz=anipose_io['df_xyz'])
                all_conversion_factors[reference_distance_id] = distance_to_cm_conversion_factor

        return all_conversion_factors

    def _get_xyz_distance_in_triangulation_space(self, marker_ids: Tuple[str, str], df_xyz: pd.DataFrame) -> float:
        squared_differences = [(df_xyz[f'{marker_ids[0]}_{axis}'] - df_xyz[f'{marker_ids[1]}_{axis}']) ** 2 for axis in
                               ['x', 'y', 'z']]
        return sum(squared_differences) ** 0.5

    def _get_xyz_to_cm_conversion_factor(self, reference_marker_ids: Tuple[str, str], distance_in_cm: Union[int, float],
                                         df_xyz: pd.DataFrame) -> float:
        distance_in_triangulation_space = self._get_xyz_distance_in_triangulation_space(marker_ids=reference_marker_ids,
                                                                                        df_xyz=df_xyz)
        return distance_in_triangulation_space / distance_in_cm

    def _show_3D_plot(self, frame_idx: int, anipose_io: Dict, marker_ids_to_connect: List[Tuple[str]]) -> None:
        p3d = anipose_io['p3ds'][frame_idx]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], c='black', s=100)
        self._connect_all_marker_ids(ax=ax, points=p3d, scheme=marker_ids_to_connect, bodyparts=anipose_io['bodyparts'])
        for i in range(len(anipose_io['bodyparts'])):
            ax.text(p3d[i, 0], p3d[i, 1] + 0.01, p3d[i, 2], anipose_io['bodyparts'][i], size=9)
        plt.show()

    def _get_length_in_3d_space(self, PointA: np.array, PointB: np.array) -> float:
        length = math.sqrt((PointA[0] - PointB[0]) ** 2 + (PointA[1] - PointB[1]) ** 2 + (PointA[2] - PointB[2]) ** 2)
        return length

    def _get_angle_from_law_of_cosines(self, length_a: float, length_b: float, length_c: float) -> float:
        cos_angle = (length_c ** 2 + length_b ** 2 - length_a ** 2) / (2 * length_b * length_c)
        return math.degrees(math.acos(cos_angle))

    def _get_angle_between_three_points_at_PointA(self, PointA: np.array, PointB: np.array, PointC: np.array) -> float:
        length_c = self._get_length_in_3d_space(PointA, PointB)
        length_b = self._get_length_in_3d_space(PointA, PointC)
        length_a = self._get_length_in_3d_space(PointB, PointC)
        return self._get_angle_from_law_of_cosines(length_a, length_b, length_c)

    def _get_coordinates_plane_equation_from_three_points(self, PointA: np.array, PointB: np.array,
                                                          PointC: np.array) -> np.array:
        R1 = self._get_Richtungsvektor_from_two_points(PointA, PointB)
        R2 = self._get_Richtungsvektor_from_two_points(PointA, PointC)
        # check for linear independency
        # np.solve: R2 * x != R1
        plane_equation_coordinates = np.asarray([PointA, R1, R2])
        return plane_equation_coordinates

    def _get_vector_product(self, A: np.array, B: np.array) -> np.array:
        # Kreuzprodukt
        N = np.asarray([A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]])
        return N

    def _get_Richtungsvektor_from_two_points(self, PointA: np.array, PointB: np.array) -> np.array:
        R = np.asarray([PointA[0] - PointB[0], PointA[1] - PointB[1], PointA[2] - PointB[2]])
        return R

    def _get_vector_length(self, vector: np.array) -> float:
        length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        return length

    def _get_angle_between_plane_and_line(self, N: np.array, R: np.array) -> float:
        """

        :param N: normal vector of the plane
        :param R:
        :return:
        """
        cosphi = self._get_vector_length(vector=self._get_vector_product(A=N, B=R)) / (
                self._get_vector_length(N) * self._get_vector_length(R))
        phi = math.degrees(math.acos(cosphi))
        angle = 90 - phi
        return angle

    def _get_angle_between_two_points_and_plane(self, PointA: np.array, PointB: np.array, N: np.array) -> float:
        R = self._get_Richtungsvektor_from_two_points(PointA, PointB)
        return self._get_angle_between_plane_and_line(N=N, R=R)

    def _get_vector_from_label(self, label: str) -> np.array:
        return np.asarray([self.anipose_io['df_xyz'][label + '_x'], self.anipose_io['df_xyz'][label + '_y'],
                           self.anipose_io['df_xyz'][label + '_z']])
    
    def _get_distance_between_plane_and_point(self, N: np.array, PointOnPlane: np.array,
                                              DistantPoint: np.array) -> float:
        a = N[0] * PointOnPlane[0] + N[1] * PointOnPlane[1] + N[2] * PointOnPlane[2]
        distance = abs(N[0] * DistantPoint[0] + N[1] * DistantPoint[1] + N[2] * DistantPoint[2] - a) / math.sqrt(
            N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
        return distance