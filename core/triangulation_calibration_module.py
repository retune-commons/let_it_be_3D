from typing import List, Tuple, Dict, Union, Optional, OrderedDict, Any, Set
from pathlib import Path
from abc import ABC, abstractmethod
import itertools as it

from scipy.spatial.transform import Rotation
import aniposelib as ap_lib
import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoClip

from .video_metadata import VideoMetadata
from .video_interface import VideoInterface
from .video_synchronization import (
    RecordingVideoDownSynchronizer,
    RecordingVideoUpSynchronizer,
    CharucoVideoSynchronizer,
)
from .plotting import (
    AlignmentPlotCrossvalidation,
    PredictionsPlot,
    CalibrationValidationPlot,
    TriangulationVisualization,
    RotationVisualization,
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
    get_subsets_of_two_lists,
    KEYS_TO_CHECK_PROJECT,
    KEYS_TO_CHECK_RECORDING,
    KEYS_TO_CHECK_CAMERA,
    STANDARD_ATTRIBUTES_TRIANGULATION,
    STANDARD_ATTRIBUTES_CALIBRATION,
    SYNCHRO_METADATA_KEYS
)
from .angles_and_distances import (
    add_reprojection_errors_of_all_calibration_validation_markers,
    set_distances_and_angles_for_evaluation,
    fill_in_distances,
    add_all_real_distances_errors,
    set_angles_error_between_screws_and_plane,
)

def _get_metadata_from_configs(recording_config_filepath: Path, project_config_filepath: Path) -> Tuple[dict, dict]:
    project_config_dict = read_config(path=project_config_filepath)
    recording_config_dict = read_config(path=recording_config_filepath)

    missing_keys_project = check_keys(
        dictionary=project_config_dict, list_of_keys=KEYS_TO_CHECK_PROJECT
    )
    if missing_keys_project:
        raise KeyError(
            f"Missing metadata information in the project_config_file {project_config_filepath} for {missing_keys_project}."
        )
    missing_keys_recording = check_keys(
        dictionary=recording_config_dict, list_of_keys=KEYS_TO_CHECK_RECORDING
    )
    if missing_keys_recording:
        raise KeyError(
            f"Missing information for {missing_keys_recording} in the config_file {recording_config_filepath}!"
        )

    for dictionary_key in KEYS_TO_CHECK_CAMERA:
        cameras_with_missing_keys = check_keys(
            dictionary=project_config_dict[dictionary_key],
            list_of_keys=project_config_dict["valid_cam_IDs"],
        )
        if cameras_with_missing_keys:
            raise KeyError(
                f"Missing information {dictionary_key} for cam {cameras_with_missing_keys} in the config_file {project_config_filepath}!"
            )
    return recording_config_dict, project_config_dict

def _validate_metadata(metadata_from_videos: Dict,
                       attributes_to_check: List[str]) -> \
        Tuple[Any, ...]:
    sets_of_attributes = []
    for attribute_to_check in attributes_to_check:
        set_of_attribute = set(getattr(video_metadata, attribute_to_check)
                               for video_metadata in metadata_from_videos.values()
                               )
        sets_of_attributes.append(set_of_attribute)
    attribute: Set[str]
    for attribute in sets_of_attributes:
        if len(attribute) > 1:
            raise ValueError(
                f"The filenames of the calibration_validation images give different metadata! Reasons could be:\n"
                f"  - image belongs to another calibration\n"
                f"  - image filename is valid, but wrong\n"
                f"You should run the filename_checker before to avoid such Errors!"
            )
    return tuple(list(set_of_attribute)[0] for set_of_attribute in sets_of_attributes)

def exclude_by_framenum(metadata_from_videos: Dict, allowed_num_diverging_frames: int) -> List[Any]:
    synch_framenum_median = np.median(
        [video_metadata.framenum_synchronized for video_metadata in metadata_from_videos.values()])

    videos_to_exclude = []
    for video_metadata in metadata_from_videos.values():
        if video_metadata.framenum_synchronized < (
                synch_framenum_median - allowed_num_diverging_frames) or video_metadata.framenum_synchronized > (
                synch_framenum_median + allowed_num_diverging_frames):
            video_metadata.exclusion_state = "exclude"
            videos_to_exclude.append(video_metadata.cam_id)
    if videos_to_exclude:
        print(f"{videos_to_exclude} were excluded!")
    return videos_to_exclude


def _create_output_directory(project_config_filepath: Path) -> Path:
    unnamed_idx = 0
    for file in project_config_filepath.parent.iterdir():
        if str(file.name).startswith("unnamed_calibration_"):
            idx = int(file.stem[20:])
            if idx > unnamed_idx:
                unnamed_idx = idx
    output_directory = project_config_filepath.parent.joinpath(
        f"unnamed_calibration_{str(unnamed_idx + 1)}"
    )
    if not output_directory.exists():
        Path.mkdir(output_directory)
    return output_directory


def _check_output_directory(project_config_filepath: Path, output_directory: Optional[Path] = None) -> Path:
    if output_directory is not None:
        if not output_directory.exists():
            output_directory = _create_output_directory(
                project_config_filepath=project_config_filepath
            )
    else:
        output_directory = _create_output_directory(
            project_config_filepath=project_config_filepath
        )
    return output_directory


def _create_video_objects(
        directory: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict,
        videometadata_tag: str,
        output_directory: Path,
        filetypes: List[str],
        filename_tag: str = "",
        test_mode: bool = False,
) -> Tuple[Dict, Dict]:
    videofiles = [file for file in directory.iterdir() if filename_tag.lower() in file.name.lower()
                  and "synchronized" not in file.name and file.suffix in filetypes]

    video_interfaces = {}
    metadata_from_videos = {}
    for filepath in videofiles:
        video_metadata = VideoMetadata(
            video_filepath=filepath,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            tag=videometadata_tag,
        )

        video_interfaces[video_metadata.cam_id] = VideoInterface(
            video_metadata=video_metadata,
            output_dir=output_directory,
            test_mode=test_mode,
        )
        metadata_from_videos[video_metadata.cam_id] = video_metadata
    return video_interfaces, metadata_from_videos

class Calibration():
    def __init__(
            self,
            calibration_directory: Path,
            project_config_filepath: Path,
            recording_config_filepath: Path,
            output_directory: Optional[Path] = None,
            test_mode: bool = False,
    ) -> None:
        for attribute in STANDARD_ATTRIBUTES_CALIBRATION:
            setattr(self, attribute, None)
        self.calibration_directory = convert_to_path(calibration_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self.output_directory = _check_output_directory(output_directory=output_directory, project_config_filepath=project_config_filepath)
        recording_config_dict, project_config_dict = _get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self.synchro_metadata = {key: project_config_dict[key] for key in SYNCHRO_METADATA_KEYS}
        for attribute in ["valid_cam_IDs", "calibration_tag", "calibration_validation_tag",
                          "score_threshold", "triangulation_type", "allowed_num_diverging_frames"]:
            setattr(self, attribute, project_config_dict[attribute])
        for attribute in ["recording_date", "led_pattern", "calibration_index", "target_fps"]:
            setattr(self, attribute, recording_config_dict[attribute])

        self.video_interfaces, self.metadata_from_videos = _create_video_objects(
            directory=self.calibration_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="calibration",
            output_directory=self.output_directory,
            filename_tag=self.calibration_tag,
            filetypes=[".AVI", ".avi", ".mov", ".mp4"],
            test_mode=test_mode,
        )
        self.recording_date, *_ = _validate_metadata(metadata_from_videos=self.metadata_from_videos,
                                                     attributes_to_check=['recording_date'])
        self.target_fps = min([video_metadata.fps for video_metadata in self.metadata_from_videos.values()])
        for video_metadata in self.metadata_from_videos.values():
            video_metadata.target_fps = self.target_fps

    def run_synchronization(self, test_mode: bool = False) -> None:
        for video_interface in self.video_interfaces.values():
            video_interface.run_synchronizer(
                synchronizer=CharucoVideoSynchronizer,
                output_directory=self.output_directory,
                synchronize_only=True,
                test_mode=test_mode,
                synchro_metadata=self.synchro_metadata,
            )
            self.synchronized_charuco_videofiles[
                video_interface.video_metadata.cam_id
            ] = str(video_interface.synchronized_video_filepath)
            self.camera_objects.append(video_interface.export_for_aniposelib())

        template = list(self.video_interfaces.values())[
            0].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(
            fps=self.target_fps)[0][0]
        led_timeseries_crossvalidation = {}
        for video_interface in self.video_interfaces.values():
            if hasattr(video_interface.synchronizer_object, "led_timeseries_for_cross_video_validation"):
                led_timeseries_crossvalidation[
                    video_interface.video_metadata.cam_id
                ] = video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
        if list(led_timeseries_crossvalidation.keys()):
            synchronization_crossvalidation = AlignmentPlotCrossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                metadata={
                    "recording_date": self.recording_date,
                    "charuco_video": True,
                    "fps": self.target_fps,
                },
                output_directory=self.output_directory,
            )
        cameras = [camera_object.name for camera_object in self.camera_objects]
        _, duplicate_cams, *_ = get_subsets_of_two_lists(cameras, [])
        if duplicate_cams:
            raise ValueError(
                f"You added multiple cameras with the cam_id {duplicate_cams}, "
                "however, all cam_ids must be unique! Please check for duplicates "
                "in the calibration directory and rename them!"
            )
        self.camera_objects.sort(key=lambda x: x.name, reverse=False)
        cams_to_exclude = exclude_by_framenum(metadata_from_videos=self.metadata_from_videos,
                                              allowed_num_diverging_frames=self.allowed_num_diverging_frames)
        self.valid_videos = [cam.name for cam in self.camera_objects if cam.name not in cams_to_exclude]
        for cam in cams_to_exclude:
            self.camera_objects.remove(cam)
        self.initialize_camera_group(camera_objects=self.camera_objects)

    # STOPPED Function controlling here
    def run_calibration(
            self,
            use_own_intrinsic_calibration: bool = True,
            verbose: int = 0,
            charuco_calibration_board: Optional = None,
            test_mode: bool = False,
            iteration: Optional[int] = None,
    ) -> Path:
        filename = f"{create_calibration_key(videos=self.valid_videos, recording_date=self.recording_date, calibration_index=self.calibration_index, iteration=iteration)}.toml"

        calibration_filepath = self.output_directory.joinpath(filename)
        if (not test_mode) or (
                test_mode and not calibration_filepath.exists()
        ):
            if charuco_calibration_board is None:
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                charuco_calibration_board = ap_lib.boards.CharucoBoard(
                    7,
                    5,
                    square_length=1,
                    marker_length=0.8,
                    marker_bits=6,
                    aruco_dict=aruco_dict,
                )
            sorted_videos = OrderedDict(sorted(self.synchronized_charuco_videofiles.items()))
            videos = [[video] for video in sorted_videos.values()]
            self.reprojerr, _ = self.camera_group.calibrate_videos(
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

    def initialize_camera_group(self, camera_objects: List) -> None:
        self.camera_group = ap_lib.cameras.CameraGroup(camera_objects)

    def _save_calibration(self, calibration_filepath: Path) -> None:
        if calibration_filepath.exists():
            calibration_filepath.unlink()
        self.camera_group.dump(calibration_filepath)

    def calibrate_optimal(
            self,
            calibration_validation: "CalibrationValidation",
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
        good_calibration_filepath = self.output_directory.joinpath(
            f"{create_calibration_key(videos=self.valid_videos, recording_date=self.recording_date, calibration_index=self.calibration_index)}.toml")

        calibration_filepath = None
        for cal in range(max_iters):
            if good_calibration_filepath.exists() and test_mode:
                calibration_filepath = good_calibration_filepath
                self.reprojerr = 0
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

            all_angle_errors, all_percentage_errors = [], []
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
                calibration_filepath = good_calibration_filepath
                print(f"Good Calibration reached at iteration {cal}! Named it {good_calibration_filepath}.")
                break

        self.report_filepath = self.output_directory.joinpath(
            f"{self.recording_date}_calibration_report.csv"
        )
        report.to_csv(self.report_filepath, index=False)

        if not calibration_found:
            print("No optimal calibration found with given thresholds! Returned last executed calibration!")
        return calibration_filepath


def _add_missing_marker_ids_to_prediction(
        missing_marker_ids: List[str], df: pd.DataFrame(), framenum: int = 1
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
        marker_ids_to_match: List[str], template_marker_ids: List[str]
) -> List:
    return [
        marker_id
        for marker_id in marker_ids_to_match
        if marker_id not in template_marker_ids
    ]


def _remove_marker_ids_not_in_ground_truth(
        marker_ids_to_remove: List[str], df: pd.DataFrame()
) -> None:
    columns_to_remove = [
        column_name
        for column_name in df.columns
        if column_name[1] in marker_ids_to_remove
    ]
    df.drop(columns=columns_to_remove, inplace=True)


def _save_dataframe_as_csv(filepath: str, df: pd.DataFrame) -> None:
    filepath = convert_to_path(filepath)
    if filepath.exists():
        filepath.unlink()
    df.to_csv(filepath, index=False)


class Triangulation(ABC):
    @abstractmethod
    def _create_csv_filepath(self) -> Path:
        pass

    @property
    @abstractmethod
    def _metadata_keys(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def _allowed_filetypes(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def _videometadata_tag(self) -> str:
        pass

    def __init__(self,
                 project_config_filepath: Path,
                 directory: Path,
                 recording_config_filepath: Path,
                 test_mode: bool = False,
                 output_directory: Optional[Path] = None):
        for attribute in STANDARD_ATTRIBUTES_TRIANGULATION:
            setattr(self, attribute, None)
        self.directory = convert_to_path(directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        output_directory = convert_to_path(output_directory)
        self.output_directory = _check_output_directory(output_directory=output_directory,
                                                        project_config_filepath=project_config_filepath)
        recording_config_dict, project_config_dict = _get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self.synchro_metadata = {key: project_config_dict[key] for key in SYNCHRO_METADATA_KEYS}
        for attribute in ["use_gpu", "valid_cam_IDs", "calibration_tag", "calibration_validation_tag",
                          "score_threshold", "triangulation_type", "allowed_num_diverging_frames"]:
            setattr(self, attribute, project_config_dict[attribute])
        for attribute in ["recording_date", "led_pattern", "calibration_index", "target_fps"]:
            setattr(self, attribute, recording_config_dict[attribute])

        self.video_interfaces, self.metadata_from_videos = _create_video_objects(
            directory=self.directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag=self._videometadata_tag,
            output_directory=self.output_directory,
            filename_tag=self.calibration_validation_tag if self._videometadata_tag == "calvin" else "",
            test_mode=test_mode,
        )

        metadata = _validate_metadata(metadata_from_videos=self.metadata_from_videos,
                                      attributes_to_check=self._metadata_keys)
        for attribute, value in zip(self._metadata_keys, metadata):
            setattr(self, attribute, value)

        self.csv_output_filepath = self._create_csv_filepath()

    def run_triangulation(
            self,
            calibration_toml_filepath: Path,
            test_mode: bool = False,
    ):
        self.calibration_toml_filepath = convert_to_path(calibration_toml_filepath)
        self._load_calibration(filepath=self.calibration_toml_filepath)

        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.all_cameras = [camera.name for camera in self.camera_group.cameras]
        self.all_cameras.sort()
        _, duplicate_cams, _, missing_cams_in_all_cameras = get_subsets_of_two_lists(filepath_keys, self.all_cameras)
        if missing_cams_in_all_cameras:
            min_framenum = min([pd.read_hdf(path).shape[0] for path in self.triangulation_dlc_cams_filepaths.values()])
            self._fake_missing_files(missing_cams=missing_cams_in_all_cameras, framenum=min_framenum,
                                     markers=self.markers)
        for cam in missing_cams_in_all_cameras:
            self.triangulation_dlc_cams_filepaths.pop(cam)

        self._preprocess_dlc_predictions_for_anipose(test_mode=test_mode)

        if self.triangulation_type == "triangulate":
            p3ds_flat = self.camera_group.triangulate(self.anipose_io["points_flat"], progress=True)
        elif self.triangulation_type == "triangulate_optim_ransac_False":
            p3ds_flat = self.camera_group.triangulate_optim(self.anipose_io["points"], init_ransac=False,
                                                            init_progress=True).reshape(
                self.anipose_io["n_points"] * self.anipose_io["n_joints"], 3)
        elif self.triangulation_type == "triangulate_optim_ransac_True":
            p3ds_flat = self.camera_group.triangulate_optim(self.anipose_io["points"], init_ransac=True,
                                                            init_progress=True).reshape(
                self.anipose_io["n_points"] * self.anipose_io["n_joints"], 3)
        else:
            raise ValueError(
                "Supported methods for triangulation are triangulate, triangulate_optim_ransac_True, triangulate_optim_ransac_False!")

        self._postprocess_triangulations_and_calculate_reprojection_error(p3ds_flat=p3ds_flat)

        self._get_dataframe_of_triangulated_points()
        if (not test_mode) or (test_mode and not self.csv_output_filepath.exists()):
            _save_dataframe_as_csv(filepath=self.csv_output_filepath, df=self.df)
        for path in self.triangulation_dlc_cams_filepaths.values():
            if "_temp" in path.name:
                path.unlink()

    def exclude_markers(self, all_markers_to_exclude_config_path: Path, verbose: bool = True):
        all_markers_to_exclude = read_config(all_markers_to_exclude_config_path)

        missing_cams = check_keys(all_markers_to_exclude, list(self.triangulation_dlc_cams_filepaths))
        if missing_cams:
            if verbose:
                print(f"Found no markers to exclude for {missing_cams} in {str(all_markers_to_exclude_config_path)}!")

        for cam_id in self.triangulation_dlc_cams_filepaths:
            h5_file = self.triangulation_dlc_cams_filepaths[cam_id]
            df = pd.read_hdf(h5_file)
            markers = set(b for a, b, c in df.keys())
            markers_to_exclude_per_cam = all_markers_to_exclude[cam_id]
            existing_markers_to_exclude = list(set(markers) & set(markers_to_exclude_per_cam))
            not_existing_markers = [marker for marker in markers_to_exclude_per_cam if marker not in markers]
            if not_existing_markers:
                if verbose:
                    print(
                        f"The following markers were not found in the dataframe, but were given as markers to exclude for {cam_id}: {not_existing_markers}!")
            if existing_markers_to_exclude:
                for i, keys in enumerate(df.columns):
                    a, b, c = keys
                    if b in existing_markers_to_exclude and c == "likelihood":
                        df.isetitem(i, 0)
                df.to_hdf(h5_file, key="key", mode="w")

        self.markers_excluded_manually = True

    def _fake_missing_files(
            self, missing_cams: List[str], framenum: int, markers: List[str]) -> None:
        for cam in missing_cams:
            print(f"Creating empty .h5 file for {cam}!")
            h5_output_filepath = self.output_directory.joinpath(
                self.csv_output_filepath.stem + f"empty_{cam}.h5"
            )
            cols = get_multi_index(markers)
            df = pd.DataFrame(data=np.zeros((framenum, len(cols))), columns=cols, dtype=int)
            df.to_hdf(str(h5_output_filepath), "empty")
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

        prediction_marker_ids = list(
            set([marker_id for scorer, marker_id, key in calibration_validation_markers_df.columns]))

        marker_ids_not_in_ground_truth = _find_non_matching_marker_ids(
            prediction_marker_ids, defined_marker_ids
        )
        marker_ids_not_in_prediction = _find_non_matching_marker_ids(
            defined_marker_ids, prediction_marker_ids
        )
        if add_missing_marker_ids_with_0_likelihood & (
                len(marker_ids_not_in_prediction) > 0
        ):
            _add_missing_marker_ids_to_prediction(
                missing_marker_ids=marker_ids_not_in_prediction,
                df=calibration_validation_markers_df,
                framenum=framenum,
            )
            print(
                "The following marker_ids were missing and added to the dataframe with a "
                f"likelihood of 0: {marker_ids_not_in_prediction}."
            )
        if marker_ids_not_in_ground_truth:
            _remove_marker_ids_not_in_ground_truth(
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
            anipose_io["n_points"] = b - a
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


def _get_best_frame_for_normalisation(config: Dict, df: pd.DataFrame) -> int:
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

    if valid_frames_for_normalization:
        return valid_frames_for_normalization[0]
    else:
        raise ValueError("Could not normalize the dataframe!")


class TriangulationRecordings(Triangulation):

    @property
    def _metadata_keys(self)->List[str]:
        return ["recording_date", "paradigm", "mouse_id"]

    @property
    def _videometadata_tag(self) -> str:
        return "recording"

    @property
    def _allowed_filetypes(self) -> List[str]:
        return [".AVI", ".avi", ".mov", ".mp4"]

    def run_synchronization(
            self, synchronize_only: bool = False, test_mode: bool = False
    ) -> None:
        for video_interface in self.video_interfaces.values():
            if (
                    video_interface.video_metadata.fps
                    >= video_interface.video_metadata.target_fps
            ):
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoDownSynchronizer,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
                    test_mode=test_mode,
                    synchro_metadata=self.synchro_metadata,
                )
            else:
                video_interface.run_synchronizer(
                    synchronizer=RecordingVideoUpSynchronizer,
                    output_directory=self.output_directory,
                    synchronize_only=synchronize_only,
                    test_mode=test_mode,
                    synchro_metadata=self.synchro_metadata,
                )
            self.synchronized_videos[
                video_interface.video_metadata.cam_id
            ] = video_interface.synchronized_video_filepath

        template = list(self.video_interfaces.values())[
            0].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(fps=self.target_fps)[0][0]

        led_timeseries_crossvalidation = {}
        for video_interface in self.video_interfaces.values():
            if hasattr(video_interface.synchronizer_object, "led_timeseries_for_cross_video_validation"):
                led_timeseries_crossvalidation[
                    video_interface.video_metadata.cam_id
                ] = video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
        if list(led_timeseries_crossvalidation.keys()):
            synchronization_crossvalidation = AlignmentPlotCrossvalidation(
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

        self.markers = list(pd.read_hdf(list(self.triangulation_dlc_cams_filepaths.values())[0]).columns.levels[1])

        cams_to_exclude = exclude_by_framenum(metadata_from_videos=self.metadata_from_videos,
                                              allowed_num_diverging_frames=self.allowed_num_diverging_frames)
        for cam in self.metadata_from_videos:
            if cam in cams_to_exclude:
                self.triangulation_dlc_cams_filepaths.pop(cam)

    def _create_csv_filepath(self) -> Path:
        filepath_out = self.output_directory.joinpath(
            f"{self.mouse_id}_{self.recording_date}_{self.paradigm}_{self.target_fps}fps_{self.score_threshold}p_excludedmarkers{self.markers_excluded_manually}_filtered{self.synchro_metadata['use_2D_filter']}_normalised{self.normalised_dataframe}_{self.triangulation_type}.csv"
        )
        return filepath_out

    def normalize(self, normalization_config_path: Path, test_mode: bool = False) -> None:
        normalization_config_path = convert_to_path(normalization_config_path)
        config = read_config(normalization_config_path)

        best_frame = _get_best_frame_for_normalisation(config=config, df=self.df)

        x, y, z = get_3D_array(self.df, config['center'], best_frame)
        for key in self.df.keys():
            if '_x' in key:
                self.df[key] = self.df[key] - x
            if '_y' in key:
                self.df[key] = self.df[key] - y
            if '_z' in key:
                self.df[key] = self.df[key] - z

        marker0 = get_3D_array(self.df, config['ReferenceLengthMarkers'][0], best_frame)
        marker1 = get_3D_array(self.df, config['ReferenceLengthMarkers'][1], best_frame)
        lengthleftside = np.sqrt(
            (marker0[0] - marker1[0]) ** 2 + (marker0[1] - marker1[1]) ** 2 + (marker0[2] - marker1[2]) ** 2)
        conversionfactor = config['ReferenceLengthCm'] / lengthleftside

        bp_keys_unflat = set(get_3D_df_keys(key[:-2]) for key in self.df.keys() if
                             'error' not in key and 'score' not in key and "M_" not in key and 'center' not in key and 'fn' not in key)
        bp_keys = list(it.chain(*bp_keys_unflat))

        normalised = self.df
        normalised[bp_keys] *= conversionfactor
        reference_rotation_markers = []
        for marker in config['ReferenceRotationMarkers']:
            reference_rotation_markers.append(get_3D_array(normalised, marker, best_frame))
        # The rotation matrix between the referencespace and the reconstructedspace is being calculated.
        r, self.rotation_error = Rotation.align_vectors(config["ReferenceRotationCoords"], reference_rotation_markers)

        rotated = normalised
        for key in bp_keys_unflat:
            rot_points = r.apply(normalised.loc[:, [key[0], key[1], key[2]]])
            rotated.loc[:, key[0]] = rot_points[:, 0]
            rotated.loc[:, key[1]] = rot_points[:, 1]
            rotated.loc[:, key[2]] = rot_points[:, 2]

        rotated_markers = []
        for marker in config['ReferenceRotationMarkers']:
            rotated_markers.append(get_3D_array(rotated, marker, best_frame))

        self.normalised_dataframe = True
        self.rotated_filepath = self._create_csv_filepath()
        if (not test_mode) or (test_mode and not self.rotated_filepath.exists()):
            _save_dataframe_as_csv(filepath=str(self.rotated_filepath), df=rotated)
        RotationVisualization(rotated_markers=rotated_markers, config=config, filepath=self.rotated_filepath,
                              rotation_error=self.rotation_error)

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

        t = TriangulationVisualization(df_filepath=self.rotated_filepath, output_directory=self.output_directory,
                                       idx=idx, config=self.video_plotting_config, plot=False, save=False)
        return t.return_fig()


class CalibrationValidation(Triangulation):

    @property
    def _metadata_keys(self)->List[str]:
        return ["recording_date"]

    @property
    def _videometadata_tag(self) -> str:
        return "calvin"

    @property
    def _allowed_filetypes(self) -> List[str]:
        return [".bmp", ".tiff", ".png", ".jpg", ".AVI", ".avi"]

    def add_ground_truth_config(self, ground_truth_config_filepath: Path) -> None:
        ground_truth_config_filepath = convert_to_path(ground_truth_config_filepath)
        self.ground_truth_config = read_config(ground_truth_config_filepath)
        self.markers = self.ground_truth_config["unique_ids"]

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
            PredictionsPlot(
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
            CalibrationValidationPlot(
                p3d=self.anipose_io["p3ds"][0],
                bodyparts=self.anipose_io["bodyparts"],
                output_directory=self.output_directory,
                marker_ids_to_connect=self.ground_truth_config[
                    "marker_ids_to_connect_in_3D_plot"
                ],
                plot=True,
                save=True,
            )

    def _create_csv_filepath(self) -> Path:
        filepath_out = self.output_directory.joinpath(
            f"Calvin_{self.recording_date}_{self.score_threshold}p_excludedmarkers{self.markers_excluded_manually}_filteredFalse_{self.triangulation_type}.csv")
        return filepath_out
