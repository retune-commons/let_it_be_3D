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
    set_angles_error_between_screws_and_plane, get_xyz_distance_in_triangulation_space,
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
            list_of_keys=project_config_dict["valid_cam_ids"],
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


def initialize_camera_group(camera_objects: List) -> ap_lib.cameras.CameraGroup:
    return ap_lib.cameras.CameraGroup(camera_objects)


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
        for attribute in ["valid_cam_ids", "calibration_tag", "calibration_validation_tag",
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
        self.synchronized_charuco_videofiles = {}
        self.camera_objects = []
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
            if video_interface.synchronizer_object.led_timeseries_for_cross_video_validation is not None:
                led_timeseries_crossvalidation[
                    video_interface.video_metadata.cam_id
                ] = video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
        if list(led_timeseries_crossvalidation.keys()):
            filename = f'{self.recording_date}_charuco_synchronization_crossvalidation_{self.target_fps}'
            synchronization_crossvalidation = AlignmentPlotCrossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                output_directory=self.output_directory,
                filename=filename,
            )
            synchronization_crossvalidation.create_plot(save=True, plot=True)
        cameras = [camera_object.name for camera_object in self.camera_objects]
        duplicate_cams = _get_duplicate_elems_in_list(cameras)
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
        self.camera_group = initialize_camera_group(camera_objects=self.camera_objects)

    def run_calibration(
            self,
            use_own_intrinsic_calibration: bool = True,
            verbose: int = 0,
            charuco_calibration_board: Optional[ap_lib.boards.CharucoBoard] = None,
            test_mode: bool = False,
            iteration: Optional[int] = None,
    ) -> Path:
        calibration_key = create_calibration_key(videos=self.valid_videos, recording_date=self.recording_date, calibration_index=self.calibration_index, iteration=iteration)
        filename = f"{calibration_key}.toml"

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
            self._save_calibration(calibration_filepath=calibration_filepath, camera_group=self.camera_group)
        else:
            self.reprojerr = 0
        return calibration_filepath

    def _save_calibration(self, calibration_filepath: Path, camera_group: ap_lib.cameras.CameraGroup) -> None:
        if calibration_filepath.exists():
            calibration_filepath.unlink()
        camera_group.dump(calibration_filepath)

    def calibrate_optimal(
            self,
            calibration_validation: "CalibrationValidation",
            max_iters: int = 5,
            p_threshold: float = 0.1,
            angle_threshold: float = 5.,
            verbose: int = 1,
            test_mode: bool = False,
    ):
        """finds optimal calibration through repeated optimisations of anipose"""
        report = pd.DataFrame()
        calibration_found = False
        calibration_key = create_calibration_key(videos=self.valid_videos, recording_date=self.recording_date, calibration_index=self.calibration_index)
        good_calibration_filepath = self.output_directory.joinpath(f"{calibration_key}.toml")
        calibration_filepath = None
        for cal in range(max_iters):
            if good_calibration_filepath.exists() and test_mode:
                calibration_filepath = good_calibration_filepath
                self.reprojerr = 0
            else:
                calibration_filepath = self.run_calibration(verbose=verbose, test_mode=test_mode, iteration=cal)

            calibration_validation.run_triangulation(calibration_toml_filepath=calibration_filepath)

            calibration_validation.evaluate_triangulation_of_calibration_validation_markers()
            calibration_errors = calibration_validation.anipose_io["distance_errors_in_cm"]
            calibration_angles_errors = calibration_validation.anipose_io["angles_error_screws_plan"]
            reprojerr_nonan = calibration_validation.anipose_io["reproj_nonan"].mean()

            all_angle_errors, all_percentage_errors = [], []
            for reference in calibration_errors.keys():
                all_percentage_errors = [
                    percentage_error
                    for marker_id_a, marker_id_b, distance_error, percentage_error in calibration_errors[
                        reference]["individual_errors"]]
            all_angle_errors = list(calibration_angles_errors.values())

            mean_dist_err_percentage = np.nanmean(np.asarray(all_percentage_errors))
            mean_angle_err = np.nanmean(np.asarray(all_angle_errors))

            if verbose > 0:
                print(f"Calibration {cal}\n mean percentage error: {mean_dist_err_percentage}\n mean angle error: {mean_angle_err}")

            report.loc[cal, "mean_distance_error_percentage"] = mean_dist_err_percentage
            report.loc[cal, "mean_angle_error"] = mean_angle_err
            report.loc[cal, "reprojerror"] = reprojerr_nonan

            if (mean_dist_err_percentage < p_threshold and mean_angle_err < angle_threshold):
                calibration_found = True
                calibration_filepath.rename(good_calibration_filepath)
                calibration_filepath = good_calibration_filepath
                print(f"Good Calibration reached at iteration {cal}! Named it {good_calibration_filepath}.")
                break

        self.report_filepath = self.output_directory.joinpath(f"{self.recording_date}_calibration_report.csv")
        report.to_csv(self.report_filepath, index=False)

        if not calibration_found:
            print("No optimal calibration found with given thresholds! Returned last executed calibration!")
        return calibration_filepath


def _add_missing_marker_ids_to_prediction(
        missing_marker_ids: List[str], df: pd.DataFrame(), framenum: int = 1
) -> pd.DataFrame():
    try:
        scorer = list(df.columns)[0][0]
    except IndexError:
        scorer = "zero_likelihood_markers"
    for marker_id in missing_marker_ids:
        for key in ["x", "y", "likelihood"]:
            df.loc[[i for i in range(framenum)], (scorer, marker_id, key)] = 0
    return df


def _find_non_matching_list_elements(list1: List[str], list2: List[str]) -> List[str]:
    return [marker_id for marker_id in list1 if marker_id not in list2]


def _get_duplicate_elems_in_list(list1: List[str])->List[str]:
    individual_elems, duplicate_elems = [], []
    for elem in list1:
        if elem in individual_elems:
            duplicate_elems.append(elem)
        else:
            individual_elems.append(elem)


def _remove_marker_ids_not_in_ground_truth(
        marker_ids_to_remove: List[str], df: pd.DataFrame()
) -> pd.DataFrame():
    columns_to_remove = [column_name for column_name in df.columns if column_name[1] in marker_ids_to_remove]
    return df.drop(columns=columns_to_remove)


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
        for attribute in ["use_gpu", "valid_cam_ids", "calibration_tag", "calibration_validation_tag",
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
            filetypes=self._allowed_filetypes,
        )
        metadata = _validate_metadata(metadata_from_videos=self.metadata_from_videos,
                                      attributes_to_check=self._metadata_keys)
        for attribute, value in zip(self._metadata_keys, metadata):
            setattr(self, attribute, value)
        self.csv_output_filepath = self._create_csv_filepath()

    def run_triangulation(
            self,
            calibration_toml_filepath: Union[Path, str],
            test_mode: bool = False,
    ):
        calibration_toml_filepath = convert_to_path(calibration_toml_filepath)
        self.camera_group = self._load_calibration(filepath=calibration_toml_filepath)

        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.all_cameras = [camera.name for camera in self.camera_group.cameras]
        self.all_cameras.sort()
        missing_cams_in_all_cameras = _find_non_matching_list_elements(filepath_keys, self.all_cameras)
        if missing_cams_in_all_cameras:
            min_framenum = min([pd.read_hdf(path).shape[0] for path in self.triangulation_dlc_cams_filepaths.values()])
            self._create_empty_files(cams_to_create_empty_files=missing_cams_in_all_cameras, framenum=min_framenum,
                                     markers=self.markers)
        for cam in missing_cams_in_all_cameras:
            self.triangulation_dlc_cams_filepaths.pop(cam)

        self.anipose_io = self._preprocess_dlc_predictions_for_anipose(test_mode=test_mode)
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
        self.anipose_io["p3ds"] = p3ds_flat.reshape(self.anipose_io["n_points"], self.anipose_io["n_joints"], 3)

        self.anipose_io["reprojerr"], self.anipose_io["reproj_nonan"], self.anipose_io["reprojerr_flat"] = self._get_reprojection_errors(
            p3ds_flat=p3ds_flat)

        self.df = self._get_dataframe_of_triangulated_points(anipose_io=self.anipose_io)
        if (not test_mode) or (test_mode and not self.csv_output_filepath.exists()):
            _save_dataframe_as_csv(filepath=self.csv_output_filepath, df=self.df)
        self.delete_temp_files()

    def delete_temp_files(self)->None:
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
                        df.iloc[:, i] = 0
                df.to_hdf(h5_file, key="key", mode="w")
        self.markers_excluded_manually = True

    def _create_empty_files(
            self, cams_to_create_empty_files: List[str], framenum: int, markers: List[str]) -> None:
        for cam in cams_to_create_empty_files:
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
                defined_marker_ids=markers
            )
            self.triangulation_dlc_cams_filepaths[cam] = h5_output_filepath

    def _validate_calibration_validation_marker_ids(
            self,
            calibration_validation_markers_df_filepath: Path,
            framenum: int,
            defined_marker_ids: List[str],
            add_missing_marker_ids_with_0_likelihood: bool = True,
    ) -> None:
        calibration_validation_markers_df = pd.read_hdf(calibration_validation_markers_df_filepath)
        prediction_marker_ids = list(
            set([marker_id for scorer, marker_id, key in calibration_validation_markers_df.columns]))
        marker_ids_not_in_ground_truth = _find_non_matching_list_elements(prediction_marker_ids, defined_marker_ids)
        marker_ids_not_in_prediction = _find_non_matching_list_elements(defined_marker_ids, prediction_marker_ids)
        if add_missing_marker_ids_with_0_likelihood & bool(marker_ids_not_in_prediction):
            calibration_validation_markers_df = _add_missing_marker_ids_to_prediction(
                missing_marker_ids=marker_ids_not_in_prediction,
                df=calibration_validation_markers_df,
                framenum=framenum,
            )
            print(
                "The following marker_ids were missing and added to the dataframe with a "
                f"likelihood of 0: {marker_ids_not_in_prediction}."
            )
        if marker_ids_not_in_ground_truth:
            calibration_validation_markers_df = _remove_marker_ids_not_in_ground_truth(
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

    def _load_calibration(self, filepath: Path) -> ap_lib.cameras.CameraGroup:
        if filepath.name.endswith(".toml") and filepath.exists():
            return ap_lib.cameras.CameraGroup.load(filepath)
        else:
            raise FileNotFoundError(
                f"The path, given as calibration_toml_filepath\n"
                "does not end with .toml or does not exist!\n"
                "Make sure, that you enter the correct path!"
            )

    def _preprocess_dlc_predictions_for_anipose(self, test_mode: bool = False) -> Dict:
        anipose_io = ap_lib.utils.load_pose2d_fnames(
            fname_dict=self.triangulation_dlc_cams_filepaths
        )
        return self._add_additional_information_and_continue_preprocessing(anipose_io=anipose_io, test_mode=test_mode)

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

    def _get_reprojection_errors(
            self, p3ds_flat: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        reprojerr_flat = self.camera_group.reprojection_error(p3ds_flat, self.anipose_io["points_flat"], mean=True)
        reprojerr = reprojerr_flat.reshape(self.anipose_io["n_points"], self.anipose_io["n_joints"])
        reprojerr_nonan = reprojerr[np.logical_not(np.isnan(reprojerr))]
        return reprojerr, reprojerr_nonan, reprojerr_flat

    def _get_dataframe_of_triangulated_points(self, anipose_io: Dict) -> pd.DataFrame:
        """
        The following function was taken from https://github.com/lambdaloop/anipose/blob/d20091550dc8b901f460f914544ecfc66c116329/anipose/triangulate.py.
        Changes were made to match our needs here.

        BSD 2-Clause License

        Copyright (c) 2019, Pierre Karashchuk
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this
           list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice,
           this list of conditions and the following disclaimer in the documentation
           and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """
        all_points_raw = anipose_io["points"]
        all_scores = anipose_io["scores"]
        _cams, n_frames, n_joints, _ = all_points_raw.shape
        good_points = ~np.isnan(all_points_raw[:, :, :, 0])
        num_cams = np.sum(good_points, axis=0).astype("float")
        all_points_3d = anipose_io['p3ds'].reshape(n_frames, n_joints, 3)
        all_errors = anipose_io['reprojerr_flat'].reshape(n_frames, n_joints)
        all_scores[~good_points] = 2
        scores_3d = np.min(all_scores, axis=0)

        scores_3d[num_cams < 2] = np.nan
        all_errors[num_cams < 2] = np.nan
        num_cams[num_cams < 2] = np.nan

        all_points_3d_adj = all_points_3d
        M = np.identity(3)
        center = np.zeros(3)
        df = pd.DataFrame()
        for bp_num, bp in enumerate(anipose_io["bodyparts"]):
            for ax_num, axis in enumerate(["x", "y", "z"]):
                df[bp + "_" + axis] = all_points_3d_adj[:, bp_num, ax_num]
            df[bp + "_error"] = anipose_io['reprojerr'][:, bp_num]
            df[bp + "_score"] = scores_3d[:, bp_num]
        for i in range(3):
            for j in range(3):
                df["M_{}{}".format(i, j)] = M[i, j]
        for i in range(3):
            df["center_{}".format(i)] = center[i]
        df["fnum"] = np.arange(n_frames)
        return df


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
                synchronizer = RecordingVideoDownSynchronizer
            else:
                synchronizer = RecordingVideoUpSynchronizer,
            video_interface.run_synchronizer(
                synchronizer=synchronizer,
                output_directory=self.output_directory,
                synchronize_only=synchronize_only,
                test_mode=test_mode,
                synchro_metadata=self.synchro_metadata,
            )
        self._plot_synchro_crossvalidation()
        if not synchronize_only:
            self.csv_output_filepath = self._create_csv_filepath()
            self.triangulation_dlc_cams_filepaths = {
                video_interface: self.video_interfaces[
                    video_interface
                ].export_for_aniposelib()
                for video_interface in self.video_interfaces
            }

        all_markers = set()
        for file in self.triangulation_dlc_cams_filepaths.values():
            df = pd.read_hdf(file)
            markers = list(df.columns.levels[1])
            all_markers = all_markers.union(markers)
        self.markers = list(all_markers)
        cams_to_exclude = exclude_by_framenum(metadata_from_videos=self.metadata_from_videos,
                                              allowed_num_diverging_frames=self.allowed_num_diverging_frames)
        for cam in self.metadata_from_videos:
            if cam in cams_to_exclude:
                self.triangulation_dlc_cams_filepaths.pop(cam)

    def _plot_synchro_crossvalidation(self)->None:
        template = list(self.video_interfaces.values())[
            0].synchronizer_object.template_blinking_motif.adjust_template_timeseries_to_fps(fps=self.target_fps)[0][0]

        led_timeseries_crossvalidation = {}
        for video_interface in self.video_interfaces.values():
            if video_interface.synchronizer_object.led_timeseries_for_cross_video_validation is not None:
                led_timeseries_crossvalidation[
                    video_interface.video_metadata.cam_id
                ] = video_interface.synchronizer_object.led_timeseries_for_cross_video_validation
        if list(led_timeseries_crossvalidation.keys()):
            filename = f'{self.mouse_id}_{self.recording_date}_{self.paradigm}_synchronization_crossvalidation'
            synchronization_crossvalidation = AlignmentPlotCrossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                filename=filename,
                output_directory=self.output_directory,
            )
            synchronization_crossvalidation.create_plot(save=True, plot=True)

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

        lengthleftside = get_xyz_distance_in_triangulation_space(marker_ids=tuple(config['ReferenceLengthMarkers']), df_xyz=self.df.iloc[best_frame, :])
        conversionfactor = config['ReferenceLengthCm'] / lengthleftside

        bp_keys_unflat = set(get_3D_df_keys(key[:-2]) for key in self.df.keys() if
                             'error' not in key and 'score' not in key and "M_" not in key and 'center' not in key and 'fn' not in key)
        bp_keys = list(it.chain(*bp_keys_unflat))

        normalised = self.df.copy()
        normalised[bp_keys] *= conversionfactor
        reference_rotation_markers = []
        for marker in config['ReferenceRotationMarkers']:
            reference_rotation_markers.append(get_3D_array(normalised, marker, best_frame))
        r, self.rotation_error = Rotation.align_vectors(config["ReferenceRotationCoords"], reference_rotation_markers)

        rotated = normalised.copy()
        for key in bp_keys_unflat:
            rot_points = r.apply(normalised.loc[:, [key[0], key[1], key[2]]])
            for axis in range(3):
                rotated.loc[:, key[axis]] = rot_points[:, axis]
        rotated_markers = []
        for marker in config['ReferenceRotationMarkers']:
            rotated_markers.append(get_3D_array(rotated, marker, best_frame))

        self.normalised_dataframe = True
        self.rotated_filepath = self._create_csv_filepath()
        if (not test_mode) or (test_mode and not self.rotated_filepath.exists()):
            _save_dataframe_as_csv(filepath=str(self.rotated_filepath), df=rotated)
        visualization = RotationVisualization(
            rotated_markers=rotated_markers, config=config,
            output_filepath=self.rotated_filepath,
            rotation_error=self.rotation_error
        )
        visualization.create_plot(plot=False, save=True)

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

        t = TriangulationVisualization(df_3D_filepath=self.rotated_filepath, output_directory=self.output_directory,
                                       idx=idx, config=self.video_plotting_config)
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

    def get_marker_predictions(self, test_mode: bool=False) -> None:
        self.markers_excluded_manually = False
        self.csv_output_filepath = self._create_csv_filepath()
        self.triangulation_dlc_cams_filepaths = {}
        for cam in self.metadata_from_videos.values():
            h5_output_filepath = self.output_directory.joinpath(
                f"Calvin_{self.recording_date}_{cam.cam_id}.h5"
            )
            self.triangulation_dlc_cams_filepaths[cam.cam_id] = h5_output_filepath
            if not test_mode or (test_mode and not h5_output_filepath.exists()):
                if cam.calibration_evaluation_type == "manual":
                    config = cam.calibration_evaluation_filepath
                    manual_interface = ManualAnnotation(
                            object_to_analyse=cam.filepath,
                            output_directory=self.output_directory,
                            marker_detection_directory=config,
                        )
                    manual_interface.analyze_objects(filepath=h5_output_filepath, only_first_frame=True)
                elif cam.calibration_evaluation_type == "DLC":
                    config = cam.calibration_evaluation_filepath
                    dlc_interface = DeeplabcutInterface(
                            object_to_analyse=cam.filepath,
                            output_directory=self.output_directory,
                            marker_detection_directory=config,
                        )
                    dlc_interface.analyze_objects(filepath=h5_output_filepath,
                                                      filtering=False)  # filtering is not supported and not necessary for single frame predictions!
                else:
                    raise ValueError(
                        "For calibration_evaluation only manual and DLC are supported!"
                    )
            self._validate_calibration_validation_marker_ids(
                calibration_validation_markers_df_filepath=h5_output_filepath, framenum=1, defined_marker_ids=self.markers
            )
            predictions = PredictionsPlot(
                image=cam.filepath,
                predictions=h5_output_filepath,
                output_directory=self.output_directory,
                cam_id=cam.cam_id,
            )
            predictions.create_plot(plot=False, save=True)

    def evaluate_triangulation_of_calibration_validation_markers(
            self, show_3D_plot: bool = True, verbose: int = 1
    ) -> None:
        self.anipose_io = add_reprojection_errors_of_all_calibration_validation_markers(
            anipose_io=self.anipose_io, df_xyz=self.df
        )
        self.anipose_io = set_distances_and_angles_for_evaluation(self.ground_truth_config, self.anipose_io, df_xyz=self.df)
        gt_distances = fill_in_distances(self.ground_truth_config["distances"])
        self.anipose_io = add_all_real_distances_errors(
            anipose_io=self.anipose_io, ground_truth_distances=gt_distances, df_xyz=self.df
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
            calibrationvalidation = CalibrationValidationPlot(
                p3d=self.anipose_io["p3ds"][0],
                bodyparts=self.anipose_io["bodyparts"],
                output_directory=self.output_directory,
                marker_ids_to_connect=self.ground_truth_config["marker_ids_to_connect_in_3D_plot"],
                filename_tag="calvin"
            )
            calibrationvalidation.create_plot(plot=True, save=True)

    def _create_csv_filepath(self) -> Path:
        filepath_out = self.output_directory.joinpath(
            f"Calvin_{self.recording_date}_{self.score_threshold}p_excludedmarkers{self.markers_excluded_manually}_filteredFalse_{self.triangulation_type}.csv")
        return filepath_out
