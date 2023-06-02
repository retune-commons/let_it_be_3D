import itertools as it
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, OrderedDict, Any, Set

import aniposelib as ap_lib
import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoClip
from numpy import ndarray
from scipy.spatial.transform import Rotation

from .angles_and_distances import (
    add_reprojection_errors_of_all_calibration_validation_markers,
    set_distances_and_angles_for_evaluation,
    load_distances_from_ground_truth,
    add_errors_between_computed_and_ground_truth_distances_for_different_references,
    add_errors_between_computed_and_ground_truth_angles, get_xyz_distance_in_triangulation_space,
)
from .marker_detection import ManualAnnotation, DeeplabcutInterface
from .plotting import (
    AlignmentPlotCrossvalidation,
    PredictionsPlot,
    CalibrationValidationPlot,
    TriangulationVisualization,
    RotationVisualization,
)
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
    KEYS_TO_CHECK_CAMERA_PROJECT,
    STANDARD_ATTRIBUTES_TRIANGULATION,
    STANDARD_ATTRIBUTES_CALIBRATION,
    SYNCHRO_METADATA_KEYS
)
from .video_interface import VideoInterface
from .video_metadata import VideoMetadata
from .video_synchronization import (
    RecordingVideoDownSynchronizer,
    RecordingVideoUpSynchronizer,
    CharucoVideoSynchronizer,
)


def _get_metadata_from_configs(recording_config_filepath: Path, project_config_filepath: Path) -> \
Tuple[dict, dict]:
    project_config_dict = read_config(path=project_config_filepath)
    recording_config_dict = read_config(path=recording_config_filepath)

    missing_keys_project = check_keys(
        dictionary=project_config_dict, list_of_keys=KEYS_TO_CHECK_PROJECT
    )
    if missing_keys_project:
        raise KeyError(
            f"Missing metadata information in the project_config_file"
            f" {project_config_filepath} for {missing_keys_project}."
        )
    missing_keys_recording = check_keys(
        dictionary=recording_config_dict, list_of_keys=KEYS_TO_CHECK_RECORDING
    )
    if missing_keys_recording:
        raise KeyError(
            f"Missing information for {missing_keys_recording} "
            f"in the config_file {recording_config_filepath}!"
        )

    for dictionary_key in KEYS_TO_CHECK_CAMERA_PROJECT:
        cameras_with_missing_keys = check_keys(
            dictionary=project_config_dict[dictionary_key],
            list_of_keys=project_config_dict["valid_cam_ids"],
        )
        if cameras_with_missing_keys:
            raise KeyError(
                f"Missing information {dictionary_key} for cam {cameras_with_missing_keys} "
                f"in the config_file {project_config_filepath}!"
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
                f"The filenames of the calibration_validation images "
                f"give different metadata! Reasons could be:\n"
                f"  - image belongs to another calibration\n"
                f"  - image filename is valid, but wrong\n"
                f"You should run the filename_checker before to avoid such Errors!"
            )
    return tuple(list(set_of_attribute)[0] for set_of_attribute in sets_of_attributes)


def _exclude_by_framenum(metadata_from_videos: Dict[str, VideoMetadata], allowed_num_diverging_frames: int) -> List[Any]:
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


def _check_output_directory(project_config_filepath: Path,
                            output_directory: Optional[Path] = None) -> Path:
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
        recreate_undistorted_plots: bool = True,
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
            recreate_undistorted_plots=recreate_undistorted_plots,
        )
        metadata_from_videos[video_metadata.cam_id] = video_metadata
    return video_interfaces, metadata_from_videos


def _initialize_camera_group(camera_objects: List) -> ap_lib.cameras.CameraGroup:
    return ap_lib.cameras.CameraGroup(camera_objects)


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


def _get_duplicate_elems_in_list(list1: List[str]) -> List[str]:
    individual_elems, duplicate_elems = [], []
    for elem in list1:
        if elem in individual_elems:
            duplicate_elems.append(elem)
        else:
            individual_elems.append(elem)


def _remove_marker_ids_not_in_ground_truth(
        marker_ids_to_remove: List[str], df: pd.DataFrame()
) -> pd.DataFrame():
    columns_to_remove = [column_name for column_name in df.columns if
                         column_name[1] in marker_ids_to_remove]
    return df.drop(columns=columns_to_remove)


def _save_dataframe_as_csv(filepath: Union[str, Path], df: pd.DataFrame) -> None:
    filepath = convert_to_path(filepath)
    if filepath.exists():
        filepath.unlink()
    df.to_csv(filepath, index=False)
    

def _get_best_frame_for_normalisation(config: Dict, df: pd.DataFrame) -> int:
    all_normalization_markers = [config['CENTER']]
    for marker in config["REFERENCE_LENGTH_MARKERS"]:
        all_normalization_markers.append(marker)
    for marker in config["REFERENCE_ROTATION_MARKERS"]:
        all_normalization_markers.append(marker)
    if "EQUAL_LENGTHS_GROUND_TRUTH" in config:
        for equal_length in config["EQUAL_LENGTHS_GROUND_TRUTH"]:
            for length in equal_length:
                for marker in length:
                    all_normalization_markers.append(marker)
    all_normalization_markers = set(all_normalization_markers)
    normalization_keys_nested = [get_3D_df_keys(marker) for marker in all_normalization_markers]
    normalization_keys = list(set(it.chain(*normalization_keys_nested)))
    df_normalization_keys = df.loc[:, normalization_keys]
    valid_frames_for_normalization = list(df_normalization_keys.dropna(axis=0).index)

    if valid_frames_for_normalization:
        if "EQUAL_LENGTHS_GROUND_TRUTH" in config:
            list_of_ground_truths = []
            for equal_length in config["EQUAL_LENGTHS_GROUND_TRUTH"]:
                length_a, length_b = equal_length[0], equal_length[1]
                a = get_xyz_distance_in_triangulation_space((length_a[0], length_a[1]), df)
                b = get_xyz_distance_in_triangulation_space((length_b[0], length_b[1]), df)
                list_of_ground_truths.append(abs((a-b))/abs((a+b)))
            result = sum(list_of_ground_truths)
            best_frame = np.argmin(result)
            return best_frame
        else:
            return valid_frames_for_normalization[0]
    else:
        raise ValueError("Could not normalize the dataframe!")


class Calibration:
    """
    A class, in which videos are calibrated to each other.

    Temporal synchronization of the videos can be performed based on a pattern.
    Spatial calibration is performed using aniposelib (ap_lib) with additional
    methods to validate the calibration based on known ground_truth.

    Parameters
    __________
    calibration_directory: Path or string
        Directory, where the calibration videos are stored.
    project_config_filepath: Path or string
        Filepath to the project_config .yaml file.
    recording_config_filepath: Path or string
        Filepath to the recording_config .yaml file.
    output_directory: Path or string, optional
        Directory, in which the files created during the analysis are saved.
        Per default it will be set the same as the calibration_directory.
    recreate_undistorted_plots: bool, default True
        If True (default), then preexisting undistorted plots will be overwritten.

    Attributes
    __________
    project_config_filepath: Path
        Filepath to the project_config .yaml file.
    output_directory: Path
        Directory, in which the files created during the analysis are saved.
    camera_group: ap_lib.cameras.CameraGroup
        Group of ap_lib.cameras.Camera objects.
    camera_objects:
        List of ap_lib.cameras.Camera objects.
    synchronized_charuco_videofiles: {str: Path}
        Dict of synchronized calibration video per camera.
    video_interfaces: {str: VideoInterface}
        Dict of VideoInterface objects for all calibration videos.
    metadata_from_videos: {str: VideoMetadata}
        Dict of VideoMetadata objects for all calibration videos.
    valid_videos: list of str
        Videos, that were found in the calibration_directory and
        not excluded due to synchronization issues.
    cams_to_exclude: list of str
        Videos, that were excluded due to synchronization issues.
    recording_date: str
        Date at which the calibration was done.
    target_fps: int
        Fps rate, to which the videos should be synchronized.
    led_pattern: dict
        Blinking pattern to use for temporal synchronisation.
    calibration_index: int
        Index of a calibration.
        Together with recording_date, it creates a unique calibration key.
    calibration_tag: str
        Filename tag to search for in the files in calibration_directory.
    reprojerr: float
        Reprojection error (px) returned by ap_lib calibration.
    report_filepath: Path
        Filepath to the report .csv for calibration optimisation.
    allowed_num_diverging_frames: int
        Difference of framenumber to the framenumber median of all synchronised videos,
        that is allowed before a video has to be excluded.
    synchro_metadata: dict
        Dictionary used as input for synchronizer objects.
    synchronization_individuals: list of AlignmentPlotIndividual
        Container for the AlignmentPlotIndividual objects of each synchronized video.
    led_detection_individuals: list of LEDMarkerPlot
        Container for the LEDMarkerPlot objects of each synchronized video.

    Methods
    _______
    run_synchronization(overwrite_synchronisations, verbose)
        Perform synchronization of all videos to the led_pattern and
        downsampling to target_fps.
    run_calibration(use_own_intrinsic_calibration, charuco_calibration_board, iteration, verbose, overwrite_calibrations)
        Call ap_lib calibrate function.
    calibrate_optimal(calibration_validation, max_iters, p_threshold, angle_threshold, verbose, overwrite_calibrations)
        Call run_calibration repeatedly and validate the quality of the
        resulting calibration on calibration_validation images and ground_truth.

    References
    __________
    [1] Karashchuk, P., Rupp, K. L., Dickinson, E. S., et al. (2021).
    Anipose: A toolkit for robust markerless 3D pose estimation.
    Cell reports, 36(13), 109730. https://doi.org/10.1016/j.celrep.2021.109730

    See Also
    ________
    TriangulationRecordings:
        A class, in which videos are triangulated based on a calibration file.
    CalibrationValidation:
        A class, in which images are triangulated based on a calibration file
        and the triangulated coordinates are validated based on a ground_truth.
    core.checker_objects.CheckCalibration:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    core.meta.MetaInterface.create_calibrations:
        Create Calibration objects for all calibration_directories added to MetaInterface.
    core.meta.MetaInterface.synchronize_calibrations:
        Run the function run_synchronization for all calibration objects added
        to MetaInterface.
    core.meta.MetaInterface.calibrate:
        Run the function run_calibration or calibrate_optimal for all calibration
        objects added to MetaInterface.

    Examples
    ________
    >>> from pathlib import Path
    >>> from core.triangulation_calibration_module import Calibration
    >>> rec_config = Path(
    ... "test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml"
    ... )
    >>> calibration_object = Calibration(
    ... calibration_directory=rec_config.parent,
    ... recording_config_filepath=rec_config,
    ... project_config_filepath="test_data/project_config.yaml",
    ... output_directory=rec_config.parent,
    ... )
    >>> calibration_object.run_synchronization()
    >>> calibration_object.run_calibration(verbose=2)
    """
    def __init__(
            self,
            calibration_directory: Union[Path, str],
            project_config_filepath: Union[Path, str],
            recording_config_filepath: Union[Path, str],
            output_directory: Optional[Union[Path, str]] = None,
            recreate_undistorted_plots: bool = True,
    ) -> None:
        """
        Construct all necessary attributes for the Calibration class.

        Read the metadata from project-/recording config and from video filenames.
        Create representations of the videos inside the given calibration_directory.

        Parameters
        ----------
        calibration_directory: Path or string
            Directory, where the calibration videos are stored.
        project_config_filepath: Path or string
            Filepath to the project_config .yaml file.
        recording_config_filepath: Path or string
            Filepath to the recording_config .yaml file.
        output_directory: Path or string, optional
            Directory, in which the files created during the analysis are saved.
            Per default it will be set the same as the calibration_directory.
        recreate_undistorted_plots: bool, default True
            If True (default), then preexisting undistorted plots will be overwritten.
        """
        for attribute in STANDARD_ATTRIBUTES_CALIBRATION:
            setattr(self, attribute, None)
        self.calibration_directory = convert_to_path(calibration_directory)
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
        for attribute in ["calibration_tag", "allowed_num_diverging_frames"]:
            setattr(self, attribute, project_config_dict[attribute])
        for attribute in ["recording_date", "led_pattern", "calibration_index", "target_fps"]:
            setattr(self, attribute, recording_config_dict[attribute])
        self.recording_date = str(self.recording_date)
        self.calibration_index = str(self.calibration_index)

        self.video_interfaces, self.metadata_from_videos = _create_video_objects(
            directory=self.calibration_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="calibration",
            output_directory=self.output_directory,
            filename_tag=self.calibration_tag,
            filetypes=[".AVI", ".avi", ".mov", ".mp4"],
            recreate_undistorted_plots=recreate_undistorted_plots,
        )
        self.recording_date, *_ = _validate_metadata(metadata_from_videos=self.metadata_from_videos,
                                                     attributes_to_check=['recording_date'])
        self.target_fps = min(
            [video_metadata.fps for video_metadata in self.metadata_from_videos.values()])
        # limits target_fps to fps of the slowest video
        for video_metadata in self.metadata_from_videos.values():
            video_metadata.target_fps = self.target_fps

    def run_synchronization(self, overwrite_synchronisations: bool = False, verbose: bool = True) -> None:
        """
        Perform synchronization of all videos to the led_pattern and
        downsampling to target_fps.

        Call the synchronizer via VideoInterface and save the
        synchronized_video_filepaths.
        Create a plot for crossvalidation of the synchronised LED timeseries.
        Exclude videos, if there are any duplicates in camera names or
        diverging framenumbers after synchronization.

        Parameters
        ----------
        overwrite_synchronisations: bool, default False
            If True (default False), then pre-existing synchronisations will be
            overwritten during analysis.
        verbose: bool, default True
            If True (default), then Crossvalidation plot and synchronised number
            of frames for each camera are printed.
        """
        self.synchronized_charuco_videofiles = {}
        camera_objects_unexcluded = []
        for video_interface in self.video_interfaces.values():
            video_interface.run_synchronizer(
                synchronizer=CharucoVideoSynchronizer,
                output_directory=self.output_directory,
                synchronize_only=True,
                overwrite_DLC_analysis_and_synchro=overwrite_synchronisations,
                synchro_metadata=self.synchro_metadata,
                verbose=verbose
            )
            self.synchronized_charuco_videofiles[
                video_interface.video_metadata.cam_id
            ] = str(video_interface.synchronized_video_filepath)
            camera_objects_unexcluded.append(video_interface.export_for_aniposelib())
        camera_objects_unexcluded.sort(key=lambda x: x.name, reverse=False)
        self._plot_synchro_crossvalidation(verbose=verbose)
        
        cameras = [camera_object.name for camera_object in camera_objects_unexcluded]
        duplicate_cams = _get_duplicate_elems_in_list(cameras)
        if duplicate_cams:
            raise ValueError(
                f"You added multiple cameras with the cam_id {duplicate_cams}, "
                "however, all cam_ids must be unique! Please check for duplicates "
                "in the calibration directory and rename them!"
            )

        self.cams_to_exclude = _exclude_by_framenum(metadata_from_videos=self.metadata_from_videos,
                                               allowed_num_diverging_frames=self.allowed_num_diverging_frames)
        self.valid_videos = [cam.name for cam in camera_objects_unexcluded if
                             cam.name not in self.cams_to_exclude]
        self.camera_objects = [cam for cam in camera_objects_unexcluded if cam.name not in self.cams_to_exclude]
        for cam in self.cams_to_exclude:
            self.synchronized_charuco_videofiles.pop(cam)
        self.camera_group = _initialize_camera_group(camera_objects=self.camera_objects)

    def run_calibration(
            self,
            use_own_intrinsic_calibration: bool = True,
            verbose: int = 0,
            charuco_calibration_board: Optional[ap_lib.boards.CharucoBoard] = None,
            overwrite_calibrations: bool = True,
            iteration: Optional[int] = None,
    ) -> Path:
        """
        Call ap_lib calibrate function.

        Create a filename for the calibration file.
        Pass videos to ap_lib.cameras.camera_group.calibrate_videos function.

        Parameters
        ----------
        use_own_intrinsic_calibration: bool, default True
            If True (default), then the externally created intrinsic calibrations
            are passed to the calibrate function. Otherwise, the ap_lib built-in
            intrinsic calibration is used.
        verbose: int, default 0
            Show ap_lib output if > 1 or no output if <= 1.
        charuco_calibration_board: ap_lib.boards.CharucoBoard, optional
            Specify the board, that was used in the calibration videos.
        overwrite_calibrations: bool, default True
            If True (default), then pre-existing calibrations will be overwritten.
        iteration: int, optional
            Variable to be included into the filename to make the
            filepath of calibration files unique for repeated calibrations.

        Returns
        -------
        calibration_filepath: Path

        """
        calibration_key = create_calibration_key(videos=self.valid_videos,
                                                 recording_date=self.recording_date,
                                                 calibration_index=self.calibration_index,
                                                 iteration=iteration)
        filename = f"{calibration_key}.toml"

        calibration_filepath = self.output_directory.joinpath(filename)
        if (overwrite_calibrations) or (not
                overwrite_calibrations and not calibration_filepath.exists()
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
            self._save_calibration(calibration_filepath=calibration_filepath,
                                   camera_group=self.camera_group)
        else:
            self.reprojerr = 0
        return calibration_filepath

    def _save_calibration(self, calibration_filepath: Path,
                          camera_group: ap_lib.cameras.CameraGroup) -> None:
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
            overwrite_calibrations: bool = True,
    ):
        """
        Call run_calibration repeatedly and validates the quality of the
        resulting calibration on calibration_validation images and ground_truth.

        Repeat calibration until a good calibration is reached or max_iters is
        superceded.
        Check whether the triangulated data in calibration_validation matches
        the ground_truth data p_threshold and angle_threshold.
        Create a report file in which the metadata to the calibration of each
        iteration is specified.

        Parameters
        ----------
        calibration_validation: CalibrationValidation
            Object, containing images, triangulated data and ground_truth
            information for calibration validation.
        max_iters: int, default 5
            Number of iterations allowed to find a good calibration.
        p_threshold: float, default 0.1
            Threshold for errors in the triangulated distances compared to
            ground truth (mean distances in percent).
        angle_threshold: float, default 5
            Threshold for errors in the triangulated angles compared to ground
            truth (mean angles in degrees).
        verbose: int, default 1
            Show ap_lib output if > 1,
            calibration_validation output if > 0
            or no output if < 1.
        overwrite_calibrations: bool, default True
            If True (default), then pre-existing calibrations will be overwritten.

        Returns
        -------
        calibration_filepath: Path
            The filepath to the optimal calibration of if no good calibration
            was reached during iteration, the filepath of the last calibration.
        """
        report = pd.DataFrame()
        calibration_found = False
        calibration_key = create_calibration_key(videos=self.valid_videos,
                                                 recording_date=self.recording_date,
                                                 calibration_index=self.calibration_index)
        good_calibration_filepath = self.output_directory.joinpath(f"{calibration_key}.toml")
        calibration_filepath = None
        for cal in range(max_iters):
            if good_calibration_filepath.exists() and not overwrite_calibrations:
                calibration_filepath = good_calibration_filepath
                self.reprojerr = 0
            else:
                calibration_filepath = self.run_calibration(verbose=verbose, overwrite_calibrations=overwrite_calibrations,
                                                            iteration=cal)

            calibration_validation.run_triangulation(calibration_toml_filepath=calibration_filepath)
            mean_dist_err_percentage, mean_angle_err, reprojerr_nonan_mean = \
                calibration_validation.evaluate_triangulation_of_calibration_validation_markers(verbose=bool(verbose > 2), show_3D_plot=bool(verbose))

            if verbose:
                print(
                    f"Calibration {cal}\n mean percentage error: {mean_dist_err_percentage}\n "
                    f"mean angle error: {mean_angle_err}\n "
                    f"ap_lib reprojection error: {self.reprojerr}\n "
                    f'calvin mean reprojection error: {calibration_validation.anipose_io["reproj_nonan"].mean()}')
                
            report.loc[cal, "key"] = str(calibration_filepath)
            report.loc[cal, "num_cams"] = len(self.camera_objects)
            report.loc[cal, "cams_to_exclude"] = str(self.cams_to_exclude)
            report.loc[cal, "mean_distance_error_percentage"] = mean_dist_err_percentage
            report.loc[cal, "mean_angle_error"] = mean_angle_err
            report.loc[cal, "mean_reprojerror_calvin"] = reprojerr_nonan_mean
            report.loc[cal, "ap_lib_reprojerr"] = self.reprojerr

            if (mean_dist_err_percentage < p_threshold and mean_angle_err < angle_threshold):
                calibration_found = True
                calibration_filepath.rename(good_calibration_filepath)
                calibration_filepath = good_calibration_filepath
                if verbose:
                    print(
                        f"Good Calibration reached at iteration {cal}!\n"
                        f"Named it {good_calibration_filepath}.")
                break

        self.report_filepath = self.output_directory.joinpath(
                f"{self.recording_date}_calibration_report.csv")
        if overwrite_calibrations or not self.report_filepath.exists():
            report.to_csv(self.report_filepath, index=False)

        if not calibration_found:
            if verbose > 0:
                print(
                    "No optimal calibration found with given thresholds! Returned last executed calibration!")
        return calibration_filepath

    def _plot_synchro_crossvalidation(self, verbose: bool = True) -> None:
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
            filename = f'{self.recording_date}_charuco_synchronization_crossvalidation_{self.target_fps}fps'
            synchronization_crossvalidation = AlignmentPlotCrossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                output_directory=self.output_directory,
                filename=filename,
            )
            synchronization_crossvalidation.create_plot(save=True, plot=verbose)


class Triangulation(ABC):
    """
    Parent class, for triangulation of videos or images.

    Triangulation is performed using aniposelib (ap_lib).

    Parameters
    ----------
    project_config_filepath: Path or string
        Filepath to the project_config .yaml file.
    directory: Path or string
        Directory, where the videos or images are stored.
    recording_config_filepath: Path or string
        Filepath to the recording_config .yaml file.
    recreate_undistorted_plots: bool, default True
        If True (default), then preexisting undistorted plots will be overwritten.
    output_directory: Path or string, optional
        Directory, in which the files created during the analysis are saved.
        Per default it will be set the same as the directory.

    Attributes
    __________
    project_config_filepath: Path
        Filepath to the project_config .yaml file.
    output_directory: Path
        Directory, in which the files created during the analysis are saved.
    video_interfaces: {str: VideoInterface}
        Dict of VideoInterface objects for all videos or images.
    metadata_from_videos: {str: VideoMetadata}
        Dict of VideoMetadata objects for all videos or images.
    triangulation_dlc_cams_filepaths: {str: Path}
        Containing the filepath to the predictions for each camera.
    csv_output_filepath: Path
        Filepath, were the triangulated dataframe should be saved.
    recording_date: str
        Date at which the calibration was done based on recording_config and as
        read from the filenames.
    calibration_index: int
        Index of a calibration.
        Together with recording_date, it creates a unique calibration key.
    target_fps: int
        Fps rate, to which the videos should be synchronized.
    led_pattern: dict
        Blinking pattern to use for temporal synchronisation.
    mouse_id: str
        Only defined in TriangulationRecordings. The mouse_id as read
        from the filenames.
    paradigm: str
        Only defined in TriangulationRecordings. The paradigm as read
        from the filenames.
    cams_to_exclude: list of str
        Videos, that were excluded due to synchronization issues.
    all_cameras: list of str
        All camera names, that are stored in the camera_group.
    self.camera_group: ap_lib.cameras.CameraGroup
        Group of cameras loaded from calibration_toml_filepath.
    anipose_io: dict
        Containing information for ap_lib functions, such as points_flat, p3ds,
        n_joints, reprojerr, as well as distances and angles to validate
        calibration in comparison with ground truth.
    markers: list of str
        All markers that will be triangulated.
    normalised_dataframe: bool, default False
        If True (default False), then the dataframe was normalised based on
        input from normalisation config.
    markers_excluded_manually: bool, default False
        If True (default False), then markers were excluded from predictions
        based on marker exclusion config.
    rotated_filepath: Path
        Filepath, were the rotated triangulated dataframe is saved.
    video_plotting_config: dict
        Containing information from video_plotting config to create 3D videos.
    synchronization_individuals: list of AlignmentPlotIndividual
        Only defined in TriangulationRecordings. Container for the AlignmentPlotIndividual objects of each synchronized video.
    led_detection_individuals: list of LEDMarkerPlot
        Only defined in TriangulationRecordings. Container for the LEDMarkerPlot objects of each synchronized video.
    synchro_metadata: dict
        Only used in TriangulationRecordings. Dictionary used as input for synchronizer objects.
    allowed_num_diverging_frames: int
        Only used in TriangulationRecordings. Difference of framenumber to the framenumber median of all synchronised videos,
        that is allowed before a video has to be excluded.
    ground_truth_config: dict
        Only defined in CalibrationValidation to compare the triangulated data
        to ground truth data.
    calibration_validation_tag: str
        Only used in CalibrationValidation. Filename tag to search for in the
        filenames in directory.
    _allowed_filetypes: list of str
        Abstract property, specify what file endings to look for in directory.
    triangulation_visualization: TriangulationVisualization
        Object to create plots for triangulation video.
    video_start_s: int
        The second, in which the triangulation video starts.

    Methods
    _______
    run_triangulation(calibration_toml_filepath, triangulate_full_recording):
        Load and validate the calibration, triangulate and create 3D df.
    exclude_marker(all_markers_to_exclude_config_path, verbose):
        Exclude markers in prediction based on markers_to_exclude config.
    _get_dataframe_of_triangulated_points(anipose_io):
        Combine the triangulated data from anipose_io to a 3D df.

    References
    __________
    [1] Karashchuk, P., Rupp, K. L., Dickinson, E. S., et al. (2021).
    Anipose: A toolkit for robust markerless 3D pose estimation.
    Cell reports, 36(13), 109730. https://doi.org/10.1016/j.celrep.2021.109730

    See Also
    ________
    TriangulationRecordings:
        Subclass of Triangulation, in which videos are triangulated based on a
        calibration file.
    CalibrationValidation:
        Subclass of Triangulation, in which images are triangulated based on a
        calibration file and the triangulated coordinates are validated based on
        a ground_truth.
    Calibration:
        A class, in which videos are calibrated to each other.
    core.meta.MetaInterface.triangulate_recordings:
        Run the function run_triangulation for all TriangulationRecording
        objects added to MetaInterface.
    Calibration.triangulate_optim:
        Run the function run_triangulation for the CalibrationValidation
        object passed to triangulate_optim.
    """
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
                 project_config_filepath: Union[Path, str],
                 directory: Union[Path, str],
                 recording_config_filepath: Union[Path, str],
                 recreate_undistorted_plots: bool = True,
                 output_directory: Optional[Union[Path, str]] = None):
        """
        Construct all necessary attributes for the Triangulation Class.

        Read the metadata from project-/recording config and from video filenames.
        Create csv_output_filepath based on the metadata.
        Create representations of the videos inside the given directory.

        Parameters
        ----------
        project_config_filepath: Path or string
            Filepath to the project_config .yaml file.
        directory: Path or string
            Directory, where the videos or images are stored.
        recording_config_filepath: Path or string
            Filepath to the recording_config .yaml file.
        recreate_undistorted_plots: bool, default True
            If True (default), then preexisting undistorted plots will be overwritten.
        output_directory: Path or string, optional
            Directory, in which the files created during the analysis are saved.
            Per default it will be set the same as the directory.
        """
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
        for attribute in ["use_gpu", "calibration_validation_tag",
                          "score_threshold", "triangulation_type", "allowed_num_diverging_frames"]:
            setattr(self, attribute, project_config_dict[attribute])
        for attribute in ["recording_date", "led_pattern", "target_fps", "calibration_index"]:
            setattr(self, attribute, recording_config_dict[attribute])
        self.recording_date = str(self.recording_date)
        self.calibration_index = str(self.calibration_index)

        self.video_interfaces, self.metadata_from_videos = _create_video_objects(
            directory=self.directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag=self._videometadata_tag,
            output_directory=self.output_directory,
            filename_tag=self.calibration_validation_tag if self._videometadata_tag == "calvin" else "",
            recreate_undistorted_plots=recreate_undistorted_plots,
            filetypes=self._allowed_filetypes,
        )
        metadata = _validate_metadata(metadata_from_videos=self.metadata_from_videos,
                                      attributes_to_check=self._metadata_keys)
        for attribute, value in zip(self._metadata_keys, metadata):
            setattr(self, attribute, value)

    def run_triangulation(
            self,
            calibration_toml_filepath: Union[Path, str],
            triangulate_full_recording: bool = True,
            use_preexisting_csvs: bool = False
    ) -> None:
        """
        Load and validate the calibration, triangulate and create 3D df.

        Validate, that the camera names in camera_group match
        triangulation_dlc_cams_filepaths and drop or add empty files if they don't.
        Triangulate using different options as defined in the project_config via
        ap_lib functions.

        Parameters
        ----------
        calibration_toml_filepath: Path or str
            Filepath to the calibration, that should be used for triangulation.
        triangulate_full_recording: bool, default True
            If False (default True), then only the first 2 frames of the
            recording will be triangulated and the 3D dataframe won't be saved.
        use_preexisting_csvs: bool, default False
            If True (default False), then a already existing file at csv_output_filepath
            will be read in and no triangulatin will be performed.
        """
        self.csv_output_filepath = self._create_csv_filepath()
        calibration_toml_filepath = convert_to_path(calibration_toml_filepath)
        self.camera_group = self._load_calibration(filepath=calibration_toml_filepath)

        filepath_keys = list(self.triangulation_dlc_cams_filepaths.keys())
        filepath_keys.sort()
        self.all_cameras = [camera.name for camera in self.camera_group.cameras]
        self.all_cameras.sort()
        missing_cams_in_all_cameras = _find_non_matching_list_elements(filepath_keys,
                                                                       self.all_cameras)
        missing_cams_in_filepath_keys = _find_non_matching_list_elements(self.all_cameras, 
                                                                         filepath_keys)
        if len(self.all_cameras)-len(missing_cams_in_filepath_keys) < 2:
                print("All videos had to be excluded!")
                raise IndexError("The exclusion state for all cameras is 'exclude'. "
                                 "At least two cameras are necessary to perform the triangulation!")
        if missing_cams_in_filepath_keys:
            min_framenum = min([pd.read_hdf(path).shape[0] for path in
                                self.triangulation_dlc_cams_filepaths.values()])
            self._create_empty_files(cams_to_create_empty_files=missing_cams_in_filepath_keys,
                                     framenum=min_framenum,
                                     markers=self.markers)
        for cam in missing_cams_in_all_cameras:
            self.triangulation_dlc_cams_filepaths.pop(cam)
            self.cams_to_exclude.append(cam)
        if use_preexisting_csvs and self.csv_output_filepath.exists():
            self.df = pd.read_csv(self.csv_output_filepath)
            self.anipose_io = {"reprojerr": np.array([0]), "reproj_nonan": np.array([0]), "reprojerr_flat": np.array([0])}
            print(f"Found a file at {self.csv_output_filepath}!\n"
                  "No triangulation will be performed.")
        else:
            self.anipose_io = self._preprocess_dlc_predictions_for_anipose(triangulate_full_recording=triangulate_full_recording)
            if self.triangulation_type == "triangulate":
                p3ds_flat = self.camera_group.triangulate(self.anipose_io["points_flat"], progress=True)
            elif self.triangulation_type == "triangulate_optim_ransac_False":
                p3ds_flat = self.camera_group.triangulate_optim(self.anipose_io["points"],
                                                                init_ransac=False,
                                                                init_progress=True).reshape(
                    self.anipose_io["n_points"] * self.anipose_io["n_joints"], 3)
            elif self.triangulation_type == "triangulate_optim_ransac_True":
                p3ds_flat = self.camera_group.triangulate_optim(self.anipose_io["points"],
                                                                init_ransac=True,
                                                                init_progress=True).reshape(
                    self.anipose_io["n_points"] * self.anipose_io["n_joints"], 3)
            else:
                raise ValueError(
                    "Supported methods for triangulation are triangulate, "
                    "triangulate_optim_ransac_True, triangulate_optim_ransac_False!")
            self.anipose_io["p3ds"] = p3ds_flat.reshape(self.anipose_io["n_points"],
                                                        self.anipose_io["n_joints"], 3)

            self.anipose_io["reprojerr"], self.anipose_io["reproj_nonan"], self.anipose_io[
                "reprojerr_flat"] = self._get_reprojection_errors(
                p3ds_flat=p3ds_flat)

            self.df = self._get_dataframe_of_triangulated_points(anipose_io=self.anipose_io)
            if (triangulate_full_recording) or (not self.csv_output_filepath.exists()):
                _save_dataframe_as_csv(filepath=self.csv_output_filepath, df=self.df)
        self._delete_temp_files()

    def _delete_temp_files(self) -> None:
        cams_to_delete = []
        for cam in self.triangulation_dlc_cams_filepaths:
            path = self.triangulation_dlc_cams_filepaths[cam]
            if "_temp" in path.name:
                path.unlink()
                cams_to_delete.append(cam)
        for cam in cams_to_delete:
            self.triangulation_dlc_cams_filepaths.pop(cam)

    def exclude_markers(self, all_markers_to_exclude_config_path: Union[Path, str], verbose: bool = True):
        """
        Exclude markers in prediction based on markers_to_exclude config.

        Parameters
        ----------
        all_markers_to_exclude_config_path: Path or str
            Filepath to the config used for exclusion of markers.
        verbose: bool, default True
            If True (default), print if exclusion of markers worked without any
            abnormalities.

        Notes
        _____
        The all_markers_to_exclude_config_path has to be a path to a yaml file
        representing a dictionary with the camera names as keys and a list of
        the markers, that should be excluded as value.
        """
        all_markers_to_exclude = read_config(all_markers_to_exclude_config_path)
        missing_cams = check_keys(all_markers_to_exclude,
                                  list(self.triangulation_dlc_cams_filepaths))
        if missing_cams:
            if verbose:
                print(
                    f"Found no markers to exclude for {missing_cams} in {str(all_markers_to_exclude_config_path)}!")

        for cam_id in self.triangulation_dlc_cams_filepaths:
            h5_file = self.triangulation_dlc_cams_filepaths[cam_id]
            df = pd.read_hdf(h5_file)
            markers = set(b for a, b, c in df.keys())
            markers_to_exclude_per_cam = all_markers_to_exclude[cam_id]
            existing_markers_to_exclude = list(set(markers) & set(markers_to_exclude_per_cam))
            not_existing_markers = [marker for marker in markers_to_exclude_per_cam if
                                    marker not in markers]
            if not_existing_markers:
                if verbose:
                    print(
                        f"The following markers were not found in the dataframe, "
                        f"but were given as markers to exclude for {cam_id}: {not_existing_markers}!")
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
                f"empty_temp_{cam}.h5"
            )
            cols = get_multi_index(markers)
            df = pd.DataFrame(data=np.zeros((framenum, len(cols))), columns=cols, dtype=int)
            df.to_hdf(str(h5_output_filepath), "empty")
            self._validate_triangulation_marker_ids(
                triangulation_markers_df_filepath=h5_output_filepath, framenum=framenum,
                defined_marker_ids=markers)
            self.triangulation_dlc_cams_filepaths[cam] = h5_output_filepath

    def _validate_triangulation_marker_ids(self, triangulation_markers_df_filepath: Path,
                                           framenum: int, defined_marker_ids: List[str],
                                           add_missing_marker_ids_with_0_likelihood: bool = True) -> None:
        triangulation_markers_df = pd.read_hdf(triangulation_markers_df_filepath)
        prediction_marker_ids = list(
            set([marker_id for scorer, marker_id, key in
                 triangulation_markers_df.columns]))
        marker_ids_not_in_ground_truth = _find_non_matching_list_elements(prediction_marker_ids,
                                                                          defined_marker_ids)
        marker_ids_not_in_prediction = _find_non_matching_list_elements(defined_marker_ids,
                                                                        prediction_marker_ids)
        if add_missing_marker_ids_with_0_likelihood & bool(marker_ids_not_in_prediction):
            triangulation_markers_df = _add_missing_marker_ids_to_prediction(
                missing_marker_ids=marker_ids_not_in_prediction,
                df=triangulation_markers_df,
                framenum=framenum,
            )
            print(
                "The following marker_ids were missing and added to the dataframe with a "
                f"likelihood of 0: {marker_ids_not_in_prediction}."
            )
        if marker_ids_not_in_ground_truth:
            triangulation_markers_df = _remove_marker_ids_not_in_ground_truth(
                marker_ids_to_remove=marker_ids_not_in_ground_truth,
                df=triangulation_markers_df,
            )
            print(
                "The following marker_ids were deleted from the dataframe, since they were "
                f"not present in the ground truth: {marker_ids_not_in_ground_truth}."
            )
        triangulation_markers_df.to_hdf(
            triangulation_markers_df_filepath, "empty", mode="w"
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

    def _preprocess_dlc_predictions_for_anipose(self, triangulate_full_recording: bool = True) -> Dict:
        anipose_io = ap_lib.utils.load_pose2d_fnames(
            fname_dict=self.triangulation_dlc_cams_filepaths
        )
        return self._add_additional_information_and_continue_preprocessing(anipose_io=anipose_io,
                                                                           triangulate_full_recording=triangulate_full_recording)

    def _add_additional_information_and_continue_preprocessing(
            self, anipose_io: Dict, triangulate_full_recording: bool = True
    ) -> Dict:
        n_cams, anipose_io["n_points"], anipose_io["n_joints"], _ = anipose_io["points"].shape
        if not triangulate_full_recording:
            start_idx, end_idx = 0, 2
            anipose_io["points"] = anipose_io["points"][:, start_idx:end_idx, :, :]
            anipose_io["n_points"] = (end_idx - start_idx) if end_idx < anipose_io['n_points'] else anipose_io['n_points']
            anipose_io["scores"] = anipose_io["scores"][:, start_idx:end_idx, :]
        anipose_io["points"][anipose_io["scores"] < self.score_threshold] = np.nan

        anipose_io["points_flat"] = anipose_io["points"].reshape(n_cams, -1, 2)
        anipose_io["scores_flat"] = anipose_io["scores"].reshape(n_cams, -1)
        return anipose_io

    def _get_reprojection_errors(
            self, p3ds_flat: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        reprojerr_flat = self.camera_group.reprojection_error(p3ds_flat,
                                                              self.anipose_io["points_flat"],
                                                              mean=True)
        reprojerr = reprojerr_flat.reshape(self.anipose_io["n_points"], self.anipose_io["n_joints"])
        reprojerr_nonan = reprojerr[np.logical_not(np.isnan(reprojerr))]
        return reprojerr, reprojerr_nonan, reprojerr_flat

    def _get_dataframe_of_triangulated_points(self, anipose_io: Dict) -> pd.DataFrame:
        """
        The following function was taken from
        https://github.com/lambdaloop/anipose/blob/d20091550dc8b901f460f914544ecfc66c116329/anipose/triangulate.py.
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
        #M = np.identity(3)
        #center = np.zeros(3)
        df = pd.DataFrame()
        for bp_num, bp in enumerate(anipose_io["bodyparts"]):
            for ax_num, axis in enumerate(["x", "y", "z"]):
                df[bp + "_" + axis] = all_points_3d_adj[:, bp_num, ax_num]
            df[bp + "_error"] = anipose_io['reprojerr'][:, bp_num]
            df[bp + "_score"] = scores_3d[:, bp_num]
        #for i in range(3):
        #    for j in range(3):
        #        df["M_{}{}".format(i, j)] = M[i, j]
        #for i in range(3):
        #    df["center_{}".format(i)] = center[i]
        df["fnum"] = np.arange(n_frames)
        return df


class TriangulationRecordings(Triangulation):
    """
    Subclass of Triangulation, in which videos are triangulated based on a
    calibration file.

    Temporal synchronization of the videos can be performed based on a pattern.
    The triangulated dataframe can be normalised (rotated and translated).
    For visalization, a triangulated video can be created.

    Methods
    _______
    run_synchronization(overwrite_DLC_analysis_and_synchro, verbose):
        Perform analysis of all videos using DLC or other methods,
        synchronization to the led_pattern and downsampling to target_fps.
    create_triangulated_video(filename, config_path):
        Create video of the triangulated data.
    normalize(normalization_config_path, save_dataframe):
        Rotate and translate the triangulated dataframe.

    See Also
    ________
    Triangulation:
        Parent class, for triangulation of videos or images.
    core.meta.MetaInterface.create_recordings:
        Create TriangulationRecording objects for all recording_directories
        added to MetaInterface.
    core.meta.MetaInterface.synchronize_recordings:
        Run the function run_synchronization for all TriangulationRecording
        objects added to MetaInterface.
    core.meta.MetaInterface.triangulate_recordings:
        Run the function run_triangulation for all TriangulationRecording
        objects added to MetaInterface.
    core.checker_objects.CheckRecording:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.

    Examples
    ________
    >>> from core.triangulation_calibration_module import TriangulationRecordings
    >>> rec_config = "test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml"
    >>> directory = "test_data/Server_structure/VGlut2-flp/September2022/206_F2-63/220922_OTE/"
    >>> triangulation_object = TriangulationRecordings(
        ... directory=directory,
        ... recording_config_filepath=rec_config,
        ... project_config_filepath="test_data/project_config.yaml",
        ... recreate_undistorted_plots = True,
        ... output_directory=directory
        ... )
    >>> triangulation_object.run_synchronization()
    >>> triangulation_object.exclude_markers(
        ... all_markers_to_exclude_config_path="test_data/markers_to_exclude_config.yaml",
        ... verbose=False,
        ... )
    >>> triangulation_object.run_triangulation(
        ... calibration_toml_filepath="test_data/Server_structure/Calibrations/220922/220922_0_Bottom_Ground1_Ground2_Side1_Side2_Side3.toml"
        ... )
    >>> normalised_path, normalisation_error = triangulation_object.normalize(
        ... normalization_config_path="test_data/normalization_config.yaml"
        ... )
    """
    @property
    def _metadata_keys(self) -> List[str]:
        return ["recording_date", "paradigm", "mouse_id"]

    @property
    def _videometadata_tag(self) -> str:
        return "recording"

    @property
    def _allowed_filetypes(self) -> List[str]:
        return [".AVI", ".avi", ".mov", ".mp4"]

    def run_synchronization(
            self, overwrite_DLC_analysis_and_synchro: bool = False, verbose: bool = True
    ) -> None:
        """
        Perform analysis of all videos using DLC or other methods,
        synchronization to the led_pattern and downsampling to target_fps.

        Call the synchronizer via VideoInterface and save the prediction file in
        triangulation_dlc_cams_filepaths.
        Create a plot for crossvalidation of the synchronised LED timeseries.
        Define self.markers as unique markers found in all prediction files.
        Exclude videos, if there are diverging framenumbers after synchronization.

        Parameters
        ----------
        overwrite_DLC_analysis_and_synchro: bool, default False
            If True (default False), then pre-existing DLC files and
            synchronisations will be overwritten during analysis.
        verbose: bool, default True
            If True (default), then Crossvalidation plot and synchronised number
            of frames for each camera are printed.
        """
        cams_not_to_analyse = []
        for video_interface in self.video_interfaces.values():
            if video_interface.video_metadata.processing_type == "exclude":
                cams_not_to_analyse.append(video_interface.video_metadata.cam_id)
            else:
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
                    synchronize_only=False,
                    overwrite_DLC_analysis_and_synchro=overwrite_DLC_analysis_and_synchro,
                    synchro_metadata=self.synchro_metadata,
                    verbose=verbose
                )
        for cam in cams_not_to_analyse:
            self.video_interfaces.pop(cam)
            self.metadata_from_videos.pop(cam)
        self._plot_synchro_crossvalidation(verbose=verbose)
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
        self.cams_to_exclude = _exclude_by_framenum(metadata_from_videos=self.metadata_from_videos,
                                               allowed_num_diverging_frames=self.allowed_num_diverging_frames)
        for cam in self.metadata_from_videos:
            if cam in self.cams_to_exclude:
                self.triangulation_dlc_cams_filepaths.pop(cam)

    def _plot_synchro_crossvalidation(self, verbose: bool) -> None:
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
            filename = f'{self.mouse_id}_{self.recording_date}_{self.paradigm}_synchronization_crossvalidation_{self.target_fps}fps'
            synchronization_crossvalidation = AlignmentPlotCrossvalidation(
                template=template,
                led_timeseries=led_timeseries_crossvalidation,
                filename=filename,
                output_directory=self.output_directory,
            )
            synchronization_crossvalidation.create_plot(save=True, plot=verbose)

    def _create_csv_filepath(self) -> Path:
        filepath_out = self.output_directory.joinpath(
            f"{self.mouse_id}_{self.recording_date}_{self.paradigm}_{self.target_fps}fps"
            f"_{self.score_threshold}p_excludedmarkers{self.markers_excluded_manually}_"
            f"filtered{self.synchro_metadata['use_2D_filter']}_"
            f"normalised{self.normalised_dataframe}_{self.triangulation_type}.csv"
        )
        return filepath_out

    def normalize(self, normalization_config_path: Union[Path, str], save_dataframe: bool = True, verbose: bool = False) -> Tuple[Path, float]:
        """
        Rotate and translate the triangulated dataframe.

        Find frame, in which all markers given in the config are defined.
        Translate all points to center. Convert into cm. Rotate dataframe using
        scipy.transform.Rotation.align_vectors to align triangulated vectors
        and ground truth vectors. Create a plot for Visualization of Rotation.

        Parameters
        ----------
        normalization_config_path: Path or str
            The path to the config used for normalisation.
        save_dataframe: bool, default True
            If True (default), then the dataframe will be saved and overwrites
            the pre-existing one.

        Returns
        -------
        self.rotated_filepath: Path
            Filepath, were the rotated triangulated dataframe is saved.
        rotation_error: flot
            Error returned by scipy.transform.Rotation.align_vectors, representing
            whether the alignment worked well.
        verbose: bool, default False
            If True (default False), then the rotation visualization plot is shown.

        Notes
        -----
        The normalization_config_path is a path to a yaml file, representing a
        dictionary with the following key-value pairs:
            CENTER: str
                The marker, at which to set (0, 0, 0).
            REFERENCE_LENGTH_CM: int
                The reference length in cm.
            REFERENCE_LENGTH_MARKERS: list
                The two markers for defining the reference length. The distance
                between those markers in px will be set to ReferenceLengthCm.
            REFERENCE_ROTATION_COORDS: list of list of int
                List of reference rotation markers (at least 3), to align the
                real world space and the triangulated space.
                Each element is a list of int, defining their x, y and z coordinate.
            REFERENCE_ROTATION_MARKERS: list of str
                List of triangulated markers, that should be aligned with
                ReferenceRotationCoords. Their lengths have to be equal!
            INVISIBLE_MARKERS: {str: list of int}
                Keys are 'x', 'y' and 'z', values are lists of ints. All lists
                have to match in length. The length is equal to the number of
                points to be plotted.
                Markers to plot in rotation visualization plot invisiblly. Can
                be used to set the axis aspect equal, since this feature is not
                established for 3D axes in matplotlib.
        """
        normalization_config_path = convert_to_path(normalization_config_path)
        config = read_config(normalization_config_path)
        try:
            best_frame = _get_best_frame_for_normalisation(config=config, df=self.df)
        except ValueError:
            print(f"could not normalize the dataframe {self.csv_output_filepath}!")
            self.rotated_filepath = None
            return Path(""), 1000
        x, y, z = get_3D_array(self.df, config['CENTER'], best_frame)
        for key in self.df.keys():
            if key.endswith('_x'):
                self.df[key] = self.df[key] - x
            if key.endswith('_y'):
                self.df[key] = self.df[key] - y
            if key.endswith('_z'):
                self.df[key] = self.df[key] - z

        reference_length_px = get_xyz_distance_in_triangulation_space(
            marker_ids=tuple(config['REFERENCE_LENGTH_MARKERS']),
            df_xyz=self.df.iloc[best_frame, :])
        conversionfactor = config['REFERENCE_LENGTH_CM'] / reference_length_px
        bp_keys_unflat = set(get_3D_df_keys(key[:-2]) for key in self.df.keys() if
                             'error' not in key and 'score' not in key and "M_" not in key and 'center' not in key and 'fn' not in key)
        bp_keys = list(it.chain(*bp_keys_unflat))
        normalised = self.df.copy()
        normalised[bp_keys] *= conversionfactor
        if config['FLIP_AXIS_TO_ADJUST_CHIRALITY'] is not None:
            keys_to_flip = [key for key in bp_keys if key.endswith(config['FLIP_AXIS_TO_ADJUST_CHIRALITY'])]
            normalised[keys_to_flip] *= -1

        reference_rotation_markers = []
        for marker in config['REFERENCE_ROTATION_MARKERS']:
            reference_rotation_markers.append(get_3D_array(normalised, marker, best_frame))
        r, rotation_error = Rotation.align_vectors(config["REFERENCE_ROTATION_COORDS"],
                                                        reference_rotation_markers)
        rotated = normalised.copy()
        for key in bp_keys_unflat:
            rot_points = r.apply(normalised.loc[:, [key[0], key[1], key[2]]])
            for axis in range(3):
                rotated.loc[:, key[axis]] = rot_points[:, axis]
        rotated_markers = []
        for marker in config['REFERENCE_ROTATION_MARKERS']:
            rotated_markers.append(get_3D_array(rotated, marker, best_frame))

        self.normalised_dataframe = True
        self.rotated_filepath = self._create_csv_filepath()
        if (save_dataframe) or (not self.rotated_filepath.exists()):
            _save_dataframe_as_csv(filepath=str(self.rotated_filepath), df=rotated)
        rotation_plot_filename = self.output_directory.joinpath(f"{self.mouse_id}_{self.recording_date}_{self.paradigm}_rotation_visualization")
        visualization = RotationVisualization(
            rotated_markers=rotated_markers, config=config,
            output_filepath=rotation_plot_filename,
            rotation_error=rotation_error
        )
        visualization.create_plot(plot=verbose, save=True)
        return self.rotated_filepath, rotation_error

    def create_triangulated_video(
            self,
            filename: str,
            config_path: Union[Path, str],
            start_s: int,
            end_s: int
    ) -> None:
        """
        Create video of the triangulated data.

        Parameters
        ----------
        filename: str
            The filename, where the video should be saved.
        config_path: Path or str
            The path to the config used to create triangulated videos.
        start_s:
            The second in the recording to start video creation.
        end_s: 
            The second in the recording to end video creation.

        Notes
        _____
        The yaml file at config_path has to have the following keys:
            body_marker_size, body_label_size: int, int
            body_marker_color, body_label_color: str, str
                matplotlib color
            body_marker_alpha, body_label_alpha: float 0 < 1, float 0 < 1
            markers_to_exclude: list of str
                Markers that should not be plotted. Recommended is, giving at
                least "M_", "center_", "fnum" and "Unnamed" to it, since those
                labels are created from aniposelib and can not be plotted.
            markers_to_connect: list of list of str, optional
                Each element consists of a list of markers, that will be
                connected in the video.
            markers_to_fill: list of dict, optional
                Each element consists of a dict, that will be filled in the
                video. The keys of the dict have to be:
                    markers: list of str
                        The markers, to create a polygon in between.
                    color: str
                        matplotlib color, to fill the polygon.
                    alpha: float 0 < 1
            additional_markers_to_plot: list of dict, optional
                Markers to plot in addition to the triangulated points.
                The elements of the lists are dictionaries containing the
                following key-value pairs:
                    name: str
                    x, y, z: int, int, int
                    alpha: float 0 < 1
                    size: int
                    color: str
                        matplotlib color
                Can be used to set the axis aspect equal, since this feature is
                not established for 3D axes in matplotlib.
                All markers can be used to be connected or filled.
        """
        config_path = convert_to_path(config_path)
        self.video_plotting_config = read_config(config_path)
        self.triangulation_visualization = TriangulationVisualization(df_3D_filepath=self.rotated_filepath,
                                       output_directory=self.output_directory,
                                       config=self.video_plotting_config)
        self.triangulated_video_start_s = start_s
        triangulated_video = VideoClip(
            self._get_triangulated_plots,
            duration=(end_s - start_s),
        )
        triangulated_video.write_videofile(
            f"{filename}.mp4", fps=self.target_fps, logger=None
        )

    def _get_triangulated_plots(self, idx: int) -> np.ndarray:
        idx = int(
            (self.triangulated_video_start_s + idx) * self.target_fps)
        
        return self.triangulation_visualization.return_fig(idx=idx)


class CalibrationValidation(Triangulation):
    """
    Subclass of Triangulation, in which images are triangulated based on a
    calibration file and the triangulated coordinates are validated based on
    a ground_truth.

    Methods
    _______
    add_ground_truth_config(ground_truth_config_filepath)
        Read the metadata from ground_truth_config_filepath and create list of
        markers.
    get_marker_predictions(overwrite_analysed_markers)
        Run marker detection for all images in metadata_from_videos.
    evaluate_triangulation_of_calibration_validation_markers(show_3D_plot, verbose)
        Evaluate the triangulated data and return mean errors.

    See Also
    ________
    Triangulation:
        Parent class, for triangulation of videos or images.
    core.meta.MetaInterface.create_calibrations:
        Create CalibrationValidation objects and run add_ground_truth_config for
        all calibration_directories added to MetaInterface.
    core.meta.MetaInterface.synchronize_calibrations:
        Run get_marker_predictions for all calibration_validation objects added
        to MetaInterface.
    core.checker_objects.CheckCalibrationValidation:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    Calibration.triangulate_optim:
        Run the function run_triangulation for the CalibrationValidation
        object passed to triangulate_optim.

    Examples
    ________
    >>> from core.triangulation_calibration_module import CalibrationValidation
    >>> from pathlib import Path
    >>> rec_config = Path("test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml")
    >>> calibration_validation_object = CalibrationValidation(
        ... project_config_filepath="test_data/project_config.yaml",
        ... directory=rec_config.parent, recording_config_filepath=rec_config,
        ... recreate_undistorted_plots = True, output_directory=rec_config.parent)
    >>> calibration_validation_object.add_ground_truth_config("test_data/ground_truth_config.yaml")
    >>> calibration_validation_object.get_marker_predictions()
    >>> calibration_validation_object.run_triangulation(
        ... calibration_toml_filepath="test_data/Server_structure/Calibrations/220922/220922_0_Bottom_Ground1_Ground2_Side1_Side2_Side3.toml",
        ... triangulate_full_recording = True)
    >>> mean_dist_err_percentage, mean_angle_err, reprojerr_nonan_mean = calibration_validation_object.evaluate_triangulation_of_calibration_validation_markers()
    """
    @property
    def _metadata_keys(self) -> List[str]:
        return ["recording_date"]

    @property
    def _videometadata_tag(self) -> str:
        return "calvin"

    @property
    def _allowed_filetypes(self) -> List[str]:
        return [".bmp", ".tiff", ".png", ".jpg", ".AVI", ".avi", ".mp4"]

    def add_ground_truth_config(self, ground_truth_config_filepath: Union[Path, str]) -> None:
        """
        Read the metadata from ground_truth_config_filepath and create list of
        markers.

        Parameters
        ----------
        ground_truth_config_filepath: str or Path
            The path to the ground_truth config file.

        Notes
        _____
        The ground_truth yaml file at ground_truth_config_filepath has to have
        the following structure:
            distances: {str: {str: float}}
                Dictionary with first markers as key and dictionaries as values,
                that have second markers as key and the known distances between
                first and second markers as values.
                Distances are floats in cm.
            angles:
                Dictionary with vertex markers as keys and dictionaries as values,
                that have the following keys:
                    value: float
                        The value of the calculated angle in degrees.
                    marker: list of str
                        The markers between that draw the triangle (if 3 markers
                        given: 0: vertex, 1/2: ray) or a plane and a line (if 5
                        markers given: 1/2/3: plane, 4/5: line).
            unique_ids: list of str
                A list with all marker_ids to take into account for ground_truth
                validation and plotting.
            marker_ids_to_connect_in_3D_plot: list of list of str, optional
                Each element consists of a list of markers, that will be
                connected in the 3D calibration validation plot.
        """
        ground_truth_config_filepath = convert_to_path(ground_truth_config_filepath)
        self.ground_truth_config = read_config(ground_truth_config_filepath)
        self.markers = self.ground_truth_config["unique_ids"]

    def get_marker_predictions(self, overwrite_analysed_markers: bool = False) -> None:
        """
        Run marker detection for all images in metadata_from_videos.

        Save predictions in triangulation_dlc_cams_filepaths, validate
        predictions and create predictions plots.

        Parameters
        ----------
        overwrite_analysed_markers: bool, default False
            If True (default False), then pre-existing files won't be overwritten
            during the analysis.
        """
        self.markers_excluded_manually = False
        self.triangulation_dlc_cams_filepaths = {}
        for cam in self.metadata_from_videos.values():
            h5_output_filepath = self._run_marker_detection(cam=cam, overwrite_analysed_markers=overwrite_analysed_markers)
            self.triangulation_dlc_cams_filepaths[cam.cam_id] = h5_output_filepath
            self._validate_triangulation_marker_ids(
                triangulation_markers_df_filepath=h5_output_filepath,
                framenum=1,
                defined_marker_ids=self.markers
            )
            predictions = PredictionsPlot(
                image=cam.filepath,
                predictions=h5_output_filepath,
                output_directory=self.output_directory,
                cam_id=cam.cam_id,
                likelihood_threshold=self.score_threshold,
            )
            predictions.create_plot(plot=False, save=True)
        self.cams_to_exclude = []

    def evaluate_triangulation_of_calibration_validation_markers(
            self, show_3D_plot: bool = True, verbose: bool = True,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Evaluate the triangulated data and return mean errors.

        Calculate the distances and angles for all references in ground truth
        and get the differences between triangulated and ground truth data.
        Print these differences and show the plot of the triangulated data.
        Calculate the mean of distance and angle error and reprojection error.

        Parameters
        ----------
        show_3D_plot: bool, default True
            If True (default), then a plot of the triangulated
            calibration_validation data is shown.
        verbose: bool, default True
            If True (default), then all angles and distances compared to their ground truth
            will be printed.

        Returns
        -------
        mean_dist_err_percentage: np.float64
            The mean error of all triangulated distances compared to their
            ground truth.
        mean_angle_err: np.float64
            The mean error of all triangulated errors compared to their ground
            truth.
        reprojerr_nonan_mean: np.float64
            The mean reprojection error of all triangulated points.

        Notes
        _____
        The path directing to the ground_truth yaml file, that is saved as
        self.ground_truth_config, has to have the following structure:
            distances: {str: {str: float}}
                Dictionary with first markers as key and dictionaries as values,
                that have second markers as key and the known distances between
                first and second markers as values.
                Distances are floats in cm.
            angles:
                Dictionary with vertex markers as keys and dictionaries as values,
                that have the following keys:
                    value: float
                        The value of the calculated angle in degrees.
                    marker: list of str
                        The markers between that draw the triangle (if 3 markers
                        given: 0: vertex, 1/2: ray) or a plane and a line (if 5
                        markers given: 1/2/3: plane, 4/5: line).
            unique_ids: list of str
                A list with all marker_ids to take into account for ground_truth
                validation and plotting.
            marker_ids_to_connect_in_3D_plot: list of list of str, optional
                Each element consists of a list of markers, that will be
                connected in the 3D calibration validation plot.
        """
        self.anipose_io = add_reprojection_errors_of_all_calibration_validation_markers(
            anipose_io=self.anipose_io, df_xyz=self.df
        )
        self.anipose_io = set_distances_and_angles_for_evaluation(parameters=self.ground_truth_config,
                                                                  anipose_io=self.anipose_io,
                                                                  df_xyz=self.df)
        gt_distances = load_distances_from_ground_truth(self.ground_truth_config["distances"])
        self.anipose_io = add_errors_between_computed_and_ground_truth_distances_for_different_references(
            anipose_io=self.anipose_io, ground_truth_distances=gt_distances)
        self.anipose_io = add_errors_between_computed_and_ground_truth_angles(
            self.ground_truth_config["angles"], self.anipose_io)

        if verbose:
            for reference_distance_id, distance_errors in self.anipose_io[
                "distance_errors_in_cm"
            ].items():
                print(
                    f'Using {reference_distance_id} as reference distance, '
                    f'the mean distance error is: {distance_errors["mean_error"]} cm.'
                )
            for angle, angle_error in self.anipose_io[
                "angles_error_ground_truth_vs_triangulated"
            ].items():
                print(f"Considering {angle}, the angle error is: {angle_error}")
                
        if show_3D_plot:
            calibration_validation_plot = CalibrationValidationPlot(
                p3d=self.anipose_io["p3ds"][0],
                bodyparts=self.anipose_io["bodyparts"],
                output_directory=self.output_directory,
                marker_ids_to_connect=self.ground_truth_config["marker_ids_to_connect_in_3D_plot"],
                filename_tag=f"{self.csv_output_filepath.stem}"
            )
            calibration_validation_plot.create_plot(plot=True, save=True)

        all_percentage_errors = []
        for reference in self.anipose_io["distance_errors_in_cm"].keys():
            all_percentage_errors = [percentage_error for *_, percentage_error
                                     in self.anipose_io["distance_errors_in_cm"][reference]["individual_errors"]]
        all_angle_errors = list(self.anipose_io["angles_error_ground_truth_vs_triangulated"].values())

        mean_dist_err_percentage = np.nanmean(np.asarray(all_percentage_errors))
        mean_angle_err = np.nanmean(np.asarray(all_angle_errors))
        reprojerr_nonan_mean = self.anipose_io["reproj_nonan"].mean()
        return mean_dist_err_percentage, mean_angle_err, reprojerr_nonan_mean

    def _create_csv_filepath(self) -> Path:
        filepath_out = self.output_directory.joinpath(
            f"Calvin_{self.recording_date}_{self.score_threshold}p_excludedmarkers"
            f"{self.markers_excluded_manually}_filteredFalse_{self.triangulation_type}.csv")
        return filepath_out

    def _run_marker_detection(self, cam: VideoMetadata, overwrite_analysed_markers: bool=False) -> Path:
        h5_output_filepath = self.output_directory.joinpath(
            f"Calvin_{self.recording_date}_{cam.cam_id}.h5"
        )
        if overwrite_analysed_markers or (not h5_output_filepath.exists()):
            if cam.calibration_evaluation_type == "manual":
                config = cam.calibration_evaluation_filepath
                manual_interface = ManualAnnotation(
                    object_to_analyse=cam.filepath,
                    output_directory=self.output_directory,
                    marker_detection_directory=config,
                )
                h5_output_filepath = manual_interface.analyze_objects(filepath=h5_output_filepath,
                                                                      only_first_frame=True)
            elif cam.calibration_evaluation_type == "DLC":
                config = cam.calibration_evaluation_filepath
                dlc_interface = DeeplabcutInterface(
                    object_to_analyse=cam.filepath,
                    output_directory=self.output_directory,
                    marker_detection_directory=config,
                )
                h5_output_filepath = dlc_interface.analyze_objects(filepath=h5_output_filepath,
                                                                   filtering=False, 
                                                                   use_gpu = "low")
                # filtering is not supported and not necessary for single frame predictions!
            else:
                raise ValueError(
                    "For calibration_evaluation only manual and DLC are supported!"
                )
        return h5_output_filepath
