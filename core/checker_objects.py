from pathlib import Path
from typing import Tuple, Dict, List, Any, Set, Union

import imageio as iio

from .plotting import Intrinsics
from .user_specific_rules import user_specific_rules_on_triangulation_calibration_videos
from .utils import convert_to_path, read_config, check_keys, KEYS_TO_CHECK_PROJECT, \
    KEYS_TO_CHECK_RECORDING, KEYS_TO_CHECK_CAMERA_RECORDING, KEYS_TO_CHECK_CAMERA_PROJECT
from .video_metadata import VideoMetadataChecker

def _check_for_missing_or_duplicate_cameras(
        valid_cam_ids: List[str],
        metadata_from_videos: List[VideoMetadataChecker],
        directory: Path,
        recording_config_filepath: Path,
        recording_config_dict: Dict
) -> List[VideoMetadataChecker]:
    files_per_cam = {}
    cams_not_found = []
    for cam in valid_cam_ids:
        files_per_cam[cam] = []
    for video_metadata in metadata_from_videos:
        files_per_cam[video_metadata.cam_id].append(video_metadata.filepath)

    for key in files_per_cam:
        if len(files_per_cam[key]) == 1:
            pass
        elif len(files_per_cam[key]) == 0:
            cams_not_found.append(key)
        elif len(files_per_cam[key]) > 1:
            information_duplicates = [
                (
                    i,
                    file,
                    f"Framenum: {iio.v2.get_reader(file).count_frames()}",
                )
                for i, file in enumerate(files_per_cam[key])
            ]
            print(
                f"Found {len(files_per_cam[key])} videos for {key} in {directory}!"
                f"\n {information_duplicates}"
            )
            file_idx_to_keep = input(
                "Enter the number of the file you want to keep (other files will be deleted!)!\n"
                "Enter c if you want to abort and move the file manually!"
            )
            if file_idx_to_keep == "c":
                print(
                    f"You have multiple videos for cam {key} in {directory}, "
                    f"but you decided to abort. If you dont move them manually, "
                    f"this can lead to wrong videos in the analysis!"
                )
            else:
                for i, file in enumerate(files_per_cam[key]):
                    if i != int(file_idx_to_keep):
                        for video_metadata in metadata_from_videos:
                            if video_metadata.filepath == file:
                                metadata_from_videos.remove(video_metadata)
                        if file.exists():
                            file.unlink()
    cameras_missing_in_recording_config = check_keys(
        dictionary=recording_config_dict, list_of_keys=valid_cam_ids
    )

    for video_metadata in metadata_from_videos:
        user_specific_rules_on_triangulation_calibration_videos(video_metadata)

    for cam in cams_not_found:
        if cam in cameras_missing_in_recording_config:
            cams_not_found.remove(cam)
            cameras_missing_in_recording_config.remove(cam)
    if cams_not_found:
        print(
            f"At {directory}\nFound no video for {cams_not_found}!"
        )
    if cameras_missing_in_recording_config:
        print(
            f"No information for {cameras_missing_in_recording_config} "
            f"in the config_file {recording_config_filepath}!"
        )
    return metadata_from_videos


def _validate_metadata(metadata_from_videos: List,
                       attributes_to_check: List[str]) -> Tuple[Any, ...]:
    sets_of_attributes = []
    for attribute_to_check in attributes_to_check:
        set_of_attribute = set(getattr(video_metadata, attribute_to_check)
                               for video_metadata in metadata_from_videos
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

def _create_video_objects(
        directory: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict,
        videometadata_tag: str,
        filetypes: List[str],
        filename_tag: str = "",
) -> List:
    videofiles = [file for file in directory.iterdir() if
                  filename_tag.lower() in file.name.lower()
                  and "synchronized" not in file.name and file.suffix in filetypes]

    metadata_from_videos = []
    for filepath in videofiles:
        try:
            iio.v3.imread(filepath, index=0)
        except:
            print(
                f"Could not open file {filepath}. Check, whether the file is corrupted and delete it manually!"
            )
            videofiles.remove(filepath)
    for filepath in videofiles:
        try:
            video_metadata = VideoMetadataChecker(
                video_filepath=filepath,
                recording_config_dict=recording_config_dict,
                project_config_dict=project_config_dict,
                tag=videometadata_tag,
            )
            metadata_from_videos.append(video_metadata)
        except TypeError:
            pass
    return metadata_from_videos


class CheckCalibration:
    """
    A class, that checks the metadata and filenames of videos in a given
    folder and allows for filename changing via user input.

    Parameters
    ----------
    calibration_directory: Path or str
        Directory, where the calibration videos are stored.
    project_config_filepath: Path or str
        Filepath to the project_config .yaml file.
    recording_config_filepath: Path or str
        Filepath to the recording_config .yaml file.
    plot: bool, default True
        If True (default), then the undistorted images are plotted.

    Attributes
    __________
    calibration_index: int
        Index of a calibration.
        Together with recording_date, it creates a unique calibration key.
    recording_date: str
        Date at which the calibration was done.

    See Also
    ________
    CheckRecording:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    CheckCalibrationValidation:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    core.triangulation_calibration_module.Calibration:
        A class, in which videos are calibrated to each other.
    core.filename_checker.FilenameCheckerInterface.create_calibrations:
        Create CheckCalibration and CheckCalibrationValidation objects for
        all calibration_directories added to FilenameCheckerInterface.
    """
    def __init__(
            self,
            calibration_directory: Union[Path, str],
            project_config_filepath: Union[Path, str],
            recording_config_filepath: Union[Path, str],
            plot: bool = True,
    ) -> None:
        """
        Construct all necessary attributes for the CheckCalibration class.

        Read the metadata from project-/recording config and from video filenames.
        Check for errors in filenames, for duplicate and missing cameras.

        Parameters
        ----------
        calibration_directory: Path or string
            Directory, where the calibration videos are stored.
        project_config_filepath: Path or string
            Filepath to the project_config .yaml file.
        recording_config_filepath: Path or string
            Filepath to the recording_config .yaml file.
        plot: bool, default True
            If True (default), then the undistorted images are plotted.
        """
        calibration_directory = convert_to_path(calibration_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)

        print("\n")
        recording_config_dict, project_config_dict = _get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self.recording_date = recording_config_dict['recording_date']
        self.calibration_index = recording_config_dict["calibration_index"]
        metadata_from_videos = _create_video_objects(
            directory=calibration_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="calibration",
            filetypes=[".AVI", ".avi", ".mov", ".mp4"],
            filename_tag=project_config_dict['calibration_tag'],
        )
        if plot:
            print(f"Intrinsic calibrations for calibration of {self.recording_date}")
            for video_metadata in metadata_from_videos:
                print(video_metadata.cam_id)
                intrinsics = Intrinsics(video_filepath=video_metadata.filepath,
                                        intrinsic_calibration=video_metadata.intrinsic_calibration, filename="",
                                        fisheye=video_metadata.fisheye)
                intrinsics.create_plot(plot=True, save=False)

        metadata_from_videos = _check_for_missing_or_duplicate_cameras(
            valid_cam_ids=project_config_dict['valid_cam_ids'],
            metadata_from_videos=metadata_from_videos,
            directory=calibration_directory,
            recording_config_filepath=recording_config_filepath,
            recording_config_dict=recording_config_dict)
        for video_metadata in metadata_from_videos:
            user_specific_rules_on_triangulation_calibration_videos(video_metadata)
        self.recording_date, *_ = _validate_metadata(metadata_from_videos=metadata_from_videos,
                                                     attributes_to_check=['recording_date'])


class CheckRecording:
    """
    A class, that checks the metadata and filenames of videos in a given
    folder and allows for filename changing via user input.

    Parameters
    ----------
    recording_directory: Path or str
        Directory, where the recording videos are stored.
    recording_config_filepath: Path or str
        Filepath to the recording_config .yaml file.
    project_config_filepath: Path or str
        Filepath to the project_config .yaml file.
    plot: bool, default True
        If True (default), then the undistorted images are plotted.

    Attributes
    __________
    calibration_index: int
        Index of a calibration.
        Together with recording_date, it creates a unique calibration key.
    recording_date: str
        Date at which the calibration was done.
    mouse_id: str
        The mouse_id as read from the filenames.
    paradigm: str
        The paradigm as read from the filenames.

    See Also
    ________
    CheckCalibration:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    CheckCalibrationValidation:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    core.triangulation_calibration_module.TriangulationRecordings:
        Subclass of Triangulation, in which videos are triangulated based on a
        calibration file.
    core.filename_checker.FilenameCheckerInterface.create_recordings:
        Create CheckRecording objects for all recording_directories
        added to FilenameCheckerInterface.
    """
    def __init__(
            self,
            recording_directory: Union[Path, str],
            recording_config_filepath: Union[Path, str],
            project_config_filepath: Union[Path, str],
            plot: bool = False,
    ) -> None:
        """
        Construct all necessary attributes for the CheckRecording class.

        Read the metadata from project-/recording config and from video filenames.
        Check for errors in filenames, for duplicate and missing cameras.

        Parameters
        ----------
        recording_directory: Path or str
            Directory, where the recording videos are stored.
        recording_config_filepath: Path or str
            Filepath to the recording_config .yaml file.
        project_config_filepath: Path or str
            Filepath to the project_config .yaml file.
        plot: bool, default True
            If True (default), then the undistorted images are plotted.
        """
        recording_directory = convert_to_path(recording_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        print("\n")
        recording_config_dict, project_config_dict = _get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self.recording_date = recording_config_dict['recording_date']
        self.calibration_index = recording_config_dict["calibration_index"]
        metadata_from_videos = _create_video_objects(
            directory=recording_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="recording",
            filetypes=[".AVI", ".avi", ".mov", ".mp4"],
        )
        if plot:
            print(f"Intrinsic calibrations for {self.recording_date}")
            for video_metadata in metadata_from_videos:
                print(video_metadata.cam_id)
                intrinsics = Intrinsics(video_filepath=video_metadata.filepath,
                                        intrinsic_calibration=video_metadata.intrinsic_calibration,
                                        filename="",
                                        fisheye=video_metadata.fisheye)
                intrinsics.create_plot(save=False, plot=True)

        metadata_from_videos = _check_for_missing_or_duplicate_cameras(
            valid_cam_ids=project_config_dict['valid_cam_ids'],
            metadata_from_videos=metadata_from_videos,
            directory=recording_directory,
            recording_config_filepath=recording_config_filepath,
            recording_config_dict=recording_config_dict)
        for video_metadata in metadata_from_videos:
            user_specific_rules_on_triangulation_calibration_videos(video_metadata)
        self.recording_date, self.paradigm, self.mouse_id = _validate_metadata(
            metadata_from_videos=metadata_from_videos,
            attributes_to_check=["recording_date", "paradigm", "mouse_id"]
        )


class CheckCalibrationValidation:
    """
    A class, that checks the metadata and filenames of videos in a given
    folder and allows for filename changing via user input.

    Parameters
    ----------
    calibration_validation_directory: Path or str
        Directory, where the calibration_validation iamges are stored.
    recording_config_filepath: Path or str
        Filepath to the recording_config .yaml file.
    ground_truth_config_filepath: Path or str
        The path to the ground_truth config file.
    project_config_filepath: Path or str
        Filepath to the project_config .yaml file.
    plot: bool, default True
        If True (default), then the undistorted images are plotted.

    Attributes
    __________
    calibration_index: int
        Index of a calibration.
        Together with recording_date, it creates a unique calibration key.
    recording_date: str
        Date at which the calibration was done.

    See Also
    ________
    CheckCalibration:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    CheckRecording:
        A class, that checks the metadata and filenames of videos in a given
        folder and allows for filename changing via user input.
    core.triangulation_calibration_module.CalibrationValidation:
        A class, in which images are triangulated based on a calibration file
        and the triangulated coordinates are validated based on a ground_truth.
    core.filename_checker.FilenameCheckerInterface.create_calibrations:
        Create CheckCalibration and CheckCalibrationValidation objects for
        all calibration_directories added to FilenameCheckerInterface.
    """
    def __init__(
            self,
            calibration_validation_directory: Union[Path, str],
            recording_config_filepath: Union[Path, str],
            ground_truth_config_filepath: Union[Path, str],
            project_config_filepath: Union[Path, str],
            plot: bool = True,
    ) -> None:
        """
        Construct all necessary attributes for the CheckCalibrationValidation class.

        Read the metadata from project-/recording config and from image filenames.
        Check for errors in filenames, for duplicate and missing cameras.

        Parameters
        ----------
        calibration_validation_directory: Path or str
            Directory, where the calibration_validation iamges are stored.
        recording_config_filepath: Path or str
            Filepath to the recording_config .yaml file.
        ground_truth_config_filepath: Path or str
            The path to the ground_truth config file.
        project_config_filepath: Path or str
            Filepath to the project_config .yaml file.
        plot: bool, default True
            If True (default), then the undistorted images are plotted.
        """
        print("\n")
        ground_truth_config_filepath = convert_to_path(ground_truth_config_filepath)
        calibration_validation_gt = read_config(ground_truth_config_filepath)
        calibration_validation_directory = convert_to_path(calibration_validation_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)

        recording_config_dict, project_config_dict = _get_metadata_from_configs(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
        )
        self.recording_date = recording_config_dict["recording_date"]
        self.calibration_index = recording_config_dict["calibration_index"]
        metadata_from_videos = _create_video_objects(
            directory=calibration_validation_directory,
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            videometadata_tag="calvin",
            filetypes=[".bmp", ".tiff", ".png", ".jpg", ".AVI", ".avi"],
            filename_tag=project_config_dict["calibration_validation_tag"],
        )
        if plot:
            print(f"Intrinsic calibrations for calvin of {self.recording_date}")
            for video_metadata in metadata_from_videos:
                print(video_metadata.cam_id)
                intrinsics = Intrinsics(video_filepath=video_metadata.filepath,
                                        intrinsic_calibration=video_metadata.intrinsic_calibration,
                                        filename="",
                                        fisheye=video_metadata.fisheye)
                intrinsics.create_plot(plot=True, save=False)

        metadata_from_videos = _check_for_missing_or_duplicate_cameras(
            valid_cam_ids=project_config_dict['valid_cam_ids'],
            metadata_from_videos=metadata_from_videos,
            directory=calibration_validation_directory,
            recording_config_filepath=recording_config_filepath,
            recording_config_dict=recording_config_dict)
        for video_metadata in metadata_from_videos:
            user_specific_rules_on_triangulation_calibration_videos(video_metadata)
        self.recording_date, *_ = _validate_metadata(metadata_from_videos=metadata_from_videos,
                                                     attributes_to_check=['recording_date'])
