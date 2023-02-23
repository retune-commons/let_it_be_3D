from typing import List, Tuple, Dict, Union, Optional
import datetime
from pathlib import Path
from abc import ABC, abstractmethod

import imageio as iio

from .utils import convert_to_path, read_config, check_keys
from .video_metadata import VideoMetadataChecker
from .plotting import Intrinsics
from .user_specific_rules import user_specific_rules_on_triangulation_calibration_videos


class Check(ABC):
    """
    @property
    def whatever(self) -> float:
        return 0.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

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
        ]
        missing_keys = check_keys(
            dictionary=project_config_dict, list_of_keys=keys_to_check_project
        )
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing metadata information in the project_config_file {project_config_filepath} for {missing_keys}."
            )

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

        self.valid_cam_ids = project_config_dict["valid_cam_IDs"]
        self.recording_date = recording_config_dict["recording_date"]
        self.led_pattern = recording_config_dict["led_pattern"]
        self.calibration_index = recording_config_dict["calibration_index"]
        self.target_fps = recording_config_dict["target_fps"]
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

    def _create_video_objects(
        self,
        directory: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict,
        videometadata_tag: str,
        filetypes: [str],
        filename_tag: str = "",
    ) -> None:
        videofiles = [
            file
            for file in directory.iterdir()
            if filename_tag.lower() in file.name.lower()
            and file.suffix in filetypes
            and "synchronized" not in file.name
        ]

        self.metadata_from_videos = []
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
                self.metadata_from_videos.append(video_metadata)
            except TypeError:
                pass


class Check_Calibration(Check):
    def __init__(
        self,
        calibration_directory: Path,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        plot: bool = True,
    ) -> None:
        self.calibration_directory = convert_to_path(calibration_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)

        print("\n")
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
        )
        if plot:
            print(f"Intrinsic calibrations for calibration of {self.recording_date}")
            for video_metadata in self.metadata_from_videos:
                print(video_metadata.cam_id)
                Intrinsics(video_metadata=video_metadata, plot=True, save=False)

        files_per_cam = {}
        cams_not_found = []
        for cam in self.valid_cam_ids:
            files_per_cam[cam] = []
        for video_metadata in self.metadata_from_videos:
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
                        f"Size: {file.stat().st_size} bytes",
                        f"Framenum: {iio.v2.get_reader(file).count_frames()}",
                    )
                    for i, file in enumerate(files_per_cam[key])
                ]
                print(
                    f"Found {len(files_per_cam[key])} videos for {key} in {self.calibration_directory}!\n {information_duplicates}"
                )
                file_idx_to_keep = input(
                    "Enter the number of the file you want to keep (other files will be deleted!)!\nEnter c if you want to abort and move the file manually!"
                )
                if file_idx_to_keep == "c":
                    print(
                        f"You have multiple videos for cam {key} in {self.calibration_directory}, but you decided to abort. If you dont move them manually, this can lead to wrong videos in the analysis!"
                    )
                else:
                    for i, file in enumerate(files_per_cam[key]):
                        if i != int(file_idx_to_keep):
                            for video_metadata in self.metadata_from_videos:
                                if video_metadata.filepath == file:
                                    self.metadata_from_videos.remove(video_metadata)
                            file.unlink()
        
        for video_metadata in self.metadata_from_videos:
            user_specific_rules_on_triangulation_calibration_videos(video_metadata)

        self._validate_and_save_metadata_for_recording()
        for cam in cams_not_found:
            if cam in self.cameras_missing_in_recording_config:
                cams_not_found.remove(cam)
                self.cameras_missing_in_recording_config.remove(cam)
        if len(cams_not_found) > 0:
            print(
                f"At {self.calibration_directory}\nFound no video for {cams_not_found}!"
            )
        if len(self.cameras_missing_in_recording_config) > 0:
            print(
                f"No information for {self.cameras_missing_in_recording_config} in the config_file {recording_config_filepath}!"
            )

        synch_videofiles = [
            file
            for file in self.calibration_directory.iterdir()
            if file.suffix == ".mp4"
            and "synchronized" in file.name
            and "Front" not in file.name
            and "downsampled" not in file.name
        ]

        for file in synch_videofiles:
            file.rename(
                file.parent.joinpath(
                    file.stem + f"_downsampled{self.target_fps}fps" + ".mp4"
                )
            )

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(
            video_metadata.recording_date
            for video_metadata in self.metadata_from_videos
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


class Check_Recording(Check):
    def __init__(
        self,
        recording_directory: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        plot: bool = False,
    ) -> None:
        self.recording_directory = convert_to_path(recording_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)
        print("\n")
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
        )
        if plot:
            print(f"Intrinsic calibrations for {self.recording_date}")
            for video_metadata in self.metadata_from_videos:
                print(video_metadata.cam_id)
                Intrinsics(video_metadata=video_metadata, plot=True, save=False)

        files_per_cam = {}
        cams_not_found = []
        for cam in self.valid_cam_ids:
            files_per_cam[cam] = []
        for video_metadata in self.metadata_from_videos:
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
                        f"Size: {file.stat().st_size} bytes",
                        f"Framenum: {iio.v2.get_reader(file).count_frames()}",
                    )
                    for i, file in enumerate(files_per_cam[key])
                ]
                print(
                    f"Found {len(files_per_cam[key])} videos for {key} in {self.recording_directory}!\n {information_duplicates}"
                )
                file_idx_to_keep = input(
                    "Enter the number of the file you want to keep (other files will be deleted!)!\nEnter c if you want to abort and move the file manually!"
                )
                if file_idx_to_keep == "c":
                    print(
                        f"You have multiple videos for cam {key} in {self.recording_directory}, but you decided to abort. If you dont move them manually, this can lead to wrong videos in the analysis!"
                    )
                else:
                    for i, file in enumerate(files_per_cam[key]):
                        if i != int(file_idx_to_keep):
                            for video_metadata in self.metadata_from_videos:
                                if video_metadata.filepath == file:
                                    self.metadata_from_videos.remove(video_metadata)
                            if file.exists():
                                file.unlink()
                                
        for video_metadata in self.metadata_from_videos:
            user_specific_rules_on_triangulation_calibration_videos(video_metadata)

        self._validate_and_save_metadata_for_recording()
        for cam in cams_not_found:
            if cam in self.cameras_missing_in_recording_config:
                cams_not_found.remove(cam)
                self.cameras_missing_in_recording_config.remove(cam)
        if len(cams_not_found) > 0:
            print(
                f"At {self.recording_directory}\nFound no video for {cams_not_found}!"
            )
        if len(self.cameras_missing_in_recording_config) > 0:
            print(
                f"No information for {self.cameras_missing_in_recording_config} in the config_file {recording_config_filepath}!"
            )

        synch_videofiles = [
            file
            for file in self.recording_directory.iterdir()
            if file.suffix == ".mp4"
            and "synchronized" in file.name
            and "Front" not in file.name
            and "downsampled" not in file.name
        ]

        for file in synch_videofiles:
            file.rename(
                file.parent.joinpath(
                    file.stem + f"_downsampled{self.target_fps}fps" + ".mp4"
                )
            )

        synch_hdffiles = [
            file
            for file in self.recording_directory.iterdir()
            if file.suffix == ".h5"
            and "Front" not in file.name
            and "filtered" not in file.name
            and "downsampled" not in file.name
            and "SyncData" not in file.name
        ]

        for file in synch_hdffiles:
            file.rename(
                file.parent.joinpath(
                    file.stem + f"_downsampled{self.target_fps}fps_synchronized" + ".h5"
                )
            )

        filtered_hdffiles = [
            file
            for file in self.recording_directory.iterdir()
            if file.suffix == ".h5"
            and "Front" not in file.name
            and "_filtered" in file.name
            and "downsampled" not in file.name
            and "SyncData" not in file.name
        ]

        for file in filtered_hdffiles:
            file.rename(
                file.parent.joinpath(
                    file.stem.replace("_filtered", "")
                    + f"_downsampled{self.target_fps}fps_synchronized_filtered"
                    + ".h5"
                )
            )

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(
            video_metadata.recording_date
            for video_metadata in self.metadata_from_videos
        )
        paradigms = set(
            video_metadata.paradigm for video_metadata in self.metadata_from_videos
        )
        mouse_ids = set(
            video_metadata.mouse_id for video_metadata in self.metadata_from_videos
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


class Check_Calibration_Validation(Check):
    def __init__(
        self,
        calibration_validation_directory: Path,
        recording_config_filepath: Path,
        ground_truth_config_filepath: Path,
        project_config_filepath: Path,
        plot: bool = True,
    ) -> None:
        print("\n")
        ground_truth_config_filepath = convert_to_path(ground_truth_config_filepath)
        calibration_validation_gt = read_config(ground_truth_config_filepath)
        self.calibration_validation_directory = convert_to_path(calibration_validation_directory)
        project_config_filepath = convert_to_path(project_config_filepath)
        recording_config_filepath = convert_to_path(recording_config_filepath)

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
        )
        if plot:
            print(f"Intrinsic calibrations for calvin of {self.recording_date}")
            for video_metadata in self.metadata_from_videos:
                print(video_metadata.cam_id)
                Intrinsics(video_metadata=video_metadata, plot=True, save=False)

        files_per_cam = {}
        cams_not_found = []
        for cam in self.valid_cam_ids:
            files_per_cam[cam] = []
        for video_metadata in self.metadata_from_videos:
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
                        f"Size: {file.stat().st_size} bytes",
                        f"Framenum: {iio.v2.get_reader(file).count_frames()}",
                    )
                    for i, file in enumerate(files_per_cam[key])
                ]
                print(
                    f"Found {len(files_per_cam[key])} videos for {key} in {self.calibration_validation_directory}!\n {information_duplicates}"
                )
                file_idx_to_keep = input(
                    "Enter the number of the file you want to keep (other files will be deleted!)!\nEnter c if you want to abort and move the file manually!"
                )
                if file_idx_to_keep == "c":
                    print(
                        f"You have multiple videos for cam {key} in {self.calibration_validation_directory}, but you decided to abort. If you dont move them manually, this can lead to wrong videos in the analysis!"
                    )
                else:
                    for i, file in enumerate(files_per_cam[key]):
                        if i != int(file_idx_to_keep):
                            for video_metadata in self.metadata_from_videos:
                                if video_metadata.filepath == file:
                                    self.metadata_from_videos.remove(video_metadata)
                            file.unlink()
        
        for video_metadata in self.metadata_from_videos:
            user_specific_rules_on_triangulation_calibration_videos(video_metadata)

        self._validate_and_save_metadata_for_recording()
        for cam in cams_not_found:
            if cam in self.cameras_missing_in_recording_config:
                cams_not_found.remove(cam)
                self.cameras_missing_in_recording_config.remove(cam)
        if len(cams_not_found) > 0:
            print(
                f"At {self.calibration_validation_directory}\nFound no video for {cams_not_found}!"
            )
        if len(self.cameras_missing_in_recording_config) > 0:
            print(
                f"No information for {self.cameras_missing_in_recording_config} in the config_file {recording_config_filepath}!"
            )

    def _validate_and_save_metadata_for_recording(self) -> None:
        recording_dates = set(
            video_metadata.recording_date
            for video_metadata in self.metadata_from_videos
        )
        for attribute in [recording_dates]:
            if len(attribute) > 1:
                raise ValueError(
                    f"The filenames of the calvin images give different metadata! Reasons could be:\n"
                    f"  - image belongs to another calibration\n"
                    f"  - image filename is valid, but wrong\n"
                    f"Go the folder {self.calibration_validation_directory} and check the filenames manually!"
                )
        self.recording_date = list(recording_dates)[0]
