import time
from abc import ABC
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from typing import Tuple, Optional, Dict, Union, List

import pandas as pd
import yaml

from .triangulation_calibration_module import (
    Calibration,
    CalibrationValidation,
    TriangulationRecordings,
)
from .utils import convert_to_path, check_keys, read_config
from .video_metadata import VideoMetadata


class MetaInterface(ABC):
    def __init__(
            self,
            project_config_filepath: Path,
            project_name: Optional[str] = None,
            overwrite: bool = False,
    ) -> None:
        self.project_config_filepath = convert_to_path(project_config_filepath)
        self._create_standard_yaml_filepath(
            project_name=project_name, overwrite=overwrite
        )
        self.paradigms = self._read_project_config()
        self.recording_configs = []
        self.recording_dates = []
        self.meta = {
            "project_config_filepath": str(self.project_config_filepath),
            "recording_days": {},
        }

    def select_recording_configs(self) -> None:
        Tk().withdraw()
        selected_recording_configs = askopenfilenames(
            title="Select recording_config.yaml"
        )

        for filepath_to_recording_config in selected_recording_configs:
            self.add_recording_config(
                filepath_to_recording_config=filepath_to_recording_config
            )

    def add_recording_config(self, filepath_to_recording_config: Union[str, Path]) -> None:
        filepath_to_recording_config = convert_to_path(filepath_to_recording_config)
        if (
                filepath_to_recording_config.suffix == ".yaml"
                and filepath_to_recording_config.exists()
        ):
            if filepath_to_recording_config not in self.recording_configs:
                self.recording_configs.append(filepath_to_recording_config)
                recording_date, calibration_index = self._read_recording_config(
                    recording_config_filepath=filepath_to_recording_config
                )
                self.meta["recording_days"][
                    f"Recording_Day_{recording_date}_{str(calibration_index)}"
                ] = {
                    "recording_config_filepath": str(filepath_to_recording_config),
                    "recording_date": recording_date,
                    "recording_directories": [],
                    "recordings": {},
                    "calibrations": {},
                    "calibration_directory": str(filepath_to_recording_config.parent),
                    "calibration_index": calibration_index,
                }
            else:
                print("The config file was already added!")
        else:
            raise FileNotFoundError(
                f"The path doesn't exist or is not linked to a .yaml file!"
            )

    def initialize_meta_config(self) -> None:
        self.objects = {}
        for recording_day in self.meta["recording_days"].values():
            for file in Path(
                    recording_day["recording_config_filepath"]
            ).parent.parent.parent.glob("**"):
                if (
                        file.name[: len(recording_day["recording_date"])]
                        == recording_day["recording_date"]
                        and file.name[-3:] in self.paradigms
                ):  # hardcoded length of paradigm and file structure
                    recording_day["recording_directories"].append(str(file))
            recording_day["num_recordings"] = len(
                recording_day["recording_directories"]
            )
            print(
                f"Found {recording_day['num_recordings']} recordings at "
                f"recording day {recording_day['recording_date']}!"
            )
        self.meta["meta_step"] = 1
        self.export_meta_to_yaml(filepath=self.standard_yaml_filepath)

    def add_recording_manually(self, file: Path, recording_day: str) -> None:
        file = convert_to_path(file)
        if not file.is_dir() or recording_day not in self.meta["recording_days"].keys():
            raise FileNotFoundError(
                f"couldn't add recording directory! \n"
                f"Check your filepath and make sure the recording_day is "
                f"in {self.meta['recording_days'].keys()}!")
        else:
            self.meta["recording_days"][recording_day]["recording_directories"].append(
                str(file)
            )
            self.meta["recording_days"][recording_day]["num_recordings"] = len(
                self.meta["recording_days"][recording_day]["recording_directories"]
            )
            print("added recording directory succesfully!")

    def create_recordings(self, test_mode: bool = False) -> None:
        self.objects["triangulation_recordings_objects"] = {}
        for recording_day in self.meta["recording_days"]:
            for recording in self.meta["recording_days"][recording_day][
                "recording_directories"
            ]:
                triangulation_recordings_object = TriangulationRecordings(
                    directory=Path(recording),
                    recording_config_filepath=self.meta["recording_days"][recording_day]["recording_config_filepath"],
                    project_config_filepath=Path(self.meta["project_config_filepath"]),
                    output_directory=recording,
                    test_mode=test_mode,
                )
                individual_key = f"{triangulation_recordings_object.mouse_id}_" \
                                 f"{triangulation_recordings_object.recording_date}_" \
                                 f"{triangulation_recordings_object.paradigm}"
                videos = {
                    video: self._create_video_dict(
                        video=triangulation_recordings_object.metadata_from_videos[
                            video
                        ]
                    )
                    for video in triangulation_recordings_object.metadata_from_videos
                }
                self.meta["recording_days"][recording_day]["recordings"][
                    individual_key
                ] = {
                    "recording_directory": recording,
                    "key": individual_key,
                    "target_fps": triangulation_recordings_object.target_fps,
                    "led_pattern": triangulation_recordings_object.led_pattern,
                    "calibration_to_use": f'{triangulation_recordings_object.recording_date}'
                                          f'_{triangulation_recordings_object.calibration_index}',
                    "videos": videos,
                }
                self.objects["triangulation_recordings_objects"][
                    individual_key
                ] = triangulation_recordings_object
        self.meta["meta_step"] = 2
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def synchronize_recordings(
            self,
            verbose: bool = False,
            test_mode: bool = False,
            synchronize_only: bool = False,
    ) -> None:
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                start_time_recording = time.time()
                recording_object = self.objects["triangulation_recordings_objects"][recording]
                recording_meta = recording_day["recordings"][recording]
                recording_object.run_synchronization(
                    test_mode=test_mode, verbose=verbose
                )
                for video in recording_meta["videos"]:
                    try:
                        recording_meta["videos"][video]["synchronized_video"] = str(
                            recording_object.synchronized_videos[video])
                        recording_meta["videos"][video]["framenum_synchronized"] = (
                            recording_object.metadata_from_videos[video].framenum_synchronized
                        )
                        recording_meta["videos"][video]["marker_detection_filepath"] = str(
                            recording_object.triangulation_dlc_cams_filepaths[video])
                    except:
                        pass
                end_time_recording = time.time()
                duration = end_time_recording - start_time_recording
                if verbose:
                    print(
                        f"The analysis of this recording {recording} took {duration}.\n"
                    )

        self.meta["meta_step"] = 3
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def create_calibrations(
            self, ground_truth_config_filepath: Path, test_mode: bool = False
    ) -> None:
        self.objects["calibration_objects"] = {}
        self.objects["calibration_validation_objects"] = {}
        for recording_day in self.meta["recording_days"].values():
            calibration_object = Calibration(
                calibration_directory=recording_day["calibration_directory"],
                project_config_filepath=self.project_config_filepath,
                recording_config_filepath=recording_day["recording_config_filepath"],
                output_directory=recording_day["calibration_directory"],
                test_mode=test_mode,
            )

            unique_calibration_key = f'{calibration_object.recording_date}_' \
                                     f'{calibration_object.calibration_index}'

            self.objects["calibration_objects"][unique_calibration_key] = calibration_object

            video_dict = {
                video: self._create_video_dict(
                    calibration_object.metadata_from_videos[video], intrinsics=True
                )
                for video in calibration_object.metadata_from_videos
            }
            recording_day["calibrations"]["calibration_key"] = unique_calibration_key
            recording_day["calibrations"]["target_fps"] = calibration_object.target_fps
            recording_day["calibrations"]["led_pattern"] = calibration_object.led_pattern
            recording_day["calibrations"]["videos"] = video_dict

            calibration_validation_object = CalibrationValidation(
                directory=recording_day["calibration_directory"],
                recording_config_filepath=recording_day["recording_config_filepath"],
                project_config_filepath=self.project_config_filepath,
                output_directory=recording_day["calibration_directory"],
                test_mode=test_mode,
            )
            calibration_validation_object.add_ground_truth_config(
                ground_truth_config_filepath=ground_truth_config_filepath)
            self.objects["calibration_validation_objects"][unique_calibration_key] = calibration_validation_object

            for video in calibration_validation_object.metadata_from_videos.values():
                try:
                    recording_day["calibrations"]["videos"][video.cam_id][
                        "calibration_validation_image_filepath"
                    ] = str(video.filepath)
                except:
                    pass
        self.meta["meta_step"] = 4
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def synchronize_calibrations(self, test_mode: bool = False) -> None:
        for recording_day in self.meta["recording_days"].values():
            calibration_object = self.objects["calibration_objects"][recording_day["calibrations"]["calibration_key"]]
            calibration_object.run_synchronization(test_mode=test_mode)
            for video in recording_day["calibrations"]["videos"]:
                recording_day["calibrations"]["videos"][video]["synchronized_video"] = str(
                    calibration_object.synchronized_charuco_videofiles[video])
                recording_day["calibrations"]["videos"][video]["framenum_synchronized"] = \
                calibration_object.metadata_from_videos[video].framenum_synchronized

            self.objects["calibration_validation_objects"][
                recording_day["calibrations"]["calibration_key"]
            ].get_marker_predictions(test_mode=test_mode)

            for video in recording_day["calibrations"]["videos"]:
                try:
                    recording_day["calibrations"]["videos"][video][
                        "calibration_validation_marker_detection_filepath"
                    ] = str(
                        self.objects["calibration_validation_objects"][
                            recording_day["calibrations"]["calibration_key"]
                        ].triangulation_dlc_cams_filepaths[video]
                    )
                except:
                    recording_day["calibrations"]["videos"][video][
                        "calibration_validation_marker_detection_filepath"
                    ] = None
        self.meta["meta_step"] = 5
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def calibrate(
            self, p_threshold: float = 0.1, angle_threshold: float = 5., max_iters: int = 5,
            calibrate_optimal: bool = True, verbose: int = 1, test_mode: bool = False
    ) -> None:
        for recording_day in self.meta["recording_days"].values():
            if calibrate_optimal:
                recording_day["calibrations"]["toml_filepath"] = str(
                    self.objects["calibration_objects"][
                        recording_day["calibrations"]["calibration_key"]
                    ].calibrate_optimal(
                        calibration_validation=self.objects["calibration_validation_objects"][
                            recording_day["calibrations"]["calibration_key"]],
                        verbose=verbose,
                        test_mode=test_mode,
                        max_iters=max_iters,
                        p_threshold=p_threshold,
                        angle_threshold=angle_threshold))
                recording_day["calibrations"]['report'] = str(self.objects["calibration_objects"][
                                                                  recording_day["calibrations"][
                                                                      "calibration_key"]].report_filepath)
            else:
                recording_day["calibrations"]["toml_filepath"] = str(self.objects["calibration_objects"][
                    recording_day["calibrations"][
                        "calibration_key"]].run_calibration(
                    verbose=verbose, test_mode=test_mode))
            recording_day["calibrations"]['reprojerr'] = self.objects["calibration_objects"][
                recording_day["calibrations"]["calibration_key"]].reprojerr
        self.meta["meta_step"] = 6
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def triangulate_recordings(self, test_mode: bool = False) -> None:
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                toml_filepath = recording_day['calibrations']['toml_filepath']
                self.objects["triangulation_recordings_objects"][recording].run_triangulation(
                    calibration_toml_filepath=toml_filepath,
                    test_mode=test_mode,
                )
                recording_day["recordings"][recording]["3D_csv"] = str(
                    self.objects["triangulation_recordings_objects"][
                        recording
                    ].csv_output_filepath
                )
                recording_day["recordings"][recording]["reprojerr_mean"] = \
                self.objects["triangulation_recordings_objects"][recording].anipose_io["reproj_nonan"].mean()
        self.meta["meta_step"] = 7
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def exclude_markers(self, all_markers_to_exclude_config_path: Path, verbose: bool = True) -> None:
        all_markers_to_exclude_config_path = convert_to_path(all_markers_to_exclude_config_path)

        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                self.objects["triangulation_recordings_objects"][
                    recording
                ].exclude_markers(all_markers_to_exclude_config_path=all_markers_to_exclude_config_path,
                                  verbose=verbose)

        for recording_day in self.meta["recording_days"].values():
            self.objects['calibration_validation_objects'][
                recording_day['calibrations']['calibration_key']
            ].exclude_markers(all_markers_to_exclude_config_path=all_markers_to_exclude_config_path, verbose=verbose)

    def normalize_recordings(self, normalization_config_path: Path, test_mode: bool = False) -> None:
        normalization_config_path = convert_to_path(normalization_config_path)
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                rotated_filepath, rotation_error = self.objects["triangulation_recordings_objects"][
                    recording
                ].normalize(normalization_config_path=normalization_config_path, test_mode=test_mode)
                recording_day["recordings"][recording]["normalised_3D_csv"] = rotated_filepath
                recording_day["recordings"][recording]["normalisation_rotation_error"] = rotation_error
        self.meta["meta_step"] = 8
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def add_triangulated_csv_to_database(
            self, data_base_path: str, overwrite: bool = True
    ) -> None:
        data_base_path = convert_to_path(data_base_path)
        data_base = pd.read_csv(data_base_path, dtype="str")
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                filename = self.objects["triangulation_recordings_objects"][
                    recording
                ].csv_output_filepath.stem
                if data_base["recording"] == filename and not overwrite:
                    print(
                        f"{filename} was already in test! if you want to add it anyways use overwrite=True!"
                    )

                new_df = pd.DataFrame(
                    {},
                    columns=[
                        "recording",
                        "date",
                        "session_id",
                        "paradigm",
                        "subject_id",
                        "group_id",
                        "batch",
                        "trial_id",
                    ],
                )

                subject_id = self.objects["triangulation_recordings_objects"][
                    recording
                ].mouse_id
                recording_date = self.objects["triangulation_recordings_objects"][
                    recording
                ].recording_date
                paradigm = self.objects["triangulation_recordings_objects"][
                    recording
                ].paradigm

                if overwrite:
                    data_base = data_base[data_base["recording"] != filename]

                new_df.loc[0, ["recording", "date", "paradigm", "subject_id"]] = (
                    filename,
                    recording_date,
                    paradigm,
                    subject_id,
                )
                data_base = pd.concat([data_base, new_df])
        data_base.to_csv(data_base_path, index=False)

    def load_meta_from_yaml(self, filepath: Path) -> None:
        filepath = convert_to_path(filepath)
        with open(filepath, "r") as ymlfile:
            self.meta = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for recording_day in self.meta["recording_days"].values():
            recording_day["num_recordings"] = len(
                recording_day["recording_directories"]
            )

        if self.meta["meta_step"] == 2:
            for recording_day in self.meta["recording_days"].values():
                for recording in recording_day["recordings"]:
                    self.objects["triangulation_recordings_objects"][
                        recording
                    ].target_fps = recording_day["recordings"][recording]["target_fps"]
                    for video_metadata in self.objects[
                        "triangulation_recordings_objects"
                    ][recording].metadata_from_videos.values():
                        video_metadata.fps = recording_day["recordings"][recording][
                            "videos"
                        ][video_metadata.cam_id]["fps"]
                        video_metadata.filepath = recording_day["recordings"][
                            recording
                        ]["videos"][video_metadata.cam_id]["filepath"]

        elif self.meta["meta_step"] == 4:
            for recording_day in self.meta["recording_days"].values():
                self.objects["calibration_objects"][
                    recording_day["calibrations"]["calibration_key"]
                ].target_fps = recording_day["calibrations"]["target_fps"]
                for video_metadata in self.objects["calibration_objects"][
                    recording_day["calibrations"]["calibration_key"]
                ].metadata_from_videos.values():
                    video_metadata.fps = recording_day["calibrations"]["videos"][
                        video_metadata.cam_id
                    ]["fps"]
                    video_metadata.filepath = recording_day["calibrations"][
                        "videos"
                    ][video_metadata.cam_id]["filepath"]
                for video_metadata in self.objects["calibration_validation_objects"][
                    recording_day["calibrations"]["calibration_key"]
                ].metadata_from_videos.values():
                    video_metadata.filepath = recording_day["calibrations"][
                        "videos"
                    ][video_metadata.cam_id]["calibration_validation_image_filepath"]

        """ currently not supported!
        elif self.meta["meta_step"] == 5:
            for recording_day in self.meta["recording_days"].values():
                calibration_index = list(
                    self.meta["recording_days"]["Recording_Day_220922_0"][
                        "calibrations"
                    ]["calibration_keys"].keys()
                )[0]
                full_calibrations = self.objects["calibration_objects"][
                    calibration_index
                ]

                for calibration in recording_day["calibrations"][
                    "calibration_keys"
                ].values():
                    for video_metadata in self.objects["calibration_validation_objects"][
                        calibration
                    ].metadata_from_video.values():
                        video_metadata.filepath = recording_day["calibrations"][
                            "videos"
                        ][video_metadata.cam_id]["calibration_validation_marker_detection_filepath"]"""

    def export_meta_to_yaml(self, filepath: Path) -> None:
        filepath = convert_to_path(filepath)
        with open(filepath, "w") as file:
            yaml.dump(self.meta, file)

    def _read_project_config(self) -> List:
        project_config = read_config(self.project_config_filepath)
        missing_keys = check_keys(project_config, ["paradigms"])
        if missing_keys:
            raise KeyError(
                f"Missing metadata information in the project_config_file "
                f"{self.project_config_filepath} for {missing_keys}."
            )
        return project_config["paradigms"]

    def _read_recording_config(self, recording_config_filepath: Path) -> Tuple[str, str]:
        recording_config = read_config(recording_config_filepath)
        missing_keys = check_keys(
            recording_config, ["recording_date", "calibration_index"]
        )
        if missing_keys:
            raise KeyError(
                f"Missing metadata information in the recording_config_file "
                f"{recording_config_filepath} for {missing_keys}."
            )
        recording_date = str(recording_config["recording_date"])
        if recording_date not in self.recording_dates:
            self.recording_dates.append(recording_config["recording_date"])
        return recording_date, str(recording_config["calibration_index"])

    def _create_video_dict(
            self, video: VideoMetadata, intrinsics: bool = False
    ) -> Dict:
        dictionary = {
            "cam_id": video.cam_id,
            "filepath": str(video.filepath),
            "fps": video.fps,
            "framenum": video.framenum,
            "exclusion_state": video.exclusion_state,
        }
        if intrinsics:
            dictionary["intrinsic_calibration_filepath"] = str(
                video.intrinsic_calibration_filepath
            )
        return dictionary

    def _create_standard_yaml_filepath(self, project_name: str, overwrite: bool):
        if project_name is None:
            project_name = "My_project"
        self.project_name = project_name
        self.standard_yaml_filepath = self.project_config_filepath.parent.joinpath(
            self.project_name + ".yaml"
        )
        while True:
            if self.standard_yaml_filepath.exists() and overwrite is False:
                self.standard_yaml_filepath = (
                    self.project_config_filepath.parent.joinpath(
                        self.standard_yaml_filepath.stem + "_01.yaml"
                    )
                )
            else:
                break
