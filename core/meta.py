from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import time

import yaml
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import imageio as iio

from .utils import convert_to_path, create_calibration_key
from .triangulation_calibration_module import (
    Calibration,
    Triangulation_Positions,
    Triangulation_Recordings,
)
from .video_metadata import VideoMetadata


class meta_interface(ABC):
    def __init__(
        self, project_config_filepath: Path, project_name: Optional[str] = None, overwrite: bool=False
    ) -> None:
        self.project_config_filepath = convert_to_path(project_config_filepath)
        if project_name == None:
            project_name = "My_project"
        self.project_name = project_name
        self.standard_yaml_filepath = self.project_config_filepath.parent.joinpath(
            self.project_name + ".yaml"
        )
        if self.standard_yaml_filepath.exists() and overwrite == False:
            self.standard_yaml_filepath = Path(str(self.standard_yaml_filepath)[:-5] + "_01.yaml")
        if not self.project_config_filepath.exists():
            raise FileNotFoundError("The file doesn't exist. Check your path!")
        self._read_project_config()
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

    def add_recording_config(self, filepath_to_recording_config: Path) -> None:
        filepath_to_recording_config = convert_to_path(filepath_to_recording_config)
        if (
            filepath_to_recording_config.suffix == ".yaml"
            and filepath_to_recording_config.exists()
        ):
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

    def initialize_meta_config(self) -> None:
        for recording_day in self.meta["recording_days"].values():
            for file in Path(
                recording_day["recording_config_filepath"]
            ).parent.parent.parent.glob("**"):
                if (
                    file.name[: len(recording_day["recording_date"])]
                    == recording_day["recording_date"]
                    and file.name[-3:] in self.paradigms
                ):  # hardcoded length of paradigm
                    recording_day["recording_directories"].append(str(file))
            recording_day["num_recordings"] = len(
                recording_day["recording_directories"]
            )
        self.meta["meta_step"] = 1
        self.export_meta_to_yaml(filepath = self.standard_yaml_filepath)

    def add_recording_manually(self, file: Path, recording_day: str) -> None:
        file = convert_to_path(file)
        if not file.exists() or recording_day not in self.meta["recording_days"].keys():
            print(
                f"couldn't add recording directory! \nCheck your filepath and make sure the recording_day is in {self.meta['recording_days'].keys()}!"
            )
        else:
            self.meta["recording_days"][recording_day]["recording_directories"].append(
                str(file)
            )
            self.meta["recording_days"][recording_day]["num_recordings"] = len(
                recording_day["recording_directories"]
            )
            print("added recording directory succesfully!")

    def create_recordings(self) -> None:
        self.objects = {"triangulation_recordings_objects": {}}
        # optional: create output_directories?
        for recording_day in self.meta["recording_days"]:
            for recording in self.meta["recording_days"][recording_day][
                "recording_directories"
            ]:
                triangulation_recordings_object = Triangulation_Recordings(
                    recording_directory=Path(recording),
                    calibration_directory=self.meta["recording_days"][recording_day][
                        "calibration_directory"
                    ],
                    recording_config_filepath=self.meta["recording_days"][
                        recording_day
                    ]["recording_config_filepath"],
                    project_config_filepath=self.meta["project_config_filepath"],
                    output_directory=recording,
                )
                individual_key = f"{triangulation_recordings_object.mouse_id}_{triangulation_recordings_object.recording_date}_{triangulation_recordings_object.paradigm}"
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
                    "videos": videos,
                }
                self.objects["triangulation_recordings_objects"][
                    individual_key
                ] = triangulation_recordings_object
        self.meta["meta_step"] = 2
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def synchronize_recordings(self, verbose: bool = False, test_mode: bool = False, synchronize_only: bool = False) -> None:
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                if verbose:
                    start_time_recording = time.time()
                self.objects["triangulation_recordings_objects"][
                    recording
                ].run_synchronization(test_mode=test_mode, synchronize_only = synchronize_only)
                for video in recording_day["recordings"][recording]["videos"]:
                    try:
                        recording_day["recordings"][recording]["videos"][video][
                            "synchronized_video"
                        ] = str(
                            self.objects["triangulation_recordings_objects"][
                                recording
                            ].synchronized_videos[video]
                        )
                        recording_day["recordings"][recording]["videos"][video][
                            "framenum_synchronized"
                        ] = (
                            self.objects["triangulation_recordings_objects"][recording]
                            .metadata_from_videos[video]
                            .framenum_synchronized
                        )
                        recording_day["recordings"][recording]["videos"][video][
                            "marker_detection_filepath"
                        ] = str(
                            self.objects["triangulation_recordings_objects"][
                                recording
                            ].triangulation_dlc_cams_filepaths[video]
                        )
                    except:
                        pass
                if verbose:
                    end_time_recording = time.time()
                    duration = start_time_recording - end_time_recording
                    print(f"The analysis of this recording {recording} took {duration}.\n")
                    
        self.meta["meta_step"] = 3
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def create_calibrations(self) -> None:
        self.objects["calibration_objects"] = {}
        self.objects["position_objects"] = {}
        self.necessary_calibrations = {}
        for recording_day in self.meta["recording_days"].values():

            recording_day["calibrations"]["calibration_keys"] = {}

            calibration_object = Calibration(
                calibration_directory=recording_day["calibration_directory"],
                project_config_filepath=self.project_config_filepath,
                recording_config_filepath=recording_day["recording_config_filepath"],
                output_directory=recording_day["calibration_directory"],
            )

            cams = [video for video in calibration_object.metadata_from_videos]

            necessary_calibrations, all_cams_key = self._get_necessary_calibrations(
                possible_videos=cams, recording_day=recording_day
            )
            necessary_calibrations.pop(all_cams_key)

            self.objects["calibration_objects"][all_cams_key] = calibration_object

            video_dict = {
                video: self._create_video_dict(
                    calibration_object.metadata_from_videos[video], intrinsics=True
                )
                for video in calibration_object.metadata_from_videos
            }
            recording_day["calibrations"]["calibration_keys"][all_cams_key] = {
                "key": all_cams_key
            }
            recording_day["calibrations"]["target_fps"] = calibration_object.target_fps
            recording_day["calibrations"][
                "led_pattern"
            ] = calibration_object.led_pattern
            recording_day["calibrations"]["videos"] = video_dict
            self.necessary_calibrations[all_cams_key] = necessary_calibrations

            positions_object = Triangulation_Positions(
                positions_directory=recording_day["calibration_directory"],
                calibration_directory=recording_day["calibration_directory"],
                recording_config_filepath=recording_day["recording_config_filepath"],
                project_config_filepath=self.project_config_filepath,
                output_directory=recording_day["calibration_directory"],
            )
            self.objects["position_objects"][all_cams_key] = positions_object
            for video in positions_object.metadata_from_videos.values():
                try:
                    recording_day["calibrations"]["videos"][video.cam_id][
                        "positions_image_filepath"
                    ] = str(video.filepath)
                except:
                    pass
        self.meta["meta_step"] = 4
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def synchronize_calibrations(self, test_mode: bool = False) -> None:
        for recording_day in self.meta["recording_days"].values():
            for calibration in recording_day["calibrations"][
                "calibration_keys"
            ].values():
                calibration_object = self.objects["calibration_objects"][
                    calibration["key"]
                ]
                calibration_object.run_synchronization(test_mode=test_mode)
                for video in recording_day["calibrations"]["videos"]:
                    recording_day["calibrations"]["videos"][video][
                        "synchronized_video"
                    ] = str(calibration_object.synchronized_charuco_videofiles[video])
                    recording_day["calibrations"]["videos"][video][
                        "framenum_synchronized"
                    ] = calibration_object.metadata_from_videos[
                        video
                    ].framenum_synchronized
                self._create_subgroups_for_necessary_calibrations(
                    necessary_calibrations=self.necessary_calibrations[
                        calibration["key"]
                    ],
                    calibration_object=calibration_object,
                    calibrations=recording_day["calibrations"]["calibration_keys"],
                )

                self.objects["position_objects"][
                    calibration["key"]
                ].get_marker_predictions()
                for video in recording_day["calibrations"]["videos"]:
                    recording_day["calibrations"]["videos"][video][
                        "positions_marker_detection_filepath"
                    ] = str(
                        self.objects["position_objects"][
                            calibration["key"]
                        ].triangulation_dlc_cams_filepaths[video]
                    )
        self.meta["meta_step"] = 5
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def calibrate(self, verbose: bool = False) -> None:
        for recording_day in self.meta["recording_days"].values():
            for calibration in recording_day["calibrations"][
                "calibration_keys"
            ].values():
                self.objects["calibration_objects"][calibration["key"]].run_calibration(
                    verbose=verbose
                )
                calibration["toml_filepath"] = str(
                    self.objects["calibration_objects"][
                        calibration["key"]
                    ].calibration_output_filepath
                )
                # add calibration log? reprojerr, etc.
        self.meta["meta_step"] = 6
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def triangulate_recordings(self) -> None:
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                toml_filepath = recording_day["calibrations"]["calibration_keys"][
                    recording_day["recordings"][recording]["calibration_to_use"]
                ]["toml_filepath"]
                self.objects["triangulation_recordings_objects"][
                    recording
                ].run_triangulation(
                    calibration_toml_filepath=toml_filepath, adapt_to_calibration=True
                )
                recording_day["recordings"][recording]["3D_csv"] = str(
                    self.objects["triangulation_recordings_objects"][
                        recording
                    ].csv_output_filepath
                )
                # add reprojection_error
        self.meta["meta_step"] = 7
        self.export_meta_to_yaml(self.standard_yaml_filepath)

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
                for calibration in recording_day["calibrations"][
                    "calibration_keys"
                ].values():
                    self.objects["calibration_objects"][
                        calibration["key"]
                    ].target_fps = recording_day["calibrations"]["target_fps"]
                    for video_metadata in self.objects["calibration_objects"][
                        calibration["key"]
                    ].metadata_from_videos.values():
                        video_metadata.fps = recording_day["calibrations"]["videos"][
                            video_metadata.cam_id
                        ]["fps"]
                        video_metadata.filepath = recording_day["calibrations"][
                            "videos"
                        ][video_metadata.cam_id]["filepath"]
                    for video_metadata in self.objects["position_objects"][
                        calibration["key"]
                    ].metadata_from_videos.values():
                        video_metadata.filepath = recording_day["calibrations"][
                            "videos"
                        ][video_metadata.cam_id]["positions_image_filepath"]

        elif self.meta["meta_step"] == 5:
            for recording_day in self.meta["recording_days"].values():
                videos = [
                    video
                    for video in recording_day["calibrations"]["videos"].keys()
                    if recording_day["calibrations"]["videos"][video]["exclusion_state"]
                    == "valid"
                ]
                all_videos = list(recording_day["calibrations"]["videos"].keys())
                calibration_key = create_calibration_key(
                    videos=videos,
                    recording_date=recording_day["recording_date"],
                    calibration_index=recording_day["calibration_index"],
                )
                all_calibrations_key = create_calibration_key(
                    videos=all_videos,
                    recording_date=recording_day["recording_date"],
                    calibration_index=recording_day["calibration_index"],
                )
                full_calibrations = self.objects["calibration_objects"][
                    all_calibrations_key
                ]

                if not calibration_key == all_calibrations_key:
                    calibration_object = full_calibrations.create_subgroup(
                        cam_ids=videos
                    )
                    self.objects["calibration_objects"][
                        calibration_key
                    ] = calibration_object
                    recording_day["calibrations"]["calibration_keys"][
                        calibration_key
                    ] = {"key": calibration_key}
                    necessary_calibrations, _ = self._get_necessary_calibrations(
                        possible_videos=videos, recording_day=recording_day
                    )
                    self._create_subgroups_for_necessary_calibrations(
                        necessary_calibrations=necessary_calibrations,
                        calibration_object=calibration_object,
                        calibrations=recording_day["calibrations"]["calibration_keys"],
                    )
                    self.objects["calibration_objects"].pop(all_calibrations_key)
                    recording_day["calibrations"]["calibration_keys"].pop(
                        all_calibrations_key
                    )

                for calibration in recording_day["calibrations"][
                    "calibration_keys"
                ].values():
                    for video_metadata in self.objects["position_objects"][
                        calibration["key"]
                    ].metadata_from_video.values():
                        video_metadata.filepath = recording_day["calibrations"][
                            "videos"
                        ][video_metadata.cam_id]["positions_marker_detection_filepath"]

    def export_meta_to_yaml(self, filepath: Path) -> None:
        filepath = convert_to_path(filepath)
        with open(filepath, "w") as file:
            yaml.dump(self.meta, file)

    def _read_project_config(self) -> None:
        with open(self.project_config_filepath, "r") as ymlfile:
            project_config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for key in [
            "paradigms",
        ]:
            try:
                project_config[key]
            except KeyError:
                raise KeyError(
                    f"Missing metadata information in the project_config_file {self.project_config_filepath} for {key}."
                )
        self.paradigms = project_config["paradigms"]

    def _read_recording_config(self, recording_config_filepath: Path) -> str:
        with open(recording_config_filepath, "r") as ymlfile:
            recording_config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for key in ["recording_date"]:
            try:
                recording_config[key]
            except KeyError:
                raise KeyError(
                    f"Missing metadata information in the recording_config_file {recording_config_filepath} for {key}."
                )
        self.recording_dates.append(recording_config["recording_date"])
        return str(recording_config["recording_date"]), str(
            recording_config["calibration_index"]
        )

    def _get_necessary_calibrations(
        self, possible_videos: List[str], recording_day: Dict
    ) -> Tuple[List, str]:
        necessary_calibrations = {}
        for recording in recording_day["recordings"]:
            videos = [
                video
                for video in recording_day["recordings"][recording]["videos"].keys()
                if recording_day["recordings"][recording]["videos"][video][
                    "exclusion_state"
                ]
                == "valid"
                and video in possible_videos
            ]
            calibration_key = create_calibration_key(
                videos=videos,
                recording_date=recording_day["recording_date"],
                calibration_index=recording_day["calibration_index"],
            )
            recording_day["recordings"][recording][
                "calibration_to_use"
            ] = calibration_key
            if not calibration_key in necessary_calibrations.keys():
                necessary_calibrations[calibration_key] = videos
        all_cams_key = create_calibration_key(
            videos=videos,
            recording_date=recording_day["recording_date"],
            calibration_index=recording_day["calibration_index"],
        )
        return necessary_calibrations, all_cams_key

    def _create_subgroups_for_necessary_calibrations(
        self,
        necessary_calibrations: Dict,
        calibration_object: Calibration,
        calibrations: Dict,
    ) -> None:
        for key in necessary_calibrations:
            calibration_subgroup = calibration_object.create_subgroup(
                cam_ids=necessary_calibrations[key]
            )

            cams = [video for video in calibration_subgroup.metadata_from_videos]
            calibration_key = create_calibration_key(
                videos=cams,
                recording_date=subgroup.recording_date,
                calibration_index=subgroup.calibration_index,
            )
            self.objects["calibration_objects"][calibration_key] = calibration_subgroup
            calibrations[calibration_key] = {"key": calibration_key}

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
