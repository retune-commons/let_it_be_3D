from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import imageio.v3 as iio

from .video_metadata import VideoMetadata
from .video_interface import VideoInterface
from .utils import convert_to_path



class filename_checking_interface():
    def __init__(
        self, project_config_filepath: Path
    ) -> None:
        self.project_config_filepath = convert_to_path(project_config_filepath)
        if not self.project_config_filepath.exists():
            raise FileNotFoundError("The file doesn't exist. Check your path!")
        self._read_project_config()
        self.recording_configs = []
        self.recording_dates = []

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
            
    def create_recordings(self) -> None:
        self.objects = {"triangulation_recordings_objects": {}}
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
                self.objects["triangulation_recordings_objects"][
                    individual_key
                ] = triangulation_recordings_object
                
     
    
# add:
# create recording_objects, calibration_objects and positions_objects
# plot one image undistorted per day per cam
# print how many recordings were found
# try to read each video
# print if multiple videos were found and let choose the best fitting one
# print if no video (rec, cal, pos) was found 
# rename files in code?
 
# improve:
# implement cam_ids to avoid in project_config and triangulation_calibration_module
# read project_config in triangulation_calibration_module, not in video_metadata
# inspect_intrinsic_calibration in video_interface with plot=False

#test:
# check function! _validate_and_save_metadata_for_recording



                
                
                