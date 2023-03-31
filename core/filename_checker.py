from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from typing import Tuple

import yaml

from .utils import convert_to_path, create_calibration_key
from .checker_objects import CheckRecording, CheckCalibration, CheckCalibrationValidation


class FilenameCheckerInterface:
    def __init__(self, project_config_filepath: Path) -> None:
        self.project_config_filepath = convert_to_path(project_config_filepath)
        if not self.project_config_filepath.exists():
            raise FileNotFoundError("The file doesn't exist. Check your path!")
        self._read_project_config()
        self.recording_configs = []
        self.recording_dates = []
        self.objects = {}
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

    def _read_project_config(self) -> None:
        if self.project_config_filepath.exists():
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

    def _read_recording_config(self, recording_config_filepath: Path) -> Tuple[str, str]:
        if recording_config_filepath.exists():
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

    def initialize_meta_config(self) -> None:
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
                f"Found {recording_day['num_recordings']} recordings at recording day {recording_day['recording_date']}!"
            )

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
        self.objects["check_recordings_objects"] = {}
        for recording_day in self.meta["recording_days"]:
            plot = True
            for recording in self.meta["recording_days"][recording_day][
                "recording_directories"
            ]:
                check_recordings_object = CheckRecording(
                    recording_directory=Path(recording),
                    recording_config_filepath=self.meta["recording_days"][
                        recording_day
                    ]["recording_config_filepath"],
                    project_config_filepath=self.meta["project_config_filepath"],
                    plot=plot,
                )
                individual_key = f"{check_recordings_object.mouse_id}_{check_recordings_object.recording_date}_{check_recordings_object.paradigm}"
                self.objects["check_recordings_objects"][
                    individual_key
                ] = check_recordings_object
                plot = False

    def create_calibrations(self, ground_truth_config_filepath: Path) -> None:
        self.objects["calibration_objects"] = {}
        self.objects["calibration_validation_objects"] = {}
        for recording_day in self.meta["recording_days"].values():
            recording_day["calibrations"]["calibration_keys"] = {}

            calibration_object = CheckCalibration(
                calibration_directory=recording_day["calibration_directory"],
                project_config_filepath=self.project_config_filepath,
                recording_config_filepath=recording_day["recording_config_filepath"],
            )

            cams = [video.cam_id for video in calibration_object.metadata_from_videos]
            all_cams_key = create_calibration_key(
                videos=cams,
                recording_date=calibration_object.recording_date,
                calibration_index=calibration_object.calibration_index,
            )

            self.objects["calibration_objects"][all_cams_key] = calibration_object

            calibration_validation_object = CheckCalibrationValidation(
                calibration_validation_directory=recording_day["calibration_directory"],
                recording_config_filepath=recording_day["recording_config_filepath"],
                project_config_filepath=self.project_config_filepath,
                ground_truth_config_filepath=ground_truth_config_filepath,
            )
            self.objects["calibration_validation_objects"][all_cams_key] = calibration_validation_object
