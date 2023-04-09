from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from typing import Tuple, Union, List

import yaml

from .checker_objects import CheckRecording, CheckCalibration, CheckCalibrationValidation
from .utils import convert_to_path, create_calibration_key


class FilenameCheckerInterface:
    """
        Interface to load all files and check filename and metadata.

        Parameters
        ----------
        project_config_filepath: Path or str
            Filepath to the project_config .yaml file.

        Attributes
        __________
        objects: dict
            Dictionary of all objects added to the FilenameCheckerInterface.
        project_config_filepath: Path
            Filepath to the project_config .yaml file.
        paradigms: list of str
            List of all paradigms to search for in directories.
        recording_configs: list of Path
            List of all recording_configs added to the FilenameCheckerInterface.
        recording_dates: list of str
            List of all recording_dates to search for in directories.
        meta: dict
            Dictionary of metadata of all objects added to the
            FilenameCheckerInterface.

        Methods
        _______
        select_recording_configs():
            Open a window to select recording_config files in filedialog.
        add_recording_config(filepath_to_recording_config)
            Add recording_config file via method.
        initialize_meta_config():
            Append all directories to metadata, that match appended
            paradigms and recording_dates in directory name.
        add_recording_manually(file, recording_day):
            Adds recordings to metadata that don't match directory name structure.
        create_recordings():
            Create CheckRecording objects for all recording_directories
            added to FilenameCheckerInterface.
        create_calibrations(ground_truth_config_filepath):
            Create CheckCalibration and CheckCalibrationValidation objects for
            all calibration_directories added to FilenameCheckerInterface.

        See Also
        ________
        core.meta.MetaInterface:
            Interface to load all files and run analysis.
        core.checker_objects.CheckRecording:
            A class, that checks the metadata and filenames of videos in a given
            folder and allows for filename changing via user input.
        core.checker_objects.CheckCalibration:
            A class, that checks the metadata and filenames of videos in a given
            folder and allows for filename changing via user input.
        core.checker_objects.CheckCalibrationValidation:
            A class, that checks the metadata and filenames of videos in a given
            folder and allows for filename changing via user input.

        Examples
        ________
        >>> from core.filename_checker import FilenameCheckerInterface
        >>> filename_checker = FilenameCheckerInterface(project_config_filepath="test_data/project_config.yaml")
        >>> filename_checker.add_recording_config("test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml")
        >>> filename_checker.initialize_meta_config()
        >>> filename_checker.create_recordings()
        >>> filename_checker.create_calibrations(ground_truth_config_filepath="test_data/ground_truth_config.yaml")
        """
    def __init__(self, project_config_filepath: Union[Path, str]) -> None:
        """
        Construct all necessary attributes for the FilenameCheckerInterface
        class.

        Parameters
        ----------
        project_config_filepath: Path or str
            Filepath to the project_config .yaml file.
        """
        self.project_config_filepath = convert_to_path(project_config_filepath)
        if not self.project_config_filepath.exists():
            raise FileNotFoundError("The file doesn't exist. Check your path!")
        self.paradigms = self._read_project_config()
        self.recording_configs = []
        self.recording_dates = []
        self.objects = {}
        self.meta = {
            "project_config_filepath": str(self.project_config_filepath),
            "recording_days": {},
        }

    def select_recording_configs(self) -> None:
        """
        Open a window to select recording_config files in filedialog.

        Add it to recording_configs.
        """
        Tk().withdraw()
        selected_recording_configs = askopenfilenames(
            title="Select recording_config.yaml"
        )

        for filepath_to_recording_config in selected_recording_configs:
            self.add_recording_config(
                filepath_to_recording_config=filepath_to_recording_config
            )

    def add_recording_config(self, filepath_to_recording_config: Union[Path, str]) -> None:
        """
        Add recording_config via method.

        Parameters
        ----------
        filepath_to_recording_config: Path or str
            The path to the recording_config, that should be added to the
            MetaInterface.
        """
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
        """
        Append all directories to metadata, that match appended
        paradigms and recording_dates in directory name.

        See Also
        ________
        FilenameCheckerInterface.add_recording_manually:
            Adds recordings to metadata that don't match directory name structure.

        Notes
        _____
        Demands for adding directories automatically:
            - recording directory name has to start with a recording date
            (YYMMDD) that is added to the FilenameCheckerInterface
            - recording directory name has to end with any of the paradigms (as
            defined in project_config)
        If you want to add recording directories, that don't match this structure,
        use FilenameCheckerInterface.add_recording_manually instead.
        """
        for recording_day in self.meta["recording_days"].values():
            parents = Path(recording_day["recording_config_filepath"]).parents
            for file in parents[len(parents) - 1].glob("**"):
                if file.name.startswith(recording_day["recording_date"]) and any(
                        [file.stem.endswith(paradigm) for paradigm in self.paradigms]):
                    recording_day["recording_directories"].append(str(file))
            recording_day["num_recordings"] = len(
                recording_day["recording_directories"]
            )
            print(
                f"Found {recording_day['num_recordings']} recordings at recording day {recording_day['recording_date']}!"
            )

    def add_recording_manually(self, file: Path, recording_day: str) -> None:
        """
        Adds recordings to metadata that don't match directory name structure.

        Parameters
        ----------
        file: Path or str
            The path to the recording directory, that should be added.
        recording_day: str
            The date of the recording.
        """
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
        """
        Create CheckRecording objects for all recording_directories
        added to FilenameCheckerInterface.
        """
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
                individual_key = f"{check_recordings_object.mouse_id}_" \
                                 f"{check_recordings_object.recording_date}_" \
                                 f"{check_recordings_object.paradigm}"
                self.objects["check_recordings_objects"][
                    individual_key
                ] = check_recordings_object
                plot = False

    def create_calibrations(self, ground_truth_config_filepath: Path) -> None:
        """
        Create CheckCalibration and CheckCalibrationValidation objects for
        all calibration_directories added to FilenameCheckerInterface.

        Parameters
        ----------
        ground_truth_config_filepath
            The path to the ground_truth config file.
        """
        self.objects["check_calibration_objects"] = {}
        self.objects["check_calibration_validation_objects"] = {}
        for recording_day in self.meta["recording_days"].values():
            recording_day["calibrations"]["calibration_keys"] = {}

            calibration_object = CheckCalibration(
                calibration_directory=recording_day["calibration_directory"],
                project_config_filepath=self.project_config_filepath,
                recording_config_filepath=recording_day["recording_config_filepath"],
            )

            unique_calibration_key = f'{calibration_object.recording_date}_' \
                                     f'{calibration_object.calibration_index}'

            self.objects["check_calibration_objects"][unique_calibration_key] = calibration_object

            check_calibration_validation_object = CheckCalibrationValidation(
                calibration_validation_directory=recording_day["calibration_directory"],
                recording_config_filepath=recording_day["recording_config_filepath"],
                project_config_filepath=self.project_config_filepath,
                ground_truth_config_filepath=ground_truth_config_filepath,
            )
            self.objects["check_calibration_validation_objects"][unique_calibration_key] = check_calibration_validation_object

    def _read_project_config(self) -> List[str]:
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
        return project_config["paradigms"]

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
        return str(recording_config["recording_date"]), str(recording_config["calibration_index"])
