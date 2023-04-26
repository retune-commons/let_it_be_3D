import time
from abc import ABC
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from typing import Tuple, Optional, Dict, Union, List

import numpy as np
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
    """
    Interface to load all files and run analysis.

    Run (optimised) calibrations, triangulation of recordings, create Database
    and save/load the whole project to/from meta .yaml-file.

    Parameters
    ----------
    project_config_filepath: Path or str
        Filepath to the project_config .yaml file.
    project_name: str, optional
        The name of the meta .yaml-file path.
    overwrite: bool, default False
        If True (default False), then the meta.yaml will be overwritten if
        already existing.

    Attributes
    __________
    objects: dict
        Dictionary of all objects added to the MetaInterface.
    project_config_filepath: Path
        Filepath to the project_config .yaml file.
    paradigms: list of str
        List of all paradigms to search for in directories.
    recording_configs: list of Path
        List of all recording_configs added to the MetaInterface.
    recording_dates: list of str
        List of all recording_dates to search for in directories.
    meta: dict
        Dictionary of metadata of all objects added to the MetaInterface.
    project_name: str
        The name of the meta .yaml-file. Default is 'My_project'.
    standard_yaml_filepath: Path
        The filepath to the meta .yaml-file. Stored in the same directory as
        the project_config.

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
    remove_recordings():
        Remove recordings from analysis via user input dialog.
    create_recordings(recreate_undistorted_plots):
        Create TriangulationRecording objects for all recording_directories
        added to MetaInterface.
    synchronize_recordings(verbose, overwrite_DLC_analysis_and_synchro):
        Run the function run_synchronization for all TriangulationRecording
        objects added to MetaInterface.
    create_calibrations(ground_truth_config_filepath, recreate_undistorted_plots):
        Create Calibration and CalibrationValidation objects and add
        ground_truth_config for all calibration_directories added to
        MetaInterface.
    synchronize_calibrations(overwrite_synchronisations_and_calvin_predictions):
        Run get_marker_predictions for all calibration_validation objects and
        run_synchronization for all calibration objects added to MetaInterface.
    calibrate(p_threshold, angle_threshold, max_iters, calibrate_optimal, verbose, overwrite_calibrations):
        Run the function run_calibration or calibrate_optimal for all
        calibration objects added to MetaInterface.
    triangulate_recordings(triangulate_full_recording):
        Run the function run_triangulation for all TriangulationRecording
        objects added to MetaInterface.
    exclude_markers(all_markers_to_exclude_config_path, verbose):
        Run the function exclude_marker for all TriangulationRecordings and
        CalibrationValidation objects added to MetaInterface.
    normalize_recordings(normalization_config_path, save_dataframe):
        Run the function normalize for all TriangulationRecordings objects and
        saves the normalisation metadata.
    add_triangulated_csv_to_database(data_base_path, overwrite):
        Add the 3D dataframes to a common data_base.
    export_meta_to_yaml(filepath):
        Store MetaInterface objects as .yaml-file.

    See Also
    ________
    core.filename_checker.FilenameCheckerInterface:
        Interface to load all files and check filename and metadata.
    core.triangulation_calibration_module.TriangulationRecordings:
        A class, in which videos are triangulated based on a calibration file.
    core.triangulation_calibration_module.CalibrationValidation:
        A class, in which images are triangulated based on a calibration file
        and the triangulated coordinates are validated based on a ground_truth.
    core.triangulation_calibration_module.Calibration:
        A class, in which videos are calibrated to each other.

    Examples
    ________
    >>> from core.meta import MetaInterface
    >>> meta_interface = MetaInterface(
        ... project_config_filepath="test_data/project_config.yaml",
        ... project_name="test_data", overwrite=False)
    >>> meta_interface.add_recording_config("test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml")
    >>> meta_interface.initialize_meta_config()
    >>> meta_interface.create_recordings()
    >>> meta_interface.synchronize_recordings(verbose=True)
    >>> meta_interface.create_calibrations(ground_truth_config_filepath="test_data/ground_truth_config_only_corners.yaml")
    >>> meta_interface.synchronize_calibrations()
    >>> meta_interface.exclude_markers(all_markers_to_exclude_config_path = "test_data/markers_to_exclude_config.yaml", verbose=False)
    >>> meta_interface.calibrate(calibrate_optimal=True, verbose=2)
    >>> meta_interface.triangulate_recordings()
    >>> meta_interface.normalize_recordings(normalization_config_path="test_data/normalization_config.yaml")
    """
    def __init__(
            self,
            project_config_filepath: Union[Path, str],
            project_name: Optional[str] = None,
            overwrite: bool = False,
    ) -> None:
        """
        Construct all necessary attributes for the MetaInterface class.

        Parameters
        ----------
        project_config_filepath: Path or str
            Filepath to the project_config .yaml file.
        project_name: str, optional
            The filename for the meta .yaml-file.
        overwrite: bool, default False
            If True (default False), then the meta .yaml-file will be
            overwritten if already existing.
        """
        self.objects = {}
        self.project_config_filepath = convert_to_path(project_config_filepath)
        self.project_name, self.standard_yaml_filepath = self._create_standard_yaml_filepath(
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
        """
        Open a window to select recording_config files in filedialog.

        Add it to recording_configs.
        """
        Tk().withdraw()
        selected_recording_configs = askopenfilenames(
            title="Select recording_config.yaml"
        )

        for filepath_to_recording_config in selected_recording_configs:
            self.add_recording_config(filepath_to_recording_config=filepath_to_recording_config)

    def add_recording_config(self, filepath_to_recording_config: Union[str, Path]) -> None:
        """
        Add recording_config via method.

        Parameters
        ----------
        filepath_to_recording_config: Path or str
            The path to the recording_config, that should be added to the
            MetaInterface.

        Raises
        ______
        FileNotFoundError:
            If the path is not linked to a .yaml file or doesn't exist.
        """
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

    def initialize_meta_config(self, num_recording_config_parents: int = 2) -> None:
        """
        Append all directories to metadata, that match appended
        paradigms and recording_dates in directory name.
        
        Parameters:
        ___________
        num_recording_config_parents: int, default 2
            The number of levels above the recording config file to look for matching
            recording directories.

        See Also
        ________
        MetaInterface.add_recording_manually:
            Adds recordings to metadata that don't match directory name structure.

        Notes
        _____
        Demands for adding directories automatically:
            - recording directory name has to start with a recording date
            (YYMMDD) that is added to the MetaInterface
            - recording directory name has to end with any of the paradigms (as
            defined in project_config)
        If you want to add recording directories, that don't match this structure,
        use MetaInterface.add_recording_manually.
        """
        for recording_day in self.meta["recording_days"].values():
            recording_config_parents = Path(recording_day["recording_config_filepath"]).parents
            if num_recording_config_parents >= len(recording_config_parents):
                num_recording_config_parents = len(recording_config_parents)-1
            for file in recording_config_parents[num_recording_config_parents].glob("**"):
                if file.name.startswith(recording_day["recording_date"]) and any(
                        [file.stem.endswith(paradigm) for paradigm in self.paradigms]):
                    recording_day["recording_directories"].append(str(file))
            recording_day["num_recordings"] = len(
                recording_day["recording_directories"]
            )
            print(
                f"\nFound {recording_day['num_recordings']} recordings at "
                f"recording day {recording_day['recording_date']}!"
            )
        self.meta["meta_step"] = 1
        self.export_meta_to_yaml(filepath=self.standard_yaml_filepath)

    def add_recording_manually(self, file: Union[Path, str], recording_day: str) -> None:
        """
        Adds recordings to metadata that don't match directory name structure.

        Parameters
        ----------
        file: Path or str
            The path to the recording directory, that should be added.
        recording_day: str
            The date of the recording.

        Raises
        ______
        FileNotFoundError:
            If the path is no directory or if there's no recording_config added
            for the recording_day.
        """
        file = convert_to_path(file)
        if not file.is_dir() or recording_day not in self.meta["recording_days"].keys():
            raise FileNotFoundError(
                f"Couldn't add recording directory! \n"
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
            
    def remove_recordings(self) -> None:
        """
        Remove recordings from analysis via user input dialog.
        """
        for recording_day in self.meta["recording_days"].values():
            print(f"\n{recording_day['recording_date']}:\n")
            recordings_to_remove = []
            for recording_dir in recording_day["recording_directories"]:
                print(recording_dir)
                remove = input("Remove from analysis: y, keep: n, skip recording_day: x")
                if remove == "y":
                    recordings_to_remove.append(recording_dir)
                elif remove == "n":
                    pass
                elif remove == "x":
                    break
                else:
                    raise ValueError("Invalid input! Please enter 'y', 'n' or 'x'!")
            for recording_dir in recordings_to_remove:
                recording_day["recording_directories"].remove(recording_dir)
                recording_day['num_recordings'] -= 1
            print(
                    f"\nFound {recording_day['num_recordings']} recordings at "
                    f"recording day {recording_day['recording_date']}!"
                )

    def create_recordings(self, recreate_undistorted_plots: bool = True, specify_calibration_to_use: bool = False) -> None:
        """
        Create TriangulationRecording objects for all recording_directories
        added to MetaInterface.

        Parameters
        ----------
        recreate_undistorted_plots
            If True (default), then preexisting undistorted plots will be overwritten.
        specify_calibration_to_use: bool, default False
            If True (default False), then you will be asked to specify the 
            calibration index for each recording.
        """
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
                    recreate_undistorted_plots=recreate_undistorted_plots,
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
                if specify_calibration_to_use:
                    print(f"{individual_key}:") 
                    calibration_index = input("Specify the calibration_index, you want to use!")
                    triangulation_recordings_object.calibration_index = calibration_index
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
            verbose: bool = True,
            overwrite_DLC_analysis_and_synchro: bool = False,
    ) -> None:
        """
        Run the function run_synchronization for all TriangulationRecording
        objects added to MetaInterface.

        Parameters
        ----------
        verbose: bool, default True:
            If True (default), then the duration of a analysis is printed and
            the attribute is passed to the TriangulationRecordings objects.
        overwrite_DLC_analysis_and_synchro: bool, default False
            If True (default False), then pre-existing DLC files and
            synchronisations will be overwritten during analysis.
        """
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                if verbose: 
                    start_time_recording = time.time()
                    print(f"\nNow analysing {recording}!")
                recording_object = self.objects["triangulation_recordings_objects"][recording]
                recording_meta = recording_day["recordings"][recording]
                recording_object.run_synchronization(
                    overwrite_DLC_analysis_and_synchro=overwrite_DLC_analysis_and_synchro, verbose=verbose
                )
                for video in recording_meta["videos"]:
                    try:
                        recording_meta["videos"][video]["framenum_synchronized"] = int(
                            recording_object.metadata_from_videos[video].framenum_synchronized
                        )
                        recording_meta["videos"][video]["exclusion_state"] = str(
                            recording_object.metadata_from_videos[video].exclusion_state)
                        recording_meta["videos"][video]["marker_detection_filepath"] = str(
                            recording_object.triangulation_dlc_cams_filepaths[video])
                    except:
                        print(f"Synchronization metadata could not be added for {video}!")
                if verbose:
                    end_time_recording = time.time()
                    duration = end_time_recording - start_time_recording
                    print(
                        f"The analysis of this recording {recording} took {duration}."
                    )

        self.meta["meta_step"] = 3
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def create_calibrations(
            self, ground_truth_config_filepath: Union[Path or str], recreate_undistorted_plots: bool = True
    ) -> None:
        """
        Create Calibration and CalibrationValidation objects and add
        ground_truth_config for all calibration_directories added to
        MetaInterface.

        Parameters
        ----------
        ground_truth_config_filepath: Path or str
            The path to the ground_truth config file.
        recreate_undistorted_plots
            If True (default), then preexisting undistorted plots will be overwritten.
        """
        self.objects["calibration_objects"] = {}
        self.objects["calibration_validation_objects"] = {}
        for recording_day in self.meta["recording_days"].values():
            calibration_object = Calibration(
                calibration_directory=recording_day["calibration_directory"],
                project_config_filepath=self.project_config_filepath,
                recording_config_filepath=recording_day["recording_config_filepath"],
                output_directory=recording_day["calibration_directory"],
                recreate_undistorted_plots=recreate_undistorted_plots,
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
                recreate_undistorted_plots=recreate_undistorted_plots)
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

    def synchronize_calibrations(self, overwrite_synchronisations_and_calvin_predictions: bool = False, verbose: bool = True) -> None:
        """
        Run get_marker_predictions for all calibration_validation objects and
        run_synchronization for all calibration objects added to MetaInterface.

        Parameters
        ----------
        overwrite_synchronisations_and_calvin_predictions: bool, default False
            If True (default False), then pre-existing synchronisations and
            calvin predictions will be overwritten during analysis.
        verbose: bool, default True
            If True (default), then the attribute is passed to the Calibration objects.
        """
        for recording_day in self.meta["recording_days"].values():
            if verbose:
                print(f'\nNow analysing {recording_day["calibrations"]["calibration_key"]}!')
            calibration_object = self.objects["calibration_objects"][recording_day["calibrations"]["calibration_key"]]
            calibration_object.run_synchronization(overwrite_synchronisations=overwrite_synchronisations_and_calvin_predictions, verbose=verbose)
            for video in recording_day["calibrations"]["videos"]:
                if video in calibration_object.synchronized_charuco_videofiles:
                    recording_day["calibrations"]["videos"][video]["synchronized_video"] = str(
                        calibration_object.synchronized_charuco_videofiles[video])
                    recording_day["calibrations"]["videos"][video]["exclusion_state"] = str(
                            calibration_object.metadata_from_videos[video].exclusion_state)
                    recording_day["calibrations"]["videos"][video]["framenum_synchronized"] = int(
                        calibration_object.metadata_from_videos[video].framenum_synchronized)
            
            recording_day["calibrations"]["cams_to_exclude"] = str(calibration_object.cams_to_exclude)
            self.objects["calibration_validation_objects"][
                recording_day["calibrations"]["calibration_key"]
            ].get_marker_predictions(overwrite_analysed_markers=overwrite_synchronisations_and_calvin_predictions)

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

    def exclude_markers(self, all_markers_to_exclude_config_path: Union[Path, str], verbose: bool = True) -> None:
        """
        Run the function exclude_marker for all TriangulationRecordings and
        CalibrationValidation objects added to MetaInterface.

        Parameters
        ----------
        all_markers_to_exclude_config_path: Path or str
            Filepath to the config used for exclusion of markers.
        verbose: bool, default True
            If True (default), print if exclusion of markers worked without any
            abnormalities.
        """
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

    def calibrate(
            self, p_threshold: float = 0.1, angle_threshold: float = 5., max_iters: int = 5,
            calibrate_optimal: bool = True, verbose: int = 1, overwrite_calibrations: bool = False
    ) -> None:
        """
        Run the function run_calibration or calibrate_optimal for all
        calibration objects added to MetaInterface.

        Parameters
        ----------
        p_threshold: float, default 0.1
            Threshold for errors in the triangulated distances compared to
            ground truth (mean distances in percent). Won't be used if
            calibrate_optimal is False.
        angle_threshold: float, default 5
            Threshold for errors in the triangulated angles compared to ground
            truth (mean angles in degrees). Won't be used if calibrate_optimal
            is False.
        max_iters: int, default 5
            Number of iterations allowed to find a good calibration. Won't be
            used if calibrate_optimal is False.
        calibrate_optimal: bool, default True
            If True (default), then calibrate_optimal will be run for all
            calibration objects added to MetaInterface. If False, then
            run_calibration will be run.
        verbose: int, default 1
            Show ap_lib output if > 1,
            calibration_validation output if > 0
            or no output if < 1.
        overwrite_calibrations: bool, default False
            If True (default False), then pre-existing calibrations will be overwritten.
        """
        for recording_day in self.meta["recording_days"].values():
            if verbose:
                print(f"\nNow analysing {recording_day['calibrations']['calibration_key']}!")
            if calibrate_optimal:
                recording_day["calibrations"]["toml_filepath"] = str(
                    self.objects["calibration_objects"][
                        recording_day["calibrations"]["calibration_key"]
                    ].calibrate_optimal(
                        calibration_validation=self.objects["calibration_validation_objects"][
                            recording_day["calibrations"]["calibration_key"]],
                        verbose=verbose,
                        overwrite_calibrations=overwrite_calibrations,
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
                    verbose=verbose, overwrite_calibrations=overwrite_calibrations))
            recording_day["calibrations"]['reprojerr'] = self.objects["calibration_objects"][
                recording_day["calibrations"]["calibration_key"]].reprojerr
        self.meta["meta_step"] = 6
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def triangulate_recordings(self, triangulate_full_recording: bool = True, verbose: bool = True) -> None:
        """
        Run the function run_triangulation for all TriangulationRecording
        objects added to MetaInterface.

        Parameters
        ----------
        triangulate_full_recording: bool, default True
            If False (default True), then only the first 2 frames of the
            recording will be triangulated and the 3D dataframe won't be saved.
        verbose: bool, default True
            If True (default), then the recording, that is currently analysed, and the
            duration of an analysis will be printed.
        """
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                if verbose:
                    print(f"\nNow analysing {recording}!")
                    start_time_recording = time.time()
                toml_filepath = recording_day['calibrations']['toml_filepath']
                self.objects["triangulation_recordings_objects"][recording].run_triangulation(
                    calibration_toml_filepath=toml_filepath,
                    triangulate_full_recording=triangulate_full_recording,
                )
                recording_day["recordings"][recording]["3D_csv"] = str(
                    self.objects["triangulation_recordings_objects"][
                        recording
                    ].csv_output_filepath
                )
                recording_day["recordings"][recording]["reprojerr_mean"] = \
                float(self.objects["triangulation_recordings_objects"][recording].anipose_io["reproj_nonan"].mean())
                recording_day["recordings"][recording]["excluded_cams"] = self.objects["triangulation_recordings_objects"][recording].cams_to_exclude
                if verbose:
                    end_time_recording = time.time()
                    duration = end_time_recording - start_time_recording
                    print(
                        f"The analysis of this recording {recording} took {duration}."
                    )
        self.meta["meta_step"] = 7
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def normalize_recordings(self, normalization_config_path: Union[Path, str], save_dataframe: bool = True, verbose: bool = False) -> None:
        """
        Run the function normalize for all TriangulationRecordings objects and
        saves the normalisation metadata.

        Parameters
        ----------
        normalization_config_path: Path or str
            The path to the config used for normalisation.
        save_dataframe: bool, default True
            If True (default), then the dataframe will be saved and overwrites
            the pre-existing one.
        verbose: bool, default False
            If True (default False), then the rotation visualization plot is shown.
        """
        normalization_config_path = convert_to_path(normalization_config_path)
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                if verbose:
                    print(f"\nRotation plot for {recording}:")
                rotated_filepath, rotation_error = self.objects["triangulation_recordings_objects"][
                    recording
                ].normalize(normalization_config_path=normalization_config_path, save_dataframe=save_dataframe, verbose=verbose)
                recording_day["recordings"][recording]["normalised_3D_csv"] = str(rotated_filepath)
                recording_day["recordings"][recording]["normalisation_rotation_error"] = float(rotation_error)
        self.meta["meta_step"] = 8
        self.export_meta_to_yaml(self.standard_yaml_filepath)

    def add_triangulated_csv_to_database(
            self, data_base_path: Union[str, Path], overwrite: bool = True
    ) -> None:
        """
        Add the 3D dataframes to a common data_base.

        Parameters
        ----------
        data_base_path: str or Path
            The path to the data_base, to which the 3D df metadata will be added.
        overwrite: bool, default True
            If True (default), then metadata for recordings in the MetaInterface,
            that were already added to the data_base, will be overwritten.
        """
        data_base_path = convert_to_path(data_base_path)
        data_base = pd.read_csv(data_base_path, dtype="str")
        for recording_day in self.meta["recording_days"].values():
            for recording in recording_day["recordings"]:
                filename = self.objects["triangulation_recordings_objects"][
                    recording
                ].csv_output_filepath.stem
                if filename in data_base["recording"].unique() and not overwrite:
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

    def export_meta_to_yaml(self, filepath: Union[str, Path]) -> None:
        """
        Store MetaInterface objects as .yaml-file.

        Parameters
        ----------
        filepath: str or Path
            The path, where the meta .yaml-file should be saved.
        """
        filepath = convert_to_path(filepath)
        with open(filepath, "w") as file:
            yaml.dump(self.meta, file)

    def _read_project_config(self) -> List[str]:
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
        }
        if intrinsics:
            dictionary["intrinsic_calibration_filepath"] = str(
                video.intrinsic_calibration_filepath
            )
        return dictionary

    def _create_standard_yaml_filepath(self, project_name: str, overwrite: bool) -> Tuple[str, Path]:
        if project_name is None:
            project_name = "My_project"
        standard_yaml_filepath = self.project_config_filepath.parent.joinpath(
            project_name + ".yaml"
        )
        while True:
            if standard_yaml_filepath.exists() and overwrite is False:
                standard_yaml_filepath = (
                    self.project_config_filepath.parent.joinpath(
                        standard_yaml_filepath.stem + "_01.yaml"
                    )
                )
            else:
                break
        return project_name, standard_yaml_filepath
