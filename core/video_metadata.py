from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod

from pathlib import Path
import datetime
import pickle
import itertools as it
import imageio.v3 as iio
import numpy as np
import aniposelib as ap_lib
import cv2
import matplotlib.pyplot as plt
import yaml

from .camera_intrinsics import (
    IntrinsicCalibratorFisheyeCamera,
    IntrinsicCalibratorRegularCameraCharuco,
)
from .utils import load_single_frame_of_video


class VideoMetadata:
    def __init__(
        self,
        video_filepath: Path,
        recording_config_filepath: Path,
        project_config_filepath: Path,
        calibration_dir: Path,
        max_calibration_frames: int = 300,
        load_calibration=False,
    ) -> None:
        self.calibration_dir = calibration_dir
        if (
            video_filepath.suffix == ".mp4"
            or video_filepath.suffix == ".AVI"
            or video_filepath.suffix == ".jpg"
        ):
            self.filepath = video_filepath
        else:
            raise ValueError("The filepath is not linked to a video or image.")
        if (
            not recording_config_filepath.exists()
            and recording_config_filepath.suffix == ".yaml"
        ):
            raise ValueError(
                "The recording_config_filepath is not linked to a .yaml-file"
            )
        if (
            not project_config_filepath.exists()
            and project_config_filepath.suffix == ".yaml"
        ):
            raise (
                f"Could not find a project_config_file at {project_config_filepath}\n Please make sure the path is correct, the file exists and is a .yaml file!"
            )

        self._read_metadata(
            recording_config_filepath=recording_config_filepath,
            project_config_filepath=project_config_filepath,
            video_filepath=video_filepath,
        )
        self._get_intrinsic_parameters(
            recording_config_filepath=recording_config_filepath,
            max_calibration_frames=max_calibration_frames,
            load_calibration=load_calibration,
        )

    def _read_metadata(
        self,
        project_config_filepath: Path,
        recording_config_filepath: Path,
        video_filepath: Path,
    ) -> None:
        with open(project_config_filepath, "r") as ymlfile:
            project_config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        with open(recording_config_filepath, "r") as ymlfile2:
            recording_config = yaml.load(ymlfile2, Loader=yaml.SafeLoader)

        for key in [
            "target_fps",
            "valid_cam_IDs",
            "paradigms",
            "animal_lines",
            "intrinsic_calibration_dir",
            "led_extraction_type",
            "led_extraction_path",
        ]:
            try:
                project_config[key]
            except KeyError:
                raise KeyError(
                    f"Missing metadata information in the project_config_file {project_config_filepath} for {key}."
                )

        self.target_fps = project_config["target_fps"]
        self.valid_cam_ids = project_config["valid_cam_IDs"]
        self.valid_paradigms = project_config["paradigms"]
        self.valid_mouse_lines = project_config["animal_lines"]
        self.intrinsic_calibrations_directory = Path(
            project_config["intrinsic_calibration_dir"]
        )

        self._extract_filepath_metadata(filepath_name=video_filepath.name)

        for key in ["led_pattern", self.cam_id]:
            try:
                recording_config[key]
            except KeyError:
                raise KeyError(
                    f"Missing information for {key} in the config_file {recording_config_filepath}!"
                )

        self.led_pattern = recording_config["led_pattern"]
        if self.recording_date != recording_config["recording_date"]:
            raise ValueError(
                f"The date of the recording_config_file {recording_config_filepath} and the provided video {self.video_filepath} do not match! Did you pass the right config-file and check the filename carefully?"
            )
        metadata_dict = recording_config[self.cam_id]

        for key in [
            "fps",
            "offset_row_idx",
            "offset_col_idx",
            "flip_h",
            "flip_v",
            "fisheye",
        ]:
            try:
                metadata_dict[key]
            except KeyError:
                raise KeyError(
                    f"Missing metadata information in the recording_config_file {recording_config_filepath} for {self.cam_id} for {key}."
                )

        self.fps = metadata_dict["fps"]
        self.offset_row_idx = metadata_dict["offset_row_idx"]
        self.offset_col_idx = metadata_dict["offset_col_idx"]
        self.flip_h = metadata_dict["flip_h"]
        self.flip_v = metadata_dict["flip_v"]
        self.fisheye = metadata_dict["fisheye"]

        self.processing_type = project_config["processing_type"][self.cam_id]
        self.calibration_evaluation_type = project_config[
            "calibration_evaluation_type"
        ][self.cam_id]
        self.processing_path = Path(project_config["processing_path"][self.cam_id])
        self.calibration_evaluation_path = Path(
            project_config["calibration_evaluation_path"][self.cam_id]
        )
        self.led_extraction_type = project_config["led_extraction_type"][self.cam_id]
        self.led_extraction_path = project_config["led_extraction_path"][self.cam_id]

    def _extract_filepath_metadata(self, filepath_name: str) -> None:
        if filepath_name[-4:] == ".AVI":
            filepath_name = filepath_name.replace(
                filepath_name[
                    filepath_name.index("00") : filepath_name.index("00") + 3
                ],
                "",
            )
            self.cam_id = "Top"

        if "Charuco" in filepath_name or "charuco" in filepath_name:
            self.charuco_video = True
            for piece in filepath_name[:-4].split("_"):
                if piece in self.valid_cam_ids:
                    self.cam_id = piece
                else:
                    try:
                        self.recording_date = datetime.date(
                            year=int("20" + piece[0:2]),
                            month=int(piece[2:4]),
                            day=int(piece[4:6]),
                        )
                    except ValueError:
                        pass

            for attribute in ["cam_id", "recording_date"]:
                if not hasattr(self, attribute):
                    self._check_attribute(attribute_to_check=attribute)

        elif "Positions" in filepath_name or "positions" in filepath_name:
            for piece in filepath_name[:-4].split("_"):
                if piece in self.valid_cam_ids:
                    self.cam_id = piece
                else:
                    try:
                        self.recording_date = datetime.date(
                            year=int("20" + piece[0:2]),
                            month=int(piece[2:4]),
                            day=int(piece[4:6]),
                        )
                    except ValueError:
                        pass

            for attribute in ["cam_id", "recording_date"]:
                if not hasattr(self, attribute):
                    self._check_attribute(attribute_to_check=attribute)

        else:
            self.charuco_video = False
            for piece in filepath_name[:-4].split("_"):
                if piece in self.valid_cam_ids:
                    self.cam_id = piece
                elif piece in self.valid_paradigms:
                    self.paradigm = piece
                elif piece in self.valid_mouse_lines:
                    self.mouse_line = piece
                elif piece.startswith("F"):
                    sub_pieces = piece.split("-")
                    if len(sub_pieces) == 2:
                        try:
                            int(sub_pieces[1])
                            self.mouse_number = piece
                        except ValueError:
                            pass
                else:
                    try:
                        self.recording_date = datetime.date(
                            year=int("20" + piece[0:2]),
                            month=int(piece[2:4]),
                            day=int(piece[4:6]),
                        )
                    except ValueError:
                        pass

            for attribute in [
                "cam_id",
                "recording_date",
                "paradigm",
                "mouse_line",
                "mouse_number",
            ]:
                if not hasattr(self, attribute):
                    self._check_attribute(attribute_to_check=attribute)
            self.mouse_id = self.mouse_line + "_" + self.mouse_number
        self.recording_date = self.recording_date.strftime("%y%m%d")

    def _check_attribute(self, attribute_to_check: str) -> None:
        # TODO: input or set_defaults?
        print(
            f"{attribute_to_check} could not be extracted automatically for the following file:\n"
            f"{self.filepath}"
        )

        messages = {}
        while True:
            entered_input = input(attribute_to_check + ": ")

            if attribute_to_check == "cam_id":
                if entered_input in self.valid_cam_ids:
                    self.cam_id = entered_input
                    break
                else:
                    messages[
                        "cam_id"
                    ] = f"Entered cam_id {entered_input} did not match any of the defined cam_ids. \nPlease enter one of the following ids: {self.valid_cam_ids}"
            if attribute_to_check == "recording_date":
                try:
                    self.recording_date = datetime.date(
                        year=int("20" + entered_input[0:2]),
                        month=int(entered_input[2:4]),
                        day=int(entered_input[4:6]),
                    )
                    break
                except ValueError:
                    messages[
                        "recording_date"
                    ] = f"Entered recording_date {entered_input} does not match the required structure for date. \nPlease enter the date as YYMMDD , e.g., 220928."
            if attribute_to_check == "paradigm":
                if entered_input in self.valid_paradigms:
                    self.paradigm = entered_input
                    break
                else:
                    messages[
                        "paradigm"
                    ] = f"Entered paradigm does not match any of the defined paradigms. \nPlease enter one of the following paradigms: {self.valid_paradigms}"
            if attribute_to_check == "mouse_line":
                if entered_input in self.valid_mouse_lines:
                    self.mouse_line = entered_input
                    break
                else:
                    messages[
                        "mouse_line"
                    ] = f"Entered mouse_line is not supported. \nPlease enter one of the following lines: {self.valid_mouse_lines}"
            if attribute_to_check == "mouse_number":
                sub_pieces = entered_input.split("-")
                if len(sub_pieces) == 2 and entered_input.startswith("F"):
                    try:
                        int(sub_pieces[1])
                        self.mouse_number = entered_input
                        break
                    except ValueError:
                        messages[
                            "mouse_number"
                        ] = f"Entered mouse_number does not match the required structure for a mouse_number. \n Please enter the mouse number as Generation-Number, e.g., F12-45"
                else:
                    messages[
                        "mouse_number"
                    ] = f"Entered mouse_number does not match the required structure for a mouse_number. \n Please enter the mouse number as Generation-Number, e.g., F12-45"
            print(messages[attribute_to_check])

    def _get_intrinsic_parameters(
        self,
        recording_config_filepath: Path,
        load_calibration: bool,
        max_calibration_frames: int,
    ) -> None:
        if self.charuco_video:
            if self.fisheye:
                try:
                    intrinsic_calibration_filepath = self.intrinsic_calibrations_directory.joinpath(
                        "Bottom_checkerboard_intrinsic_calibration_results.p"
                    )
                    with open(intrinsic_calibration_filepath, "rb") as io:
                        intrinsic_calibration = pickle.load(io)
                except FileNotFoundError:
                    try:
                        intrinsic_calibration_checkerboard_video_filepath = [
                            file
                            for file in self.intrinsic_calibrations_directory.iterdir()
                            if file.suffix == ".mp4"
                            and "checkerboard" in file.stem
                            and self.cam_id in file.stem
                        ][0]
                        calibrator = IntrinsicCalibratorFisheyeCamera(
                            filepath_calibration_video=intrinsic_calibration_checkerboard_video_filepath,
                            max_frame_count=max_calibration_frames,
                        )
                        intrinsic_calibration = calibrator.run()
                    except IndexError:
                        raise FileNotFoundError(
                            f"Could not find a filepath for an intrinsic calibration or a checkerboard video for {self.cam_id}.\nIt is required having a intrinsic_calibration .p file or a checkerboard video in the intrinsic_calibrations_directory ({self.intrinsic_calibrations_directory}) for a fisheye-camera!"
                        )
            else:
                if load_calibration:
                    try:
                        intrinsic_calibration_filepath = [
                            file
                            for file in self.intrinsic_calibrations_directory.iterdir()
                            if file.suffix == ".p" and self.cam_id in file.stem
                        ][0]
                        with open(intrinsic_calibration_filepath, "rb") as io:
                            intrinsic_calibration = pickle.load(io)
                    except IndexError:
                        raise FileNotFoundError(
                            f'Could not find an intrinsic calibration for {self.cam_id}! Use "load_calibration = False" to calibrate now!'
                        )
                else:
                    calibrator = IntrinsicCalibratorRegularCameraCharuco(
                        filepath_calibration_video=self.filepath,
                        max_frame_count=max_calibration_frames,
                    )
                    intrinsic_calibration = calibrator.run()
            self._save_calibration(intrinsic_calibration=intrinsic_calibration)
        else:
            try:
                intrinsic_calibration_filepath = [
                    file
                    for file in self.calibration_dir.iterdir()
                    if file.suffix == ".p" and self.cam_id in file.stem
                ][0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find an intrinsic calibration for {self.cam_id} in {self.calibration_dir.parent}! \nRunning the 3D Calibration should also create an intrinsic calibration. \nMake sure, you run the 3D Calibration before Triangulation."
                )
            with open(intrinsic_calibration_filepath, "rb") as io:
                intrinsic_calibration = pickle.load(io)
                

        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(
            unadjusted_intrinsic_calibration=intrinsic_calibration
        )
        self._set_intrinsic_calibration(
            intrinsic_calibration=intrinsic_calibration,
            adjusting_required=adjusting_required,
        )

        # ToDo:
        # We might need some methods to inspect the quality of the intrinsic calibration
        # after adjusting it to the cropped video, since flipping the video streams in
        # ICcapture has different effects, depending on whether it was applied right from
        # launching the software / cameras, or whether it was manually activated once the
        # software is running and the camera was loaded. This is at least our best guess
        # for the inconsistent behavior & warrants further testing.

    def _save_calibration(self, intrinsic_calibration: Dict) -> None:
        video_filename = self.filepath.name
        filename = f'{video_filename[:video_filename.rfind(".")]}_intrinsic_calibration_results.p'
        with open(self.calibration_dir.joinpath(filename), "wb") as io:
            pickle.dump(intrinsic_calibration, io)

    def _is_adjusting_of_intrinsic_calibration_required(
        self, unadjusted_intrinsic_calibration: Dict
    ) -> bool:
        adjusting_required = False
        if any(
            [
                self.offset_col_idx != 0,
                self.offset_row_idx != 0,
                self.flip_h,
                self.flip_v,
            ]
        ):
            adjusting_required = True
        return adjusting_required

    def _set_intrinsic_calibration(
        self, intrinsic_calibration: Dict, adjusting_required: bool
    ) -> None:
        if adjusting_required:
            intrinsic_calibration = self._adjust_intrinsic_calibration(
                unadjusted_intrinsic_calibration=intrinsic_calibration
            )
        setattr(self, "intrinsic_calibration", intrinsic_calibration)

    def _adjust_intrinsic_calibration(
        self, unadjusted_intrinsic_calibration: Dict
    ) -> Dict:
        adjusted_intrinsic_calibration = unadjusted_intrinsic_calibration.copy()
        intrinsic_calibration_video_size = unadjusted_intrinsic_calibration["size"]
        new_video_size = self._get_cropped_video_size()
        self._get_correct_x_y_offsets(
            intrinsic_calibration_video_size=intrinsic_calibration_video_size,
            new_video_size=new_video_size,
        )
        adjusted_K = self._get_adjusted_K(K=unadjusted_intrinsic_calibration["K"])
        adjusted_intrinsic_calibration = self._incorporate_adjustments_in_intrinsic_calibration(
            intrinsic_calibration=unadjusted_intrinsic_calibration.copy(),
            new_size=new_video_size,
            adjusted_K=adjusted_K,
        )
        return adjusted_intrinsic_calibration

    def _get_cropped_video_size(self) -> Tuple[int, int]:
        props = iio.improps(self.filepath, index=0)
        size = props.shape[0:2]
        return size

    def _get_correct_x_y_offsets(
        self,
        intrinsic_calibration_video_size: Tuple[int, int],
        new_video_size: Tuple[int, int],
    ) -> None:
        if self.flip_v:
            self.offset_row_idx = (
                intrinsic_calibration_video_size[0]
                - new_video_size[1]
                - self.offset_row_idx
            )
            # rows or cols first in intrinsic_calibration_video_size? (rows for now, but maybe this will be changed?)
        if self.flip_h:
            self.offset_col_idx = (
                intrinsic_calibration_video_size[1]
                - new_video_size[0]
                - self.offset_col_idx
            )

    def _get_adjusted_K(self, K: np.ndarray) -> np.ndarray:
        adjusted_K = K.copy()
        adjusted_K[0][2] = adjusted_K[0][2] - self.offset_row_idx
        adjusted_K[1][2] = adjusted_K[1][2] - self.offset_col_idx
        return adjusted_K

    def _incorporate_adjustments_in_intrinsic_calibration(
        self,
        intrinsic_calibration: Dict,
        new_size: Tuple[int, int],
        adjusted_K: np.ndarray,
    ) -> Dict:
        intrinsic_calibration["size"] = new_size
        intrinsic_calibration["K"] = adjusted_K
        return intrinsic_calibration
