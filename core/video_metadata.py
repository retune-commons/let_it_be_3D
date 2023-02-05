from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
from pathlib import Path
import datetime

import pickle
import imageio as iio
import numpy as np
import aniposelib as ap_lib
import cv2
import matplotlib.pyplot as plt
import yaml

from .camera_intrinsics import (
    IntrinsicCalibratorFisheyeCamera,
    IntrinsicCalibratorRegularCameraCharuco,
)
from .utils import load_single_frame_of_video, convert_to_path, check_keys, read_config


class VideoMetadata:
    def __init__(
        self,
        video_filepath: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict,
        tag: str
    ) -> None:
        self._get_video_identity(tag=tag)
        self.exclusion_state = "valid"

        self._check_filepaths(
            video_filepath=video_filepath,
        )

        state = self._read_metadata(
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            video_filepath=video_filepath,
        )
        
        self._get_intrinsic_parameters(
            max_calibration_frames=self.max_calibration_frames,
        )
        try:
            self.framenum = iio.v2.get_reader(video_filepath).count_frames()
        except:
            self.framenum = 0
            

    def _check_filepaths(
        self,
        video_filepath: Path,
    ) -> None:
        if (
            (
            video_filepath.suffix == ".mp4"
            or video_filepath.suffix == ".mov"
            or video_filepath.suffix == ".AVI"
            or video_filepath.suffix == ".avi"
            or video_filepath.suffix == ".jpg"
            or video_filepath.suffix == ".png"
            or video_filepath.suffix == ".tiff"
            or video_filepath.suffix == ".bmp"
            )
            and video_filepath.exists()
        ):
            self.filepath = video_filepath
        else:
            raise ValueError("The filepath is not linked to a video or image.")

    def _read_metadata(
        self,
        project_config_dict: Dict,
        recording_config_dict: Dict,
        video_filepath: Path,
    ) -> None:

        self.valid_cam_ids = project_config_dict["valid_cam_IDs"]
        self.valid_paradigms = project_config_dict["paradigms"]
        self.valid_mouse_lines = project_config_dict["animal_lines"]
        self.load_calibration = project_config_dict["load_calibration"]
        if self.load_calibration:
            try:
                self.intrinsic_calibrations_directory = Path(
                    project_config_dict["intrinsic_calibration_directory"]
                )
            except:
                raise ValueError(
                    "If you use load_calibration = True, you need to set an intrinsic calibrations directory!"
                )
        self.max_calibration_frames = project_config_dict["max_calibration_frames"]
        self.max_ram_digestible_frames = project_config_dict["max_ram_digestible_frames"]
        self.max_cpu_cores_to_pool = project_config_dict["max_cpu_cores_to_pool"]
        
        while True:
            undefined_attributes = self._extract_filepath_metadata(self.filepath)
            if len(undefined_attributes)>0:
                self._print_message(attributes = undefined_attributes)
                self._rename_file()
                if self.filepath.name == "x":
                    self.filepath.unlink()
                    return "del"
                if self.filepath.name == "y":
                    print(f"{video_filepath} needs to be moved!")
                    return "del"
            else:
                break
        self.recording_date = self.recording_date.strftime("%y%m%d")
        if self.recording:
            self.mouse_id = self.mouse_line + "_" + self.mouse_number

        self.led_pattern = recording_config_dict["led_pattern"]
        self.target_fps = recording_config_dict["target_fps"]
        self.calibration_index = recording_config_dict["calibration_index"]
        if self.recording_date != recording_config_dict["recording_date"]:
            raise ValueError(
                f"The date of the recording_config_file and the provided video {video_filepath} do not match! Did you pass the right config-file and check the filename carefully?"
            )

        metadata_dict = recording_config_dict[self.cam_id]
        keys_to_check = [
            "fps",
            "offset_row_idx",
            "offset_col_idx",
            "flip_h",
            "flip_v",
            "fisheye",
        ]
        missing_keys = check_keys(dictionary = metadata_dict, list_of_keys = keys_to_check)
        if len(missing_keys) > 0:
            raise KeyError(
                f"Missing metadata information in the recording_config_file for {self.cam_id} for {missing_keys}."
            )

        self.fps = metadata_dict["fps"]
        self.offset_row_idx = metadata_dict["offset_row_idx"]
        self.offset_col_idx = metadata_dict["offset_col_idx"]
        self.flip_h = metadata_dict["flip_h"]
        self.flip_v = metadata_dict["flip_v"]
        self.fisheye = metadata_dict["fisheye"]

        self.processing_type = project_config_dict["processing_type"][self.cam_id]
        self.calibration_evaluation_type = project_config_dict[
            "calibration_evaluation_type"
        ][self.cam_id]
        self.processing_filepath = Path(
            project_config_dict["processing_filepath"][self.cam_id]
        )
        self.calibration_evaluation_filepath = Path(
            project_config_dict["calibration_evaluation_filepath"][self.cam_id]
        )
        self.led_extraction_type = project_config_dict["led_extraction_type"][self.cam_id]
        self.led_extraction_filepath = project_config_dict["led_extraction_filepath"][
            self.cam_id
        ]
        return "valid"
    
    def _get_video_identity(self, tag: str)->None:
        self.charuco_video = False
        self.positions = False
        self.recording = False
        if tag == "calibration":
            self.charuco_video = True
        elif tag == "positions":
            self.positions = True
        elif tag == "recording":
            self.recording = True

    def _extract_filepath_metadata(self, filepath_name: Path) -> List:
        if filepath_name.suffix == ".AVI":
            try:
                filepath_name = Path(filepath_name.name.replace(
                    filepath_name.name[
                        filepath_name.name.index("00") : filepath_name.name.index("00") + 3
                    ],
                    "",
                ))
            except:
                pass
            self.cam_id = "Top"

        if self.charuco_video:
            for piece in filepath_name.stem.split("_"):
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
                    raise ValueError(f"{attribute} was not found in {self.filepath}! Rename the path manually or use the filename_checker!")

        elif self.positions:
            for piece in filepath_name.stem.split("_"):
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
                    raise ValueError(f"{attribute} was not found in {self.filepath}! Rename the path manually or use the filename_checker!")

        elif self.recording:
            for piece in filepath_name.stem.split("_"):
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
                    raise ValueError(f"{attribute} was not found in {self.filepath}! Rename the path manually or use the filename_checker!")
        return []
                    
    def _get_intrinsic_parameters(
        self,
        max_calibration_frames: int,
    ) -> None:
        if self.fisheye:
            try:
                intrinsic_calibration_filepath = [file for file in self.intrinsic_calibrations_directory.iterdir() if file.suffix == ".p" and self.cam_id in file.stem][0]
                with open(intrinsic_calibration_filepath, "rb") as io:
                    intrinsic_calibration = pickle.load(io)
            except IndexError:
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
                        max_calibration_frames=self.max_calibration_frames,
                    )
                    intrinsic_calibration = calibrator.run()
                except IndexError:
                    raise FileNotFoundError(
                        f"Could not find a filepath for an intrinsic calibration or a checkerboard video for {self.cam_id}.\nIt is required having a intrinsic_calibration .p file or a checkerboard video in the intrinsic_calibrations_directory ({self.intrinsic_calibrations_directory}) for a fisheye-camera!"
                    )
        else:
            if self.load_calibration:
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
                        f'Could not find an intrinsic calibration for {self.cam_id}! Use "load_calibration = False" in project_config to calibrate now!'
                    )
            else:
                raise NotImplementedError("currently not working!")
                """
                calibrator = IntrinsicCalibratorRegularCameraCharuco(
                    filepath_calibration_video=self.filepath,
                    max_calibration_frames=self.max_calibration_frames,
                )
                intrinsic_calibration = calibrator.run()
                with open(intrinsic_calibration_filepath, "rb") as io:
                    intrinsic_calibration = pickle.load(io)
                """     
            
        self.intrinsic_calibration_filepath = intrinsic_calibration_filepath

        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(
            unadjusted_intrinsic_calibration=intrinsic_calibration
        )
        self._set_intrinsic_calibration(
            intrinsic_calibration=intrinsic_calibration,
            adjusting_required=adjusting_required,
        )

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
        adjusted_intrinsic_calibration = (
            self._incorporate_adjustments_in_intrinsic_calibration(
                intrinsic_calibration=unadjusted_intrinsic_calibration.copy(),
                new_size=new_video_size,
                adjusted_K=adjusted_K,
            )
        )
        return adjusted_intrinsic_calibration

    def _get_cropped_video_size(self) -> Tuple[int, int]:
        try:
            size = iio.v3.immeta(self.filepath, exclude_applied=False)["size"]
        except KeyError:
            size = iio.v3.immeta(self.filepath, exclude_applied=False)["shape"]
        return size

    def _get_correct_x_y_offsets(
        self,
        intrinsic_calibration_video_size: Tuple[int, int],
        new_video_size: Tuple[int, int],
    ) -> None:
        if self.flip_v:
            self.offset_row_idx = (
                intrinsic_calibration_video_size[1]
                - new_video_size[0]
                - self.offset_row_idx
            )
            # rows or cols first in intrinsic_calibration_video_size? (rows for now, but maybe this will be changed?)
        if self.flip_h:
            self.offset_col_idx = (
                intrinsic_calibration_video_size[0]
                - new_video_size[1]
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
    
    
class VideoMetadataChecker(VideoMetadata):
    def __init__(
        self,
        video_filepath: Path,
        recording_config_dict: Dict,
        project_config_dict: Dict,
        tag: str
    ) -> None:
        self._get_video_identity(tag=tag)
        self._check_filepaths(
            video_filepath=video_filepath
        )

        state = self._read_metadata(
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            video_filepath=video_filepath,
        )
        
        if state == "del":
            return state
        else:
            self._get_intrinsic_parameters(
                max_calibration_frames=self.max_calibration_frames,
            )
            
            
    def _extract_filepath_metadata(self, filepath_name: Path) -> List[str]:
        undefined_attributes = []
        if filepath_name.suffix == ".AVI":
            try:
                filepath_name = Path(filepath_name.name.replace(
                    filepath_name.name[
                        filepath_name.name.index("00") : filepath_name.name.index("00") + 3
                    ],
                    "",
                ))
            except:
                pass
            self.cam_id = "Top"

        if self.charuco_video:
            for piece in filepath_name.stem.split("_"):
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
                    undefined_attributes.append(attribute)

        elif self.positions:
            for piece in filepath_name.stem.split("_"):
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
                    undefined_attributes.append(attribute)

        elif self.recording:
            for piece in filepath_name.stem.split("_"):
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
                    undefined_attributes.append(attribute)
        
        return undefined_attributes
    
    def _print_message(self, attributes: List[str])->None:
        print(
                f"The information {attributes} could not be extracted automatically from the following file:\n"
                f"{self.filepath}"
            )
        for attribute in attributes:
            if attribute == "cam_id":
                print(f"Cam_id was not found in filename or did not match any of the defined cam_ids. \nPlease include one of the following ids into the filename: {self.valid_cam_ids} or add the cam_id to valid_cam_ids!")
            elif attribute == "recording_date":
                print(f"Recording_date was not found in filename or did not match the required structure for date. \nPlease include the date as YYMMDD , e.g., 220928, into the filename!")
            elif atribute == "paradigm":
                f"Paradigm was not found in filename or did not match any of the defined paradigms. \nPlease Please include one of the following paradigms into the filename: {self.valid_paradigms} or add the paradigm to valid_paradigmes!"
            elif attribute == "mouse_line":
                print(f"Mouse_line was not found in filename or is not supported. \nPlease include one of the following lines into the filename: {self.valid_mouse_lines} or add the line to valid_mouse_lines!")
            elif attribute == "mouse_number":
                print("Mouse_number was not found in filename or did not match the required structure for a mouse_number. \n Please include the mouse number as Generation-Number, e.g., F12-45, into the filename!")
    
    def _rename_file(self) -> None:
        suffix = self.filepath.suffix
        new_filename = Path(input(f"Enter new filename! \nIf the video is invalid, enter x and it will be deleted!\n If the video belongs to another folder, enter y, and move it manually!\n{self.filepath.parent}/"))
        new_filepath = self.filepath.parent.joinpath(new_filename.with_suffix(suffix))
        if new_filepath == self.filepath:
            print("The entered filename and the real filename are identical.")
        elif new_filepath.exists():
            print("Couldn't rename file, since the entered filename does already exist.")
        else:
            self.filepath.rename(new_filepath)
            self.filepath = new_filepath